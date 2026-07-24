package signal

import (
	"context"
	"fmt"
	"regexp"
	"strings"
)

// SchemaOptions configures EnsureSchema.
type SchemaOptions struct {
	// Database is the dedicated ClickHouse database the signal plane owns
	// (NOT the host app's analytics database). Required.
	Database string

	// Cluster, when set, renders ON CLUSTER DDL and Replicated* engines
	// (ZooKeeper/Keeper paths under /clickhouse/tables/{database}/<table>).
	// Leave empty for single-node ClickHouse.
	Cluster string
}

var identRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Exec is the minimal ClickHouse execution surface EnsureSchema needs;
// clickhouse-go's driver.Conn satisfies it.
type Exec interface {
	Exec(ctx context.Context, query string, args ...any) error
}

// EnsureSchema creates the signal-plane database, tables, and materialized
// view. Idempotent (IF NOT EXISTS everywhere); safe to run at startup. The
// connection must have DDL privileges.
//
// Tables:
//   - signal_events  — append-only event stream, one row per session or
//     interaction; ReplacingMergeTree deduplicates replayed event_ids.
//   - signal_state   — durable current-state projection (no TTL), one row per
//     (tenant, subject, entity_type, entity_id); ReplacingMergeTree(last_updated).
//   - entity_daily   — daily per-entity rollup (AggregatingMergeTree) feeding
//     popularity/trending over fixed windows and arbitrary date slices; tiny
//     (one row/entity/day), retained long-term.
//   - mv_entity_daily — materialized view from signal_events into entity_daily.
//   - search_impressions — one row per SERP/shelf render (the shown entity refs
//     and their positions); clicks link back to it via a query_id carried on the
//     click signal's attribution payload. Feeds learned-ranking training data.
func EnsureSchema(ctx context.Context, conn Exec, opts SchemaOptions) error {
	db := strings.TrimSpace(opts.Database)
	if db == "" {
		return fmt.Errorf("signal: SchemaOptions.Database is required")
	}
	if !identRe.MatchString(db) {
		return fmt.Errorf("signal: invalid database name %q", db)
	}
	cluster := strings.TrimSpace(opts.Cluster)
	if cluster != "" && !identRe.MatchString(cluster) {
		return fmt.Errorf("signal: invalid cluster name %q", cluster)
	}

	onCluster := ""
	if cluster != "" {
		onCluster = fmt.Sprintf(" ON CLUSTER '%s'", cluster)
	}
	engine := func(kind, args string) string {
		// kind: "ReplacingMergeTree" | "AggregatingMergeTree" | "MergeTree"
		if cluster == "" {
			if args == "" {
				return kind + "()"
			}
			return fmt.Sprintf("%s(%s)", kind, args)
		}
		path := "'/clickhouse/tables/{database}/{table}', '{replica}'"
		if args == "" {
			return fmt.Sprintf("Replicated%s(%s)", kind, path)
		}
		return fmt.Sprintf("Replicated%s(%s, %s)", kind, path, args)
	}

	stmts := []string{
		fmt.Sprintf("CREATE DATABASE IF NOT EXISTS %s%s", db, onCluster),

		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s.signal_events%s (
    tenant LowCardinality(String),
    entity_type LowCardinality(String),
    entity_id String,
    subject_kind LowCardinality(String),
    subject String,
    signal_type LowCardinality(String),
    event_id String,
    occurred_at DateTime('UTC'),
    duration_s UInt32 DEFAULT 0,
    progress UInt32 DEFAULT 0,
    progress_max UInt32 DEFAULT 0,
    value Float64 DEFAULT 0,
    label LowCardinality(String) DEFAULT '',
    weight Float64 DEFAULT 1,
    score Int16 DEFAULT 0,
    completed Bool DEFAULT false,
    resume String DEFAULT '',
    payload String DEFAULT '',
    recorded_at DateTime('UTC') DEFAULT now(),
    INDEX idx_signal_events_subject (subject_kind, subject) TYPE bloom_filter GRANULARITY 4
) ENGINE = %s
PARTITION BY toYYYYMM(occurred_at)
ORDER BY (tenant, entity_type, entity_id, subject_kind, subject, occurred_at, event_id)
SETTINGS index_granularity = 8192`, db, onCluster, engine("ReplacingMergeTree", "recorded_at")),

		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s.signal_state%s (
    tenant LowCardinality(String),
    subject_kind LowCardinality(String),
    subject String,
    entity_type LowCardinality(String),
    entity_id String,
    first_seen_at DateTime('UTC'),
    last_signal_at DateTime('UTC'),
    total_events UInt32 DEFAULT 0,
    max_progress UInt32 DEFAULT 0,
    progress_max UInt32 DEFAULT 0,
    completed Bool DEFAULT false,
    resume String DEFAULT '',
    has_interacted Bool DEFAULT false,
    last_score Int16 DEFAULT 0,
    net_value Float64 DEFAULT 0,
    last_updated DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = %s
ORDER BY (tenant, subject_kind, subject, entity_type, entity_id)
SETTINGS index_granularity = 8192`, db, onCluster, engine("ReplacingMergeTree", "last_updated")),

		// Idempotent column addition for deployments created before net_value.
		fmt.Sprintf(`ALTER TABLE %s.signal_state%s ADD COLUMN IF NOT EXISTS net_value Float64 DEFAULT 0 AFTER last_score`, db, onCluster),

		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s.entity_daily%s (
    tenant LowCardinality(String),
    entity_type LowCardinality(String),
    entity_id String,
    day Date,
    subjects AggregateFunction(uniqExact, String),
    signals SimpleAggregateFunction(sum, UInt64),
    engagement_sum SimpleAggregateFunction(sum, Int64),
    scored_signals SimpleAggregateFunction(sum, UInt64),
    completions SimpleAggregateFunction(sum, UInt64),
    value_sum SimpleAggregateFunction(sum, Float64),
    signal_counts SimpleAggregateFunction(sumMap, Map(String, UInt64))
) ENGINE = %s
PARTITION BY toYear(day)
ORDER BY (tenant, entity_type, entity_id, day)
SETTINGS index_granularity = 8192`, db, onCluster, engine("AggregatingMergeTree", "")),

		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s.item_pairs%s (
    tenant LowCardinality(String),
    entity_type_a LowCardinality(String),
    entity_id_a String,
    entity_type_b LowCardinality(String),
    entity_id_b String,
    strength Int64,
    refreshed_at DateTime('UTC') DEFAULT now()
) ENGINE = %s
ORDER BY (tenant, entity_type_a, entity_id_a, entity_type_b, entity_id_b)
SETTINGS index_granularity = 8192`, db, onCluster, engine("ReplacingMergeTree", "refreshed_at")),

		// One row per SERP/shelf render. The shown items are stored as parallel
		// arrays (entity ref + its position), never one row per item. query_id is
		// unique per render and is what click signals carry to link back.
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s.search_impressions%s (
    tenant LowCardinality(String),
    query_id String,
    surface LowCardinality(String) DEFAULT 'search',
    normalized_query String DEFAULT '',
    language LowCardinality(String) DEFAULT '',
    subject_kind LowCardinality(String) DEFAULT '',
    subject String DEFAULT '',
    shown_entity_types Array(LowCardinality(String)),
    shown_entity_ids Array(String),
    shown_positions Array(UInt32),
    occurred_at DateTime('UTC'),
    recorded_at DateTime('UTC') DEFAULT now()
) ENGINE = %s
PARTITION BY toYYYYMM(occurred_at)
ORDER BY (tenant, occurred_at, query_id)
SETTINGS index_granularity = 8192`, db, onCluster, engine("ReplacingMergeTree", "recorded_at")),

		fmt.Sprintf(`CREATE MATERIALIZED VIEW IF NOT EXISTS %s.mv_entity_daily%s TO %s.entity_daily AS
SELECT
    tenant,
    entity_type,
    entity_id,
    toDate(occurred_at) AS day,
    uniqExactState(concat(subject_kind, ':', subject)) AS subjects,
    toUInt64(count()) AS signals,
    toInt64(sum(score)) AS engagement_sum,
    toUInt64(countIf(score != 0)) AS scored_signals,
    toUInt64(countIf(completed)) AS completions,
    sum(value) AS value_sum,
    sumMap(map(signal_type, toUInt64(1))) AS signal_counts
FROM %s.signal_events
GROUP BY tenant, entity_type, entity_id, day`, db, onCluster, db, db),
	}

	for _, stmt := range stmts {
		if err := conn.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("signal: schema setup failed: %w\nstatement:\n%s", err, stmt)
		}
	}
	return nil
}
