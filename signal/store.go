package signal

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
)

// Conn is the minimal ClickHouse surface the Store needs; clickhouse-go's
// driver.Conn satisfies it.
type Conn interface {
	Exec(ctx context.Context, query string, args ...any) error
	Query(ctx context.Context, query string, args ...any) (driver.Rows, error)
}

// Store reads and writes the signal plane for one ClickHouse database.
// Every method is tenant-scoped via its tenant argument; the embedded hub
// pins a single tenant value.
type Store struct {
	conn Conn
	db   string
}

// NewStore returns a Store over an existing ClickHouse connection. Run
// EnsureSchema (once, with DDL privileges) before first use.
func NewStore(conn Conn, database string) (*Store, error) {
	if conn == nil {
		return nil, fmt.Errorf("signal: conn is required")
	}
	db := strings.TrimSpace(database)
	if db == "" {
		return nil, fmt.Errorf("signal: database is required")
	}
	if !identRe.MatchString(db) {
		return nil, fmt.Errorf("signal: invalid database name %q", db)
	}
	return &Store{conn: conn, db: db}, nil
}

// RecordSignal appends one event and reprojects the durable current-state row
// for that (subject, entity). The projection is recomputed ClickHouse-side
// from the (deduplicated) event stream, so replaying an event converges to
// identical state instead of double-counting.
func (st *Store) RecordSignal(ctx context.Context, tenant string, s Signal) error {
	if strings.TrimSpace(tenant) == "" {
		return fmt.Errorf("signal: tenant is required")
	}
	if err := s.validate(); err != nil {
		return err
	}
	if s.OccurredAt.IsZero() {
		s.OccurredAt = time.Now().UTC()
	}
	if s.Weight == 0 {
		s.Weight = 1
	}
	payload := ""
	if len(s.Payload) > 0 {
		b, err := json.Marshal(s.Payload)
		if err != nil {
			return fmt.Errorf("signal: marshal payload: %w", err)
		}
		payload = string(b)
	}

	insert := fmt.Sprintf(`INSERT INTO %s.signal_events
(tenant, entity_type, entity_id, subject_kind, subject, signal_type, event_id, occurred_at,
 duration_s, progress, progress_max, value, label, weight, score, completed, resume, payload)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, st.db)
	if err := st.conn.Exec(ctx, insert,
		tenant, s.EntityType, s.EntityID, s.Subject.Kind(), s.Subject.Key(),
		s.Type, s.eventID(), s.OccurredAt.UTC(),
		s.DurationS, s.Progress, s.ProgressMax, s.Value, s.Label, s.Weight,
		s.Score, s.Completed, s.Resume, payload,
	); err != nil {
		return fmt.Errorf("signal: insert event: %w", err)
	}

	return st.reprojectState(ctx, tenant, s.Subject, s.EntityRef)
}

// reprojectState recomputes the current-state row for one (subject, entity)
// key by aggregating its events. Deterministic and replay-idempotent:
// total_events deduplicates by event_id, the rest are min/max/argMax.
func (st *Store) reprojectState(ctx context.Context, tenant string, subject Subject, ref EntityRef) error {
	q := fmt.Sprintf(`INSERT INTO %s.signal_state
(tenant, subject_kind, subject, entity_type, entity_id, first_seen_at, last_signal_at,
 total_events, max_progress, progress_max, completed, resume, has_interacted, last_score, last_updated)
SELECT
    tenant, subject_kind, subject, entity_type, entity_id,
    min(occurred_at),
    max(occurred_at),
    toUInt32(uniqExact(event_id)),
    max(progress),
    max(progress_max),
    max(completed),
    argMaxIf(resume, occurred_at, resume != ''),
    max(value != 0 OR label != ''),
    argMax(score, occurred_at),
    now64(3)
FROM %s.signal_events
WHERE tenant = ? AND entity_type = ? AND entity_id = ? AND subject_kind = ? AND subject = ?
GROUP BY tenant, subject_kind, subject, entity_type, entity_id`, st.db, st.db)
	if err := st.conn.Exec(ctx, q, tenant, ref.EntityType, ref.EntityID, subject.Kind(), subject.Key()); err != nil {
		return fmt.Errorf("signal: reproject state: %w", err)
	}
	return nil
}

const stateColumns = `entity_type, entity_id, first_seen_at, last_signal_at, total_events,
 max_progress, progress_max, completed, resume, has_interacted, last_score`

func scanStateRow(rows driver.Rows) (StateRow, error) {
	var (
		r         StateRow
		completed bool
		inter     bool
	)
	if err := rows.Scan(
		&r.EntityType, &r.EntityID, &r.FirstSeenAt, &r.LastSignalAt, &r.TotalEvents,
		&r.MaxProgress, &r.ProgressMax, &completed, &r.Resume, &inter, &r.LastScore,
	); err != nil {
		return StateRow{}, err
	}
	r.Completed = completed
	r.HasInteracted = inter
	r.Seen = r.MaxProgress > 0
	return r, nil
}

// States is the bulk "annotate this list with view context" read: for each
// requested entity, the subject's standing (seen?, progress bar, completed?,
// resume pointer). Entities the subject has no signals for are absent from
// the result map. A point/IN lookup on current-state — cheap per page of
// cards.
func (st *Store) States(ctx context.Context, tenant string, subject Subject, refs []EntityRef) (map[EntityRef]State, error) {
	if err := subject.Validate(); err != nil {
		return nil, err
	}
	out := make(map[EntityRef]State, len(refs))
	if len(refs) == 0 {
		return out, nil
	}

	// Group ids by entity type: (entity_type = ? AND entity_id IN ?) OR ...
	byType := map[string][]string{}
	for _, r := range refs {
		if err := r.validate(); err != nil {
			return nil, err
		}
		byType[r.EntityType] = append(byType[r.EntityType], r.EntityID)
	}
	clauses := make([]string, 0, len(byType))
	args := []any{tenant, subject.Kind(), subject.Key()}
	for _, t := range sortedKeys(byType) {
		clauses = append(clauses, "(entity_type = ? AND entity_id IN ?)")
		args = append(args, t, byType[t])
	}

	q := fmt.Sprintf(`SELECT %s
FROM %s.signal_state FINAL
WHERE tenant = ? AND subject_kind = ? AND subject = ? AND (%s)`,
		stateColumns, st.db, strings.Join(clauses, " OR "))

	rows, err := st.conn.Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("signal: states: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		r, err := scanStateRow(rows)
		if err != nil {
			return nil, fmt.Errorf("signal: states scan: %w", err)
		}
		out[r.EntityRef] = r.State
	}
	return out, rows.Err()
}

// History returns the subject's current-state rows, most recent signal first.
func (st *Store) History(ctx context.Context, tenant string, subject Subject, opts HistoryOptions) ([]StateRow, error) {
	if err := subject.Validate(); err != nil {
		return nil, err
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	var sb strings.Builder
	args := []any{tenant, subject.Kind(), subject.Key()}
	fmt.Fprintf(&sb, `SELECT %s
FROM %s.signal_state FINAL
WHERE tenant = ? AND subject_kind = ? AND subject = ?`, stateColumns, st.db)
	if t := strings.TrimSpace(opts.EntityType); t != "" {
		sb.WriteString(" AND entity_type = ?")
		args = append(args, t)
	}
	switch opts.Status {
	case HistoryAny:
	case HistorySeen:
		sb.WriteString(" AND max_progress > 0")
	case HistoryInProgress:
		sb.WriteString(" AND max_progress > 0 AND NOT completed")
	case HistoryCompleted:
		sb.WriteString(" AND completed")
	default:
		return nil, fmt.Errorf("signal: invalid HistoryOptions.Status %q", opts.Status)
	}
	sb.WriteString(" ORDER BY last_signal_at DESC, entity_type ASC, entity_id ASC LIMIT ? OFFSET ?")
	args = append(args, limit, opts.Offset)

	rows, err := st.conn.Query(ctx, sb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("signal: history: %w", err)
	}
	defer rows.Close()
	out := make([]StateRow, 0, limit)
	for rows.Next() {
		r, err := scanStateRow(rows)
		if err != nil {
			return nil, fmt.Errorf("signal: history scan: %w", err)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// SeenIDs returns the subject's seen-set for one entity type: entity ids with
// max_progress > 0. This is the signal-plane half of the unseen anti-join;
// the caller diffs it against the host catalog's universe.
func (st *Store) SeenIDs(ctx context.Context, tenant string, subject Subject, entityType string) (map[string]struct{}, error) {
	if err := subject.Validate(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(entityType) == "" {
		return nil, fmt.Errorf("signal: entityType is required")
	}
	q := fmt.Sprintf(`SELECT entity_id
FROM %s.signal_state FINAL
WHERE tenant = ? AND subject_kind = ? AND subject = ? AND entity_type = ? AND max_progress > 0`, st.db)
	rows, err := st.conn.Query(ctx, q, tenant, subject.Kind(), subject.Key(), entityType)
	if err != nil {
		return nil, fmt.Errorf("signal: seen ids: %w", err)
	}
	defer rows.Close()
	out := map[string]struct{}{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("signal: seen ids scan: %w", err)
		}
		out[id] = struct{}{}
	}
	return out, rows.Err()
}

// TopStates returns the subject's highest-signal entities (recommendation
// seeds): ordered by last_score DESC, then recency.
func (st *Store) TopStates(ctx context.Context, tenant string, subject Subject, opts TopStatesOptions) ([]StateRow, error) {
	if err := subject.Validate(); err != nil {
		return nil, err
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	var sb strings.Builder
	args := []any{tenant, subject.Kind(), subject.Key()}
	fmt.Fprintf(&sb, `SELECT %s
FROM %s.signal_state FINAL
WHERE tenant = ? AND subject_kind = ? AND subject = ?`, stateColumns, st.db)
	if types := trimAll(opts.EntityTypes); len(types) > 0 {
		sb.WriteString(" AND entity_type IN ?")
		args = append(args, types)
	}
	sb.WriteString(" ORDER BY last_score DESC, last_signal_at DESC, entity_type ASC, entity_id ASC LIMIT ?")
	args = append(args, limit)

	rows, err := st.conn.Query(ctx, sb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("signal: top states: %w", err)
	}
	defer rows.Close()
	out := make([]StateRow, 0, limit)
	for rows.Next() {
		r, err := scanStateRow(rows)
		if err != nil {
			return nil, fmt.Errorf("signal: top states scan: %w", err)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// Engagement aggregates signal quality for one entity over the event stream.
func (st *Store) Engagement(ctx context.Context, tenant string, ref EntityRef) (EntityEngagement, error) {
	if err := ref.validate(); err != nil {
		return EntityEngagement{}, err
	}
	q := fmt.Sprintf(`SELECT
    uniqExactIf(subject, subject_kind = 'user') AS unique_users,
    uniqExactIf(subject, subject_kind = 'anon') AS unique_anon,
    uniqExact(event_id) AS signals,
    uniqExactIf(concat(subject_kind, ':', subject), completed) AS completions,
    uniqExact(concat(subject_kind, ':', subject)) AS subjects_total,
    ifNotFinite(avgIf(score, score != 0), 0) AS avg_score,
    sumMap(map(signal_type, toUInt64(1))) AS signal_counts
FROM %s.signal_events
WHERE tenant = ? AND entity_type = ? AND entity_id = ?`, st.db)

	rows, err := st.conn.Query(ctx, q, tenant, ref.EntityType, ref.EntityID)
	if err != nil {
		return EntityEngagement{}, fmt.Errorf("signal: engagement: %w", err)
	}
	defer rows.Close()
	var (
		e             EntityEngagement
		subjectsTotal uint64
	)
	if rows.Next() {
		if err := rows.Scan(
			&e.UniqueUsers, &e.UniqueAnon, &e.Signals,
			&e.Completions, &subjectsTotal, &e.AvgScore, &e.SignalCounts,
		); err != nil {
			return EntityEngagement{}, fmt.Errorf("signal: engagement scan: %w", err)
		}
	}
	if subjectsTotal > 0 {
		e.CompletionRate = float64(e.Completions) / float64(subjectsTotal)
	}
	return e, rows.Err()
}

// Popular ranks entities of one type over a time window. Day-aligned windows
// (and all-time) merge the tiny entity_daily rollup; sub-day windows scan the
// month-partitioned event stream.
//
// IDs, when non-empty, restricts ranking to that candidate set (used for
// signal-aware re-ranking of search results).
func (st *Store) Popular(ctx context.Context, tenant string, entityType string, opts PopularOptions) ([]PopularHit, error) {
	return st.popular(ctx, tenant, entityType, nil, opts)
}

// PopularityFor scores a fixed candidate set by the popularity ranking and
// returns entity_id -> score. Candidates with no signals in the window are
// absent.
func (st *Store) PopularityFor(ctx context.Context, tenant string, entityType string, ids []string, window Window) (map[string]float64, error) {
	ids = trimAll(ids)
	if len(ids) == 0 {
		return map[string]float64{}, nil
	}
	hits, err := st.popular(ctx, tenant, entityType, ids, PopularOptions{Window: window, Limit: len(ids)})
	if err != nil {
		return nil, err
	}
	out := make(map[string]float64, len(hits))
	for _, h := range hits {
		out[h.EntityID] = h.Score
	}
	return out, nil
}

func (st *Store) popular(ctx context.Context, tenant string, entityType string, ids []string, opts PopularOptions) ([]PopularHit, error) {
	if strings.TrimSpace(entityType) == "" {
		return nil, fmt.Errorf("signal: entityType is required")
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	w := opts.Weights.withDefaults()
	rankExpr := strings.TrimSpace(opts.RankExpr)
	useDecay := rankExpr == "" && w.HalfLifeDays > 0

	var (
		q    string
		args []any
	)
	switch {
	case !opts.Window.dayAligned():
		q, args = st.popularFromEvents(tenant, entityType, ids, opts.Window, rankExpr, w, limit)
	case useDecay:
		q, args = st.popularDecayed(tenant, entityType, ids, opts.Window, w, limit)
	default:
		q, args = st.popularFromRollup(tenant, entityType, ids, opts.Window, rankExpr, w, limit)
	}

	rows, err := st.conn.Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("signal: popular: %w", err)
	}
	defer rows.Close()
	out := make([]PopularHit, 0, limit)
	for rows.Next() {
		h := PopularHit{EntityRef: EntityRef{EntityType: entityType}}
		if err := rows.Scan(&h.EntityID, &h.Subjects, &h.Signals, &h.Completions, &h.Score); err != nil {
			return nil, fmt.Errorf("signal: popular scan: %w", err)
		}
		if math.IsNaN(h.Score) || math.IsInf(h.Score, 0) {
			h.Score = 0
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// defaultRankExpr builds the default ranking over the aggregate aliases
// subjects / engagement_sum / scored_signals (see RankWeights).
func defaultRankExpr(w RankWeights, volumeExpr string) string {
	return fmt.Sprintf(
		"log10(1 + %s) * greatest(%g, (toFloat64(engagement_sum) + %g) / (toFloat64(scored_signals) + %g))",
		volumeExpr, w.QualityFloor, w.PriorScore*w.PriorWeight, w.PriorWeight,
	)
}

func (st *Store) popularFromRollup(tenant, entityType string, ids []string, win Window, rankExpr string, w RankWeights, limit int) (string, []any) {
	if rankExpr == "" {
		rankExpr = defaultRankExpr(w, "toFloat64(subjects)")
	}
	var sb strings.Builder
	args := []any{tenant, entityType}
	// Output aliases must not collide with the source columns (subjects /
	// signals / completions): ClickHouse expands same-named aliases inside
	// the rank expression, nesting the merge combinators.
	fmt.Fprintf(&sb, `SELECT
    entity_id,
    uniqExactMerge(subjects) AS out_subjects,
    toUInt64(sum(signals)) AS out_signals,
    toUInt64(sum(completions)) AS out_completions,
    (%s) AS rank_score
FROM %s.entity_daily
WHERE tenant = ? AND entity_type = ?`, rewriteRankAliases(rankExpr, map[string]string{
		"subjects":       "uniqExactMerge(subjects)",
		"signals":        "toUInt64(sum(signals))",
		"engagement_sum": "toInt64(sum(engagement_sum))",
		"scored_signals": "toUInt64(sum(scored_signals))",
		"completions":    "toUInt64(sum(completions))",
	}), st.db)
	if len(ids) > 0 {
		sb.WriteString(" AND entity_id IN ?")
		args = append(args, ids)
	}
	if !win.From.IsZero() {
		sb.WriteString(" AND day >= toDate(?)")
		args = append(args, win.From.UTC())
	}
	if !win.To.IsZero() {
		sb.WriteString(" AND day <= toDate(?)")
		args = append(args, win.To.UTC())
	}
	sb.WriteString("\nGROUP BY entity_id\nORDER BY rank_score DESC, entity_id ASC\nLIMIT ?")
	args = append(args, limit)
	return sb.String(), args
}

func (st *Store) popularDecayed(tenant, entityType string, ids []string, win Window, w RankWeights, limit int) (string, []any) {
	// Per-day uniq subjects, decayed by age before the log; exact uniq across
	// the window is still reported via the merged state.
	rank := fmt.Sprintf(
		"log10(1 + decayed_volume) * greatest(%g, (toFloat64(engagement_sum) + %g) / (toFloat64(scored_signals) + %g))",
		w.QualityFloor, w.PriorScore*w.PriorWeight, w.PriorWeight,
	)
	var sb strings.Builder
	args := []any{tenant, entityType}
	fmt.Fprintf(&sb, `SELECT
    entity_id,
    uniqExactMerge(subjects_state) AS subjects,
    toUInt64(sum(day_signals)) AS signals,
    toUInt64(sum(day_completions)) AS completions,
    (%s) AS rank_score
FROM (
    SELECT
        entity_id,
        day,
        uniqExactMergeState(subjects) AS subjects_state,
        uniqExactMerge(subjects) AS day_subjects,
        sum(signals) AS day_signals,
        sum(engagement_sum) AS day_engagement,
        sum(scored_signals) AS day_scored,
        sum(completions) AS day_completions
    FROM %s.entity_daily
    WHERE tenant = ? AND entity_type = ?`,
		rewriteRankAliases(rank, map[string]string{
			"decayed_volume": fmt.Sprintf("sum(toFloat64(day_subjects) * exp2(-toFloat64(dateDiff('day', day, toDate(now()))) / %g))", w.HalfLifeDays),
			"engagement_sum": "toInt64(sum(day_engagement))",
			"scored_signals": "toUInt64(sum(day_scored))",
		}), st.db)
	if len(ids) > 0 {
		sb.WriteString(" AND entity_id IN ?")
		args = append(args, ids)
	}
	if !win.From.IsZero() {
		sb.WriteString(" AND day >= toDate(?)")
		args = append(args, win.From.UTC())
	}
	if !win.To.IsZero() {
		sb.WriteString(" AND day <= toDate(?)")
		args = append(args, win.To.UTC())
	}
	sb.WriteString(`
    GROUP BY entity_id, day
)
GROUP BY entity_id
ORDER BY rank_score DESC, entity_id ASC
LIMIT ?`)
	args = append(args, limit)
	return sb.String(), args
}

func (st *Store) popularFromEvents(tenant, entityType string, ids []string, win Window, rankExpr string, w RankWeights, limit int) (string, []any) {
	if rankExpr == "" {
		rankExpr = defaultRankExpr(w, "toFloat64(subjects)")
	}
	var sb strings.Builder
	args := []any{tenant, entityType}
	fmt.Fprintf(&sb, `SELECT
    entity_id,
    uniqExact(concat(subject_kind, ':', subject)) AS subjects,
    uniqExact(event_id) AS signals,
    toUInt64(countIf(completed)) AS completions,
    (%s) AS rank_score
FROM %s.signal_events
WHERE tenant = ? AND entity_type = ?`, rewriteRankAliases(rankExpr, map[string]string{
		"subjects":       "uniqExact(concat(subject_kind, ':', subject))",
		"signals":        "uniqExact(event_id)",
		"engagement_sum": "toInt64(sum(score))",
		"scored_signals": "toUInt64(countIf(score != 0))",
		"completions":    "toUInt64(countIf(completed))",
	}), st.db)
	if len(ids) > 0 {
		sb.WriteString(" AND entity_id IN ?")
		args = append(args, ids)
	}
	if !win.From.IsZero() {
		sb.WriteString(" AND occurred_at >= ?")
		args = append(args, win.From.UTC())
	}
	if !win.To.IsZero() {
		sb.WriteString(" AND occurred_at <= ?")
		args = append(args, win.To.UTC())
	}
	sb.WriteString("\nGROUP BY entity_id\nORDER BY rank_score DESC, entity_id ASC\nLIMIT ?")
	args = append(args, limit)
	return sb.String(), args
}

// rewriteRankAliases substitutes whole-word aggregate aliases in a rank
// expression with their concrete aggregate expressions, so the same host
// expression works over both the rollup and the raw event paths.
func rewriteRankAliases(expr string, subs map[string]string) string {
	for _, alias := range sortedKeys(subs) {
		expr = replaceWord(expr, alias, subs[alias])
	}
	return expr
}

// replaceWord replaces whole-identifier occurrences of word in expr.
func replaceWord(expr, word, with string) string {
	isIdent := func(b byte) bool {
		return b == '_' || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
	}
	var out strings.Builder
	for i := 0; i < len(expr); {
		j := strings.Index(expr[i:], word)
		if j < 0 {
			out.WriteString(expr[i:])
			break
		}
		j += i
		before := j == 0 || !isIdent(expr[j-1])
		afterIdx := j + len(word)
		after := afterIdx >= len(expr) || !isIdent(expr[afterIdx])
		if before && after {
			out.WriteString(expr[i:j])
			out.WriteString(with)
		} else {
			out.WriteString(expr[i:afterIdx])
		}
		i = afterIdx
	}
	return out.String()
}

// CoEngaged returns entities co-engaged with the anchor: "subjects who
// engaged with X also engaged with Y", strength = co-engaged subject count.
// Works without a logged-in viewer; feeds the "more like this" fusion.
func (st *Store) CoEngaged(ctx context.Context, tenant string, ref EntityRef, opts CoEngagedOptions) ([]CoEngagedHit, error) {
	if err := ref.validate(); err != nil {
		return nil, err
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	maxSubjects := opts.MaxSubjects
	if maxSubjects <= 0 {
		maxSubjects = 10000
	}

	var inner strings.Builder
	innerArgs := []any{tenant, ref.EntityType, ref.EntityID}
	fmt.Fprintf(&inner, `SELECT DISTINCT subject_kind, subject FROM %s.signal_events
WHERE tenant = ? AND entity_type = ? AND entity_id = ?`, st.db)
	if !opts.Window.From.IsZero() {
		inner.WriteString(" AND occurred_at >= ?")
		innerArgs = append(innerArgs, opts.Window.From.UTC())
	}
	if !opts.Window.To.IsZero() {
		inner.WriteString(" AND occurred_at <= ?")
		innerArgs = append(innerArgs, opts.Window.To.UTC())
	}
	inner.WriteString(" LIMIT ?")
	innerArgs = append(innerArgs, maxSubjects)

	var sb strings.Builder
	args := []any{tenant}
	fmt.Fprintf(&sb, `SELECT entity_type, entity_id, uniqExact(concat(subject_kind, ':', subject)) AS strength
FROM %s.signal_events
WHERE tenant = ?
  AND (subject_kind, subject) IN (%s)
  AND NOT (entity_type = ? AND entity_id = ?)`, st.db, inner.String())
	args = append(args, innerArgs...)
	args = append(args, ref.EntityType, ref.EntityID)
	if types := trimAll(opts.EntityTypes); len(types) > 0 {
		sb.WriteString(" AND entity_type IN ?")
		args = append(args, types)
	}
	if !opts.Window.From.IsZero() {
		sb.WriteString(" AND occurred_at >= ?")
		args = append(args, opts.Window.From.UTC())
	}
	if !opts.Window.To.IsZero() {
		sb.WriteString(" AND occurred_at <= ?")
		args = append(args, opts.Window.To.UTC())
	}
	sb.WriteString("\nGROUP BY entity_type, entity_id\nORDER BY strength DESC, entity_type ASC, entity_id ASC\nLIMIT ?")
	args = append(args, limit)

	rows, err := st.conn.Query(ctx, sb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("signal: co-engaged: %w", err)
	}
	defer rows.Close()
	out := make([]CoEngagedHit, 0, limit)
	for rows.Next() {
		var h CoEngagedHit
		if err := rows.Scan(&h.EntityType, &h.EntityID, &h.Strength); err != nil {
			return nil, fmt.Errorf("signal: co-engaged scan: %w", err)
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

func trimAll(in []string) []string {
	out := make([]string, 0, len(in))
	seen := map[string]struct{}{}
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
