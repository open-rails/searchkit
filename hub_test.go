package searchkit

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	chdriver "github.com/ClickHouse/clickhouse-go/v2/lib/driver"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/open-rails/searchkit/signal"
)

// --- fakes ---

type hubFakeConn struct {
	execs   []hubCapturedCall
	queries []hubCapturedCall
	rowsFor map[string][][]any // query substring -> rows
}

type hubCapturedCall struct {
	query string
	args  []any
}

func (f *hubFakeConn) Exec(_ context.Context, query string, args ...any) error {
	f.execs = append(f.execs, hubCapturedCall{query: query, args: args})
	return nil
}

func (f *hubFakeConn) Query(_ context.Context, query string, args ...any) (chdriver.Rows, error) {
	f.queries = append(f.queries, hubCapturedCall{query: query, args: args})
	for sub, rows := range f.rowsFor {
		if strings.Contains(query, sub) {
			return &hubFakeRows{rows: rows}, nil
		}
	}
	return &hubFakeRows{}, nil
}

type hubFakeRows struct {
	rows [][]any
	idx  int
	cur  []any
}

func (r *hubFakeRows) Next() bool {
	if r.idx >= len(r.rows) {
		return false
	}
	r.cur = r.rows[r.idx]
	r.idx++
	return true
}

func (r *hubFakeRows) Scan(dest ...any) error {
	if len(dest) != len(r.cur) {
		return fmt.Errorf("hubFakeRows: scan %d dests, row has %d values", len(dest), len(r.cur))
	}
	for i, d := range dest {
		dv := reflect.ValueOf(d)
		sv := reflect.ValueOf(r.cur[i])
		if !sv.Type().AssignableTo(dv.Elem().Type()) {
			return fmt.Errorf("hubFakeRows: dest %d: cannot assign %s to %s", i, sv.Type(), dv.Elem().Type())
		}
		dv.Elem().Set(sv)
	}
	return nil
}

func (r *hubFakeRows) HasData() bool                      { return len(r.rows) > 0 }
func (r *hubFakeRows) ScanStruct(any) error               { return fmt.Errorf("not implemented") }
func (r *hubFakeRows) ColumnTypes() []chdriver.ColumnType { return nil }
func (r *hubFakeRows) Totals(...any) error                { return fmt.Errorf("not implemented") }
func (r *hubFakeRows) Columns() []string                  { return nil }
func (r *hubFakeRows) Close() error                       { return nil }
func (r *hubFakeRows) Err() error                         { return nil }

func lazyPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	// pgxpool connects lazily; these unit tests never touch Postgres.
	pool, err := pgxpool.New(context.Background(), "postgres://unused:unused@127.0.0.1:9/unused")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func newTestHub(t *testing.T, fc *hubFakeConn, mutate func(*EmbeddedConfig)) *EmbeddedHub {
	t.Helper()
	cfg := EmbeddedConfig{
		PG:       lazyPool(t),
		PGSchema: "hub",
		Tenant:   "doujins",
	}
	if fc != nil {
		cfg.CH = fc
		cfg.CHDatabase = "hub"
	}
	if mutate != nil {
		mutate(&cfg)
	}
	h, err := NewEmbedded(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// --- tests ---

func TestHubSignalPlaneDisabled(t *testing.T) {
	h := newTestHub(t, nil, nil)
	ctx := context.Background()
	sub := signal.Subject{UserID: "u1"}

	if err := h.RecordSignal(ctx, signal.Signal{}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("RecordSignal: %v", err)
	}
	if _, err := h.History(ctx, sub, signal.HistoryOptions{}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("History: %v", err)
	}
	if _, err := h.States(ctx, sub, nil); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("States: %v", err)
	}
	if _, err := h.Engagement(ctx, signal.EntityRef{}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("Engagement: %v", err)
	}
	if _, err := h.Popular(ctx, "gallery", signal.PopularOptions{}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("Popular: %v", err)
	}
	if _, err := h.Unseen(ctx, sub, UnseenOptions{EntityType: "gallery"}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("Unseen: %v", err)
	}
	if _, err := h.Recommend(ctx, sub, RecommendOptions{EntityTypes: []string{"gallery"}}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("Recommend: %v", err)
	}
	if _, err := h.Search(ctx, "query", HubSearchOptions{Personalize: &Personalization{Subject: sub}}); !errors.Is(err, ErrSignalPlaneDisabled) {
		t.Fatalf("personalized Search: %v", err)
	}
}

func TestHubDefaultTenant(t *testing.T) {
	h := newTestHub(t, nil, func(c *EmbeddedConfig) { c.Tenant = "" })
	if h.Tenant() != "default" {
		t.Fatalf("tenant: %q", h.Tenant())
	}
}

func TestHubRecordSignalAppliesScorer(t *testing.T) {
	fc := &hubFakeConn{}
	h := newTestHub(t, fc, func(c *EmbeddedConfig) {
		c.Scorers = map[string]signal.Scorer{
			"blog_post": signal.ScorerFunc(func(_ context.Context, s signal.Signal) (signal.Scored, error) {
				// e.g. read-time + scroll-depth scoring.
				return signal.Scored{Score: 77, Progress: 95, ProgressMax: 100, Completed: true}, nil
			}),
		}
	})

	err := h.RecordSignal(context.Background(), signal.Signal{
		EntityRef:  signal.EntityRef{EntityType: "blog_post", EntityID: "42"},
		Subject:    signal.Subject{UserID: "u1"},
		Type:       "view",
		OccurredAt: time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC),
		Progress:   95, ProgressMax: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(fc.execs) != 2 {
		t.Fatalf("expected event insert + state reproject, got %d execs", len(fc.execs))
	}
	args := fc.execs[0].args
	// ... value, label, weight, score, completed, resume, payload
	if args[14] != int16(77) || args[15] != true {
		t.Fatalf("scorer result not applied to insert args: %v", args)
	}
	if args[0] != "doujins" {
		t.Fatalf("tenant not pinned: %v", args[0])
	}
}

func TestHubRecordSignalScorerError(t *testing.T) {
	fc := &hubFakeConn{}
	h := newTestHub(t, fc, func(c *EmbeddedConfig) {
		c.Scorers = map[string]signal.Scorer{
			"blog_post": signal.ScorerFunc(func(context.Context, signal.Signal) (signal.Scored, error) {
				return signal.Scored{}, fmt.Errorf("boom")
			}),
		}
	})
	err := h.RecordSignal(context.Background(), signal.Signal{
		EntityRef: signal.EntityRef{EntityType: "blog_post", EntityID: "42"},
		Subject:   signal.Subject{UserID: "u1"},
		Type:      "view",
	})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("scorer error must propagate: %v", err)
	}
	if len(fc.execs) != 0 {
		t.Fatal("nothing must be recorded when the scorer fails")
	}
}

func TestHubUnseenDiffsUniverseAgainstSeen(t *testing.T) {
	fc := &hubFakeConn{rowsFor: map[string][][]any{
		"max_progress > 0": {{"b"}, {"d"}}, // SeenIDs
	}}
	universeCalls := 0
	h := newTestHub(t, fc, func(c *EmbeddedConfig) {
		c.Catalogs = map[string]EntityCatalog{
			"gallery": EntityCatalogFunc(func(_ context.Context, tenant, entityType string, q CatalogQuery) ([]string, error) {
				universeCalls++
				if tenant != "doujins" || entityType != "gallery" {
					return nil, fmt.Errorf("unexpected catalog call %s/%s", tenant, entityType)
				}
				return []string{"a", "b", "c", "d", "e"}, nil
			}),
		}
	})

	got, err := h.Unseen(context.Background(), signal.Subject{UserID: "u1"}, UnseenOptions{EntityType: "gallery"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"a", "c", "e"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unseen: got %v want %v", got, want)
	}
	if universeCalls != 1 {
		t.Fatalf("universe calls: %d", universeCalls)
	}

	// Limit respected, universe order preserved.
	got, err = h.Unseen(context.Background(), signal.Subject{UserID: "u1"}, UnseenOptions{EntityType: "gallery", Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"a", "c"}) {
		t.Fatalf("unseen limited: got %v", got)
	}

	// Unregistered type errors.
	if _, err := h.Unseen(context.Background(), signal.Subject{UserID: "u1"}, UnseenOptions{EntityType: "video"}); err == nil {
		t.Fatal("missing catalog must error")
	}
}

func TestHubRecommendColdStartFallsBackToPopular(t *testing.T) {
	fc := &hubFakeConn{rowsFor: map[string][][]any{
		// TopStates + SeenIDs both hit signal_state and return nothing
		// (cold user). Popular returns ranked entities.
		"entity_daily": {
			{"g1", uint64(10), uint64(12), uint64(8), float64(3.5)},
			{"g2", uint64(5), uint64(6), uint64(1), float64(2.0)},
		},
	}}
	h := newTestHub(t, fc, nil)

	got, err := h.Recommend(context.Background(), signal.Subject{UserID: "newbie"}, RecommendOptions{
		EntityTypes: []string{"gallery"},
		Limit:       5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].EntityID != "g1" || got[1].EntityID != "g2" {
		t.Fatalf("cold-start recs: %+v", got)
	}
	if got[0].Score <= got[1].Score {
		t.Fatalf("fill scores must decay: %+v", got)
	}
}

func TestHubRecommendUsesCoEngagementAndExcludesSeen(t *testing.T) {
	now := time.Now().UTC()
	stateRow := func(id string, score int16) []any {
		return []any{"gallery", id, now, now, uint32(3), uint32(10), uint32(10), true, "", true, score}
	}
	fc := &hubFakeConn{rowsFor: map[string][][]any{
		// TopStates -> one strong seed.
		"ORDER BY last_score DESC": {stateRow("seed1", int16(90))},
		// SeenIDs -> the subject has seen gSeen (and the seed).
		"max_progress > 0": {{"gSeen"}, {"seed1"}},
		// CoEngaged for the seed.
		"NOT (entity_type = ? AND entity_id = ?)": {
			{"gallery", "gNew", uint64(7)},
			{"gallery", "gSeen", uint64(5)},
		},
	}}
	h := newTestHub(t, fc, nil) // no DefaultModel -> vector source skipped

	got, err := h.Recommend(context.Background(), signal.Subject{UserID: "u1"}, RecommendOptions{
		EntityTypes: []string{"gallery"},
		Limit:       2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].EntityID != "gNew" {
		t.Fatalf("recs: %+v", got)
	}
	for _, r := range got {
		if r.EntityID == "gSeen" || r.EntityID == "seed1" {
			t.Fatalf("seen/seed leaked into recs: %+v", got)
		}
	}
}

func TestHubRecommendRequiresEntityTypes(t *testing.T) {
	h := newTestHub(t, &hubFakeConn{}, nil)
	if _, err := h.Recommend(context.Background(), signal.Subject{UserID: "u"}, RecommendOptions{}); err == nil {
		t.Fatal("EntityTypes must be required")
	}
}

func TestHubSimilarToCoEngagementOnly(t *testing.T) {
	fc := &hubFakeConn{rowsFor: map[string][][]any{
		"NOT (entity_type = ? AND entity_id = ?)": {
			{"gallery", "g2", uint64(4)},
			{"gallery", "g3", uint64(2)},
		},
	}}
	h := newTestHub(t, fc, nil)

	got, err := h.SimilarTo(context.Background(), "gallery", "g1", HubSimilarOptions{CoEngagement: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].EntityID != "g2" || got[1].EntityID != "g3" {
		t.Fatalf("similar: %+v", got)
	}

	// Without a model and without co-engagement there is no source.
	if _, err := h.SimilarTo(context.Background(), "gallery", "g1", HubSimilarOptions{}); err == nil {
		t.Fatal("no source must error")
	}
}
