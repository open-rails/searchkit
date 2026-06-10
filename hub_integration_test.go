package searchkit

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/open-rails/searchkit/signal"
	"github.com/pgvector/pgvector-go"
)

// Round-trip integration test for the embedded hub against a real Postgres
// AND ClickHouse. Opt-in: requires both SEARCHKIT_TEST_URL (PG DSN) and
// SEARCHKIT_TEST_CH_ADDR (CH native addr). Owns the PG schema and CH database
// named below.
const (
	hubTestPGSchema = "searchkit_hub_test"
	hubTestCHDB     = "searchkit_hub_test"
)

func TestHubIntegrationRoundTrip(t *testing.T) {
	dsn := os.Getenv("SEARCHKIT_TEST_URL")
	chAddr := os.Getenv("SEARCHKIT_TEST_CH_ADDR")
	if dsn == "" || chAddr == "" {
		t.Skip("SEARCHKIT_TEST_URL / SEARCHKIT_TEST_CH_ADDR not set")
	}
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool: %v", err)
	}
	defer pool.Close()

	chUser := os.Getenv("SEARCHKIT_TEST_CH_USER")
	if chUser == "" {
		chUser = "default"
	}
	ch, err := clickhouse.Open(&clickhouse.Options{
		Addr: []string{chAddr},
		Auth: clickhouse.Auth{Username: chUser, Password: os.Getenv("SEARCHKIT_TEST_CH_PASSWORD")},
	})
	if err != nil {
		t.Fatalf("clickhouse open: %v", err)
	}
	defer ch.Close()

	// --- Postgres content plane (minimal schema, like client tests) ---
	_, err = pool.Exec(ctx, fmt.Sprintf(`
		DROP SCHEMA IF EXISTS %[1]s CASCADE;
		CREATE SCHEMA %[1]s;
		CREATE EXTENSION IF NOT EXISTS pg_trgm;
		CREATE EXTENSION IF NOT EXISTS vector;

		CREATE OR REPLACE FUNCTION %[1]s.searchkit_regconfig_for_language(lang text)
		RETURNS regconfig LANGUAGE sql IMMUTABLE AS $$ SELECT 'simple'::regconfig $$;

		CREATE TABLE %[1]s.search_documents (
			entity_type text NOT NULL,
			entity_id text NOT NULL,
			language text NOT NULL,
			raw_document text,
			tsv tsvector,
			created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (entity_type, entity_id, language)
		);
		CREATE TABLE %[1]s.embedding_vectors (
			entity_type text NOT NULL,
			entity_id text NOT NULL,
			model text NOT NULL,
			language text NOT NULL,
			embedding halfvec,
			created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (entity_type, entity_id, model, language)
		);
	`, hubTestPGSchema))
	if err != nil {
		t.Fatalf("pg setup: %v", err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), "DROP SCHEMA IF EXISTS "+hubTestPGSchema+" CASCADE")
	})

	docs := []struct {
		id, text string
		vec      []float32
	}{
		{"g1", "two factor authentication guide", []float32{1, 0, 0}},
		{"g2", "two factor backup codes", []float32{0.9, 0.1, 0}},
		{"g3", "cooking with cast iron", []float32{0, 1, 0}},
	}
	for _, d := range docs {
		if _, err := pool.Exec(ctx, fmt.Sprintf(`
			INSERT INTO %[1]s.search_documents(entity_type, entity_id, language, raw_document, tsv)
			VALUES ('gallery', $1, 'en', $2, to_tsvector(%[1]s.searchkit_regconfig_for_language('en'), $2))`, hubTestPGSchema),
			d.id, d.text); err != nil {
			t.Fatalf("insert doc %s: %v", d.id, err)
		}
		if _, err := pool.Exec(ctx, fmt.Sprintf(`
			INSERT INTO %[1]s.embedding_vectors(entity_type, entity_id, model, language, embedding)
			VALUES ('gallery', $1, 'm', 'en', $2::halfvec(3))`, hubTestPGSchema),
			d.id, pgvector.NewHalfVector(d.vec)); err != nil {
			t.Fatalf("insert vec %s: %v", d.id, err)
		}
	}

	// --- ClickHouse signal plane ---
	if err := ch.Exec(ctx, "DROP DATABASE IF EXISTS "+hubTestCHDB); err != nil {
		t.Fatalf("drop ch db: %v", err)
	}
	if err := signal.EnsureSchema(ctx, ch, signal.SchemaOptions{Database: hubTestCHDB}); err != nil {
		t.Fatalf("ensure ch schema: %v", err)
	}
	t.Cleanup(func() {
		_ = ch.Exec(context.Background(), "DROP DATABASE IF EXISTS "+hubTestCHDB)
	})

	// --- The hub ---
	emb := &recordingEmbedder{vec: []float32{1, 0, 0}}
	hub, err := NewEmbedded(EmbeddedConfig{
		PG:           pool,
		PGSchema:     hubTestPGSchema,
		Embedder:     emb,
		DefaultModel: "m",
		CH:           ch,
		CHDatabase:   hubTestCHDB,
		Tenant:       "doujins",
		Scorers: map[string]signal.Scorer{
			"gallery": signal.ScorerFunc(func(_ context.Context, s signal.Signal) (signal.Scored, error) {
				score := int16(0)
				if s.ProgressMax > 0 {
					score = int16(100 * s.Progress / s.ProgressMax)
				}
				return signal.Scored{
					Score:       score,
					Progress:    s.Progress,
					ProgressMax: s.ProgressMax,
					Completed:   s.ProgressMax > 0 && 10*s.Progress >= 9*s.ProgressMax,
				}, nil
			}),
		},
		Catalogs: map[string]EntityCatalog{
			"gallery": EntityCatalogFunc(func(context.Context, string, string, CatalogQuery) ([]string, error) {
				return []string{"g3", "g2", "g1"}, nil // newest first
			}),
		},
	})
	if err != nil {
		t.Fatalf("NewEmbedded: %v", err)
	}

	var _ Hub = hub

	user := signal.Subject{UserID: "u1"}
	day := func(d, h int) time.Time { return time.Date(2026, 6, d, h, 0, 0, 0, time.UTC) }

	// Record signals: u1 completes g1; many anons view g2 (popular).
	if err := hub.RecordSignal(ctx, signal.Signal{
		EntityRef:  signal.EntityRef{EntityType: "gallery", EntityID: "g1"},
		Subject:    user,
		Type:       "view",
		OccurredAt: day(1, 10),
		Progress:   20, ProgressMax: 20,
		Resume: "p:20",
	}); err != nil {
		t.Fatalf("RecordSignal: %v", err)
	}
	for i := 0; i < 8; i++ {
		if err := hub.RecordSignal(ctx, signal.Signal{
			EntityRef:  signal.EntityRef{EntityType: "gallery", EntityID: "g2"},
			Subject:    signal.Subject{AnonKey: fmt.Sprintf("a%d", i)},
			Type:       "view",
			OccurredAt: day(2, 9+i%6),
			Progress:   18, ProgressMax: 20,
		}); err != nil {
			t.Fatalf("RecordSignal anon: %v", err)
		}
	}

	// Scorer applied: g1 state must be completed with score 100.
	states, err := hub.States(ctx, user, []signal.EntityRef{{EntityType: "gallery", EntityID: "g1"}})
	if err != nil {
		t.Fatalf("States: %v", err)
	}
	g1 := states[signal.EntityRef{EntityType: "gallery", EntityID: "g1"}]
	if !g1.Completed || g1.LastScore != 100 || g1.Resume != "p:20" {
		t.Fatalf("scorer not applied: %+v", g1)
	}

	// History.
	hist, err := hub.History(ctx, user, signal.HistoryOptions{EntityType: "gallery"})
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	if len(hist) != 1 || hist[0].EntityID != "g1" {
		t.Fatalf("history: %+v", hist)
	}

	// Unseen: u1 saw g1 -> g3, g2 remain (catalog order).
	unseen, err := hub.Unseen(ctx, user, UnseenOptions{EntityType: "gallery"})
	if err != nil {
		t.Fatalf("Unseen: %v", err)
	}
	if len(unseen) != 2 || unseen[0] != "g3" || unseen[1] != "g2" {
		t.Fatalf("unseen: %v", unseen)
	}

	// Engagement on g2.
	eng, err := hub.Engagement(ctx, signal.EntityRef{EntityType: "gallery", EntityID: "g2"})
	if err != nil {
		t.Fatalf("Engagement: %v", err)
	}
	if eng.UniqueAnon != 8 || eng.Signals != 8 {
		t.Fatalf("engagement: %+v", eng)
	}

	// Popular: g2 must lead (8 subjects vs 1).
	pop, err := hub.Popular(ctx, "gallery", signal.PopularOptions{Limit: 10})
	if err != nil {
		t.Fatalf("Popular: %v", err)
	}
	if len(pop) < 2 || pop[0].EntityID != "g2" {
		t.Fatalf("popular: %+v", pop)
	}

	// Plain search: content ranking only.
	plain, err := hub.Search(ctx, "two factor", HubSearchOptions{SearchOptions: SearchOptions{
		Language:    "en",
		EntityTypes: []string{"gallery"},
		Limit:       10,
	}})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(plain) < 2 {
		t.Fatalf("plain search: %+v", plain)
	}

	// Personalized search: popularity blend + demote-seen. g1 is completed
	// by u1 and g2 is popular, so g2 must outrank g1.
	pers, err := hub.Search(ctx, "two factor", HubSearchOptions{
		SearchOptions: SearchOptions{
			Language:    "en",
			EntityTypes: []string{"gallery"},
			Limit:       10,
		},
		Personalize: &Personalization{
			Subject:          user,
			PopularityWeight: 1,
			DemoteSeen:       true,
			PopularityWindow: signal.AllTime(),
		},
	})
	if err != nil {
		t.Fatalf("personalized Search: %v", err)
	}
	rank := map[string]int{}
	for i, h := range pers {
		rank[h.EntityID] = i
	}
	if rank["g2"] > rank["g1"] {
		t.Fatalf("personalization must rank popular g2 above completed g1: %+v", pers)
	}

	// SimilarTo with co-engagement enabled still returns vector neighbours.
	sim, err := hub.SimilarTo(ctx, "gallery", "g1", HubSimilarOptions{
		SimilarOptions: SimilarOptions{Language: "en"},
		CoEngagement:   true,
	})
	if err != nil {
		t.Fatalf("SimilarTo: %v", err)
	}
	if len(sim) == 0 || sim[0].EntityID != "g2" {
		t.Fatalf("similar: %+v", sim)
	}
	for _, s := range sim {
		if s.EntityID == "g1" {
			t.Fatalf("anchor leaked into similar: %+v", sim)
		}
	}

	// Recommend: u1's seed is g1 -> vector + co-engagement suggest g2; but
	// fall back gracefully too. g1 (seen) must never be recommended.
	recs, err := hub.Recommend(ctx, user, RecommendOptions{
		EntityTypes: []string{"gallery"},
		Limit:       5,
		Language:    "en",
	})
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if len(recs) == 0 {
		t.Fatal("expected recommendations")
	}
	if recs[0].EntityID != "g2" {
		t.Fatalf("g2 should lead recs: %+v", recs)
	}
	for _, r := range recs {
		if r.EntityID == "g1" {
			t.Fatalf("seen entity recommended: %+v", recs)
		}
	}

	// Replay idempotency through the hub: re-record u1's g1 session.
	if err := hub.RecordSignal(ctx, signal.Signal{
		EntityRef:  signal.EntityRef{EntityType: "gallery", EntityID: "g1"},
		Subject:    user,
		Type:       "view",
		OccurredAt: day(1, 10),
		Progress:   20, ProgressMax: 20,
		Resume: "p:20",
	}); err != nil {
		t.Fatalf("replay RecordSignal: %v", err)
	}
	states, err = hub.States(ctx, user, []signal.EntityRef{{EntityType: "gallery", EntityID: "g1"}})
	if err != nil {
		t.Fatalf("States after replay: %v", err)
	}
	if got := states[signal.EntityRef{EntityType: "gallery", EntityID: "g1"}]; got.TotalEvents != 1 {
		t.Fatalf("replay double-counted: %+v", got)
	}
}
