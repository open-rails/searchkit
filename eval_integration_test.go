package searchkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"

	"github.com/open-rails/searchkit/eval"
)

const (
	// evalBaselinePath is the committed golden report the regression gate
	// compares against. Regenerate with SEARCHKIT_EVAL_UPDATE=1.
	evalBaselinePath = "eval/testdata/golden_gallery_baseline.json"
	evalUpdateEnv    = "SEARCHKIT_EVAL_UPDATE"
)

// mapEmbedder is a deterministic, query-aware stub: it maps known query text to
// a fixed vector so semantic ranking is reproducible across cases. An unknown
// query fails loudly rather than silently embedding to a zero vector.
type mapEmbedder struct {
	vecs map[string][]float32
}

func (m mapEmbedder) EmbedQueryText(_ context.Context, _ string, text string) ([]float32, error) {
	key := strings.ToLower(strings.TrimSpace(text))
	if vec, ok := m.vecs[key]; ok {
		return vec, nil
	}
	return nil, fmt.Errorf("mapEmbedder: no vector for query %q", key)
}

// seedDoc is one corpus row: FTS document text plus its semantic vector.
type seedDoc struct {
	id  string
	doc string
	vec []float32
}

// newEvalTestClient provisions an isolated schema, seeds a small deterministic
// corpus, and returns a client wired to a query-aware embedder. Shared by the
// baseline-gate and config-diff tests so the corpus stays identical.
func newEvalTestClient(t *testing.T) (context.Context, *Client) {
	t.Helper()
	dsn := os.Getenv("SEARCHKIT_TEST_URL")
	if dsn == "" {
		t.Skip("SEARCHKIT_TEST_URL not set")
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool: %v", err)
	}
	t.Cleanup(pool.Close)

	schema := fmt.Sprintf("searchkit_eval_%d_%d", os.Getpid(), time.Now().UnixNano())
	quotedSchema := pgx.Identifier{schema}.Sanitize()
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(cleanupCtx, "DROP SCHEMA IF EXISTS "+quotedSchema+" CASCADE"); err != nil {
			t.Errorf("drop eval schema %s: %v", schema, err)
		}
	})

	if _, err := pool.Exec(ctx, fmt.Sprintf(`
		CREATE EXTENSION IF NOT EXISTS pg_trgm;
		CREATE EXTENSION IF NOT EXISTS vector;
		CREATE SCHEMA %s;

		CREATE OR REPLACE FUNCTION %s.searchkit_regconfig_for_language(lang text)
		RETURNS regconfig LANGUAGE sql IMMUTABLE AS $$ SELECT 'simple'::regconfig $$;

		CREATE TABLE %s.search_documents (
			entity_type text NOT NULL,
			entity_id text NOT NULL,
			language text NOT NULL,
			document text,
			raw_document text,
			tsv tsvector,
			created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (entity_type, entity_id, language)
		);

		CREATE TABLE %s.embedding_vectors (
			entity_type text NOT NULL,
			entity_id text NOT NULL,
			model text NOT NULL,
			language text NOT NULL,
			embedding halfvec,
			created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (entity_type, entity_id, model, language)
		);
	`, quotedSchema, quotedSchema, quotedSchema, quotedSchema)); err != nil {
		t.Fatalf("setup schema: %v", err)
	}

	corpus := []seedDoc{
		{id: "1", doc: "two factor authentication", vec: []float32{1, 0, 0}},
		{id: "2", doc: "two factor backup codes", vec: []float32{0.9, 0.1, 0}},
		{id: "3", doc: "single sign on saml", vec: []float32{0, 1, 0}},
		{id: "4", doc: "password reset email flow", vec: []float32{0, 0.9, 0.1}},
		{id: "5", doc: "unrelated cooking recipe", vec: []float32{-1, 0, 0}},
	}
	for _, d := range corpus {
		if _, err := pool.Exec(ctx, fmt.Sprintf(`
			INSERT INTO %s.search_documents(entity_type, entity_id, language, document, raw_document, tsv)
			VALUES ('gallery', $1, 'en', lower($2), $2, to_tsvector(%s.searchkit_regconfig_for_language('en'), $2))
		`, quotedSchema, quotedSchema), d.id, d.doc); err != nil {
			t.Fatalf("insert search_documents %s: %v", d.id, err)
		}
		if _, err := pool.Exec(ctx, fmt.Sprintf(`
			INSERT INTO %s.embedding_vectors(entity_type, entity_id, model, language, embedding)
			VALUES ('gallery', $1, 'm', 'en', $2::halfvec(3))
		`, quotedSchema), d.id, pgvector.NewHalfVector(d.vec)); err != nil {
			t.Fatalf("insert embedding_vectors %s: %v", d.id, err)
		}
	}

	emb := mapEmbedder{vecs: map[string][]float32{
		"two factor":       {1, 0, 0},
		"single sign on":   {0, 1, 0},
		"authentication":   {1, 0, 0},
		"zzzq nonexistent": {0, 0, 1},
	}}
	client, err := NewClient(ClientConfig{
		Pool:         pool,
		Schema:       schema,
		Embedder:     emb,
		DefaultModel: "m",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return ctx, client
}

func loadGoldenSuite(t *testing.T) eval.Suite {
	t.Helper()
	f, err := os.Open("eval/testdata/golden_gallery.json")
	if err != nil {
		t.Fatalf("open golden suite: %v", err)
	}
	t.Cleanup(func() { _ = f.Close() })
	suite, err := eval.ParseSuite(f)
	if err != nil {
		t.Fatalf("parse golden suite: %v", err)
	}
	return suite
}

// runGolden executes the golden suite under one search configuration and builds
// a report. floor <= 0 disables the semantic floor.
func runGolden(ctx context.Context, t *testing.T, client *Client, suite eval.Suite, candidateID string, floor float32) eval.Report {
	t.Helper()
	runner := NewEvalRunner(client, SearchOptions{
		Mode:                  SearchModeDual,
		Language:              "en",
		SemanticMinSimilarity: floor,
		RRFK:                  60,
	})
	identity := eval.ReportIdentity{DatasetID: "gallery-smoke", SuiteID: suite.ID, CandidateID: candidateID}
	report, err := eval.RunSuite(ctx, suite, runner, identity, "query_type")
	if err != nil {
		t.Fatalf("RunSuite(%s): %v", candidateID, err)
	}
	return report
}

// TestEvalRunSuite_Integration runs the committed golden suite through
// client.Search against a real Postgres, asserts a clean baseline, and gates on
// the committed baseline report so a future quality regression fails CI.
// Regenerate the baseline with SEARCHKIT_EVAL_UPDATE=1.
func TestEvalRunSuite_Integration(t *testing.T) {
	ctx, client := newEvalTestClient(t)
	suite := loadGoldenSuite(t)

	// Dual mode with a semantic floor: the floor removes the low-cosine tail so
	// the nonsense query returns nothing instead of polluting with neighbors.
	report := runGolden(ctx, t, client, suite, "dual+floor0.5", 0.5)

	if report.Metrics.Cases != 4 {
		t.Fatalf("Cases = %d, want 4", report.Metrics.Cases)
	}
	if report.Metrics.FailedCases != 0 {
		t.Fatalf("FailedCases = %d, want 0", report.Metrics.FailedCases)
	}
	if report.Metrics.JudgedCases != 3 {
		t.Fatalf("JudgedCases = %d, want 3", report.Metrics.JudgedCases)
	}
	if report.Metrics.RecallAtK != 1 {
		t.Fatalf("RecallAtK = %v, want 1 (all relevant retrieved within k)", report.Metrics.RecallAtK)
	}
	if report.Metrics.SuccessAtK != 1 {
		t.Fatalf("SuccessAtK = %v, want 1 (every judged case hits)", report.Metrics.SuccessAtK)
	}
	if report.Metrics.NDCGAtK < 0.7 {
		t.Fatalf("NDCGAtK = %v, want >= 0.7 (top-graded docs rank near top)", report.Metrics.NDCGAtK)
	}
	if report.Metrics.EmptyCases != 1 || report.Metrics.ExactEmptyRate != 1 {
		t.Fatalf("empty metrics = {cases:%d rate:%v}, want {1, 1} (floor keeps nonsense empty)",
			report.Metrics.EmptyCases, report.Metrics.ExactEmptyRate)
	}
	if _, ok := report.Breakdowns["query_type"]; !ok {
		t.Fatalf("missing query_type breakdown: %+v", report.Breakdowns)
	}
	if report.ContentID == "" {
		t.Fatal("report ContentID is empty")
	}

	// Golden-file regression gate: update on demand, otherwise compare.
	if os.Getenv(evalUpdateEnv) != "" {
		writeBaselineReport(t, evalBaselinePath, report)
		t.Logf("wrote baseline %s (SEARCHKIT_EVAL_UPDATE set)", evalBaselinePath)
		return
	}
	baseline := loadReport(t, evalBaselinePath)
	tolerances := eval.Tolerances{
		RecallAtKDrop:      0,
		SuccessAtKDrop:     0,
		MRRAtKDrop:         0.05,
		NDCGAtKDrop:        0.05,
		ExactEmptyRateDrop: 0,
		FailedCaseIncrease: 0,
	}
	comparison, err := eval.Compare(baseline, report, tolerances)
	if err != nil {
		t.Fatalf("Compare(baseline, current): %v", err)
	}
	if !comparison.Compatible {
		t.Fatalf("baseline incompatible with current report (regenerate with %s=1): %v", evalUpdateEnv, comparison.Mismatches)
	}
	if comparison.Regressed() {
		t.Fatalf("search quality regressed vs committed baseline: %+v", comparison.Regressions)
	}
}

// TestEvalConfigDiff_Integration proves the eval can compare two configurations
// side by side: disabling the semantic floor regresses exact-empty-rate because
// the nonsense query's low-cosine tail is no longer filtered.
func TestEvalConfigDiff_Integration(t *testing.T) {
	ctx, client := newEvalTestClient(t)
	suite := loadGoldenSuite(t)

	floorOn := runGolden(ctx, t, client, suite, "dual+floor0.5", 0.5)
	floorOff := runGolden(ctx, t, client, suite, "dual+nofloor", 0)

	// Ground the claim directly: the floor keeps the nonsense case empty; without
	// it the semantic tail leaks in.
	if floorOn.Metrics.ExactEmptyRate != 1 {
		t.Fatalf("floor-on ExactEmptyRate = %v, want 1", floorOn.Metrics.ExactEmptyRate)
	}
	if floorOff.Metrics.ExactEmptyRate != 0 {
		t.Fatalf("floor-off ExactEmptyRate = %v, want 0 (semantic tail should leak without the floor)", floorOff.Metrics.ExactEmptyRate)
	}

	// The comparator must flag floor-off as a regression against floor-on.
	strict := eval.Tolerances{} // zero tolerance on every metric
	comparison, err := eval.Compare(floorOn, floorOff, strict)
	if err != nil {
		t.Fatalf("Compare(floorOn, floorOff): %v", err)
	}
	if !comparison.Compatible {
		t.Fatalf("floor-on and floor-off reports should be comparable: %v", comparison.Mismatches)
	}
	if !comparison.Regressed() {
		t.Fatal("expected floor-off to regress vs floor-on, comparator saw none")
	}
	var sawEmptyRate bool
	for _, r := range comparison.Regressions {
		if r.Metric == "exact_empty_rate" {
			sawEmptyRate = true
		}
	}
	if !sawEmptyRate {
		t.Fatalf("expected an exact_empty_rate regression, got %+v", comparison.Regressions)
	}
}

func loadReport(t *testing.T, path string) eval.Report {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open baseline %s (generate with %s=1): %v", path, evalUpdateEnv, err)
	}
	defer func() { _ = f.Close() }()
	var report eval.Report
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&report); err != nil {
		t.Fatalf("decode baseline %s: %v", path, err)
	}
	return report
}

func writeBaselineReport(t *testing.T, path string, report eval.Report) {
	t.Helper()
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatalf("marshal baseline: %v", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write baseline %s: %v", path, err)
	}
}
