package searchkit

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

func TestClientSearch_Integration_LexicalAndSemantic(t *testing.T) {
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

	schema := fmt.Sprintf("searchkit_test_%d_%d", os.Getpid(), time.Now().UnixNano())
	quotedSchema := pgx.Identifier{schema}.Sanitize()
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(cleanupCtx, "DROP SCHEMA IF EXISTS "+quotedSchema+" CASCADE"); err != nil {
			t.Errorf("drop integration schema %s: %v", schema, err)
		}
	})

	// Minimal isolated schema for trigram + FTS + semantic search.
	_, err = pool.Exec(ctx, fmt.Sprintf(`
		CREATE EXTENSION IF NOT EXISTS pg_trgm;
		CREATE EXTENSION IF NOT EXISTS vector;
		CREATE SCHEMA %s;

		CREATE OR REPLACE FUNCTION %s.searchkit_regconfig_for_language(lang text)
		RETURNS regconfig
		LANGUAGE sql
		IMMUTABLE
		AS $$
			SELECT 'simple'::regconfig
		$$;

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
	`, quotedSchema, quotedSchema, quotedSchema, quotedSchema))
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	_, err = pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.search_documents(entity_type, entity_id, language, document, raw_document, tsv)
		VALUES ('gallery', '1', 'en', lower('Two factor authentication'), 'Two factor authentication', to_tsvector(%s.searchkit_regconfig_for_language('en'), 'Two factor authentication'))
	`, quotedSchema, quotedSchema))
	if err != nil {
		t.Fatalf("insert search_documents: %v", err)
	}
	_, err = pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.search_documents(entity_type, entity_id, language, document, raw_document, tsv)
		VALUES ('gallery', '2', 'en', lower('Two factor backup codes'), 'Two factor backup codes', to_tsvector(%s.searchkit_regconfig_for_language('en'), 'Two factor backup codes'))
	`, quotedSchema, quotedSchema))
	if err != nil {
		t.Fatalf("insert search_documents 2: %v", err)
	}

	_, err = pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.embedding_vectors(entity_type, entity_id, model, language, embedding)
		VALUES ('gallery', '1', 'm', 'en', $1::halfvec(3))
	`, quotedSchema), pgvector.NewHalfVector([]float32{1, 0, 0}))
	if err != nil {
		t.Fatalf("insert embedding_vectors: %v", err)
	}
	_, err = pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.embedding_vectors(entity_type, entity_id, model, language, embedding)
		VALUES ('gallery', '2', 'm', 'en', $1::halfvec(3))
	`, quotedSchema), pgvector.NewHalfVector([]float32{1, 0, 0}))
	if err != nil {
		t.Fatalf("insert embedding_vectors 2: %v", err)
	}
	_, err = pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.embedding_vectors(entity_type, entity_id, model, language, embedding)
		VALUES ('gallery', '3', 'm', 'en', $1::halfvec(3))
	`, quotedSchema), pgvector.NewHalfVector([]float32{-1, 0, 0}))
	if err != nil {
		t.Fatalf("insert embedding_vectors 3: %v", err)
	}

	emb := &recordingEmbedder{vec: []float32{1, 0, 0}}
	client, err := NewClient(ClientConfig{
		Pool:         pool,
		Schema:       schema,
		Embedder:     emb,
		DefaultModel: "m",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	lexHits, err := client.Search(ctx, "factor", SearchOptions{
		Mode:               SearchModeLexical,
		Language:           "en",
		LexicalEntityTypes: []string{"gallery"},
		Limit:              10,
	})
	if err != nil {
		t.Fatalf("lexical Search: %v", err)
	}
	if len(lexHits) == 0 || lexHits[0].EntityID != "1" {
		t.Fatalf("expected lexical hit entity_id=1, got %+v", lexHits)
	}
	tracedLexHits, lexTrace, err := client.SearchWithTrace(ctx, "factor", SearchOptions{
		Mode:               SearchModeLexical,
		Language:           "en",
		LexicalEntityTypes: []string{"gallery"},
		Limit:              10,
	})
	if err != nil {
		t.Fatalf("traced lexical Search: %v", err)
	}
	if !reflect.DeepEqual(tracedLexHits, lexHits) {
		t.Fatalf("traced lexical results differ: got %+v, want %+v", tracedLexHits, lexHits)
	}
	if len(lexTrace.Sources) != 1 || lexTrace.Sources[0].Backend != BackendFTS || lexTrace.Sources[0].Status != SourceStatusSucceeded || len(lexTrace.Results) != len(tracedLexHits) {
		t.Fatalf("unexpected lexical trace: %+v", lexTrace)
	}

	semHits, err := client.Search(ctx, "two-factor", SearchOptions{
		Mode:                SearchModeSemantic,
		Language:            "en",
		SemanticEntityTypes: []string{"gallery"},
		Limit:               10,
	})
	if err != nil {
		t.Fatalf("semantic Search: %v", err)
	}
	if len(semHits) == 0 || semHits[0].EntityID != "1" {
		t.Fatalf("expected semantic hit entity_id=1, got %+v", semHits)
	}
	tracedSemHits, semTrace, err := client.SearchWithTrace(ctx, "two-factor", SearchOptions{
		Mode:                SearchModeSemantic,
		Language:            "en",
		SemanticEntityTypes: []string{"gallery"},
		Limit:               10,
	})
	if err != nil {
		t.Fatalf("traced semantic Search: %v", err)
	}
	if !reflect.DeepEqual(tracedSemHits, semHits) {
		t.Fatalf("traced semantic results differ: got %+v, want %+v", tracedSemHits, semHits)
	}
	if len(semTrace.Sources) != 1 || semTrace.Sources[0].Backend != BackendSemantic || semTrace.Sources[0].ScoreKind != ScoreCosineSimilarity || len(semTrace.Sources[0].Candidates) != len(tracedSemHits) {
		t.Fatalf("unexpected semantic trace: %+v", semTrace)
	}
	for _, result := range semTrace.Results {
		var contributionSum float32
		for _, contribution := range result.Contributions {
			contributionSum += contribution.Contribution
		}
		if contributionSum != result.Score {
			t.Fatalf("result contributions = %v, score = %v", contributionSum, result.Score)
		}
	}
	for range 10 {
		_, repeatedTrace, err := client.SearchWithTrace(ctx, "two-factor", SearchOptions{
			Mode:                SearchModeSemantic,
			Language:            "en",
			SemanticEntityTypes: []string{"gallery"},
			Limit:               3,
			CandidateLimit:      3,
		})
		if err != nil {
			t.Fatalf("repeated semantic Search: %v", err)
		}
		candidates := repeatedTrace.Sources[0].Candidates
		if len(candidates) != 3 || candidates[0].Key.EntityID != "1" || candidates[1].Key.EntityID != "2" || candidates[2].Key.EntityID != "3" {
			t.Fatalf("semantic ties are not deterministic: %+v", candidates)
		}
	}

	limitedHits, limitedTrace, err := client.SearchWithTrace(ctx, "factor", SearchOptions{
		Mode:               SearchModeLexical,
		Language:           "en",
		LexicalEntityTypes: []string{"gallery"},
		Limit:              1,
		CandidateLimit:     2,
	})
	if err != nil {
		t.Fatalf("candidate-limited Search: %v", err)
	}
	if len(limitedHits) != 1 || limitedTrace.ResultLimit != 1 || limitedTrace.CandidateLimit != 2 || len(limitedTrace.Sources[0].Candidates) != 2 {
		t.Fatalf("candidate/result limits not separated: hits=%+v trace=%+v", limitedHits, limitedTrace)
	}

	flooredHits, floorTrace, err := client.SearchWithTrace(ctx, "two-factor", SearchOptions{
		Mode:                  SearchModeSemantic,
		Language:              "en",
		SemanticEntityTypes:   []string{"gallery"},
		Limit:                 10,
		SemanticMinSimilarity: 1.1,
	})
	if err != nil {
		t.Fatalf("semantic-floor Search: %v", err)
	}
	if len(flooredHits) != 0 || floorTrace.SemanticMinSimilarity != 1.1 || len(floorTrace.Sources) != 1 || len(floorTrace.Sources[0].Candidates) != 0 {
		t.Fatalf("semantic floor not applied/traced: hits=%+v trace=%+v", flooredHits, floorTrace)
	}

	for _, floor := range []float32{0, -0.5} {
		disabledHits, disabledTrace, err := client.SearchWithTrace(ctx, "two-factor", SearchOptions{
			Mode:                  SearchModeSemantic,
			Language:              "en",
			SemanticEntityTypes:   []string{"gallery"},
			Limit:                 1,
			CandidateLimit:        3,
			TwoStage:              boolPointer(true),
			OversampleFactor:      2,
			SemanticMinSimilarity: floor,
		})
		if err != nil {
			t.Fatalf("two-stage disabled-floor Search(%v): %v", floor, err)
		}
		if len(disabledHits) != 1 || disabledTrace.SemanticMinSimilarity != 0 || disabledTrace.CandidateLimit != 3 || disabledTrace.OversampleFactor != 2 {
			t.Fatalf("unexpected disabled-floor limits: floor=%v hits=%+v trace=%+v", floor, disabledHits, disabledTrace)
		}
		candidates := disabledTrace.Sources[0].Candidates
		if len(candidates) != 3 || candidates[2].Score >= 0 {
			t.Fatalf("disabled floor removed negative cosine candidate: floor=%v candidates=%+v", floor, candidates)
		}
	}

	positiveHits, positiveTrace, err := client.SearchWithTrace(ctx, "two-factor", SearchOptions{
		Mode:                  SearchModeSemantic,
		Language:              "en",
		SemanticEntityTypes:   []string{"gallery"},
		Limit:                 3,
		CandidateLimit:        3,
		TwoStage:              boolPointer(true),
		OversampleFactor:      2,
		SemanticMinSimilarity: 0.1,
	})
	if err != nil {
		t.Fatalf("two-stage positive-floor Search: %v", err)
	}
	if len(positiveHits) != 2 || len(positiveTrace.Sources[0].Candidates) != 2 {
		t.Fatalf("positive floor did not remove negative candidate: hits=%+v trace=%+v", positiveHits, positiveTrace)
	}

	filteredLex, err := client.Search(ctx, "two factor", SearchOptions{
		Mode:               SearchModeLexical,
		Language:           "en",
		LexicalEntityTypes: []string{"gallery"},
		Limit:              10,
		FilterSQL:          "sd.entity_id = @allowed_id",
		FilterArgs: map[string]any{
			"allowed_id": "1",
		},
	})
	if err != nil {
		t.Fatalf("filtered lexical Search: %v", err)
	}
	for _, h := range filteredLex {
		if h.EntityID != "1" {
			t.Fatalf("expected filtered lexical hits to contain only entity_id=1, got %+v", filteredLex)
		}
	}

	filteredSem, err := client.Search(ctx, "two-factor", SearchOptions{
		Mode:                SearchModeSemantic,
		Language:            "en",
		SemanticEntityTypes: []string{"gallery"},
		Limit:               10,
		FilterSQL:           "ev.entity_id = @allowed_id",
		FilterArgs: map[string]any{
			"allowed_id": "1",
		},
	})
	if err != nil {
		t.Fatalf("filtered semantic Search: %v", err)
	}
	for _, h := range filteredSem {
		if h.EntityID != "1" {
			t.Fatalf("expected filtered semantic hits to contain only entity_id=1, got %+v", filteredSem)
		}
	}

	filteredTypeahead, err := client.Typeahead(ctx, "two", TypeaheadOptions{
		Language:    "en",
		EntityTypes: []string{"gallery"},
		Limit:       10,
		FilterSQL:   "sd.entity_id = @allowed_id",
		FilterArgs: map[string]any{
			"allowed_id": "1",
		},
	})
	if err != nil {
		t.Fatalf("filtered typeahead: %v", err)
	}
	for _, h := range filteredTypeahead {
		if h.EntityID != "1" {
			t.Fatalf("expected filtered typeahead hits to contain only entity_id=1, got %+v", filteredTypeahead)
		}
	}

	// Default behavior is strict language (exact only).
	strictLex, err := client.Search(ctx, "factor", SearchOptions{
		Mode:               SearchModeLexical,
		Language:           "es",
		LexicalEntityTypes: []string{"gallery"},
		Limit:              10,
	})
	if err != nil {
		t.Fatalf("strict lexical Search: %v", err)
	}
	if len(strictLex) != 0 {
		t.Fatalf("expected strict lexical language mode to return no hits, got %+v", strictLex)
	}

	fallbackLex, err := client.Search(ctx, "factor", SearchOptions{
		Mode:               SearchModeLexical,
		Language:           "es",
		LanguageMode:       LanguageModeFallbackEnglish,
		LexicalEntityTypes: []string{"gallery"},
		Limit:              10,
	})
	if err != nil {
		t.Fatalf("fallback lexical Search: %v", err)
	}
	if len(fallbackLex) == 0 {
		t.Fatalf("expected fallback lexical language mode to return english hits")
	}
	for _, h := range fallbackLex {
		if h.Language != "en" {
			t.Fatalf("expected fallback lexical hits language=en, got %+v", fallbackLex)
		}
	}

	strictTypeahead, err := client.Typeahead(ctx, "two", TypeaheadOptions{
		Language:    "es",
		EntityTypes: []string{"gallery"},
		Limit:       10,
	})
	if err != nil {
		t.Fatalf("strict typeahead: %v", err)
	}
	if len(strictTypeahead) != 0 {
		t.Fatalf("expected strict typeahead language mode to return no hits, got %+v", strictTypeahead)
	}

	fallbackTypeahead, err := client.Typeahead(ctx, "two", TypeaheadOptions{
		Language:     "es",
		LanguageMode: LanguageModeFallbackEnglish,
		EntityTypes:  []string{"gallery"},
		Limit:        10,
	})
	if err != nil {
		t.Fatalf("fallback typeahead: %v", err)
	}
	if len(fallbackTypeahead) == 0 {
		t.Fatalf("expected fallback typeahead language mode to return english hits")
	}
	for _, h := range fallbackTypeahead {
		if h.Language != "en" {
			t.Fatalf("expected fallback typeahead hits language=en, got %+v", fallbackTypeahead)
		}
	}
}

func boolPointer(value bool) *bool {
	return &value
}
