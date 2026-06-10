package searchkit

import (
	"context"
	"os"
	"testing"

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
	defer pool.Close()

	// Minimal schema setup for trigram + FTS + semantic search. Dropped and
	// recreated so the shape always matches what the search paths expect.
	_, err = pool.Exec(ctx, `
		DROP SCHEMA IF EXISTS s CASCADE;
		CREATE SCHEMA s;
		CREATE EXTENSION IF NOT EXISTS pg_trgm;
		CREATE EXTENSION IF NOT EXISTS vector;

		CREATE OR REPLACE FUNCTION s.searchkit_regconfig_for_language(lang text)
		RETURNS regconfig
		LANGUAGE sql
		IMMUTABLE
		AS $$
			SELECT 'simple'::regconfig
		$$;

		CREATE TABLE s.search_documents (
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

		CREATE TABLE s.embedding_vectors (
			entity_type text NOT NULL,
			entity_id text NOT NULL,
			model text NOT NULL,
			language text NOT NULL,
			embedding halfvec,
			created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (entity_type, entity_id, model, language)
		);
	`)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), "DROP SCHEMA IF EXISTS s CASCADE")
	})

	_, err = pool.Exec(ctx, `
		INSERT INTO s.search_documents(entity_type, entity_id, language, document, raw_document, tsv)
		VALUES ('gallery', '1', 'en', lower('Two factor authentication'), 'Two factor authentication', to_tsvector(s.searchkit_regconfig_for_language('en'), 'Two factor authentication'))
	`)
	if err != nil {
		t.Fatalf("insert search_documents: %v", err)
	}
	_, err = pool.Exec(ctx, `
		INSERT INTO s.search_documents(entity_type, entity_id, language, document, raw_document, tsv)
		VALUES ('gallery', '2', 'en', lower('Two factor backup codes'), 'Two factor backup codes', to_tsvector(s.searchkit_regconfig_for_language('en'), 'Two factor backup codes'))
	`)
	if err != nil {
		t.Fatalf("insert search_documents 2: %v", err)
	}

	_, err = pool.Exec(ctx, `
		INSERT INTO s.embedding_vectors(entity_type, entity_id, model, language, embedding)
		VALUES ('gallery', '1', 'm', 'en', $1::halfvec(3))
	`, pgvector.NewHalfVector([]float32{1, 0, 0}))
	if err != nil {
		t.Fatalf("insert embedding_vectors: %v", err)
	}
	_, err = pool.Exec(ctx, `
		INSERT INTO s.embedding_vectors(entity_type, entity_id, model, language, embedding)
		VALUES ('gallery', '2', 'm', 'en', $1::halfvec(3))
	`, pgvector.NewHalfVector([]float32{1, 0, 0}))
	if err != nil {
		t.Fatalf("insert embedding_vectors 2: %v", err)
	}

	emb := &recordingEmbedder{vec: []float32{1, 0, 0}}
	client, err := NewClient(ClientConfig{
		Pool:         pool,
		Schema:       "s",
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
