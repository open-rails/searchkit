package search_test

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/open-rails/searchkit/pg"
	"github.com/open-rails/searchkit/search"
)

// TestFTSSearch_RawDocumentCorpusPersisted_Integration is a regression test for
// the raw_document storage contract (doujins tracker #158, decided 2026-07-02).
//
// History: commit f6a1365 stopped persisting raw_document, but the subsequent
// PGroonga lexical search for ja/zh/ko (3d9eb27) made raw_document its CJK
// corpus — search/pgroonga.go requires `raw_document IS NOT NULL` and matches
// with `raw_document OPERATOR(&@~)`. The owner decision is to KEEP raw_document
// persisted. This test guards both halves of that contract:
//
//  1. the upsert persists raw_document verbatim for every row (PGroonga's
//     corpus must exist — silently NULLing it again would break CJK search);
//  2. BM25/FTS still returns the expected hits, with tsv built from the raw
//     input (lowercased/tokenized), so normalization keeps working.
//
// It exercises the real public write + read APIs:
//   - write: pg.UpsertSearchDocuments
//   - read:  search.FTSSearch (the BM25/FTS path used by lexical search for
//     non-CJK languages)
//
// Run with:
//
//	SEARCHKIT_TEST_URL=postgres://postgres:pass@localhost:55432/testdb?sslmode=disable \
//	  go test ./search/... -run TestFTSSearch_RawDocumentCorpusPersisted_Integration
//
// It SKIPS cleanly when SEARCHKIT_TEST_URL is unset (same convention as the
// existing integration tests: pgroonga_integration_test.go and
// client_integration_test.go).
func TestFTSSearch_RawDocumentCorpusPersisted_Integration(t *testing.T) {
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

	// Minimal schema setup for the FTS/BM25 path only. Mirrors the schema used by
	// the existing integration tests. Uses the 'simple' regconfig so token
	// lowercasing is deterministic and language-independent.
	_, err = pool.Exec(ctx, `
		CREATE SCHEMA IF NOT EXISTS s;
		SET search_path = s, public;
		CREATE OR REPLACE FUNCTION searchkit_regconfig_for_language(lang text)
		RETURNS regconfig
		LANGUAGE sql
		IMMUTABLE
		AS $$
			SELECT 'simple'::regconfig
		$$;
		CREATE TABLE IF NOT EXISTS search_documents (
			entity_type text NOT NULL,
			entity_id text NOT NULL,
			language text NOT NULL,
			raw_document text,
			document text NOT NULL,
			tsv tsvector,
			created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (entity_type, entity_id, language)
		);
		TRUNCATE TABLE search_documents;
	`)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}

	const (
		schema     = "s"
		entityType = "gallery"
		language   = "en"
	)

	// Raw display strings passed to the public upsert API. These are intentionally
	// mixed-case with punctuation so normalization matters: the tsv must be built
	// from this raw input (lowercased/tokenized by to_tsvector) for a lowercase
	// normalized query term to match.
	docs := map[string]string{
		"1": "Two-Factor Authentication!",
		"2": "The Quick Brown Fox",
		"3": "Café del Mar (Vol. 2)",
	}

	if err := pg.UpsertSearchDocuments(ctx, pool, schema, entityType, language, docs); err != nil {
		t.Fatalf("UpsertSearchDocuments: %v", err)
	}

	// --- Storage contract: raw_document is persisted VERBATIM for every row. ---
	//
	// PGroonga's ja/zh/ko lexical search reads raw_document directly
	// (search/pgroonga.go filters `raw_document IS NOT NULL` and matches with
	// `raw_document OPERATOR(&@~)`), so the upsert must keep storing the raw
	// input. A regression to the old store-NULL behavior would silently break
	// CJK search.
	for entityID, raw := range docs {
		var got string
		if err := pool.QueryRow(ctx, `
			SELECT raw_document
			FROM s.search_documents
			WHERE entity_type = $1 AND language = $2 AND entity_id = $3
			  AND raw_document IS NOT NULL
		`, entityType, language, entityID).Scan(&got); err != nil {
			t.Fatalf("raw_document for entity %q: %v (PGroonga corpus contract violated: raw_document must be persisted)", entityID, err)
		}
		if got != raw {
			t.Fatalf("raw_document for entity %q = %q, want verbatim raw input %q", entityID, got, raw)
		}
	}

	// Sanity: exactly the upserted rows exist.
	var total int
	if err := pool.QueryRow(ctx, `
		SELECT count(*)
		FROM s.search_documents
		WHERE entity_type = $1 AND language = $2
	`, entityType, language).Scan(&total); err != nil {
		t.Fatalf("count rows: %v", err)
	}
	if total != len(docs) {
		t.Fatalf("expected %d rows written, got %d", len(docs), total)
	}

	// --- BM25/FTS lexical search must return hits after the upsert. ---
	tests := []struct {
		name       string
		query      string
		wantEntity string
		reason     string
	}{
		{
			name:       "normalized_lowercase_term_matches_mixedcase_raw",
			query:      "factor",
			wantEntity: "1",
			reason:     "raw input was 'Two-Factor Authentication!'; tsv must be built from raw input (lowercased) for a lowercase query to match",
		},
		{
			name:       "multiword_term",
			query:      "brown fox",
			wantEntity: "2",
			reason:     "multi-word FTS query against raw-derived tsv",
		},
		{
			name:       "term_next_to_punctuation_still_tokenizes",
			query:      "mar",
			wantEntity: "3",
			reason:     "raw 'Café del Mar (Vol. 2)' has punctuation around tokens; 'Mar' must survive as a lowercased tsv token so 'mar' matches",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hits, err := search.FTSSearch(ctx, pool, tt.query, search.FTSOptions{
				Schema:      schema,
				Language:    language,
				EntityTypes: []string{entityType},
				Limit:       10,
			})
			if err != nil {
				t.Fatalf("FTSSearch(%q): %v", tt.query, err)
			}
			found := false
			for _, h := range hits {
				if h.EntityID == tt.wantEntity {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("FTSSearch(%q): expected a hit for entity_id=%q but got %+v (%s)",
					tt.query, tt.wantEntity, hits, tt.reason)
			}
		})
	}
}
