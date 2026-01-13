package searchkit

import (
	"context"

	"github.com/doujins-org/searchkit/search"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Typeahead is the recommended entrypoint for trigram-based suggestions while typing.
//
// Under the hood it uses `pg_trgm` over `<schema>.search_documents.document`.
func Typeahead(ctx context.Context, pool *pgxpool.Pool, query string, opts search.LexicalOptions) ([]search.LexicalHit, error) {
	return search.LexicalSearch(ctx, pool, query, opts)
}
