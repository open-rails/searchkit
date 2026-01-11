package migrate

import (
	"context"
	"fmt"
	"io/fs"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/doujins-org/embeddingkit/migrations"
)

func quoteIdent(ident string) (string, error) {
	ident = strings.TrimSpace(ident)
	if ident == "" {
		return "", fmt.Errorf("empty identifier")
	}
	for _, r := range ident {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			continue
		}
		return "", fmt.Errorf("invalid identifier %q", ident)
	}
	return `"` + ident + `"`, nil
}

// ApplyPostgres applies embeddingkit's Postgres migrations to the given schema.
//
// This intentionally mirrors River-style embedding: the host app can call this
// during its migration phase, or delegate to its own migration runner.
func ApplyPostgres(ctx context.Context, pool *pgxpool.Pool, schema string) error {
	if strings.TrimSpace(schema) == "" {
		return fmt.Errorf("schema is required")
	}

	dirEntries, err := fs.ReadDir(migrations.Postgres, "postgres")
	if err != nil {
		return fmt.Errorf("read embedded migrations: %w", err)
	}

	var files []string
	for _, de := range dirEntries {
		if de.IsDir() {
			continue
		}
		name := de.Name()
		if strings.HasSuffix(name, ".up.sql") {
			files = append(files, name)
		}
	}
	sort.Strings(files)

	conn, err := pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire pg connection: %w", err)
	}
	defer conn.Release()

	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	quotedSchema, err := quoteIdent(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	if _, err := tx.Exec(ctx, fmt.Sprintf("SET LOCAL search_path = %s", quotedSchema)); err != nil {
		return fmt.Errorf("set search_path: %w", err)
	}

	for _, f := range files {
		raw, err := fs.ReadFile(migrations.Postgres, "postgres/"+f)
		if err != nil {
			return fmt.Errorf("read migration %s: %w", f, err)
		}
		if _, err := tx.Exec(ctx, string(raw)); err != nil {
			return fmt.Errorf("apply migration %s: %w", f, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}
