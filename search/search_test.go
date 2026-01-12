package search

import (
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestMergeNamedArgs_Conflict(t *testing.T) {
	dst := pgx.NamedArgs{"model": "x"}
	if err := mergeNamedArgs(dst, map[string]any{"model": "y"}); err == nil {
		t.Fatalf("expected conflict error")
	}
}

func TestMergeNamedArgs_EmptyKey(t *testing.T) {
	dst := pgx.NamedArgs{"model": "x"}
	if err := mergeNamedArgs(dst, map[string]any{"": 1}); err == nil {
		t.Fatalf("expected empty key error")
	}
}

