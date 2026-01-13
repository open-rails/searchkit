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

func TestFuseRRF_Basic(t *testing.T) {
	// list1: A, B
	// list2: B, C
	l1 := []RRFKey{
		{EntityType: "gallery", EntityID: "1", Language: "en", Model: ""},
		{EntityType: "gallery", EntityID: "2", Language: "en", Model: ""},
	}
	l2 := []RRFKey{
		{EntityType: "gallery", EntityID: "2", Language: "en", Model: ""},
		{EntityType: "gallery", EntityID: "3", Language: "en", Model: ""},
	}
	out := FuseRRF([][]RRFKey{l1, l2}, RRFOptions{K: 60})
	if len(out) != 3 {
		t.Fatalf("expected 3 results, got %d", len(out))
	}
	// "2" appears in both lists, so it should rank first.
	if out[0].EntityID != "2" {
		t.Fatalf("expected top entity_id=2, got %q", out[0].EntityID)
	}
}
