package search

import (
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestMergeNamedArgs_Conflict(t *testing.T) {
	dst := pgx.NamedArgs{"model": "x"}
	if err := mergeNamedArgs(dst, map[string]any{"model": "y"}); err == nil {
		t.Fatalf("expected conflict error")
	}
}

func TestFuseRRFWithTrace_MatchesFuseRRF(t *testing.T) {
	t.Parallel()

	lists := [][]RRFKey{
		{{EntityType: "gallery", EntityID: "1"}, {EntityType: "gallery", EntityID: "2"}},
		{{EntityType: "gallery", EntityID: "2"}, {EntityType: "gallery", EntityID: "3"}},
	}
	opts := RRFOptions{K: 10, Weights: []float32{1, 2}}
	want := FuseRRF(lists, opts)
	traced, err := FuseRRFWithTrace(lists, opts)
	if err != nil {
		t.Fatalf("FuseRRFWithTrace() error = %v", err)
	}
	got := make([]RRFHit, 0, len(traced))
	for _, hit := range traced {
		got = append(got, hit.Hit)
		var sum float32
		for _, contribution := range hit.Contributions {
			sum += contribution.Contribution
		}
		if sum != hit.Hit.Score {
			t.Fatalf("contributions sum = %v, score = %v", sum, hit.Hit.Score)
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("traced hits differ:\n got %#v\nwant %#v", got, want)
	}
	if len(traced[0].Contributions) != 2 || traced[0].Contributions[0].ListIndex != 0 || traced[0].Contributions[1].ListIndex != 1 {
		t.Fatalf("unexpected contributions: %#v", traced[0].Contributions)
	}
}

func TestFuseRRFWithTrace_RejectsNonfiniteAndOverflowingScores(t *testing.T) {
	t.Parallel()

	lists := [][]RRFKey{{{EntityType: "gallery", EntityID: "1"}}}
	for _, weight := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := FuseRRFWithTrace(lists, RRFOptions{Weights: []float32{weight}}); err == nil {
			t.Fatalf("FuseRRFWithTrace(weight=%v) error = nil", weight)
		}
	}

	duplicateLists := [][]RRFKey{lists[0], lists[0], lists[0]}
	if _, err := FuseRRFWithTrace(duplicateLists, RRFOptions{K: 1, Weights: []float32{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}}); err == nil {
		t.Fatal("FuseRRFWithTrace() error = nil, want score overflow")
	}
}

func TestFuseRRF_DeterministicFullKeyTieBreak(t *testing.T) {
	t.Parallel()

	lists := [][]RRFKey{{
		{EntityType: "gallery", EntityID: "1", Language: "ja", Model: "z"},
		{EntityType: "gallery", EntityID: "1", Language: "en", Model: "z"},
		{EntityType: "gallery", EntityID: "1", Language: "en", Model: "a"},
	}}
	// Put each key at the same rank in a separate list so scores tie.
	lists = [][]RRFKey{{lists[0][0]}, {lists[0][1]}, {lists[0][2]}}
	want := []RRFKey{
		{EntityType: "gallery", EntityID: "1", Language: "en", Model: "a"},
		{EntityType: "gallery", EntityID: "1", Language: "en", Model: "z"},
		{EntityType: "gallery", EntityID: "1", Language: "ja", Model: "z"},
	}
	for range 20 {
		got := FuseRRF(lists, RRFOptions{K: 60})
		if len(got) != len(want) {
			t.Fatalf("len(FuseRRF()) = %d, want %d", len(got), len(want))
		}
		for i := range want {
			if got[i].RRFKey != want[i] {
				t.Fatalf("FuseRRF()[%d] = %#v, want %#v", i, got[i].RRFKey, want[i])
			}
		}
	}
}

func TestEffectiveOversampleFactor(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		input int
		want  int
	}{{input: -1, want: 5}, {input: 0, want: 5}, {input: 1, want: 5}, {input: 2, want: 2}} {
		if got := EffectiveOversampleFactor(tt.input); got != tt.want {
			t.Errorf("EffectiveOversampleFactor(%d) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestFuseRRF_LargeKDoesNotOverflow(t *testing.T) {
	t.Parallel()

	out := FuseRRF([][]RRFKey{{{EntityType: "gallery", EntityID: "1"}}}, RRFOptions{K: int(^uint(0) >> 1)})
	if len(out) != 1 || out[0].Score < 0 || math.IsNaN(float64(out[0].Score)) || math.IsInf(float64(out[0].Score), 0) {
		t.Fatalf("FuseRRF() produced invalid large-k score: %#v", out)
	}
}

func TestSemanticSearch_RejectsOversampleOverflowBeforeQuery(t *testing.T) {
	t.Parallel()

	pool, err := pgxpool.New(context.Background(), "postgres://user:pass@127.0.0.1:1/db?sslmode=disable")
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	base := Query{
		Schema: "test", Model: "model", Language: "en", QueryVec: []float32{1}, Limit: 1,
	}
	overflow := base
	overflow.Limit = int(^uint(0) >> 1)
	overflow.Options.TwoStage = true
	overflow.Options.OversampleFactor = 2
	if _, err := SemanticSearch(context.Background(), pool, overflow); err == nil {
		t.Fatal("SemanticSearch() error = nil, want oversample overflow error")
	}
}

func TestSemanticSearch_RejectsNonfiniteMinSimilarity(t *testing.T) {
	t.Parallel()

	pool, err := pgxpool.New(context.Background(), "postgres://user:pass@127.0.0.1:1/db?sslmode=disable")
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		_, err := SemanticSearch(context.Background(), pool, Query{Options: Options{MinSimilarity: value}})
		if err == nil || err.Error() != "min similarity must be finite" {
			t.Fatalf("SemanticSearch(MinSimilarity=%v) error = %v, want finite validation error", value, err)
		}
	}
}

func TestSimilarTo_RejectsNonfiniteMinSimilarity(t *testing.T) {
	t.Parallel()

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		_, err := SimilarTo(context.Background(), nil, "test", "gallery", "1", "model", "en", 10, Options{MinSimilarity: value})
		if err == nil || err.Error() != "min similarity must be finite" {
			t.Fatalf("SimilarTo(MinSimilarity=%v) error = %v, want finite validation error", value, err)
		}
	}
}

func TestMergeNamedArgs_EmptyKey(t *testing.T) {
	dst := pgx.NamedArgs{"model": "x"}
	if err := mergeNamedArgs(dst, map[string]any{"": 1}); err == nil {
		t.Fatalf("expected empty key error")
	}
}

func TestMergeNamedArgs_ReservedSearchArgs(t *testing.T) {
	for _, name := range []string{"min_similarity", "qvec", "limit", "oversample"} {
		dst := pgx.NamedArgs{name: 1}
		if err := mergeNamedArgs(dst, map[string]any{name: 2}); err == nil {
			t.Fatalf("mergeNamedArgs() error = nil, want reserved %s conflict", name)
		}
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
