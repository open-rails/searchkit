package eval

import (
	"math"
	"strings"
	"testing"
)

func TestValidateCase(t *testing.T) {
	t.Parallel()

	key := GoldenKey{EntityType: "gallery", EntityID: "1"}
	valid := GoldenCase{ID: "case-1", Query: "query", K: 5, Expected: []GoldenKey{key}}

	tests := []struct {
		name    string
		mutate  func(*GoldenCase)
		wantErr bool
	}{
		{name: "valid", mutate: func(*GoldenCase) {}},
		{name: "missing id", mutate: func(c *GoldenCase) { c.ID = "" }, wantErr: true},
		{name: "missing query", mutate: func(c *GoldenCase) { c.Query = "" }, wantErr: true},
		{name: "nonpositive k", mutate: func(c *GoldenCase) { c.K = 0 }, wantErr: true},
		{name: "conflicting judgment forms", mutate: func(c *GoldenCase) { c.Judgments = []Judgment{{Key: key, Relevance: 2}} }, wantErr: true},
		{name: "empty with judgment", mutate: func(c *GoldenCase) { c.ExpectEmpty = true }, wantErr: true},
		{name: "duplicate expected key", mutate: func(c *GoldenCase) { c.Expected = append(c.Expected, key) }, wantErr: true},
		{name: "invalid relevance", mutate: func(c *GoldenCase) { c.Expected = nil; c.Judgments = []Judgment{{Key: key, Relevance: 4}} }, wantErr: true},
		{name: "empty label", mutate: func(c *GoldenCase) { c.Labels = map[string]string{"suite": ""} }, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			c := valid
			c.Expected = append([]GoldenKey(nil), valid.Expected...)
			tt.mutate(&c)
			if err := ValidateCase(c); (err != nil) != tt.wantErr {
				t.Fatalf("ValidateCase() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestEvaluate_GradedMetrics(t *testing.T) {
	t.Parallel()

	high := GoldenKey{EntityType: "gallery", EntityID: "high"}
	medium := GoldenKey{EntityType: "gallery", EntityID: "medium"}
	other := GoldenKey{EntityType: "gallery", EntityID: "other"}
	c := GoldenCase{
		ID:    "graded",
		Query: "graded query",
		K:     3,
		Judgments: []Judgment{
			{Key: high, Relevance: 3},
			{Key: medium, Relevance: 2},
		},
	}

	out, err := Evaluate(c, []Result{
		{Key: medium, Score: 0.9},
		{Key: high, Score: 0.8},
		{Key: other, Score: 0.7},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if !out.Judged || out.QualityStatus != QualityStatusHit || out.RecallAtK != 1 || out.SuccessAtK != 1 || out.ReciprocalRank != 1 {
		t.Fatalf("unexpected binary metrics: %#v", out)
	}
	const wantNDCG = 0.8339912323981488
	if math.Abs(out.NDCGAtK-wantNDCG) > 1e-12 {
		t.Fatalf("NDCGAtK = %.15f, want %.15f", out.NDCGAtK, wantNDCG)
	}
}

func TestEvaluate_DuplicatesDoNotCompactRawRanks(t *testing.T) {
	t.Parallel()

	a := GoldenKey{EntityType: "gallery", EntityID: "a"}
	b := GoldenKey{EntityType: "gallery", EntityID: "b"}
	results := []Result{{Key: a, Score: 1}, {Key: a, Score: 0.9}, {Key: b, Score: 0.8}}
	out, err := Evaluate(GoldenCase{ID: "duplicates-k2", Query: "query", K: 2, Expected: []GoldenKey{b}}, results)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if out.ResultCount != 3 {
		t.Fatalf("ResultCount = %d, want 3", out.ResultCount)
	}
	if out.RecallAtK != 0 || out.ReciprocalRank != 0 {
		t.Fatalf("duplicate compacted the top-k window: %#v", out)
	}
	out, err = Evaluate(GoldenCase{ID: "duplicates-k3", Query: "query", K: 3, Expected: []GoldenKey{b}}, results)
	if err != nil {
		t.Fatalf("Evaluate(k=3) error = %v", err)
	}
	if out.RecallAtK != 1 || out.ReciprocalRank != 1.0/3.0 {
		t.Fatalf("raw result rank was not preserved: %#v", out)
	}
}

func TestEvaluate_KBoundary(t *testing.T) {
	t.Parallel()

	a := GoldenKey{EntityType: "gallery", EntityID: "a"}
	b := GoldenKey{EntityType: "gallery", EntityID: "b"}
	c := GoldenKey{EntityType: "gallery", EntityID: "c"}

	tests := []struct {
		name    string
		results []Result
		wantHit float64
	}{
		{name: "hit at k", results: []Result{{Key: a}, {Key: b}}, wantHit: 1},
		{name: "miss at k plus one", results: []Result{{Key: a}, {Key: c}, {Key: b}}, wantHit: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			out, err := Evaluate(GoldenCase{ID: tt.name, Query: "query", K: 2, Expected: []GoldenKey{b}}, tt.results)
			if err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}
			if out.SuccessAtK != tt.wantHit {
				t.Fatalf("SuccessAtK = %v, want %v", out.SuccessAtK, tt.wantHit)
			}
		})
	}
}

func TestEvaluate_TiedScoresPreserveCallerOrder(t *testing.T) {
	t.Parallel()

	first := GoldenKey{EntityType: "gallery", EntityID: "first"}
	relevant := GoldenKey{EntityType: "gallery", EntityID: "relevant"}
	out, err := Evaluate(
		GoldenCase{ID: "ties", Query: "query", K: 2, Expected: []GoldenKey{relevant}},
		[]Result{{Key: first, Score: 0.5}, {Key: relevant, Score: 0.5}},
	)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if out.ReciprocalRank != 0.5 {
		t.Fatalf("ReciprocalRank = %v, want 0.5", out.ReciprocalRank)
	}
}

func TestEvaluate_EmptyAndUnjudgedAreDistinct(t *testing.T) {
	t.Parallel()

	empty, err := Evaluate(GoldenCase{ID: "empty", Query: "nonsense", K: 5, ExpectEmpty: true}, nil)
	if err != nil {
		t.Fatalf("Evaluate(empty) error = %v", err)
	}
	if !empty.EmptyExpected || !empty.ExactEmpty || empty.Judged || empty.QualityStatus != QualityStatusExactEmpty {
		t.Fatalf("unexpected empty outcome: %#v", empty)
	}

	unjudged, err := Evaluate(GoldenCase{ID: "unjudged", Query: "query", K: 5}, nil)
	if err != nil {
		t.Fatalf("Evaluate(unjudged) error = %v", err)
	}
	if unjudged.EmptyExpected || unjudged.ExactEmpty || unjudged.Judged || unjudged.QualityStatus != QualityStatusUnjudged {
		t.Fatalf("unexpected unjudged outcome: %#v", unjudged)
	}
}

func TestEvaluate_RejectsNonfiniteScore(t *testing.T) {
	t.Parallel()

	key := GoldenKey{EntityType: "gallery", EntityID: "1"}
	_, err := Evaluate(
		GoldenCase{ID: "nan", Query: "query", K: 1, Expected: []GoldenKey{key}},
		[]Result{{Key: key, Score: float32(math.NaN())}},
	)
	if err == nil {
		t.Fatal("Evaluate() error = nil, want nonfinite score error")
	}
}

func TestValidateCase_RejectsAllZeroJudgments(t *testing.T) {
	t.Parallel()

	err := ValidateCase(GoldenCase{
		ID: "zero", Query: "query", K: 1,
		Judgments: []Judgment{{Key: GoldenKey{EntityType: "gallery", EntityID: "1"}, Relevance: 0}},
	})
	if err == nil {
		t.Fatal("ValidateCase() error = nil, want positive-judgment error")
	}
}

func TestEvaluate_NormalizesKeysAndLabels(t *testing.T) {
	t.Parallel()

	out, err := Evaluate(
		GoldenCase{
			ID: " normalized ", Query: " query ", K: 1,
			Expected: []GoldenKey{{EntityType: " gallery ", EntityID: " 1 "}},
			Labels:   map[string]string{" suite ": " manual "},
		},
		[]Result{{Key: GoldenKey{EntityType: "gallery", EntityID: "1"}, Score: 1}},
	)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if out.RecallAtK != 1 || out.Case.ID != "normalized" || out.Case.Labels["suite"] != "manual" {
		t.Fatalf("normalization mismatch: %#v", out)
	}
}

func TestValidateCase_RejectsCanonicalDuplicates(t *testing.T) {
	t.Parallel()

	err := ValidateCase(GoldenCase{
		ID: "duplicates", Query: "query", K: 1,
		Expected: []GoldenKey{
			{EntityType: "gallery", EntityID: "1"},
			{EntityType: " gallery ", EntityID: "1 "},
		},
	})
	if err == nil {
		t.Fatal("ValidateCase() error = nil, want canonical duplicate error")
	}
}

func TestFailed_PreservesCaseAndCategory(t *testing.T) {
	t.Parallel()

	out := Failed(GoldenCase{
		ID:          "failed",
		Query:       "query",
		K:           1,
		ExpectEmpty: true,
		Labels:      map[string]string{"suite": "manual", "pipeline": "search"},
	}, "timeout")
	if out.ErrorCategory != "timeout" || !out.EmptyExpected || out.ExactEmpty {
		t.Fatalf("unexpected failed outcome: %#v", out)
	}
	if out.Case.ID != "failed" || out.Case.Labels["pipeline"] != "search" || out.Case.Labels["suite"] != "manual" {
		t.Fatalf("case was not preserved: %#v", out.Case)
	}
}

func TestFailed_SanitizesErrorCategory(t *testing.T) {
	t.Parallel()

	c := GoldenCase{ID: "failed", Query: "query", K: 1, ExpectEmpty: true}
	for _, category := range []string{"provider error: secret=credential", "UPPERCASE", strings.Repeat("a", 65)} {
		if got := Failed(c, category).ErrorCategory; got != "unspecified" {
			t.Fatalf("Failed(%q).ErrorCategory = %q, want unspecified", category, got)
		}
	}
	if got := Failed(c, "semantic_search").ErrorCategory; got != "semantic_search" {
		t.Fatalf("Failed(valid).ErrorCategory = %q", got)
	}
}
