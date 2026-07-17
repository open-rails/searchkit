package eval

import (
	"context"
	"math"
	"testing"
)

func TestCandidateFloors_ProducesExactKeepDropBoundaries(t *testing.T) {
	t.Parallel()

	outcomes := floorOutcomes(t)
	floors, err := CandidateFloors(outcomes, "cosine_similarity")
	if err != nil {
		t.Fatalf("CandidateFloors() error = %v", err)
	}
	want := map[float32]bool{
		0.2: true,
		math.Nextafter32(0.2, float32(math.Inf(1))): true,
		0.8: true,
		math.Nextafter32(0.8, float32(math.Inf(1))): true,
	}
	if len(floors) != len(want) {
		t.Fatalf("len(floors) = %d, want %d: %v", len(floors), len(want), floors)
	}
	for _, floor := range floors {
		if !want[floor] {
			t.Fatalf("unexpected floor %v", floor)
		}
	}
}

func TestSweepResultFloors_InclusiveAndOrderPreserving(t *testing.T) {
	t.Parallel()

	outcomes := floorOutcomes(t)
	evaluations, err := SweepResultFloors(context.Background(), outcomes, "cosine_similarity", []float32{0.2, math.Nextafter32(0.2, float32(math.Inf(1))), 0.8})
	if err != nil {
		t.Fatalf("SweepResultFloors() error = %v", err)
	}
	if evaluations[0].RetainedResults != 2 || evaluations[0].Metrics.RecallAtK != 1 {
		t.Fatalf("inclusive floor did not retain tied boundary: %#v", evaluations[0])
	}
	if evaluations[1].RetainedResults != 1 || evaluations[1].Metrics.RecallAtK != 0 {
		t.Fatalf("next boundary did not drop low score: %#v", evaluations[1])
	}
	if evaluations[2].RetainedResults != 1 || evaluations[2].Metrics.ExactEmptyRate != 1 {
		t.Fatalf("high floor metrics unexpected: %#v", evaluations[2])
	}
}

func TestFloorHelpers_RejectMixedDomainsAndNonfiniteFloor(t *testing.T) {
	t.Parallel()

	outcomes := floorOutcomes(t)
	outcomes[1].Case.Labels[ScoreDomainLabel] = "rrf"
	if _, err := CandidateFloors(outcomes, "cosine_similarity"); err == nil {
		t.Fatal("CandidateFloors() error = nil, want mixed domain error")
	}
	failed := Failed(GoldenCase{
		ID: "failed", Query: "query", K: 1,
		Labels: map[string]string{ScoreDomainLabel: "rrf"},
	}, "timeout")
	if _, err := CandidateFloors([]Outcome{failed}, "cosine_similarity"); err == nil {
		t.Fatal("CandidateFloors() error = nil, want failed-outcome domain error")
	}
	if _, err := SweepResultFloors(context.Background(), floorOutcomes(t), "cosine_similarity", []float32{float32(math.NaN())}); err == nil {
		t.Fatal("SweepResultFloors() error = nil, want nonfinite floor error")
	}
	malformed := Outcome{
		Status: OutcomeStatusSuccess,
		Case: GoldenCase{
			ID: "malformed", Query: "query", K: 1,
			Labels: map[string]string{ScoreDomainLabel: "cosine_similarity"},
		},
		Results: []Result{{Key: GoldenKey{EntityType: "gallery", EntityID: "1"}, Score: float32(math.NaN())}},
	}
	if _, err := SweepResultFloors(context.Background(), []Outcome{malformed}, "cosine_similarity", []float32{0}); err == nil {
		t.Fatal("SweepResultFloors() error = nil, want malformed score error")
	}
	malformed.Results = []Result{{Key: GoldenKey{EntityType: "gallery"}, Score: -1}}
	if _, err := SweepResultFloors(context.Background(), []Outcome{malformed}, "cosine_similarity", []float32{0}); err == nil {
		t.Fatal("SweepResultFloors() error = nil, want invalid key error before filtering")
	}
}

func TestSweepResultFloors_HonorsCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SweepResultFloors(ctx, floorOutcomes(t), "cosine_similarity", []float32{0}); err == nil {
		t.Fatal("SweepResultFloors() error = nil, want cancellation")
	}
}

func floorOutcomes(t *testing.T) []Outcome {
	t.Helper()
	relevant := GoldenKey{EntityType: "gallery", EntityID: "relevant"}
	garbage := GoldenKey{EntityType: "gallery", EntityID: "garbage"}
	labels := func() map[string]string { return map[string]string{ScoreDomainLabel: "cosine_similarity"} }
	judged, err := Evaluate(
		GoldenCase{ID: "judged", Query: "query", K: 2, Expected: []GoldenKey{relevant}, Labels: labels()},
		[]Result{{Key: garbage, Score: 0.8}, {Key: relevant, Score: 0.2}},
	)
	if err != nil {
		t.Fatalf("Evaluate(judged) error = %v", err)
	}
	empty, err := Evaluate(
		GoldenCase{ID: "empty", Query: "nonsense", K: 2, ExpectEmpty: true, Labels: labels()},
		nil,
	)
	if err != nil {
		t.Fatalf("Evaluate(empty) error = %v", err)
	}
	return []Outcome{judged, empty}
}
