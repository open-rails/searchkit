package eval

import (
	"context"
	"errors"
	"testing"
)

// stubRunner returns canned results (or an error) per case id, so RunSuite can
// be exercised without any retrieval backend.
type stubRunner struct {
	results  map[string][]Result
	failWith map[string]string // case id -> error category
}

func (s stubRunner) Run(_ context.Context, c GoldenCase) ([]Result, string, error) {
	if category, ok := s.failWith[c.ID]; ok {
		return nil, category, errors.New("boom")
	}
	return s.results[c.ID], "", nil
}

func testIdentity() ReportIdentity {
	return ReportIdentity{DatasetID: "ds", SuiteID: "suite", CandidateID: "cand"}
}

func TestRunSuite_MixedHitFailAndEmpty(t *testing.T) {
	suite := Suite{
		ID: "suite",
		Cases: []GoldenCase{
			{ID: "hit", Query: "q", K: 3, Judgments: []Judgment{
				{Key: GoldenKey{EntityType: "gallery", EntityID: "1"}, Relevance: 3},
			}},
			{ID: "fail", Query: "q", K: 3, Expected: []GoldenKey{
				{EntityType: "gallery", EntityID: "9"},
			}},
			{ID: "empty", Query: "q", K: 3, ExpectEmpty: true},
		},
	}
	runner := stubRunner{
		results: map[string][]Result{
			"hit":   {{Key: GoldenKey{EntityType: "gallery", EntityID: "1"}, Score: 0.9}},
			"empty": {},
		},
		failWith: map[string]string{"fail": "search"},
	}

	report, err := RunSuite(context.Background(), suite, runner, testIdentity())
	if err != nil {
		t.Fatalf("RunSuite() error = %v", err)
	}
	if report.Metrics.Cases != 3 {
		t.Fatalf("Cases = %d, want 3", report.Metrics.Cases)
	}
	if report.Metrics.FailedCases != 1 {
		t.Fatalf("FailedCases = %d, want 1", report.Metrics.FailedCases)
	}
	if report.Metrics.JudgedCases != 1 {
		t.Fatalf("JudgedCases = %d, want 1", report.Metrics.JudgedCases)
	}
	if report.Metrics.SuccessAtK != 1 {
		t.Fatalf("SuccessAtK = %v, want 1 (only judged case is a hit)", report.Metrics.SuccessAtK)
	}
	if report.Metrics.ExactEmptyRate != 1 {
		t.Fatalf("ExactEmptyRate = %v, want 1", report.Metrics.ExactEmptyRate)
	}

	// The failed case must carry a sanitized error category, not metrics.
	var failed *Outcome
	for i := range report.Outcomes {
		if report.Outcomes[i].Case.ID == "fail" {
			failed = &report.Outcomes[i]
		}
	}
	if failed == nil || failed.Status != OutcomeStatusFailed || failed.ErrorCategory != "search" {
		t.Fatalf("failed outcome = %+v, want status=failed category=search", failed)
	}
}

func TestRunSuite_Guards(t *testing.T) {
	suite := Suite{ID: "suite", Cases: []GoldenCase{{ID: "a", Query: "q", K: 1, ExpectEmpty: true}}}
	runner := stubRunner{results: map[string][]Result{"a": {}}}

	if _, err := RunSuite(context.Background(), suite, nil, testIdentity()); err == nil {
		t.Fatal("RunSuite(nil runner) error = nil, want error")
	}
	if _, err := RunSuite(context.Background(), Suite{ID: "empty"}, runner, testIdentity()); err == nil {
		t.Fatal("RunSuite(no cases) error = nil, want error")
	}
	if _, err := RunSuite(context.Background(), suite, runner, ReportIdentity{}); err == nil {
		t.Fatal("RunSuite(invalid identity) error = nil, want error")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := RunSuite(cancelled, suite, runner, testIdentity()); err == nil {
		t.Fatal("RunSuite(cancelled ctx) error = nil, want error")
	}
}
