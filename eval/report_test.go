package eval

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestBuildReport_NormalizesAndGroupsOutcomes(t *testing.T) {
	t.Parallel()

	a := GoldenKey{EntityType: "gallery", EntityID: "a"}
	b := GoldenKey{EntityType: "gallery", EntityID: "b"}
	manual := GoldenCase{
		ID: "manual", Query: "manual", K: 2, Expected: []GoldenKey{a},
		Labels: map[string]string{"suite": "manual", ScoreDomainLabel: "rrf"},
	}
	nonsense := GoldenCase{
		ID: "nonsense", Query: "qwerty", K: 2, ExpectEmpty: true,
		Labels: map[string]string{"suite": "nonsense", ScoreDomainLabel: "rrf"},
	}
	failure := GoldenCase{
		ID: "failure", Query: "failure", K: 2, Expected: []GoldenKey{b},
		Labels: map[string]string{"suite": "manual", ScoreDomainLabel: "rrf"},
	}

	report, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "candidate"},
		[]Outcome{
			Failed(failure, "timeout"),
			{Status: OutcomeStatusSuccess, Case: nonsense},
			{Status: OutcomeStatusSuccess, Case: manual, Results: []Result{{Key: a, Score: 0.2}}, RecallAtK: 0},
		},
		"suite",
	)
	if err != nil {
		t.Fatalf("BuildReport() error = %v", err)
	}
	if report.SchemaVersion != ReportSchemaVersion {
		t.Fatalf("SchemaVersion = %d, want %d", report.SchemaVersion, ReportSchemaVersion)
	}
	if got := []string{report.Outcomes[0].Case.ID, report.Outcomes[1].Case.ID, report.Outcomes[2].Case.ID}; !reflect.DeepEqual(got, []string{"failure", "manual", "nonsense"}) {
		t.Fatalf("outcomes not sorted: %v", got)
	}
	if report.Outcomes[1].RecallAtK != 1 {
		t.Fatalf("successful outcome was not recomputed: %#v", report.Outcomes[1])
	}
	if report.Metrics.Cases != 3 || report.Metrics.SuccessfulCases != 2 || report.Metrics.FailedCases != 1 {
		t.Fatalf("unexpected counts: %#v", report.Metrics)
	}
	if report.Metrics.JudgedCases != 1 || report.Metrics.RecallAtK != 1 || report.Metrics.ExactEmptyRate != 1 {
		t.Fatalf("unexpected quality metrics: %#v", report.Metrics)
	}
	if report.Metrics.MinResults != 0 || report.Metrics.MaxResults != 1 || report.Metrics.MeanResults != 0.5 || report.Metrics.MedianResults != 0.5 {
		t.Fatalf("unexpected result distribution: %#v", report.Metrics)
	}
	if report.Breakdowns["suite"]["manual"].Cases != 2 || report.Breakdowns["suite"]["nonsense"].Cases != 1 {
		t.Fatalf("unexpected breakdowns: %#v", report.Breakdowns)
	}
}

func TestBuildReport_RejectsDuplicateCases(t *testing.T) {
	t.Parallel()

	c := GoldenCase{ID: "same", Query: "query", K: 1}
	_, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "candidate"},
		[]Outcome{{Status: OutcomeStatusSuccess, Case: c}, {Status: OutcomeStatusSuccess, Case: c}},
	)
	if err == nil {
		t.Fatal("BuildReport() error = nil, want duplicate case error")
	}
}

func TestBuildReport_NormalizesGroupLabelsBeforeSorting(t *testing.T) {
	t.Parallel()

	report, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "candidate"},
		[]Outcome{{
			Status: OutcomeStatusSuccess,
			Case:   GoldenCase{ID: "case", Query: "query", K: 1},
		}},
		" b", "a",
	)
	if err != nil {
		t.Fatalf("BuildReport() error = %v", err)
	}
	if !reflect.DeepEqual(report.GroupLabels, []string{"a", "b"}) {
		t.Fatalf("GroupLabels = %#v, want canonical ordering", report.GroupLabels)
	}
	comparison, err := Compare(report, report, Tolerances{})
	if err != nil {
		t.Fatalf("Compare(report, report) error = %v", err)
	}
	if !comparison.Compatible {
		t.Fatalf("self-comparison incompatible: %#v", comparison)
	}
}

func TestBuildReport_EmptyFailureCannotBecomeExactEmpty(t *testing.T) {
	t.Parallel()

	c := GoldenCase{ID: "failure", Query: "nonsense", K: 1, ExpectEmpty: true}
	report, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "candidate"},
		[]Outcome{Failed(c, " ")},
	)
	if err != nil {
		t.Fatalf("BuildReport() error = %v", err)
	}
	if report.Metrics.FailedCases != 1 || report.Metrics.EmptyCases != 0 || report.Metrics.ExactEmptyRate != 0 {
		t.Fatalf("failure was scored as empty success: %#v", report.Metrics)
	}
	if report.Outcomes[0].Status != OutcomeStatusFailed || report.Outcomes[0].ErrorCategory != "unspecified" {
		t.Fatalf("failure status/category lost: %#v", report.Outcomes[0])
	}
}

func TestBuildReport_RejectsUnknownOutcomeStatus(t *testing.T) {
	t.Parallel()

	_, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "candidate"},
		[]Outcome{{Case: GoldenCase{ID: "unknown", Query: "query", K: 1}}},
	)
	if err == nil {
		t.Fatal("BuildReport() error = nil, want unknown status error")
	}
}

func TestBuildReport_RoundTripPreservesOrderAndScores(t *testing.T) {
	t.Parallel()

	key := GoldenKey{EntityType: "gallery", EntityID: "1"}
	report, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "candidate"},
		[]Outcome{{
			Status:  OutcomeStatusSuccess,
			Case:    GoldenCase{ID: "case", Query: "query", K: 1, Expected: []GoldenKey{key}},
			Results: []Result{{Key: key, Score: 0.12345679}},
		}},
	)
	if err != nil {
		t.Fatalf("BuildReport() error = %v", err)
	}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	var decoded Report
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if !reflect.DeepEqual(decoded, report) {
		t.Fatalf("round trip mismatch:\n got %#v\nwant %#v", decoded, report)
	}
}

func TestHashSuite_IsDeterministicForLabelMapOrder(t *testing.T) {
	t.Parallel()

	base := GoldenCase{
		ID: "case", Query: "query", K: 1,
		Labels: map[string]string{"suite": "manual", "pipeline": "search"},
	}
	a, err := HashSuite(Suite{ID: "suite", Cases: []GoldenCase{base}})
	if err != nil {
		t.Fatalf("HashSuite(a) error = %v", err)
	}
	base.Labels = map[string]string{"pipeline": "search", "suite": "manual"}
	b, err := HashSuite(Suite{ID: "suite", Cases: []GoldenCase{base}})
	if err != nil {
		t.Fatalf("HashSuite(b) error = %v", err)
	}
	if a != b {
		t.Fatalf("HashSuite() differs by map insertion order: %q != %q", a, b)
	}
}
