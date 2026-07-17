package eval

import "testing"

func TestCompare_CompatibleCandidateAndIdentityMismatches(t *testing.T) {
	t.Parallel()

	baseline, current := comparisonReports(t)
	comparison, err := Compare(baseline, current, Tolerances{})
	if err != nil {
		t.Fatalf("Compare() error = %v", err)
	}
	if !comparison.Compatible || comparison.Regressed() {
		t.Fatalf("equivalent candidate reports did not compare cleanly: %#v", comparison)
	}

	for _, tt := range []struct {
		name   string
		mutate func(*Report)
	}{
		{name: "dataset", mutate: func(r *Report) { r.Identity.DatasetID = "other" }},
		{name: "suite", mutate: func(r *Report) { r.Identity.SuiteID = "other" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			changed := current
			tt.mutate(&changed)
			changed, err = BuildReport(changed.Identity, changed.Outcomes, "suite")
			if err != nil {
				t.Fatalf("BuildReport(changed identity) error = %v", err)
			}
			comparison, err := Compare(baseline, changed, Tolerances{})
			if err != nil {
				t.Fatalf("Compare() error = %v", err)
			}
			if comparison.Compatible {
				t.Fatalf("identity mismatch was compatible: %#v", comparison)
			}
		})
	}
}

func TestCompare_DetectsQualityAndFailureRegressions(t *testing.T) {
	t.Parallel()

	baseline, _ := comparisonReports(t)
	judged := baseline.Outcomes[1].Case
	empty := baseline.Outcomes[0].Case
	miss, err := Evaluate(judged, []Result{{Key: GoldenKey{EntityType: "gallery", EntityID: "other"}, Score: 1}})
	if err != nil {
		t.Fatalf("Evaluate(miss) error = %v", err)
	}
	emptyOutcome, err := Evaluate(empty, nil)
	if err != nil {
		t.Fatalf("Evaluate(empty) error = %v", err)
	}
	current, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "miss"},
		[]Outcome{miss, emptyOutcome}, "suite",
	)
	if err != nil {
		t.Fatalf("BuildReport(miss) error = %v", err)
	}
	comparison, err := Compare(baseline, current, Tolerances{})
	if err != nil {
		t.Fatalf("Compare(miss) error = %v", err)
	}
	if !comparison.Compatible || !comparison.Regressed() {
		t.Fatalf("quality regression not detected: %#v", comparison)
	}
	comparison, err = Compare(baseline, current, Tolerances{
		RecallAtKDrop: 1, SuccessAtKDrop: 1, MRRAtKDrop: 1, NDCGAtKDrop: 1,
	})
	if err != nil {
		t.Fatalf("Compare(tolerated miss) error = %v", err)
	}
	if comparison.Regressed() {
		t.Fatalf("drop at tolerance boundary regressed: %#v", comparison)
	}

	failed, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "failed"},
		[]Outcome{Failed(judged, "timeout"), emptyOutcome}, "suite",
	)
	if err != nil {
		t.Fatalf("BuildReport(failed) error = %v", err)
	}
	comparison, err = Compare(baseline, failed, Tolerances{})
	if err != nil {
		t.Fatalf("Compare(failed) error = %v", err)
	}
	if !comparison.Regressed() {
		t.Fatalf("failure regression not detected: %#v", comparison)
	}
}

func TestCompare_RejectsScopeCaseAndAggregateDrift(t *testing.T) {
	t.Parallel()

	baseline, current := comparisonReports(t)
	withoutScope, err := BuildReport(current.Identity, current.Outcomes)
	if err != nil {
		t.Fatalf("BuildReport(without scope) error = %v", err)
	}
	comparison, err := Compare(baseline, withoutScope, Tolerances{})
	if err != nil {
		t.Fatalf("Compare(without scope) error = %v", err)
	}
	if comparison.Compatible {
		t.Fatalf("missing scope was compatible: %#v", comparison)
	}

	fewer, err := BuildReport(current.Identity, current.Outcomes[:1], "suite")
	if err != nil {
		t.Fatalf("BuildReport(fewer) error = %v", err)
	}
	comparison, err = Compare(baseline, fewer, Tolerances{})
	if err != nil {
		t.Fatalf("Compare(fewer) error = %v", err)
	}
	if comparison.Compatible {
		t.Fatalf("case-count mismatch was compatible: %#v", comparison)
	}

	changedOutcomes := append([]Outcome(nil), current.Outcomes...)
	changedOutcomes[0].Case.Query = "different query"
	changed, err := BuildReport(current.Identity, changedOutcomes, "suite")
	if err != nil {
		t.Fatalf("BuildReport(changed case) error = %v", err)
	}
	comparison, err = Compare(baseline, changed, Tolerances{})
	if err != nil {
		t.Fatalf("Compare(changed case) error = %v", err)
	}
	if comparison.Compatible {
		t.Fatalf("case-definition mismatch was compatible: %#v", comparison)
	}

	tampered := current
	tampered.Metrics.RecallAtK = 0
	if _, err := Compare(baseline, tampered, Tolerances{}); err == nil {
		t.Fatal("Compare() error = nil, want tampered aggregate error")
	}
	tampered = current
	tampered.Outcomes[1].Results[0].Score = 0.5
	if _, err := Compare(baseline, tampered, Tolerances{}); err == nil {
		t.Fatal("Compare() error = nil, want tampered result score error")
	}
	_, current = comparisonReports(t)
	tampered = current
	delete(tampered.Breakdowns, "suite")
	if _, err := Compare(baseline, tampered, Tolerances{}); err == nil {
		t.Fatal("Compare() error = nil, want missing declared breakdown error")
	}
}

func TestCompare_DistinguishesDelimiterLikeScopes(t *testing.T) {
	t.Parallel()

	a := GoldenCase{
		ID: "a", Query: "a", K: 1, Expected: []GoldenKey{{EntityType: "gallery", EntityID: "a"}},
		Labels: map[string]string{"a": "b=c"},
	}
	b := GoldenCase{
		ID: "b", Query: "b", K: 1, Expected: []GoldenKey{{EntityType: "gallery", EntityID: "b"}},
		Labels: map[string]string{"a=b": "c"},
	}
	aHit, _ := Evaluate(a, []Result{{Key: a.Expected[0], Score: 1}})
	bHit, _ := Evaluate(b, []Result{{Key: b.Expected[0], Score: 1}})
	identity := ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "before"}
	baseline, err := BuildReport(identity, []Outcome{aHit, bHit}, "a", "a=b")
	if err != nil {
		t.Fatalf("BuildReport(baseline) error = %v", err)
	}
	bMiss, _ := Evaluate(b, nil)
	identity.CandidateID = "after"
	current, err := BuildReport(identity, []Outcome{aHit, bMiss}, "a", "a=b")
	if err != nil {
		t.Fatalf("BuildReport(current) error = %v", err)
	}
	comparison, err := Compare(baseline, current, Tolerances{})
	if err != nil {
		t.Fatalf("Compare() error = %v", err)
	}
	found := false
	for _, regression := range comparison.Regressions {
		if regression.Scope == `a=b="c"` {
			found = true
		}
	}
	if !found {
		t.Fatalf("delimiter-like scope regression missing: %#v", comparison)
	}
}

func TestCompare_RejectsInvalidInputs(t *testing.T) {
	t.Parallel()

	if _, err := Compare(Report{}, Report{}, Tolerances{RecallAtKDrop: -1}); err == nil {
		t.Fatal("Compare() error = nil, want invalid tolerance error")
	}
	if _, err := Compare(Report{SchemaVersion: 99}, Report{SchemaVersion: 99}, Tolerances{}); err == nil {
		t.Fatal("Compare() error = nil, want invalid report error")
	}
}

func comparisonReports(t *testing.T) (Report, Report) {
	t.Helper()
	judged := GoldenCase{
		ID: "judged", Query: "query", K: 1,
		Expected: []GoldenKey{{EntityType: "gallery", EntityID: "1"}},
		Labels:   map[string]string{"suite": "manual"},
	}
	empty := GoldenCase{
		ID: "empty", Query: "nonsense", K: 1, ExpectEmpty: true,
		Labels: map[string]string{"suite": "nonsense"},
	}
	judgedOutcome, err := Evaluate(judged, []Result{{Key: judged.Expected[0], Score: 1}})
	if err != nil {
		t.Fatalf("Evaluate(judged) error = %v", err)
	}
	emptyOutcome, err := Evaluate(empty, nil)
	if err != nil {
		t.Fatalf("Evaluate(empty) error = %v", err)
	}
	baseline, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "before"},
		[]Outcome{judgedOutcome, emptyOutcome}, "suite",
	)
	if err != nil {
		t.Fatalf("BuildReport(baseline) error = %v", err)
	}
	current, err := BuildReport(
		ReportIdentity{DatasetID: "dataset", SuiteID: "suite", CandidateID: "after"},
		[]Outcome{judgedOutcome, emptyOutcome}, "suite",
	)
	if err != nil {
		t.Fatalf("BuildReport(current) error = %v", err)
	}
	return baseline, current
}
