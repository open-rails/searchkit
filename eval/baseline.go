package eval

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
)

// Tolerances defines maximum allowed absolute metric drops and failure growth.
type Tolerances struct {
	RecallAtKDrop      float64 `json:"recall_at_k_drop"`
	SuccessAtKDrop     float64 `json:"success_at_k_drop"`
	MRRAtKDrop         float64 `json:"mrr_at_k_drop"`
	NDCGAtKDrop        float64 `json:"ndcg_at_k_drop"`
	ExactEmptyRateDrop float64 `json:"exact_empty_rate_drop"`
	FailedCaseIncrease int     `json:"failed_case_increase"`
}

// Regression identifies one metric outside its configured tolerance.
type Regression struct {
	Scope    string  `json:"scope"`
	Metric   string  `json:"metric"`
	Baseline float64 `json:"baseline"`
	Current  float64 `json:"current"`
	Delta    float64 `json:"delta"`
}

// Comparison describes report compatibility and measured regressions.
type Comparison struct {
	Compatible  bool         `json:"compatible"`
	Mismatches  []string     `json:"mismatches,omitempty"`
	Regressions []Regression `json:"regressions,omitempty"`
}

// Regressed reports whether any compared metric exceeded tolerance.
func (c Comparison) Regressed() bool {
	return len(c.Regressions) > 0
}

// Compare rejects incompatible reports and checks every common metric scope.
func Compare(baseline Report, current Report, tolerances Tolerances) (Comparison, error) {
	if err := validateTolerances(tolerances); err != nil {
		return Comparison{}, err
	}
	if err := validateReport(baseline); err != nil {
		return Comparison{}, fmt.Errorf("invalid baseline report: %w", err)
	}
	if err := validateReport(current); err != nil {
		return Comparison{}, fmt.Errorf("invalid current report: %w", err)
	}
	comparison := Comparison{Compatible: true}
	if baseline.SchemaVersion != ReportSchemaVersion {
		comparison.Mismatches = append(comparison.Mismatches, "unsupported_baseline_schema")
	}
	if current.SchemaVersion != ReportSchemaVersion {
		comparison.Mismatches = append(comparison.Mismatches, "unsupported_current_schema")
	}
	if err := validateIdentity(baseline.Identity); err != nil {
		comparison.Mismatches = append(comparison.Mismatches, "invalid_baseline_identity")
	}
	if err := validateIdentity(current.Identity); err != nil {
		comparison.Mismatches = append(comparison.Mismatches, "invalid_current_identity")
	}
	if baseline.SchemaVersion != current.SchemaVersion {
		comparison.Mismatches = append(comparison.Mismatches, "schema_version")
	}
	if baseline.Identity.DatasetID != current.Identity.DatasetID {
		comparison.Mismatches = append(comparison.Mismatches, "dataset_id")
	}
	if baseline.Identity.SuiteID != current.Identity.SuiteID {
		comparison.Mismatches = append(comparison.Mismatches, "suite_id")
	}
	if baseline.Metrics.Cases != current.Metrics.Cases {
		comparison.Mismatches = append(comparison.Mismatches, "case_count")
	}
	baselineCases, err := reportCasesHash(baseline)
	if err != nil {
		return Comparison{}, fmt.Errorf("hash baseline cases: %w", err)
	}
	currentCases, err := reportCasesHash(current)
	if err != nil {
		return Comparison{}, fmt.Errorf("hash current cases: %w", err)
	}
	if baselineCases != currentCases {
		comparison.Mismatches = append(comparison.Mismatches, "case_definitions")
	}

	baselineScopes := metricScopes(baseline)
	currentScopes := metricScopes(current)
	for scope := range baselineScopes {
		if _, ok := currentScopes[scope]; !ok {
			comparison.Mismatches = append(comparison.Mismatches, "missing_current_scope:"+scope.String())
		}
	}
	for scope := range currentScopes {
		if _, ok := baselineScopes[scope]; !ok {
			comparison.Mismatches = append(comparison.Mismatches, "missing_baseline_scope:"+scope.String())
		}
	}
	sort.Strings(comparison.Mismatches)
	if len(comparison.Mismatches) > 0 {
		comparison.Compatible = false
		return comparison, nil
	}

	scopes := make([]metricScope, 0, len(baselineScopes))
	for scope := range baselineScopes {
		scopes = append(scopes, scope)
	}
	sort.Slice(scopes, func(i, j int) bool { return scopes[i].String() < scopes[j].String() })
	for _, scope := range scopes {
		compareMetrics(&comparison, scope.String(), baselineScopes[scope], currentScopes[scope], tolerances)
	}
	return comparison, nil
}

func validateReport(report Report) error {
	if report.SchemaVersion != ReportSchemaVersion {
		return fmt.Errorf("unsupported schema version %d", report.SchemaVersion)
	}
	if err := validateIdentity(report.Identity); err != nil {
		return err
	}
	rebuilt, err := BuildReport(report.Identity, report.Outcomes, report.GroupLabels...)
	if err != nil {
		return err
	}
	if report.ContentID != rebuilt.ContentID {
		return fmt.Errorf("content id does not match identity and outcomes")
	}
	if !reflect.DeepEqual(report.Metrics, rebuilt.Metrics) {
		return fmt.Errorf("aggregate metrics do not match outcomes")
	}
	if !reflect.DeepEqual(report.Breakdowns, rebuilt.Breakdowns) {
		return fmt.Errorf("breakdowns do not match outcomes")
	}
	gotOutcomes, err := json.Marshal(report.Outcomes)
	if err != nil {
		return fmt.Errorf("marshal report outcomes: %w", err)
	}
	wantOutcomes, err := json.Marshal(rebuilt.Outcomes)
	if err != nil {
		return fmt.Errorf("marshal normalized outcomes: %w", err)
	}
	if string(gotOutcomes) != string(wantOutcomes) {
		return fmt.Errorf("outcomes are not normalized or contain inconsistent metrics")
	}
	return nil
}

func reportCasesHash(report Report) (string, error) {
	cases := make([]GoldenCase, 0, len(report.Outcomes))
	for _, outcome := range report.Outcomes {
		cases = append(cases, outcome.Case)
	}
	return HashSuite(Suite{ID: "report-cases", Cases: cases})
}

func compareMetrics(comparison *Comparison, scope string, baseline Metrics, current Metrics, tolerances Tolerances) {
	addDrop := func(metric string, before float64, after float64, allowed float64) {
		if after < before-allowed {
			comparison.Regressions = append(comparison.Regressions, Regression{
				Scope: scope, Metric: metric, Baseline: before, Current: after, Delta: after - before,
			})
		}
	}
	if baseline.JudgedCases > 0 && current.JudgedCases > 0 {
		addDrop("recall_at_k", baseline.RecallAtK, current.RecallAtK, tolerances.RecallAtKDrop)
		addDrop("success_at_k", baseline.SuccessAtK, current.SuccessAtK, tolerances.SuccessAtKDrop)
		addDrop("mrr_at_k", baseline.MRRAtK, current.MRRAtK, tolerances.MRRAtKDrop)
		addDrop("ndcg_at_k", baseline.NDCGAtK, current.NDCGAtK, tolerances.NDCGAtKDrop)
	}
	if baseline.EmptyCases > 0 && current.EmptyCases > 0 {
		addDrop("exact_empty_rate", baseline.ExactEmptyRate, current.ExactEmptyRate, tolerances.ExactEmptyRateDrop)
	}
	failedIncrease := current.FailedCases - baseline.FailedCases
	if failedIncrease > tolerances.FailedCaseIncrease {
		comparison.Regressions = append(comparison.Regressions, Regression{
			Scope: scope, Metric: "failed_cases", Baseline: float64(baseline.FailedCases),
			Current: float64(current.FailedCases), Delta: float64(failedIncrease),
		})
	}
}

type metricScope struct {
	Label string
	Value string
}

func (s metricScope) String() string {
	if s.Label == "" {
		return "overall"
	}
	value, _ := json.Marshal(s.Value)
	return s.Label + "=" + string(value)
}

func metricScopes(report Report) map[metricScope]Metrics {
	out := map[metricScope]Metrics{{}: report.Metrics}
	for label, values := range report.Breakdowns {
		for value, metrics := range values {
			out[metricScope{Label: label, Value: value}] = metrics
		}
	}
	return out
}

func validateTolerances(t Tolerances) error {
	values := []float64{t.RecallAtKDrop, t.SuccessAtKDrop, t.MRRAtKDrop, t.NDCGAtKDrop, t.ExactEmptyRateDrop}
	for _, value := range values {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("metric tolerances must be finite and nonnegative")
		}
	}
	if t.FailedCaseIncrease < 0 {
		return fmt.Errorf("failed case increase must be nonnegative")
	}
	return nil
}
