package eval

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
)

// ScoreDomainLabel is the required case label used to isolate score domains.
const ScoreDomainLabel = "score_domain"

// FloorEvaluation contains aggregate quality after one inclusive score floor.
type FloorEvaluation struct {
	Floor           float32 `json:"floor"`
	RetainedResults int     `json:"retained_results"`
	Metrics         Metrics `json:"metrics"`
}

// CandidateFloors returns exact observed keep/drop boundaries for one score
// domain. The caller remains responsible for choosing a production floor.
func CandidateFloors(outcomes []Outcome, scoreDomain string) ([]float32, error) {
	if err := validateScoreDomain(outcomes, scoreDomain); err != nil {
		return nil, err
	}
	unique := make(map[float32]struct{})
	for _, outcome := range outcomes {
		if outcome.Status == OutcomeStatusFailed {
			continue
		}
		for _, result := range outcome.Results {
			if !finite32(result.Score) {
				return nil, fmt.Errorf("case %q has nonfinite score", caseID(outcome.Case))
			}
			unique[result.Score] = struct{}{}
			next := math.Nextafter32(result.Score, float32(math.Inf(1)))
			if finite32(next) {
				unique[next] = struct{}{}
			}
		}
	}
	floors := make([]float32, 0, len(unique))
	for floor := range unique {
		floors = append(floors, floor)
	}
	sort.Slice(floors, func(i, j int) bool { return floors[i] < floors[j] })
	return floors, nil
}

// SweepResultFloors re-evaluates outcomes after applying each inclusive floor.
// Callers should pass a bounded candidate set; exhaustive observed boundaries
// can be expensive for large reports.
func SweepResultFloors(ctx context.Context, outcomes []Outcome, scoreDomain string, floors []float32) ([]FloorEvaluation, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is required")
	}
	if err := validateScoreDomain(outcomes, scoreDomain); err != nil {
		return nil, err
	}
	evaluations := make([]FloorEvaluation, 0, len(floors))
	for _, floor := range floors {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("sweep floors: %w", err)
		}
		if !finite32(floor) {
			return nil, fmt.Errorf("floor must be finite")
		}
		filteredOutcomes := make([]Outcome, 0, len(outcomes))
		retained := 0
		for _, outcome := range outcomes {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("sweep floor %v: %w", floor, err)
			}
			if outcome.Status == OutcomeStatusFailed {
				filteredOutcomes = append(filteredOutcomes, Failed(outcome.Case, outcome.ErrorCategory))
				continue
			}
			results := make([]Result, 0, len(outcome.Results))
			for _, result := range outcome.Results {
				if result.Score >= floor {
					results = append(results, result)
				}
			}
			reevaluated, err := Evaluate(outcome.Case, results)
			if err != nil {
				return nil, err
			}
			retained += len(results)
			filteredOutcomes = append(filteredOutcomes, reevaluated)
		}
		evaluations = append(evaluations, FloorEvaluation{
			Floor: floor, RetainedResults: retained, Metrics: Summarize(filteredOutcomes),
		})
	}
	return evaluations, nil
}

func validateScoreDomain(outcomes []Outcome, scoreDomain string) error {
	scoreDomain = strings.TrimSpace(scoreDomain)
	if scoreDomain == "" {
		return fmt.Errorf("score domain is required")
	}
	for _, outcome := range outcomes {
		if err := ValidateCase(outcome.Case); err != nil {
			return err
		}
		actual := strings.TrimSpace(outcome.Case.Labels[ScoreDomainLabel])
		if actual != scoreDomain {
			return fmt.Errorf("case %q has score domain %q, want %q", caseID(outcome.Case), actual, scoreDomain)
		}
		switch outcome.Status {
		case OutcomeStatusSuccess:
			for _, result := range outcome.Results {
				if err := validateGoldenKey(normalizeGoldenKey(result.Key)); err != nil {
					return fmt.Errorf("case %q has invalid result key: %w", caseID(outcome.Case), err)
				}
				if !finite32(result.Score) {
					return fmt.Errorf("case %q has nonfinite score", caseID(outcome.Case))
				}
			}
		case OutcomeStatusFailed:
			continue
		default:
			return fmt.Errorf("case %q has invalid outcome status %q", caseID(outcome.Case), outcome.Status)
		}
	}
	return nil
}

func finite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
