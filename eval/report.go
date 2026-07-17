package eval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ReportSchemaVersion is the current serialized report contract.
const ReportSchemaVersion = 1

// ReportIdentity binds a report to corpus, suite, and candidate configuration.
type ReportIdentity struct {
	DatasetID   string `json:"dataset_id"`
	SuiteID     string `json:"suite_id"`
	CandidateID string `json:"candidate_id"`
}

// Metrics contains macro-averaged quality and result-count measurements.
type Metrics struct {
	Cases           int `json:"cases"`
	SuccessfulCases int `json:"successful_cases"`
	FailedCases     int `json:"failed_cases"`
	JudgedCases     int `json:"judged_cases"`
	EmptyCases      int `json:"empty_cases"`

	RecallAtK      float64 `json:"recall_at_k"`
	SuccessAtK     float64 `json:"success_at_k"`
	MRRAtK         float64 `json:"mrr_at_k"`
	NDCGAtK        float64 `json:"ndcg_at_k"`
	ExactEmptyRate float64 `json:"exact_empty_rate"`

	MinResults    int     `json:"min_results"`
	MaxResults    int     `json:"max_results"`
	MeanResults   float64 `json:"mean_results"`
	MedianResults float64 `json:"median_results"`
}

// Report is a validated deterministic evaluation artifact.
type Report struct {
	SchemaVersion int                           `json:"schema_version"`
	Identity      ReportIdentity                `json:"identity"`
	ContentID     string                        `json:"content_id"`
	GroupLabels   []string                      `json:"group_labels,omitempty"`
	Metrics       Metrics                       `json:"metrics"`
	Breakdowns    map[string]map[string]Metrics `json:"breakdowns,omitempty"`
	Outcomes      []Outcome                     `json:"outcomes"`
}

// BuildReport validates, normalizes, and deterministically orders outcomes.
func BuildReport(identity ReportIdentity, outcomes []Outcome, groupLabels ...string) (Report, error) {
	if err := validateIdentity(identity); err != nil {
		return Report{}, err
	}
	if len(outcomes) == 0 {
		return Report{}, fmt.Errorf("at least one outcome is required")
	}
	normalized := make([]Outcome, 0, len(outcomes))
	seen := make(map[string]struct{}, len(outcomes))
	for i, outcome := range outcomes {
		if err := ValidateCase(outcome.Case); err != nil {
			return Report{}, fmt.Errorf("outcome %d: %w", i, err)
		}
		id := caseID(outcome.Case)
		if _, ok := seen[id]; ok {
			return Report{}, fmt.Errorf("duplicate outcome case id %q", id)
		}
		seen[id] = struct{}{}

		switch outcome.Status {
		case OutcomeStatusFailed:
			normalized = append(normalized, Failed(outcome.Case, outcome.ErrorCategory))
		case OutcomeStatusSuccess:
			recomputed, err := Evaluate(outcome.Case, outcome.Results)
			if err != nil {
				return Report{}, fmt.Errorf("outcome %q: %w", id, err)
			}
			normalized = append(normalized, recomputed)
		default:
			return Report{}, fmt.Errorf("outcome %q: invalid status %q", id, outcome.Status)
		}
	}
	sort.Slice(normalized, func(i, j int) bool {
		return caseID(normalized[i].Case) < caseID(normalized[j].Case)
	})

	report := Report{
		SchemaVersion: ReportSchemaVersion,
		Identity:      identity,
		Metrics:       Summarize(normalized),
		Breakdowns:    make(map[string]map[string]Metrics),
		Outcomes:      normalized,
	}
	labels := append([]string(nil), groupLabels...)
	for i := range labels {
		labels[i] = strings.TrimSpace(labels[i])
	}
	sort.Strings(labels)
	for _, label := range labels {
		if label == "" {
			return Report{}, fmt.Errorf("group label is required")
		}
		if _, exists := report.Breakdowns[label]; exists {
			return Report{}, fmt.Errorf("duplicate group label %q", label)
		}
		report.Breakdowns[label] = SummarizeByLabel(normalized, label)
		report.GroupLabels = append(report.GroupLabels, label)
	}
	if len(report.Breakdowns) == 0 {
		report.Breakdowns = nil
	}
	contentID, err := hashReportContent(report.Identity, report.GroupLabels, report.Outcomes)
	if err != nil {
		return Report{}, err
	}
	report.ContentID = contentID
	return report, nil
}

// Summarize computes macro-averaged quality metrics over successful outcomes.
// Execution failures are counted but excluded from quality denominators.
func Summarize(outcomes []Outcome) Metrics {
	m := Metrics{Cases: len(outcomes)}
	resultCounts := make([]int, 0, len(outcomes))
	for _, outcome := range outcomes {
		if outcome.Status != OutcomeStatusSuccess {
			m.FailedCases++
			continue
		}
		m.SuccessfulCases++
		resultCounts = append(resultCounts, outcome.ResultCount)
		if outcome.Judged {
			m.JudgedCases++
			m.RecallAtK += outcome.RecallAtK
			m.SuccessAtK += outcome.SuccessAtK
			m.MRRAtK += outcome.ReciprocalRank
			m.NDCGAtK += outcome.NDCGAtK
		}
		if outcome.EmptyExpected {
			m.EmptyCases++
			if outcome.ExactEmpty {
				m.ExactEmptyRate++
			}
		}
	}
	if m.JudgedCases > 0 {
		denom := float64(m.JudgedCases)
		m.RecallAtK /= denom
		m.SuccessAtK /= denom
		m.MRRAtK /= denom
		m.NDCGAtK /= denom
	}
	if m.EmptyCases > 0 {
		m.ExactEmptyRate /= float64(m.EmptyCases)
	}
	if len(resultCounts) == 0 {
		return m
	}
	sort.Ints(resultCounts)
	m.MinResults = resultCounts[0]
	m.MaxResults = resultCounts[len(resultCounts)-1]
	total := 0
	for _, count := range resultCounts {
		total += count
	}
	m.MeanResults = float64(total) / float64(len(resultCounts))
	m.MedianResults = median(resultCounts)
	return m
}

// SummarizeByLabel groups outcomes by one case label value.
func SummarizeByLabel(outcomes []Outcome, label string) map[string]Metrics {
	groups := make(map[string][]Outcome)
	for _, outcome := range outcomes {
		value := strings.TrimSpace(outcome.Case.Labels[label])
		if value == "" {
			continue
		}
		groups[value] = append(groups[value], outcome)
	}
	out := make(map[string]Metrics, len(groups))
	for value, grouped := range groups {
		out[value] = Summarize(grouped)
	}
	return out
}

// HashSuite returns a stable identity for the exact suite contents.
func HashSuite(suite Suite) (string, error) {
	if strings.TrimSpace(suite.ID) == "" || len(suite.Cases) == 0 {
		return "", fmt.Errorf("suite id and cases are required")
	}
	seen := make(map[string]struct{}, len(suite.Cases))
	normalized := Suite{ID: strings.TrimSpace(suite.ID), Cases: make([]GoldenCase, 0, len(suite.Cases))}
	for _, c := range suite.Cases {
		if err := ValidateCase(c); err != nil {
			return "", err
		}
		id := caseID(c)
		if _, ok := seen[id]; ok {
			return "", fmt.Errorf("duplicate case id %q", id)
		}
		seen[id] = struct{}{}
		normalized.Cases = append(normalized.Cases, normalizeCase(c))
	}
	data, err := json.Marshal(normalized)
	if err != nil {
		return "", fmt.Errorf("marshal suite: %w", err)
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func validateIdentity(identity ReportIdentity) error {
	if strings.TrimSpace(identity.DatasetID) == "" {
		return fmt.Errorf("dataset id is required")
	}
	if strings.TrimSpace(identity.SuiteID) == "" {
		return fmt.Errorf("suite id is required")
	}
	if strings.TrimSpace(identity.CandidateID) == "" {
		return fmt.Errorf("candidate id is required")
	}
	return nil
}

func hashReportContent(identity ReportIdentity, groupLabels []string, outcomes []Outcome) (string, error) {
	payload := struct {
		Identity    ReportIdentity `json:"identity"`
		GroupLabels []string       `json:"group_labels,omitempty"`
		Outcomes    []Outcome      `json:"outcomes"`
	}{Identity: identity, GroupLabels: groupLabels, Outcomes: outcomes}
	data, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal report content: %w", err)
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func median(sorted []int) float64 {
	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return float64(sorted[middle])
	}
	return float64(sorted[middle-1]+sorted[middle]) / 2
}
