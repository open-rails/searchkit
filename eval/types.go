package eval

import "strings"

// GoldenCase is a validated query and its quality expectation. Case remains the
// original minimal compatibility type used by RecallAtK and MRR.
type GoldenCase struct {
	ID          string            `json:"id"`
	Query       string            `json:"query"`
	Language    string            `json:"language,omitempty"`
	EntityTypes []string          `json:"entity_types,omitempty"`
	K           int               `json:"k"`
	Expected    []GoldenKey       `json:"expected,omitempty"`
	Judgments   []Judgment        `json:"judgments,omitempty"`
	ExpectEmpty bool              `json:"expect_empty,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// GoldenKey is the stable JSON entity identity used by golden fixtures and
// reports. Key remains the original compatibility type.
type GoldenKey struct {
	EntityType string `json:"entity_type"`
	EntityID   string `json:"entity_id"`
}

// Judgment assigns a relevance grade to an entity for a query. Grades range
// from 0 (not relevant) to 3 (highly relevant).
type Judgment struct {
	Key       GoldenKey `json:"key"`
	Relevance int       `json:"relevance"`
}

// Result is one ordered retrieval result and its score. The score domain is
// supplied by the caller and must not be mixed with other domains in a sweep.
type Result struct {
	Key   GoldenKey `json:"key"`
	Score float32   `json:"score"`
}

// OutcomeStatus records retrieval execution success or failure.
type OutcomeStatus string

const (
	OutcomeStatusSuccess OutcomeStatus = "success"
	OutcomeStatusFailed  OutcomeStatus = "failed"
)

// QualityStatus records the relevance result independently of execution status.
type QualityStatus string

const (
	QualityStatusUnjudged          QualityStatus = "unjudged"
	QualityStatusHit               QualityStatus = "hit"
	QualityStatusMiss              QualityStatus = "miss"
	QualityStatusExactEmpty        QualityStatus = "exact_empty"
	QualityStatusUnexpectedResults QualityStatus = "unexpected_results"
)

// Outcome is the evaluation of one case. Status and QualityStatus keep
// execution failure distinct from successful empty, hit, and miss outcomes.
type Outcome struct {
	Status        OutcomeStatus `json:"status"`
	QualityStatus QualityStatus `json:"quality_status,omitempty"`
	Case          GoldenCase    `json:"case"`
	Results       []Result      `json:"results,omitempty"`
	ResultCount   int           `json:"result_count"`
	ErrorCategory string        `json:"error_category,omitempty"`

	Judged         bool    `json:"judged"`
	RecallAtK      float64 `json:"recall_at_k,omitempty"`
	SuccessAtK     float64 `json:"success_at_k,omitempty"`
	ReciprocalRank float64 `json:"reciprocal_rank,omitempty"`
	NDCGAtK        float64 `json:"ndcg_at_k,omitempty"`

	EmptyExpected bool `json:"empty_expected"`
	ExactEmpty    bool `json:"exact_empty"`
}

// Failed creates an execution-failure outcome. Category should be a stable,
// sanitized identifier such as "timeout" or "semantic_search".
func Failed(c GoldenCase, category string) Outcome {
	category = normalizeErrorCategory(category)
	return Outcome{
		Status:        OutcomeStatusFailed,
		Case:          normalizeCase(c),
		ErrorCategory: category,
		EmptyExpected: c.ExpectEmpty,
	}
}

func normalizeErrorCategory(category string) string {
	category = strings.TrimSpace(category)
	if category == "" || len(category) > 64 {
		return "unspecified"
	}
	for _, r := range category {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '_' && r != '-' {
			return "unspecified"
		}
	}
	return category
}
