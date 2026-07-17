package eval

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// ValidateCase validates one evaluation case without modifying it.
func ValidateCase(c GoldenCase) error {
	if caseID(c) == "" {
		return fmt.Errorf("case id is required")
	}
	if strings.TrimSpace(c.Query) == "" {
		return fmt.Errorf("case %q: query is required", caseID(c))
	}
	if c.K <= 0 {
		return fmt.Errorf("case %q: k must be positive", caseID(c))
	}
	if len(c.Expected) > 0 && len(c.Judgments) > 0 {
		return fmt.Errorf("case %q: expected and judgments are mutually exclusive", caseID(c))
	}
	if c.ExpectEmpty && (len(c.Expected) > 0 || len(c.Judgments) > 0) {
		return fmt.Errorf("case %q: empty expectation cannot include relevance judgments", caseID(c))
	}

	seenKeys := make(map[GoldenKey]struct{}, len(c.Expected)+len(c.Judgments))
	for _, key := range c.Expected {
		key = normalizeGoldenKey(key)
		if err := validateGoldenKey(key); err != nil {
			return fmt.Errorf("case %q: expected key: %w", caseID(c), err)
		}
		if _, duplicate := seenKeys[key]; duplicate {
			return fmt.Errorf("case %q: duplicate expected key", caseID(c))
		}
		seenKeys[key] = struct{}{}
	}
	hasPositiveJudgment := false
	for _, judgment := range c.Judgments {
		judgment.Key = normalizeGoldenKey(judgment.Key)
		if err := validateGoldenKey(judgment.Key); err != nil {
			return fmt.Errorf("case %q: judgment key: %w", caseID(c), err)
		}
		if judgment.Relevance < 0 || judgment.Relevance > 3 {
			return fmt.Errorf("case %q: relevance must be between 0 and 3", caseID(c))
		}
		if judgment.Relevance > 0 {
			hasPositiveJudgment = true
		}
		if _, duplicate := seenKeys[judgment.Key]; duplicate {
			return fmt.Errorf("case %q: duplicate judgment key", caseID(c))
		}
		seenKeys[judgment.Key] = struct{}{}
	}
	if len(c.Judgments) > 0 && !hasPositiveJudgment {
		return fmt.Errorf("case %q: at least one positive judgment is required", caseID(c))
	}

	seenLabels := make(map[string]struct{}, len(c.Labels))
	for name, value := range c.Labels {
		name = strings.TrimSpace(name)
		if name == "" || strings.TrimSpace(value) == "" {
			return fmt.Errorf("case %q: label names and values must be nonempty", caseID(c))
		}
		if _, duplicate := seenLabels[name]; duplicate {
			return fmt.Errorf("case %q: duplicate normalized label %q", caseID(c), name)
		}
		seenLabels[name] = struct{}{}
	}

	seenEntityTypes := make(map[string]struct{}, len(c.EntityTypes))
	for _, entityType := range c.EntityTypes {
		entityType = strings.TrimSpace(entityType)
		if entityType == "" {
			return fmt.Errorf("case %q: entity types must be nonempty", caseID(c))
		}
		if _, duplicate := seenEntityTypes[entityType]; duplicate {
			return fmt.Errorf("case %q: duplicate entity type %q", caseID(c), entityType)
		}
		seenEntityTypes[entityType] = struct{}{}
	}
	return nil
}

// Evaluate computes metrics for one successful retrieval while preserving the
// caller's result order. Duplicate returned keys contribute only at their first
// raw rank, but ResultCount retains the actual number returned.
func Evaluate(c GoldenCase, results []Result) (Outcome, error) {
	if err := ValidateCase(c); err != nil {
		return Outcome{}, err
	}
	c = normalizeCase(c)
	normalizedResults := make([]Result, len(results))
	for i, result := range results {
		result.Key = normalizeGoldenKey(result.Key)
		if err := validateGoldenKey(result.Key); err != nil {
			return Outcome{}, fmt.Errorf("case %q: result %d: %w", caseID(c), i, err)
		}
		if math.IsNaN(float64(result.Score)) || math.IsInf(float64(result.Score), 0) {
			return Outcome{}, fmt.Errorf("case %q: result %d: score must be finite", caseID(c), i)
		}
		normalizedResults[i] = result
	}

	out := Outcome{
		Status:        OutcomeStatusSuccess,
		Case:          cloneCase(c),
		Results:       normalizedResults,
		ResultCount:   len(normalizedResults),
		EmptyExpected: c.ExpectEmpty,
		ExactEmpty:    c.ExpectEmpty && len(normalizedResults) == 0,
		QualityStatus: QualityStatusUnjudged,
	}
	if c.ExpectEmpty {
		if out.ExactEmpty {
			out.QualityStatus = QualityStatusExactEmpty
		} else {
			out.QualityStatus = QualityStatusUnexpectedResults
		}
		return out, nil
	}

	relevance := relevanceByKey(c)
	if len(relevance) == 0 {
		return out, nil
	}
	out.Judged = true

	relevantCount := 0
	for _, grade := range relevance {
		if grade > 0 {
			relevantCount++
		}
	}
	hits, reciprocalRank, dcg := rankingMetrics(normalizedResults, relevance, c.K)
	out.ReciprocalRank = reciprocalRank
	out.RecallAtK = float64(hits) / float64(relevantCount)
	if hits > 0 {
		out.SuccessAtK = 1
		out.QualityStatus = QualityStatusHit
	} else {
		out.QualityStatus = QualityStatusMiss
	}
	out.NDCGAtK = normalizedDCG(relevance, c.K, dcg)
	return out, nil
}

func rankingMetrics(results []Result, relevance map[GoldenKey]int, k int) (hits int, reciprocalRank float64, dcg float64) {
	seenResults := make(map[GoldenKey]struct{}, min(k, len(results)))
	limit := min(k, len(results))
	for i, result := range results[:limit] {
		if _, duplicate := seenResults[result.Key]; duplicate {
			continue
		}
		seenResults[result.Key] = struct{}{}
		grade := relevance[result.Key]
		if grade <= 0 {
			continue
		}
		hits++
		if reciprocalRank == 0 {
			reciprocalRank = 1 / float64(i+1)
		}
		dcg += discountedGain(grade, i+1)
	}
	return hits, reciprocalRank, dcg
}

func caseID(c GoldenCase) string {
	return strings.TrimSpace(c.ID)
}

func validateGoldenKey(key GoldenKey) error {
	if key.EntityType == "" {
		return fmt.Errorf("entity type is required")
	}
	if key.EntityID == "" {
		return fmt.Errorf("entity id is required")
	}
	return nil
}

func relevanceByKey(c GoldenCase) map[GoldenKey]int {
	if len(c.Judgments) > 0 {
		out := make(map[GoldenKey]int, len(c.Judgments))
		for _, judgment := range c.Judgments {
			out[judgment.Key] = judgment.Relevance
		}
		return out
	}
	out := make(map[GoldenKey]int, len(c.Expected))
	for _, key := range c.Expected {
		out[key] = 1
	}
	return out
}

func discountedGain(relevance int, rank int) float64 {
	return (math.Pow(2, float64(relevance)) - 1) / math.Log2(float64(rank)+1)
}

func normalizedDCG(relevance map[GoldenKey]int, k int, dcg float64) float64 {
	grades := make([]int, 0, len(relevance))
	for _, grade := range relevance {
		if grade > 0 {
			grades = append(grades, grade)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(grades)))
	if len(grades) > k {
		grades = grades[:k]
	}
	idcg := 0.0
	for i, grade := range grades {
		idcg += discountedGain(grade, i+1)
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func cloneCase(c GoldenCase) GoldenCase {
	out := c
	out.EntityTypes = append([]string(nil), c.EntityTypes...)
	out.Expected = append([]GoldenKey(nil), c.Expected...)
	out.Judgments = append([]Judgment(nil), c.Judgments...)
	if c.Labels != nil {
		out.Labels = make(map[string]string, len(c.Labels))
		for name, value := range c.Labels {
			out.Labels[name] = value
		}
	}
	return out
}

func normalizeCase(c GoldenCase) GoldenCase {
	out := cloneCase(c)
	out.ID = strings.TrimSpace(out.ID)
	out.Query = strings.TrimSpace(out.Query)
	out.Language = strings.TrimSpace(out.Language)
	for i := range out.EntityTypes {
		out.EntityTypes[i] = strings.TrimSpace(out.EntityTypes[i])
	}
	for i := range out.Expected {
		out.Expected[i] = normalizeGoldenKey(out.Expected[i])
	}
	for i := range out.Judgments {
		out.Judgments[i].Key = normalizeGoldenKey(out.Judgments[i].Key)
	}
	if out.Labels != nil {
		normalizedLabels := make(map[string]string, len(out.Labels))
		for name, value := range out.Labels {
			normalizedLabels[strings.TrimSpace(name)] = strings.TrimSpace(value)
		}
		out.Labels = normalizedLabels
	}
	return out
}

func normalizeGoldenKey(key GoldenKey) GoldenKey {
	key.EntityType = strings.TrimSpace(key.EntityType)
	key.EntityID = strings.TrimSpace(key.EntityID)
	return key
}
