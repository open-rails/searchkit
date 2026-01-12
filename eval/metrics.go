package eval

// This package is intentionally minimal: it provides a small set of evaluation
// metrics that apps can use with their own hand-written test cases.

type Key struct {
	EntityType string
	EntityID   string
}

type Case struct {
	Name     string
	Query    string
	Expected []Key
}

// RecallAtK computes recall@k for a single case.
func RecallAtK(got []Key, expected []Key, k int) float64 {
	if len(expected) == 0 {
		return 1.0
	}
	if k <= 0 {
		return 0.0
	}
	if k > len(got) {
		k = len(got)
	}

	exp := make(map[Key]struct{}, len(expected))
	for _, e := range expected {
		exp[e] = struct{}{}
	}

	hit := 0
	for i := 0; i < k; i++ {
		if _, ok := exp[got[i]]; ok {
			hit++
		}
	}

	return float64(hit) / float64(len(expected))
}

// MRR computes mean reciprocal rank for a single case.
func MRR(got []Key, expected []Key) float64 {
	if len(expected) == 0 {
		return 1.0
	}
	exp := make(map[Key]struct{}, len(expected))
	for _, e := range expected {
		exp[e] = struct{}{}
	}
	for i, g := range got {
		if _, ok := exp[g]; ok {
			return 1.0 / float64(i+1)
		}
	}
	return 0.0
}
