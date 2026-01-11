package search

import "math"

// MMRReRank applies Maximal Marginal Relevance to an initial candidate list.
//
// This is intentionally generic: embeddingkit does not assume it can access
// stored vectors for candidates. Callers must provide a similarity function
// between two candidates (e.g. computed from vectors, or approximated by
// metadata).
//
// lambda must be in [0..1]. Higher means "more relevance, less diversity".
func MMRReRank(hits []Hit, k int, lambda float32, candidateSim func(a, b Hit) float32) []Hit {
	if k <= 0 || len(hits) == 0 {
		return []Hit{}
	}
	if k > len(hits) {
		k = len(hits)
	}
	if lambda < 0 {
		lambda = 0
	} else if lambda > 1 {
		lambda = 1
	}

	selected := make([]Hit, 0, k)
	remaining := make([]Hit, len(hits))
	copy(remaining, hits)

	// Always take the top-1 by similarity first (assumes caller pre-sorted).
	selected = append(selected, remaining[0])
	remaining = append(remaining[:0], remaining[1:]...)

	for len(selected) < k && len(remaining) > 0 {
		bestIdx := -1
		bestScore := float32(-math.MaxFloat32)

		for i, cand := range remaining {
			rel := cand.Similarity

			maxRedundancy := float32(0)
			for _, sel := range selected {
				sim := candidateSim(cand, sel)
				if sim > maxRedundancy {
					maxRedundancy = sim
				}
			}

			score := lambda*rel - (1-lambda)*maxRedundancy
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			break
		}

		selected = append(selected, remaining[bestIdx])
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}

	return selected
}

