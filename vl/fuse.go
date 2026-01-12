package vl

import "github.com/doujins-org/embeddingkit/internal/normalize"

// FuseAverageL2 averages vectors elementwise and L2-normalizes the result.
// Returns nil if vectors is empty or dimensions mismatch.
func FuseAverageL2(vectors [][]float32) []float32 {
	if len(vectors) == 0 {
		return nil
	}
	dim := len(vectors[0])
	if dim == 0 {
		return nil
	}
	sum := make([]float32, dim)
	for _, v := range vectors {
		if len(v) != dim {
			return nil
		}
		for i := 0; i < dim; i++ {
			sum[i] += v[i]
		}
	}
	inv := float32(1.0) / float32(len(vectors))
	for i := 0; i < dim; i++ {
		sum[i] *= inv
	}
	normalize.L2NormalizeInPlace(sum)
	return sum
}
