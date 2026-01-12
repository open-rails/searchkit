package normalize

import "math"

// L2NormalizeInPlace normalizes vec to unit L2 norm.
// If vec is empty or all zeros, it is left unchanged.
func L2NormalizeInPlace(vec []float32) {
	if len(vec) == 0 {
		return
	}
	var sumSq float64
	for _, v := range vec {
		f := float64(v)
		sumSq += f * f
	}
	if sumSq <= 0 {
		return
	}
	invNorm := float32(1.0 / math.Sqrt(sumSq))
	for i := range vec {
		vec[i] *= invNorm
	}
}
