package search

import (
	"sort"
	"strings"
)

// RRF (Reciprocal Rank Fusion) combines ranked lists without relying on raw
// score calibration.
//
// Typical formula:
//
//	score(doc) = Σ (weight_i / (k + rank_i))
//
// where rank_i is 1-based position in list i, and k is usually 50–60.
type RRFOptions struct {
	// K is the stabilizer constant; higher K flattens rank differences.
	// Defaults to 60 when <= 0.
	K int

	// Weights applied to each list. Empty => all 1.0.
	Weights []float32
}

type RRFKey struct {
	EntityType string
	EntityID   string
	Language   string
	Model      string
}

type RRFHit struct {
	RRFKey
	Score float32
}

func (k RRFKey) keyString() string {
	return strings.Join([]string{
		strings.TrimSpace(k.EntityType),
		strings.TrimSpace(k.EntityID),
		strings.TrimSpace(k.Language),
		strings.TrimSpace(k.Model),
	}, "\x1f")
}

// FuseRRF fuses multiple ranked lists into a single ranked list via RRF.
//
// Input lists are expected to be ordered best-first.
func FuseRRF(lists [][]RRFKey, opts RRFOptions) []RRFHit {
	k := opts.K
	if k <= 0 {
		k = 60
	}
	weights := opts.Weights
	if len(weights) == 0 {
		weights = make([]float32, len(lists))
		for i := range weights {
			weights[i] = 1.0
		}
	}

	scores := make(map[string]float32)
	example := make(map[string]RRFKey)

	for li, list := range lists {
		w := float32(1.0)
		if li < len(weights) && weights[li] > 0 {
			w = weights[li]
		}
		for i, item := range list {
			rank := i + 1
			ks := item.keyString()
			example[ks] = item
			scores[ks] += w / float32(k+rank)
		}
	}

	out := make([]RRFHit, 0, len(scores))
	for ks, sc := range scores {
		out = append(out, RRFHit{RRFKey: example[ks], Score: sc})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			if out[i].EntityType == out[j].EntityType {
				return out[i].EntityID < out[j].EntityID
			}
			return out[i].EntityType < out[j].EntityType
		}
		return out[i].Score > out[j].Score
	})
	return out
}
