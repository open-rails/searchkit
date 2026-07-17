package search

import (
	"fmt"
	"math"
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

type RRFContribution struct {
	ListIndex    int     `json:"list_index"` // zero-based source-list index
	Rank         int     `json:"rank"`       // one-based rank within the source list
	Weight       float32 `json:"weight"`
	Contribution float32 `json:"contribution"`
}

type RRFTraceHit struct {
	Hit           RRFHit            `json:"-"`
	Key           RRFTraceKey       `json:"key"`
	Score         float32           `json:"score"`
	Contributions []RRFContribution `json:"contributions"`
}

type RRFTraceKey struct {
	EntityType string `json:"entity_type"`
	EntityID   string `json:"entity_id"`
	Language   string `json:"language"`
	Model      string `json:"model"`
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
	out, _ := fuseRRF(lists, opts, false)
	return out
}

// FuseRRFWithTrace returns the same ordered hits as FuseRRF plus the exact
// source-list contributions used to compute each score.
func FuseRRFWithTrace(lists [][]RRFKey, opts RRFOptions) ([]RRFTraceHit, error) {
	if err := validateRRFTraceOptions(opts); err != nil {
		return nil, err
	}
	hits, contributions := fuseRRF(lists, opts, true)
	out := make([]RRFTraceHit, 0, len(hits))
	for _, hit := range hits {
		if !finiteFloat32(hit.Score) {
			return nil, fmt.Errorf("RRF score overflow for entity %q", hit.EntityID)
		}
		out = append(out, RRFTraceHit{
			Hit: hit,
			Key: RRFTraceKey{
				EntityType: hit.EntityType,
				EntityID:   hit.EntityID,
				Language:   hit.Language,
				Model:      hit.Model,
			},
			Score:         hit.Score,
			Contributions: contributions[hit.RRFKey.keyString()],
		})
	}
	return out, nil
}

func fuseRRF(lists [][]RRFKey, opts RRFOptions, includeTrace bool) ([]RRFHit, map[string][]RRFContribution) {
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
	var contributions map[string][]RRFContribution
	if includeTrace {
		contributions = make(map[string][]RRFContribution)
	}

	for li, list := range lists {
		w := float32(1.0)
		if li < len(weights) && weights[li] > 0 {
			w = weights[li]
		}
		for i, item := range list {
			rank := i + 1
			ks := item.keyString()
			example[ks] = item
			contribution := w / (float32(k) + float32(rank))
			scores[ks] += contribution
			if includeTrace {
				contributions[ks] = append(contributions[ks], RRFContribution{
					ListIndex: li, Rank: rank, Weight: w, Contribution: contribution,
				})
			}
		}
	}

	out := make([]RRFHit, 0, len(scores))
	for ks, sc := range scores {
		out = append(out, RRFHit{RRFKey: example[ks], Score: sc})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			if out[i].EntityType == out[j].EntityType {
				if out[i].EntityID == out[j].EntityID {
					if out[i].Language == out[j].Language {
						return out[i].Model < out[j].Model
					}
					return out[i].Language < out[j].Language
				}
				return out[i].EntityID < out[j].EntityID
			}
			return out[i].EntityType < out[j].EntityType
		}
		return out[i].Score > out[j].Score
	})
	return out, contributions
}

func validateRRFTraceOptions(opts RRFOptions) error {
	for i, weight := range opts.Weights {
		if !finiteFloat32(weight) {
			return fmt.Errorf("RRF weight %d must be finite", i)
		}
	}
	return nil
}

func finiteFloat32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
