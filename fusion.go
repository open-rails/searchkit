package searchkit

import (
	"fmt"
	"math"
	"strings"

	"github.com/open-rails/searchkit/search"
)

// FusionSource is one ordered source of hits for reciprocal rank fusion.
// ID must be non-empty after trimming and unique among sources. Weight must be
// finite and greater than zero. Hits are ordered best-first, and their
// positions define source rank; input SearchHit scores are ignored.
type FusionSource struct {
	ID     string
	Weight float32
	Hits   []SearchHit
}

// FusionOptions controls reciprocal rank fusion and result truncation.
// RRFK values less than or equal to zero use the default K of 60. Limit values
// less than or equal to zero return every fused hit.
type FusionOptions struct {
	RRFK  int
	Limit int
}

// FusionContribution records one source's contribution to a fused hit.
// SourceRank is the one-based position of the hit in that source.
type FusionContribution struct {
	SourceID     string
	SourceRank   int
	Weight       float32
	Contribution float32
}

// FusionTraceHit contains a fused hit and its ordered source contributions.
type FusionTraceHit struct {
	Hit           SearchHit
	Contributions []FusionContribution
}

type fusionInput struct {
	lists     [][]search.RRFKey
	options   search.RRFOptions
	sourceIDs []string
	limit     int
}

type fusionIdentity struct {
	entityType string
	entityID   string
	language   string
}

// FuseSources combines ordered, weighted sources using reciprocal rank fusion.
func FuseSources(sources []FusionSource, opts FusionOptions) ([]SearchHit, error) {
	input, err := prepareFusionInput(sources, opts)
	if err != nil {
		return nil, err
	}

	fused := search.FuseRRF(input.lists, input.options)
	for _, hit := range fused {
		if !finiteFusionValue(hit.Score) {
			return nil, fmt.Errorf("fusing sources: rrf score overflow for entity %q", hit.EntityID)
		}
	}
	resultCount := input.resultCount(len(fused))
	results := make([]SearchHit, 0, resultCount)
	for _, hit := range fused[:resultCount] {
		results = append(results, SearchHit{
			EntityType: hit.EntityType,
			EntityID:   hit.EntityID,
			Language:   hit.Language,
			Score:      hit.Score,
		})
	}
	return results, nil
}

// FuseSourcesWithTrace combines ordered, weighted sources and returns each
// source's exact reciprocal rank fusion contribution.
func FuseSourcesWithTrace(sources []FusionSource, opts FusionOptions) ([]FusionTraceHit, error) {
	input, err := prepareFusionInput(sources, opts)
	if err != nil {
		return nil, err
	}

	fused, err := search.FuseRRFWithTrace(input.lists, input.options)
	if err != nil {
		return nil, fmt.Errorf("fusing sources with trace: %w", err)
	}
	resultCount := input.resultCount(len(fused))
	results := make([]FusionTraceHit, 0, resultCount)
	for _, hit := range fused[:resultCount] {
		contributions := make([]FusionContribution, 0, len(hit.Contributions))
		for _, contribution := range hit.Contributions {
			if contribution.ListIndex < 0 || contribution.ListIndex >= len(input.sourceIDs) {
				return nil, fmt.Errorf(
					"fusing sources with trace: contribution source index %d is out of range",
					contribution.ListIndex,
				)
			}
			contributions = append(contributions, FusionContribution{
				SourceID:     input.sourceIDs[contribution.ListIndex],
				SourceRank:   contribution.Rank,
				Weight:       contribution.Weight,
				Contribution: contribution.Contribution,
			})
		}
		results = append(results, FusionTraceHit{
			Hit: SearchHit{
				EntityType: hit.Hit.EntityType,
				EntityID:   hit.Hit.EntityID,
				Language:   hit.Hit.Language,
				Score:      hit.Hit.Score,
			},
			Contributions: contributions,
		})
	}
	return results, nil
}

func prepareFusionInput(sources []FusionSource, opts FusionOptions) (fusionInput, error) {
	input := fusionInput{
		lists:     make([][]search.RRFKey, 0, len(sources)),
		sourceIDs: make([]string, 0, len(sources)),
		limit:     opts.Limit,
		options: search.RRFOptions{
			K:       opts.RRFK,
			Weights: make([]float32, 0, len(sources)),
		},
	}
	sourceIDs := make(map[string]struct{}, len(sources))

	for sourceIndex, source := range sources {
		normalizedSourceID := strings.TrimSpace(source.ID)
		if normalizedSourceID == "" {
			return fusionInput{}, fmt.Errorf("source %d id must be non-empty", sourceIndex)
		}
		if _, exists := sourceIDs[normalizedSourceID]; exists {
			return fusionInput{}, fmt.Errorf("source id %q must be unique", source.ID)
		}
		if !finiteFusionValue(source.Weight) {
			return fusionInput{}, fmt.Errorf("source %q weight must be finite", source.ID)
		}
		if source.Weight <= 0 {
			return fusionInput{}, fmt.Errorf("source %q weight must be greater than zero", source.ID)
		}

		list := make([]search.RRFKey, 0, len(source.Hits))
		identities := make(map[fusionIdentity]struct{}, len(source.Hits))
		for hitIndex, hit := range source.Hits {
			entityType := strings.TrimSpace(hit.EntityType)
			if entityType == "" {
				return fusionInput{}, fmt.Errorf(
					"source %q hit %d entity type must be non-empty",
					source.ID,
					hitIndex,
				)
			}
			entityID := strings.TrimSpace(hit.EntityID)
			if entityID == "" {
				return fusionInput{}, fmt.Errorf(
					"source %q hit %d entity id must be non-empty",
					source.ID,
					hitIndex,
				)
			}
			identity := fusionIdentity{
				entityType: entityType,
				entityID:   entityID,
				language:   strings.TrimSpace(hit.Language),
			}
			if _, exists := identities[identity]; exists {
				return fusionInput{}, fmt.Errorf(
					"source %q hit %d duplicates identity (%q, %q, %q)",
					source.ID,
					hitIndex,
					hit.EntityType,
					hit.EntityID,
					hit.Language,
				)
			}

			identities[identity] = struct{}{}
			list = append(list, search.RRFKey{
				EntityType: hit.EntityType,
				EntityID:   hit.EntityID,
				Language:   hit.Language,
			})
		}

		sourceIDs[normalizedSourceID] = struct{}{}
		input.lists = append(input.lists, list)
		input.options.Weights = append(input.options.Weights, source.Weight)
		input.sourceIDs = append(input.sourceIDs, source.ID)
	}

	return input, nil
}

func (input fusionInput) resultCount(total int) int {
	if input.limit > 0 && input.limit < total {
		return input.limit
	}
	return total
}

func finiteFusionValue(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
