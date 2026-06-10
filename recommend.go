package searchkit

import (
	"context"
	"fmt"
	"strings"

	"github.com/open-rails/searchkit/search"
	"github.com/open-rails/searchkit/signal"
)

// RecommendOptions controls Recommend ("for you": user → items).
type RecommendOptions struct {
	// EntityTypes are the candidate entity types to recommend. Required.
	EntityTypes []string

	// Limit caps results (default: client default limit).
	Limit int

	// SeedLimit is how many of the subject's highest-signal entities seed
	// content-based + co-engagement candidates (default 5).
	SeedLimit int

	// SeedEntityTypes limits which entity types may seed (default: any).
	SeedEntityTypes []string

	// SimilarWeight / CoEngagementWeight are RRF weights for the two
	// candidate sources (both default 1).
	SimilarWeight      float32
	CoEngagementWeight float32

	// IncludeSeen keeps already-seen entities in results (default: excluded).
	IncludeSeen bool

	// PopularWindow is the popularity window used to fill out results on
	// cold start or thin candidate sets (default: last 30 days).
	PopularWindow signal.Window

	// Language / Model override the content-plane similarity defaults.
	Language string
	Model    string

	// FilterSQL / FilterArgs apply host-owned constraints to the
	// content-based (vector) candidate source.
	FilterSQL  string
	FilterArgs map[string]any
}

// Recommend returns "for you" recommendations for a subject: content-based
// similarity seeded from the subject's high-signal entities, fused (RRF) with
// co-engagement ("subjects who engaged with your favorites also engaged
// with..."), excluding already-seen, with a popularity fallback for cold
// start. Returns ranked entity ids; the host hydrates.
func (h *EmbeddedHub) Recommend(ctx context.Context, subject signal.Subject, opts RecommendOptions) ([]RecHit, error) {
	store, err := h.requireStore()
	if err != nil {
		return nil, err
	}
	if err := subject.Validate(); err != nil {
		return nil, err
	}
	entityTypes := dedupTrim(opts.EntityTypes)
	if len(entityTypes) == 0 {
		return nil, fmt.Errorf("searchkit: RecommendOptions.EntityTypes is required")
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = h.client.defaultLimit
	}
	seedLimit := opts.SeedLimit
	if seedLimit <= 0 {
		seedLimit = 5
	}
	simWeight := opts.SimilarWeight
	if simWeight <= 0 {
		simWeight = 1
	}
	coWeight := opts.CoEngagementWeight
	if coWeight <= 0 {
		coWeight = 1
	}
	popWindow := opts.PopularWindow
	if popWindow == (signal.Window{}) {
		popWindow = signal.LastDays(30)
	}

	seeds, err := store.TopStates(ctx, h.tenant, subject, signal.TopStatesOptions{
		EntityTypes: opts.SeedEntityTypes,
		Limit:       seedLimit,
	})
	if err != nil {
		return nil, err
	}

	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = h.client.defaultModel
	}

	perSeed := clampInt(limit, 20, 100)
	lists := make([][]search.RRFKey, 0, 2*len(seeds))
	weights := make([]float32, 0, 2*len(seeds))
	seedSet := map[signal.EntityRef]struct{}{}
	for _, seed := range seeds {
		seedSet[seed.EntityRef] = struct{}{}
	}

	for _, seed := range seeds {
		// Content-based: nearest neighbours of the seed (when a semantic
		// model is available).
		if model != "" {
			sim, err := h.client.SimilarTo(ctx, seed.EntityType, seed.EntityID, SimilarOptions{
				Language:    opts.Language,
				Model:       model,
				Limit:       perSeed,
				EntityTypes: entityTypes,
				FilterSQL:   opts.FilterSQL,
				FilterArgs:  opts.FilterArgs,
			})
			if err != nil {
				return nil, err
			}
			keys := make([]search.RRFKey, 0, len(sim))
			for _, s := range sim {
				keys = append(keys, search.RRFKey{EntityType: s.EntityType, EntityID: s.EntityID})
			}
			lists = append(lists, keys)
			weights = append(weights, simWeight)
		}

		// Collaborative: co-engagement with the seed.
		co, err := store.CoEngaged(ctx, h.tenant, seed.EntityRef, signal.CoEngagedOptions{
			EntityTypes: entityTypes,
			Limit:       perSeed,
		})
		if err != nil {
			return nil, err
		}
		keys := make([]search.RRFKey, 0, len(co))
		for _, c := range co {
			keys = append(keys, search.RRFKey{EntityType: c.EntityType, EntityID: c.EntityID})
		}
		lists = append(lists, keys)
		weights = append(weights, coWeight)
	}

	fused := []search.RRFHit{}
	if len(lists) > 0 {
		fused = search.FuseRRF(lists, search.RRFOptions{K: h.client.defaultRRFK, Weights: weights})
	}

	// Exclusions: seeds and (unless IncludeSeen) everything already seen.
	seen := map[string]map[string]struct{}{}
	if !opts.IncludeSeen {
		for _, t := range entityTypes {
			s, err := store.SeenIDs(ctx, h.tenant, subject, t)
			if err != nil {
				return nil, err
			}
			seen[t] = s
		}
	}
	allowed := map[string]struct{}{}
	for _, t := range entityTypes {
		allowed[t] = struct{}{}
	}

	out := make([]RecHit, 0, limit)
	have := map[signal.EntityRef]struct{}{}
	push := func(entityType, entityID string, score float32) {
		ref := signal.EntityRef{EntityType: entityType, EntityID: entityID}
		if _, ok := allowed[entityType]; !ok {
			return
		}
		if _, ok := seedSet[ref]; ok {
			return
		}
		if _, ok := have[ref]; ok {
			return
		}
		if s, ok := seen[entityType]; ok {
			if _, isSeen := s[entityID]; isSeen {
				return
			}
		}
		have[ref] = struct{}{}
		out = append(out, RecHit{EntityType: entityType, EntityID: entityID, Score: score})
	}

	for _, f := range fused {
		if len(out) >= limit {
			break
		}
		push(f.EntityType, f.EntityID, f.Score)
	}

	// Cold start / thin results: fill from popularity. Popular scores live
	// on a different scale than RRF, so filled hits are appended after the
	// fused block with decaying tail scores.
	if len(out) < limit {
		for _, t := range entityTypes {
			if len(out) >= limit {
				break
			}
			// Oversample so exclusions still leave enough to fill.
			popLimit := clampInt((limit-len(out))*3, 20, 500)
			pop, err := store.Popular(ctx, h.tenant, t, signal.PopularOptions{
				Window: popWindow,
				Limit:  popLimit,
			})
			if err != nil {
				return nil, err
			}
			var tail float32
			if n := len(out); n > 0 {
				tail = out[n-1].Score
			}
			for i, ph := range pop {
				if len(out) >= limit {
					break
				}
				score := tail / 2
				if tail == 0 {
					score = 1 / float32(h.client.defaultRRFK+i+1)
				}
				push(ph.EntityType, ph.EntityID, score)
			}
		}
	}

	return out, nil
}

func dedupTrim(in []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}
