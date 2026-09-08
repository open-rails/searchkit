package searchkit

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/open-rails/searchkit/search"
	"github.com/open-rails/searchkit/signal"
)

// recommendSeedConcurrency bounds how many seed queries are in flight at once.
// One recommendation must not take the whole ClickHouse or Postgres pool: the
// seeds are few, and the win is already in not waiting for each in turn.
const recommendSeedConcurrency = 4

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

	// DiversityLambda enables MMR diversity re-ranking over the fused
	// candidates using stored embeddings: (0,1), higher = more relevance /
	// less diversity. 0 disables. Best-effort: skipped when no vectors exist.
	DiversityLambda float32

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
		EntityTypes:     opts.SeedEntityTypes,
		ExcludeNegative: true, // disliked items must not seed recommendations
		Limit:           seedLimit,
	})
	if err != nil {
		return nil, err
	}

	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = h.client.defaultModel
	}

	perSeed := clampInt(limit, 20, 100)
	seedSet := map[signal.EntityRef]struct{}{}
	for _, seed := range seeds {
		seedSet[seed.EntityRef] = struct{}{}
	}

	// Each seed contributes a content list and a co-engagement list, and the two
	// hit different stores. Run them together rather than one after another: a
	// five-seed recommendation was ten sequential round trips, so its latency was
	// the sum of every query rather than the slowest one. Results land in
	// per-seed slots, so the fused list order stays exactly what it was —
	// RRF weights are positional.
	type seedLists struct {
		similar   []search.RRFKey
		coEngaged []search.RRFKey
	}
	perSeedLists := make([]seedLists, len(seeds))

	group, groupCtx := errgroup.WithContext(ctx)
	group.SetLimit(recommendSeedConcurrency)
	for i, seed := range seeds {
		if model != "" {
			group.Go(func() error {
				sim, err := h.client.SimilarTo(groupCtx, seed.EntityType, seed.EntityID, SimilarOptions{
					Language:    opts.Language,
					Model:       model,
					Limit:       perSeed,
					EntityTypes: entityTypes,
					FilterSQL:   opts.FilterSQL,
					FilterArgs:  opts.FilterArgs,
				})
				if err != nil {
					return err
				}
				keys := make([]search.RRFKey, 0, len(sim))
				for _, s := range sim {
					keys = append(keys, search.RRFKey{EntityType: s.EntityType, EntityID: s.EntityID})
				}
				perSeedLists[i].similar = keys
				return nil
			})
		}

		group.Go(func() error {
			co, err := store.CoEngaged(groupCtx, h.tenant, seed.EntityRef, signal.CoEngagedOptions{
				EntityTypes: entityTypes,
				Limit:       perSeed,
			})
			if err != nil {
				return err
			}
			keys := make([]search.RRFKey, 0, len(co))
			for _, c := range co {
				keys = append(keys, search.RRFKey{EntityType: c.EntityType, EntityID: c.EntityID})
			}
			perSeedLists[i].coEngaged = keys
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		return nil, err
	}

	lists := make([][]search.RRFKey, 0, 2*len(seeds))
	weights := make([]float32, 0, 2*len(seeds))
	for _, perSeed := range perSeedLists {
		if model != "" {
			lists = append(lists, perSeed.similar)
			weights = append(weights, simWeight)
		}
		lists = append(lists, perSeed.coEngaged)
		weights = append(weights, coWeight)
	}

	fused := []search.RRFHit{}
	if len(lists) > 0 {
		fused = search.FuseRRF(lists, search.RRFOptions{K: h.client.defaultRRFK, Weights: weights})
	}

	// Exclusions: seeds, (unless IncludeSeen) everything already seen, and
	// ALWAYS everything the subject has net-negative explicit feedback for.
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
	negative, err := store.NegativeIDs(ctx, h.tenant, subject, entityTypes)
	if err != nil {
		return nil, err
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
		if _, ok := negative[ref]; ok {
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

	fusedHits := make([]RecHit, 0, len(fused))
	for _, f := range fused {
		fusedHits = append(fusedHits, RecHit{EntityType: f.EntityType, EntityID: f.EntityID, Score: f.Score})
	}
	if opts.DiversityLambda > 0 {
		fusedHits = h.diversifyRecHits(ctx, fusedHits, opts.DiversityLambda, model, opts.Language)
	}
	for _, f := range fusedHits {
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
