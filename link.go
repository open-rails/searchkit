package searchkit

import (
	"context"
	"fmt"
)

// LinkedEntity is a controlled-vocabulary entity that a raw query resolved to.
// EntityType and EntityID are opaque host identifiers — searchkit attaches no
// meaning to them (they are whatever the host indexed, e.g. its tag/artist ids).
type LinkedEntity struct {
	EntityType string
	EntityID   string
	Score      float32
}

// QueryPlan is the structured interpretation of a raw query: the controlled-
// vocabulary entities it linked to, plus the residual free text for the normal
// lexical + dense recall channels.
//
// It is host-agnostic: searchkit produces the links, but the HOST decides what
// the linked entity types mean and how to apply them (exact filter vs boost).
// It is also the structured feature source consumed by learned ranking (#35).
type QueryPlan struct {
	Query          string
	LinkedEntities []LinkedEntity
	// Residual is the free text to run through normal recall. In this increment
	// it is the whole query; span extraction (removing linked spans) is a later
	// step. The host can use LinkedEntities to filter/boost while still sending
	// Residual through lexical + dense retrieval.
	Residual string
}

// LinkOptions configures LinkQuery. VocabEntityTypes names the entity types that
// make up the controlled vocabulary to link against (e.g. a host's indexed
// tag/artist/character/series entities). searchkit treats them as opaque
// strings and never assumes what they represent.
type LinkOptions struct {
	Language string
	// VocabEntityTypes is required; with none, LinkQuery returns an empty plan.
	VocabEntityTypes []string
	// Limit caps the number of linked entities returned (default 5).
	Limit int
	// Mode selects the retrieval mode used for linking (default SearchModeDual so
	// both exact/alias lexical hits and semantic concept hits can link).
	Mode  SearchMode
	Model string
	// MinSemanticSimilarity floors the semantic side so weak concept matches do
	// not link spuriously. Zero disables the extra floor.
	MinSemanticSimilarity float32
	FilterSQL             string
	FilterArgs            map[string]any
}

// LinkQuery resolves a raw query against the controlled vocabulary (the given
// entity types) and returns a QueryPlan. It reuses the standard retrieval
// pipeline restricted to VocabEntityTypes, so lexical/alias matches and semantic
// (embedding) matches both surface as links.
//
// Note: semantic linking requires the vocabulary entities to have embeddings; a
// host that only indexes vocabulary lexically will get lexical/alias links only.
func (c *Client) LinkQuery(ctx context.Context, query string, opts LinkOptions) (QueryPlan, error) {
	vocab := cloneAndTrim(opts.VocabEntityTypes)
	if len(vocab) == 0 {
		return QueryPlan{Query: query, Residual: query}, nil
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 5
	}
	mode := opts.Mode
	if mode == "" {
		mode = SearchModeDual
	}
	hits, err := c.Search(ctx, query, SearchOptions{
		Language:                     opts.Language,
		Mode:                         mode,
		EntityTypes:                  vocab,
		Limit:                        limit,
		Model:                        opts.Model,
		SemanticMinSimilarity:        opts.MinSemanticSimilarity,
		SemanticMinSimilarityEnabled: opts.MinSemanticSimilarity > 0,
		FilterSQL:                    opts.FilterSQL,
		FilterArgs:                   opts.FilterArgs,
	})
	if err != nil {
		return QueryPlan{}, fmt.Errorf("link query: %w", err)
	}
	return planFromHits(query, hits, limit), nil
}

// planFromHits maps ranked vocabulary hits into a QueryPlan. Pure so it is unit
// testable without a database.
func planFromHits(query string, hits []SearchHit, limit int) QueryPlan {
	plan := QueryPlan{Query: query, Residual: query}
	for _, h := range hits {
		if limit > 0 && len(plan.LinkedEntities) >= limit {
			break
		}
		plan.LinkedEntities = append(plan.LinkedEntities, LinkedEntity{
			EntityType: h.EntityType,
			EntityID:   h.EntityID,
			Score:      h.Score,
		})
	}
	return plan
}
