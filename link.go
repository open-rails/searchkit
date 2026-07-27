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
	// Score is the fused (RRF) ranking score. It orders links but is NOT
	// comparable across queries — use the provenance fields below to judge
	// link confidence.
	Score float32
	// Lexical reports whether a lexical branch (fts/trigram/pgroonga) produced
	// this link. Lexical links are exact/alias/name matches and are trustworthy
	// by construction; semantic-only links are similarity guesses.
	Lexical bool
	// SemanticSimilarity is the raw cosine from the semantic branch, 0 when the
	// link did not surface semantically. Unlike Score it is comparable across
	// queries, so hosts can threshold on it (e.g. suppress weak
	// semantic-only links).
	SemanticSimilarity float32
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
	hits, trace, err := c.SearchWithTrace(ctx, query, SearchOptions{
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
	return planFromHits(query, hits, limit, trace), nil
}

// linkProvenance is the per-entity branch evidence extracted from a trace.
type linkProvenance struct {
	lexical            bool
	semanticSimilarity float32
}

// provenanceFromTrace folds the trace's per-source candidates into per-entity
// provenance: whether any lexical backend surfaced the entity, and the highest
// raw cosine the semantic backend reported for it.
func provenanceFromTrace(trace SearchTrace) map[string]linkProvenance {
	prov := make(map[string]linkProvenance)
	for _, src := range trace.Sources {
		if src.Status != SourceStatusSucceeded {
			continue
		}
		semantic := src.Backend == BackendSemantic
		for _, cand := range src.Candidates {
			key := cand.Key.EntityType + "\x00" + cand.Key.EntityID
			p := prov[key]
			if semantic {
				if cand.Score > p.semanticSimilarity {
					p.semanticSimilarity = cand.Score
				}
			} else {
				p.lexical = true
			}
			prov[key] = p
		}
	}
	return prov
}

// planFromHits maps ranked vocabulary hits into a QueryPlan, annotating each
// link with branch provenance from the trace. Pure so it is unit testable
// without a database.
func planFromHits(query string, hits []SearchHit, limit int, trace SearchTrace) QueryPlan {
	plan := QueryPlan{Query: query, Residual: query}
	prov := provenanceFromTrace(trace)
	for _, h := range hits {
		if limit > 0 && len(plan.LinkedEntities) >= limit {
			break
		}
		p := prov[h.EntityType+"\x00"+h.EntityID]
		plan.LinkedEntities = append(plan.LinkedEntities, LinkedEntity{
			EntityType:         h.EntityType,
			EntityID:           h.EntityID,
			Score:              h.Score,
			Lexical:            p.lexical,
			SemanticSimilarity: p.semanticSimilarity,
		})
	}
	return plan
}
