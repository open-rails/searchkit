package searchkit

import (
	"context"

	"github.com/open-rails/searchkit/eval"
)

// NewEvalRunner adapts a Client to eval.CaseRunner so a golden suite can be
// executed against real search. The base options carry cross-case settings
// (Mode, LanguageMode, RRFK, floors); each case overrides Language,
// EntityTypes, and Limit from its own definition.
//
// This adapter is the single seam where the client meets the dependency-free
// eval package.
func NewEvalRunner(client *Client, base SearchOptions) eval.CaseRunner {
	return clientRunner{client: client, base: base}
}

type clientRunner struct {
	client *Client
	base   SearchOptions
}

func (r clientRunner) Run(ctx context.Context, c eval.GoldenCase) ([]eval.Result, string, error) {
	opts := r.base
	if c.Language != "" {
		opts.Language = c.Language
	}
	if len(c.EntityTypes) > 0 {
		// A case's entity types apply to both retrieval planes unless the base
		// options already pinned per-plane types.
		opts.EntityTypes = c.EntityTypes
	}
	opts.Limit = c.K

	hits, err := r.client.Search(ctx, c.Query, opts)
	if err != nil {
		return nil, "search", err
	}

	results := make([]eval.Result, len(hits))
	for i, hit := range hits {
		results[i] = eval.Result{
			Key:   eval.GoldenKey{EntityType: hit.EntityType, EntityID: hit.EntityID},
			Score: hit.Score,
		}
	}
	return results, "", nil
}
