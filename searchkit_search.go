package searchkit

import (
	"context"

	"github.com/doujins-org/searchkit/search"
	"github.com/jackc/pgx/v5/pgxpool"
)

type SearchRequest struct {
	// Postgres schema where searchkit tables live (host schema).
	Schema string
	// Language of the requested search (language-specific lexical docs and vectors).
	Language string

	// Lexical entity types to include in FTS (BM25-family).
	LexicalEntityTypes []string
	// Semantic entity types to include in vector search.
	SemanticEntityTypes []string

	// Active semantic model to use.
	Model string
	// Query vector for the model (host computes via runtime embedder).
	QueryVec []float32
	// Optional explicit dims for binary two-stage.
	Dimensions int

	// Limit for each underlying retriever. Final results are also capped by this.
	Limit int

	// TwoStage enables binary_quantize oversample + halfvec rescore.
	TwoStage bool
	// OversampleFactor controls candidate fanout when TwoStage=true.
	OversampleFactor int

	// FilterSQL/Args are forwarded into the semantic KNN query.
	FilterSQL  string
	FilterArgs map[string]any

	// RRFK is the stabilizer constant for reciprocal rank fusion.
	RRFK int
}

type SearchHit struct {
	EntityType string
	EntityID   string
	Language   string
	Score      float32 // fused RRF score
}

// Search is the recommended entrypoint for “regular search”.
//
// It combines:
//   - Postgres full-text search (BM25-family) over search_documents.tsv
//   - semantic vector KNN over embedding_vectors
//
// using Reciprocal Rank Fusion (RRF), so results don’t depend on raw score scale.
func Search(ctx context.Context, pool *pgxpool.Pool, query string, req SearchRequest) ([]SearchHit, error) {
	lex, err := search.FTSSearch(ctx, pool, query, search.FTSOptions{
		Schema:      req.Schema,
		Language:    req.Language,
		EntityTypes: req.LexicalEntityTypes,
		Limit:       req.Limit,
	})
	if err != nil {
		return nil, err
	}

	sem, err := search.SemanticSearch(ctx, pool, search.Query{
		Schema:     req.Schema,
		Model:      req.Model,
		Language:   req.Language,
		QueryVec:   req.QueryVec,
		Limit:      req.Limit,
		Dimensions: req.Dimensions,
		Options: search.Options{
			EntityTypes:      req.SemanticEntityTypes,
			TwoStage:         req.TwoStage,
			OversampleFactor: req.OversampleFactor,
			FilterSQL:        req.FilterSQL,
			FilterArgs:       req.FilterArgs,
		},
	})
	if err != nil {
		return nil, err
	}

	lexKeys := make([]search.RRFKey, 0, len(lex))
	for _, h := range lex {
		lexKeys = append(lexKeys, search.RRFKey{
			EntityType: h.EntityType,
			EntityID:   h.EntityID,
			Language:   h.Language,
			Model:      "",
		})
	}
	semKeys := make([]search.RRFKey, 0, len(sem))
	for _, h := range sem {
		semKeys = append(semKeys, search.RRFKey{
			EntityType: h.EntityType,
			EntityID:   h.EntityID,
			Language:   h.Language,
			Model:      "",
		})
	}

	fused := search.FuseRRF([][]search.RRFKey{lexKeys, semKeys}, search.RRFOptions{K: req.RRFK})
	out := make([]SearchHit, 0, len(fused))
	for _, h := range fused {
		out = append(out, SearchHit{
			EntityType: h.EntityType,
			EntityID:   h.EntityID,
			Language:   h.Language,
			Score:      h.Score,
		})
		if req.Limit > 0 && len(out) >= req.Limit {
			break
		}
	}
	return out, nil
}
