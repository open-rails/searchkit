package searchkit

import (
	"context"
	"strings"

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
	q := normalizeWhitespace(query)
	if q == "" || !hasAnyLetterOrNumber(q) {
		return []SearchHit{}, nil
	}

	var lexLists [][]search.RRFKey
	{
		lang := strings.ToLower(strings.TrimSpace(req.Language))
		if lang == "ja" || lang == "zh" || lang == "ko" {
			usePGroonga := containsCJKScript(q)
			useTrigram := containsASCIIAlphaNum(q)

			// Trigram lexical (romaji/pinyin)
			if useTrigram {
				lex, err := search.LexicalSearch(ctx, pool, q, search.LexicalOptions{
					Schema:        req.Schema,
					Language:      req.Language,
					EntityTypes:   req.LexicalEntityTypes,
					Limit:         req.Limit,
					MinSimilarity: 0.1,
				})
				if err != nil {
					return nil, err
				}
				keys := make([]search.RRFKey, 0, len(lex))
				for _, h := range lex {
					keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
				}
				lexLists = append(lexLists, keys)
			}

			// PGroonga lexical (native script)
			if usePGroonga {
				lex, err := search.PGroongaSearch(ctx, pool, q, search.PGroongaOptions{
					Schema:      req.Schema,
					Language:    req.Language,
					EntityTypes: req.LexicalEntityTypes,
					Limit:       req.Limit,
					Prefix:      false,
					ScoreK:      1,
				})
				if err != nil {
					return nil, err
				}
				keys := make([]search.RRFKey, 0, len(lex))
				for _, h := range lex {
					keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
				}
				lexLists = append(lexLists, keys)
			}
		} else {
			lex, err := search.FTSSearch(ctx, pool, q, search.FTSOptions{
				Schema:      req.Schema,
				Language:    req.Language,
				EntityTypes: req.LexicalEntityTypes,
				Limit:       req.Limit,
			})
			if err != nil {
				return nil, err
			}
			keys := make([]search.RRFKey, 0, len(lex))
			for _, h := range lex {
				keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
			}
			lexLists = append(lexLists, keys)
		}
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

	semKeys := make([]search.RRFKey, 0, len(sem))
	for _, h := range sem {
		semKeys = append(semKeys, search.RRFKey{
			EntityType: h.EntityType,
			EntityID:   h.EntityID,
			Language:   h.Language,
			Model:      "",
		})
	}

	lists := make([][]search.RRFKey, 0, len(lexLists)+1)
	lists = append(lists, lexLists...)
	lists = append(lists, semKeys)

	fused := search.FuseRRF(lists, search.RRFOptions{K: req.RRFK})
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
