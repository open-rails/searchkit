package searchkit

import (
	"context"
	"fmt"
	"strings"

	"github.com/open-rails/searchkit/search"
	"github.com/open-rails/searchkit/signal"
	"github.com/pgvector/pgvector-go"
)

// fetchEmbeddings loads stored vectors for a candidate set (one model +
// language), returning ref -> unit-ish vector. Candidates without a stored
// vector are absent.
func (c *Client) fetchEmbeddings(ctx context.Context, model, language string, refs []signal.EntityRef) (map[signal.EntityRef][]float32, error) {
	out := map[signal.EntityRef][]float32{}
	if len(refs) == 0 || strings.TrimSpace(model) == "" {
		return out, nil
	}
	byType := map[string][]string{}
	for _, r := range refs {
		byType[r.EntityType] = append(byType[r.EntityType], r.EntityID)
	}
	clauses := make([]string, 0, len(byType))
	args := []any{model, language}
	i := 3
	for t, ids := range byType {
		clauses = append(clauses, fmt.Sprintf("(entity_type = $%d AND entity_id = ANY($%d))", i, i+1))
		args = append(args, t, ids)
		i += 2
	}
	q := fmt.Sprintf(`SELECT entity_type, entity_id, embedding::halfvec
FROM %q.embedding_vectors
WHERE model = $1 AND language = $2 AND (%s)`, c.schema, strings.Join(clauses, " OR "))

	rows, err := c.pool.Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("searchkit: fetch embeddings: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var (
			ref signal.EntityRef
			vec pgvector.HalfVector
		)
		if err := rows.Scan(&ref.EntityType, &ref.EntityID, &vec); err != nil {
			return nil, fmt.Errorf("searchkit: scan embedding: %w", err)
		}
		out[ref] = vec.Slice()
	}
	return out, rows.Err()
}

func cosine(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt32(na) * sqrt32(nb))
}

func sqrt32(v float32) float32 {
	if v <= 0 {
		return 0
	}
	x := v
	for i := 0; i < 16; i++ {
		x = 0.5 * (x + v/x)
	}
	return x
}

// diversifyRecHits applies MMR over a ranked candidate list using stored
// embeddings for candidate-to-candidate similarity. lambda in (0,1]: higher =
// more relevance, less diversity. Candidates without vectors contribute zero
// redundancy (they never get demoted for similarity). No-op when lambda is 0
// or vectors are unavailable.
func (h *EmbeddedHub) diversifyRecHits(ctx context.Context, hits []RecHit, lambda float32, model, language string) []RecHit {
	if lambda <= 0 || lambda >= 1 || len(hits) < 3 {
		return hits
	}
	if strings.TrimSpace(model) == "" {
		model = h.client.defaultModel
	}
	if strings.TrimSpace(language) == "" {
		language = h.client.defaultLanguage
	}
	refs := make([]signal.EntityRef, 0, len(hits))
	for _, r := range hits {
		refs = append(refs, signal.EntityRef{EntityType: r.EntityType, EntityID: r.EntityID})
	}
	vecs, err := h.client.fetchEmbeddings(ctx, model, language, refs)
	if err != nil || len(vecs) == 0 {
		return hits // diversity is best-effort
	}

	searchHits := make([]search.Hit, 0, len(hits))
	for _, r := range hits {
		searchHits = append(searchHits, search.Hit{EntityType: r.EntityType, EntityID: r.EntityID, Similarity: r.Score})
	}
	reranked := search.MMRReRank(searchHits, len(searchHits), lambda, func(a, b search.Hit) float32 {
		va, oka := vecs[signal.EntityRef{EntityType: a.EntityType, EntityID: a.EntityID}]
		vb, okb := vecs[signal.EntityRef{EntityType: b.EntityType, EntityID: b.EntityID}]
		if !oka || !okb {
			return 0
		}
		return cosine(va, vb)
	})

	scoreByRef := make(map[signal.EntityRef]float32, len(hits))
	for _, r := range hits {
		scoreByRef[signal.EntityRef{EntityType: r.EntityType, EntityID: r.EntityID}] = r.Score
	}
	out := make([]RecHit, 0, len(reranked))
	for _, s := range reranked {
		ref := signal.EntityRef{EntityType: s.EntityType, EntityID: s.EntityID}
		out = append(out, RecHit{EntityType: s.EntityType, EntityID: s.EntityID, Score: scoreByRef[ref]})
	}
	return out
}
