package pg

import "fmt"

// TwoStageHalfvecSQL returns a SQL snippet for two-stage retrieval:
// stage-1: approximate candidate retrieval using Hamming distance on binary quantization
// stage-2: exact rescoring using halfvec cosine distance.
//
// This helper is intentionally low-level: callers still need to provide table
// names, filters, and bind the query vector parameters in the right order.
//
// Expected bindings (in order):
//  1) query halfvec vector (for stage-1 binary_quantize)
//  2) query halfvec vector (for stage-2 rescoring)
//
// NOTE: this assumes pgvector functions/operators are available:
// - binary_quantize(halfvec) returning bit(K)
// - hamming distance operator (commonly <~>)
func TwoStageHalfvecSQL(table string, embeddingColumn string, dim int, oversample int, limit int) string {
	half := HalfvecType(dim)
	// We use a CTE so callers can tack on additional joins/filters as needed.
	return fmt.Sprintf(`
WITH candidates AS (
  SELECT
    t.*,
    (%s) AS similarity
  FROM %s t
  WHERE t.%s IS NOT NULL
  ORDER BY binary_quantize(t.%s::%s) <~> binary_quantize(?::%s)
  LIMIT %d
)
SELECT *
FROM candidates
ORDER BY %s
LIMIT %d
`, SimilarityExpr("candidates."+embeddingColumn, dim),
		table,
		embeddingColumn,
		embeddingColumn,
		half,
		half,
		oversample,
		DistanceOrderExpr("candidates."+embeddingColumn, dim),
		limit,
	)
}

