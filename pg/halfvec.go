package pg

import (
	"fmt"

	pgvector "github.com/pgvector/pgvector-go"
)

// HalfvecType returns the SQL type name for a halfvec of the given dimension.
func HalfvecType(dim int) string {
	return fmt.Sprintf("halfvec(%d)", dim)
}

// SimilarityExpr returns a SQL expression that computes cosine similarity from
// cosine distance (<=>) for halfvec columns.
func SimilarityExpr(column string, dim int) string {
	half := HalfvecType(dim)
	return "1 - (" + column + "::" + half + " <=> (?::" + half + "))"
}

// DistanceOrderExpr returns a SQL ORDER BY expression for cosine distance.
func DistanceOrderExpr(column string, dim int) string {
	half := HalfvecType(dim)
	return column + "::" + half + " <=> (?::" + half + ")"
}

// QueryVector wraps a []float32 for parameter binding via pgvector-go.
func QueryVector(vec []float32) pgvector.Vector {
	return pgvector.NewVector(vec)
}
