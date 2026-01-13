package embedder

import "context"

// Embedder generates text embeddings (language-agnostic; the query language is part of the text).
type Embedder interface {
	Model() string
	Dimensions() int
	EmbedText(ctx context.Context, text string) ([]float32, error)
	EmbedTexts(ctx context.Context, texts []string) ([][]float32, error)
}
