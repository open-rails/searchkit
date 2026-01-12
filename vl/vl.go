package vl

import "context"

// AssetKind indicates what kind of asset is being embedded.
type AssetKind string

const (
	AssetKindImage AssetKind = "image"
	AssetKindFrame AssetKind = "frame" // video frame
	AssetKindVideo AssetKind = "video" // full video file (URL-only)
)

type AssetURL struct {
	Kind AssetKind
	URL  string
}

// ListAssetURLs returns the assets that should be embedded for each entity
// (gallery/video) as presigned/public URLs.
//
// NOTE: embeddingkit's VL pipeline is URL-only: embeddingkit does not upload raw
// bytes to providers.
//
// The returned map should contain entries only for entities that exist. Missing
// IDs are treated as "entity not found" by the caller (and tasks may be
// dropped).
type ListAssetURLs func(ctx context.Context, entityType string, entityIDs []string) (map[string][]AssetURL, error)

// Embedder generates vision-language embeddings for text+assets (URL-only).
//
// The app supplies text + a list of URLs (images/frames and optionally a single
// video URL) and the provider returns one fused vector.
//
// NOTE: Provider wiring for Qwen3-VL-Embedding is intentionally out of scope
// here for now; embeddingkit defines the interface so apps can implement it.
type Embedder interface {
	Model() string
	Dimensions() int
	EmbedTextAndAssetURLs(ctx context.Context, text string, assets []AssetURL) ([]float32, error)
}
