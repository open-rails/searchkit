package vl

import "context"

// AssetKind indicates what kind of asset is being embedded.
type AssetKind string

const (
	AssetKindImage AssetKind = "image"
	AssetKindFrame AssetKind = "frame" // video frame
	AssetKindVideo AssetKind = "video" // full video file (URL-only)
)

// AssetRef is an opaque handle returned by the host app's AssetLister.
// It typically points to an object in S3/MinIO (bucket/key) or a DB row.
type AssetRef struct {
	Kind     AssetKind
	Key      string
	FrameIdx *int // optional: for video-derived frames
}

// AssetLister returns the assets that should be embedded for an entity (gallery/video).
type AssetLister func(ctx context.Context, entityType string, entityID string) ([]AssetRef, error)

type AssetContent struct {
	// URL is required. embeddingkit's VL pipeline is URL-only: callers must
	// provide presigned/public URLs that the provider can fetch directly.
	URL string
}

// AssetFetcher resolves an AssetRef into either a URL (preferred) or bytes.
// (Implementations may stream in practice; keep it simple for now.)
type AssetFetcher func(ctx context.Context, ref AssetRef) (AssetContent, error)

type AssetURL struct {
	Kind AssetKind
	URL  string
}

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
