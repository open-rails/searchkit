package vl

import "context"

// AssetKind indicates what kind of asset is being embedded.
type AssetKind string

const (
	AssetKindImage AssetKind = "image"
	AssetKindFrame AssetKind = "frame" // video frame
)

// AssetRef is an opaque handle returned by the host app's AssetLister.
// It typically points to an object in S3/MinIO (bucket/key) or a DB row.
type AssetRef struct {
	Kind     AssetKind
	Key      string
	FrameIdx *int // optional: for video-derived frames
}

// AssetLister returns the assets that should be embedded for an entity (gallery/video).
type AssetLister func(ctx context.Context, entityType string, entityID int64) ([]AssetRef, error)

// AssetFetcher returns the bytes for the asset referenced by AssetRef.
// (Implementations may stream in practice; keep it simple for now.)
type AssetFetcher func(ctx context.Context, ref AssetRef) (contentType string, data []byte, err error)

// Embedder generates vision-language embeddings for text+assets.
//
// NOTE: Provider wiring for Qwen3-VL-Embedding is intentionally out of scope
// here for now; embeddingkit defines the interface so apps can implement it.
type Embedder interface {
	Model() string
	Dimensions() int
	EmbedTextAndImages(ctx context.Context, text string, images []Image) ([]float32, error)
}

type Image struct {
	ContentType string
	Bytes       []byte
}

