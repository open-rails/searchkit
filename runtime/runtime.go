package runtime

import (
	"context"
	"fmt"

	"github.com/doujins-org/embeddingkit/embedder"
	"github.com/doujins-org/embeddingkit/tasks"
	"github.com/doujins-org/embeddingkit/vl"
)

// DocumentBuilder builds a single canonical text document for an entity.
// Embeddings are language-agnostic in storage and search.
type DocumentBuilder func(ctx context.Context, entityType string, entityID int64) (string, error)

// Storage is implemented by the host application and maps embeddingkit's generic
// concepts to the app's concrete Postgres tables/indexes (typically halfvec(K)).
//
// This exists because halfvec requires fixed dimensions, and apps may store
// multiple models with different dims (e.g. 2560 vs 4096) in separate tables.
type Storage interface {
	UpsertTextEmbedding(ctx context.Context, entityType string, entityID int64, model string, dim int, embedding []float32) error
	UpsertVLEmbeddingAsset(ctx context.Context, entityType string, entityID int64, model string, dim int, ref vl.AssetRef, embedding []float32) error
	UpsertVLAggregateEmbedding(ctx context.Context, entityType string, entityID int64, model string, dim int, embedding []float32) error
}

type Config struct {
	EnabledModels []ModelSpec
	MaxAssets     int
}

type ModelSpec struct {
	Name string
	Dim  int
	Kind string // "text" | "vl" (future)
}

type Runtime struct {
	textEmbedder embedder.Embedder
	vlEmbedder   vl.Embedder
	taskRepo     *tasks.Repo
	storage      Storage
	builder      DocumentBuilder
	assetLister  vl.AssetLister
	assetFetcher vl.AssetFetcher
	cfg          Config
}

type Options struct {
	TextEmbedder embedder.Embedder
	VLEmbedder   vl.Embedder

	TaskRepo *tasks.Repo
	Storage  Storage

	DocumentBuilder DocumentBuilder
	AssetLister     vl.AssetLister
	AssetFetcher    vl.AssetFetcher

	Config Config
}

func New(opts Options) (*Runtime, error) {
	if opts.TextEmbedder == nil && opts.VLEmbedder == nil {
		return nil, fmt.Errorf("at least one embedder is required")
	}
	if opts.TaskRepo == nil {
		return nil, fmt.Errorf("task repo is required")
	}
	if opts.Storage == nil {
		return nil, fmt.Errorf("storage is required")
	}
	if opts.DocumentBuilder == nil {
		return nil, fmt.Errorf("document builder is required")
	}
	if opts.VLEmbedder != nil && (opts.AssetLister == nil || opts.AssetFetcher == nil) {
		return nil, fmt.Errorf("vl embedder provided but asset lister/fetcher missing")
	}

	cfg := opts.Config
	if cfg.MaxAssets <= 0 {
		cfg.MaxAssets = 8
	}

	return &Runtime{
		textEmbedder: opts.TextEmbedder,
		vlEmbedder:   opts.VLEmbedder,
		taskRepo:     opts.TaskRepo,
		storage:      opts.Storage,
		builder:      opts.DocumentBuilder,
		assetLister:  opts.AssetLister,
		assetFetcher: opts.AssetFetcher,
		cfg:          cfg,
	}, nil
}

func (r *Runtime) isTextModel(model string) bool {
	if r.textEmbedder != nil && r.textEmbedder.Model() == model {
		return true
	}
	for _, ms := range r.cfg.EnabledModels {
		if ms.Name == model && ms.Kind == "text" {
			return true
		}
	}
	return false
}

func (r *Runtime) isVLModel(model string) bool {
	if r.vlEmbedder != nil && r.vlEmbedder.Model() == model {
		return true
	}
	for _, ms := range r.cfg.EnabledModels {
		if ms.Name == model && ms.Kind == "vl" {
			return true
		}
	}
	return false
}

// EnqueueTextEmbedding enqueues a text embedding task for an entity+model.
func (r *Runtime) EnqueueTextEmbedding(ctx context.Context, entityType string, entityID int64, model string, reason string) error {
	return r.taskRepo.Enqueue(ctx, entityType, entityID, model, reason)
}

// EnqueueEmbedding enqueues an embedding task for an entity+model (text or VL).
func (r *Runtime) EnqueueEmbedding(ctx context.Context, entityType string, entityID int64, model string, reason string) error {
	return r.taskRepo.Enqueue(ctx, entityType, entityID, model, reason)
}

// GenerateAndStoreTextEmbedding computes and upserts the embedding for an entity.
// Intended to be called from a background worker.
func (r *Runtime) GenerateAndStoreTextEmbedding(ctx context.Context, entityType string, entityID int64, model string) error {
	if !r.isTextModel(model) {
		return fmt.Errorf("model %q is not configured for text embeddings", model)
	}
	if r.textEmbedder == nil {
		return fmt.Errorf("text embedder not configured")
	}
	doc, err := r.builder(ctx, entityType, entityID)
	if err != nil {
		return err
	}
	vec, err := r.textEmbedder.EmbedText(ctx, doc)
	if err != nil {
		return err
	}
	dim := len(vec)
	return r.storage.UpsertTextEmbedding(ctx, entityType, entityID, model, dim, vec)
}

// GenerateAndStoreVLEmbedding computes VL embeddings for selected assets, stores
// per-asset embeddings, and stores an aggregate embedding for fast search.
func (r *Runtime) GenerateAndStoreVLEmbedding(ctx context.Context, entityType string, entityID int64, model string) error {
	if !r.isVLModel(model) {
		return fmt.Errorf("model %q is not configured for vl embeddings", model)
	}
	if r.vlEmbedder == nil {
		return fmt.Errorf("vl embedder not configured")
	}
	if r.assetLister == nil || r.assetFetcher == nil {
		return fmt.Errorf("vl asset lister/fetcher not configured")
	}

	doc, err := r.builder(ctx, entityType, entityID)
	if err != nil {
		return err
	}

	refs, err := r.assetLister(ctx, entityType, entityID)
	if err != nil {
		return err
	}
	if len(refs) == 0 {
		return nil
	}
	if r.cfg.MaxAssets > 0 && len(refs) > r.cfg.MaxAssets {
		refs = refs[:r.cfg.MaxAssets]
	}

	var sum []float32
	var n int
	for _, ref := range refs {
		ct, data, err := r.assetFetcher(ctx, ref)
		if err != nil {
			continue
		}
		vec, err := r.vlEmbedder.EmbedTextAndImages(ctx, doc, []vl.Image{{ContentType: ct, Bytes: data}})
		if err != nil {
			continue
		}
		dim := len(vec)
		if err := r.storage.UpsertVLEmbeddingAsset(ctx, entityType, entityID, model, dim, ref, vec); err != nil {
			continue
		}
		if sum == nil {
			sum = make([]float32, dim)
		}
		if len(sum) != dim {
			continue
		}
		for i := range vec {
			sum[i] += vec[i]
		}
		n++
	}

	if n == 0 || sum == nil {
		return nil
	}

	avg := make([]float32, len(sum))
	inv := float32(1.0) / float32(n)
	for i := range sum {
		avg[i] = sum[i] * inv
	}
	return r.storage.UpsertVLAggregateEmbedding(ctx, entityType, entityID, model, len(avg), avg)
}

// GenerateAndStoreEmbedding routes to text vs VL based on model configuration.
func (r *Runtime) GenerateAndStoreEmbedding(ctx context.Context, entityType string, entityID int64, model string) error {
	if r.isVLModel(model) {
		return r.GenerateAndStoreVLEmbedding(ctx, entityType, entityID, model)
	}
	return r.GenerateAndStoreTextEmbedding(ctx, entityType, entityID, model)
}
