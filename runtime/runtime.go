package runtime

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/doujins-org/searchkit/embedder"
	"github.com/doujins-org/searchkit/internal/normalize"
	"github.com/doujins-org/searchkit/pg"
	"github.com/doujins-org/searchkit/tasks"
	"github.com/doujins-org/searchkit/vl"
)

// ErrEntityNotFound can be returned by host callbacks when an entity no longer
// exists (e.g. deleted). Workers should treat this as a terminal success and
// drop the task.
var ErrEntityNotFound = errors.New("entity not found")

// BuildSemanticDocument builds semantic documents for a batch of entities in a
// specific language. These documents are used to generate embeddings.
//
// The returned map should contain entries only for entities that exist. Missing
// IDs are treated as "entity not found" by the caller (and tasks may be
// dropped).
type BuildSemanticDocument func(ctx context.Context, entityType string, language string, entityIDs []string) (map[string]string, error)

// BuildLexicalString builds lexical (trigram) documents for a batch of
// entities in a specific language.
//
// These strings should stay "sharp": do NOT include search hints or
// description-like text.
type BuildLexicalString func(ctx context.Context, entityType string, language string, entityIDs []string) (map[string]string, error)

type Runtime struct {
	textEmbedders map[string]embedder.Embedder
	vlEmbedders   map[string]vl.Embedder

	taskRepo *tasks.Repo
	storage  *pg.PostgresStorage

	buildSemantic BuildSemanticDocument
	buildLexical  BuildLexicalString
	listAssetURLs vl.ListAssetURLs
}

type Options struct {
	// Required.
	Pool   *pgxpool.Pool
	Schema string

	// One embedder instance per enabled model.
	TextEmbedders []embedder.Embedder
	VLEmbedders   []vl.Embedder

	// Required.
	BuildSemanticDocument BuildSemanticDocument

	// Optional: only needed if you want searchkit-managed lexical (trigram)
	// document storage/backfill.
	BuildLexicalString BuildLexicalString

	// Required if VLEmbedders is non-empty.
	ListAssetURLs vl.ListAssetURLs

	// Optional overrides (primarily for tests).
	TaskRepo *tasks.Repo
	Storage  *pg.PostgresStorage
}

func New(opts Options) (*Runtime, error) {
	if opts.Pool == nil {
		return nil, fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(opts.Schema) == "" {
		return nil, fmt.Errorf("schema is required")
	}
	if opts.BuildSemanticDocument == nil {
		return nil, fmt.Errorf("BuildSemanticDocument is required")
	}
	if len(opts.TextEmbedders) == 0 && len(opts.VLEmbedders) == 0 {
		return nil, fmt.Errorf("at least one embedder is required")
	}

	textMap := make(map[string]embedder.Embedder, len(opts.TextEmbedders))
	for _, e := range opts.TextEmbedders {
		if e == nil {
			continue
		}
		m := strings.TrimSpace(e.Model())
		if m == "" {
			return nil, fmt.Errorf("text embedder has empty model name")
		}
		textMap[m] = e
	}

	vlMap := make(map[string]vl.Embedder, len(opts.VLEmbedders))
	for _, e := range opts.VLEmbedders {
		if e == nil {
			continue
		}
		m := strings.TrimSpace(e.Model())
		if m == "" {
			return nil, fmt.Errorf("vl embedder has empty model name")
		}
		vlMap[m] = e
	}

	if len(vlMap) > 0 && opts.ListAssetURLs == nil {
		return nil, fmt.Errorf("vl embedder provided but ListAssetURLs missing")
	}

	repo := opts.TaskRepo
	if repo == nil {
		repo = tasks.NewRepo(opts.Pool, opts.Schema)
	}
	store := opts.Storage
	if store == nil {
		store = pg.NewPostgresStorage(opts.Pool, opts.Schema)
	}

	return &Runtime{
		textEmbedders: textMap,
		vlEmbedders:   vlMap,
		taskRepo:      repo,
		storage:       store,
		buildSemantic: opts.BuildSemanticDocument,
		buildLexical:  opts.BuildLexicalString,
		listAssetURLs: opts.ListAssetURLs,
	}, nil
}

// NewWithContext constructs a Runtime and ensures searchkit's model registry
// and per-model indexes exist.
//
// This is intended for host apps that want searchkit to be fully
// configuration-driven at startup (no manual index management).
//
// IMPORTANT: index creation uses CREATE INDEX CONCURRENTLY and therefore must
// not run inside a transaction.
func NewWithContext(ctx context.Context, opts Options) (*Runtime, error) {
	rt, err := New(opts)
	if err != nil {
		return nil, err
	}
	models := rt.modelSpecs()
	if err := pg.UpsertModels(ctx, opts.Pool, opts.Schema, models); err != nil {
		return nil, err
	}
	if err := pg.EnsureIndexesForModels(ctx, opts.Pool, opts.Schema, models); err != nil {
		return nil, err
	}

	return rt, nil
}

func (r *Runtime) modelSpecs() []pg.ModelSpec {
	seen := make(map[string]struct{})
	var out []pg.ModelSpec
	for name, e := range r.textEmbedders {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, pg.ModelSpec{Name: name, Dims: e.Dimensions(), Modality: "text"})
	}
	for name, e := range r.vlEmbedders {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, pg.ModelSpec{Name: name, Dims: e.Dimensions(), Modality: "vl"})
	}
	return out
}

// ActiveModels returns the configured embedding model names.
func (r *Runtime) ActiveModels() []string {
	seen := make(map[string]struct{})
	var out []string
	for name := range r.textEmbedders {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	for name := range r.vlEmbedders {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	return out
}

// EnqueueEmbedding enqueues an embedding task for an entity+model+language (text or VL).
func (r *Runtime) EnqueueEmbedding(ctx context.Context, entityType string, entityID string, model string, language string, reason string) error {
	return r.taskRepo.Enqueue(ctx, entityType, entityID, model, language, reason)
}

// BuildSemanticDocument is exposed for worker implementations that want to batch
// hydration. The returned map contains text for entities that exist.
func (r *Runtime) BuildSemanticDocument(ctx context.Context, entityType string, language string, entityIDs []string) (map[string]string, error) {
	if r.buildSemantic == nil {
		return nil, fmt.Errorf("BuildSemanticDocument not configured")
	}
	return r.buildSemantic(ctx, entityType, language, entityIDs)
}

// BuildLexicalString is exposed for worker implementations that want to batch
// hydration for trigram document storage/backfill. The returned map contains text
// for entities that exist.
func (r *Runtime) BuildLexicalString(ctx context.Context, entityType string, language string, entityIDs []string) (map[string]string, error) {
	if r.buildLexical == nil {
		return nil, fmt.Errorf("BuildLexicalString not configured")
	}
	return r.buildLexical(ctx, entityType, language, entityIDs)
}

// ListAssetURLs is exposed for worker implementations that want to batch
// hydration. The returned map contains assets for entities that exist.
func (r *Runtime) ListAssetURLs(ctx context.Context, entityType string, entityIDs []string) (map[string][]vl.AssetURL, error) {
	if r.listAssetURLs == nil {
		return nil, fmt.Errorf("ListAssetURLs not configured")
	}
	return r.listAssetURLs(ctx, entityType, entityIDs)
}

func (r *Runtime) IsVLModel(model string) bool {
	_, ok := r.vlEmbedders[model]
	return ok
}

type TextEmbeddingItem struct {
	EntityType string
	EntityID   string
	Language   string
	Document   string
}

func (r *Runtime) GenerateAndStoreTextEmbeddingWithDocument(ctx context.Context, entityType string, entityID string, model string, language string, doc string) error {
	emb, ok := r.textEmbedders[model]
	if !ok {
		return fmt.Errorf("model %q is not configured for text embeddings", model)
	}
	if strings.TrimSpace(doc) == "" {
		return ErrEntityNotFound
	}
	vec, err := emb.EmbedText(ctx, doc)
	if err != nil {
		return err
	}
	normalize.L2NormalizeInPlace(vec)
	return r.storage.UpsertTextEmbedding(ctx, entityType, entityID, model, language, len(vec), vec)
}

// GenerateAndStoreTextEmbeddingsWithDocuments generates embeddings in a batch (provider call)
// and stores them in the database (one upsert per item).
//
// Returned per-item errors align with items by index. If the provider call fails, the
// returned error is non-nil and per-item errors are only set for inputs we can classify
// locally (e.g. ErrEntityNotFound for empty docs).
func (r *Runtime) GenerateAndStoreTextEmbeddingsWithDocuments(ctx context.Context, model string, items []TextEmbeddingItem) ([]error, error) {
	emb, ok := r.textEmbedders[model]
	if !ok {
		return nil, fmt.Errorf("model %q is not configured for text embeddings", model)
	}

	errs := make([]error, len(items))
	if len(items) == 0 {
		return errs, nil
	}

	idx := make([]int, 0, len(items))
	docs := make([]string, 0, len(items))
	for i, it := range items {
		if strings.TrimSpace(it.Document) == "" {
			errs[i] = ErrEntityNotFound
			continue
		}
		idx = append(idx, i)
		docs = append(docs, it.Document)
	}
	if len(docs) == 0 {
		return errs, nil
	}

	vecs, err := emb.EmbedTexts(ctx, docs)
	if err != nil {
		return errs, err
	}
	if len(vecs) != len(docs) {
		return errs, fmt.Errorf("expected %d embeddings, got %d", len(docs), len(vecs))
	}

	for k, vec := range vecs {
		i := idx[k]
		normalize.L2NormalizeInPlace(vec)
		it := items[i]
		if err := r.storage.UpsertTextEmbedding(ctx, it.EntityType, it.EntityID, model, it.Language, len(vec), vec); err != nil {
			errs[i] = err
		}
	}
	return errs, nil
}

func (r *Runtime) GenerateAndStoreVLEmbeddingWithInputs(ctx context.Context, entityType string, entityID string, model string, language string, doc string, assets []vl.AssetURL) error {
	emb, ok := r.vlEmbedders[model]
	if !ok {
		return fmt.Errorf("model %q is not configured for vl embeddings", model)
	}
	if strings.TrimSpace(doc) == "" || len(assets) == 0 {
		return ErrEntityNotFound
	}
	vec, err := emb.EmbedTextAndAssetURLs(ctx, doc, assets)
	if err != nil {
		return err
	}
	normalize.L2NormalizeInPlace(vec)
	return r.storage.UpsertTextEmbedding(ctx, entityType, entityID, model, language, len(vec), vec)
}

func (r *Runtime) GenerateAndStoreTextEmbedding(ctx context.Context, entityType string, entityID string, model string, language string) error {
	docs, err := r.buildSemantic(ctx, entityType, language, []string{entityID})
	if err != nil {
		return err
	}
	doc, ok := docs[entityID]
	if !ok {
		return ErrEntityNotFound
	}
	return r.GenerateAndStoreTextEmbeddingWithDocument(ctx, entityType, entityID, model, language, doc)
}

func (r *Runtime) GenerateAndStoreVLEmbedding(ctx context.Context, entityType string, entityID string, model string, language string) error {
	if r.listAssetURLs == nil {
		return fmt.Errorf("ListAssetURLs not configured")
	}

	docs, err := r.buildSemantic(ctx, entityType, language, []string{entityID})
	if err != nil {
		return err
	}
	doc, ok := docs[entityID]
	if !ok {
		return ErrEntityNotFound
	}

	assetMap, err := r.listAssetURLs(ctx, entityType, []string{entityID})
	if err != nil {
		return err
	}
	assets := assetMap[entityID]
	if len(assets) == 0 {
		return ErrEntityNotFound
	}
	return r.GenerateAndStoreVLEmbeddingWithInputs(ctx, entityType, entityID, model, language, doc, assets)
}

// GenerateAndStoreEmbedding routes to text vs VL based on which embedder is configured.
func (r *Runtime) GenerateAndStoreEmbedding(ctx context.Context, entityType string, entityID string, model string, language string) error {
	if _, ok := r.vlEmbedders[model]; ok {
		return r.GenerateAndStoreVLEmbedding(ctx, entityType, entityID, model, language)
	}
	return r.GenerateAndStoreTextEmbedding(ctx, entityType, entityID, model, language)
}
