package worker

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"

	"github.com/doujins-org/searchkit/runtime"
	"github.com/doujins-org/searchkit/tasks"
	"github.com/doujins-org/searchkit/vl"
)

type Options struct {
	BatchSize int
	LockAhead time.Duration
	PollEvery time.Duration

	MaxConcurrentEmbeds  int
	MaxRequestsPerSecond float64 // 0 = unlimited

	MaxAttempts int
	BackoffBase time.Duration
	BackoffMax  time.Duration
}

const providerEmbedBatchSize = 25

func (o *Options) withDefaults() Options {
	out := *o
	if out.BatchSize <= 0 {
		out.BatchSize = 250
	}
	if out.LockAhead <= 0 {
		out.LockAhead = 30 * time.Second
	}
	if out.PollEvery <= 0 {
		out.PollEvery = 2 * time.Second
	}
	if out.MaxConcurrentEmbeds <= 0 {
		out.MaxConcurrentEmbeds = 8
	}
	if out.MaxAttempts <= 0 {
		out.MaxAttempts = 10
	}
	if out.BackoffBase <= 0 {
		out.BackoffBase = 5 * time.Second
	}
	if out.BackoffMax <= 0 {
		out.BackoffMax = 10 * time.Minute
	}
	return out
}

func isRateLimit(err error) bool {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return apiErr.HTTPStatusCode == 429
	}
	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) {
		return reqErr.HTTPStatusCode == 429
	}
	return false
}

func isRetryable(err error) bool {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		if apiErr.HTTPStatusCode == 429 || apiErr.HTTPStatusCode == 408 {
			return true
		}
		return apiErr.HTTPStatusCode >= 500 && apiErr.HTTPStatusCode <= 599
	}
	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) {
		if reqErr.HTTPStatusCode == 429 || reqErr.HTTPStatusCode == 408 {
			return true
		}
		return reqErr.HTTPStatusCode >= 500 && reqErr.HTTPStatusCode <= 599
	}
	return true
}

func expBackoff(base time.Duration, attempt int, max time.Duration) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	mult := math.Pow(2, float64(attempt-1))
	d := time.Duration(float64(base) * mult)
	if d > max {
		return max
	}
	return d
}

func addJitter(rng *rand.Rand, d time.Duration) time.Duration {
	if d <= 0 {
		return d
	}
	// Up to 25% jitter.
	j := time.Duration(rng.Int63n(int64(d / 4)))
	return d + j
}

func makeTokenBucket(rps float64, burst int) <-chan struct{} {
	ch := make(chan struct{}, burst)
	for i := 0; i < burst; i++ {
		ch <- struct{}{}
	}
	if rps <= 0 {
		return ch
	}
	interval := time.Duration(float64(time.Second) / rps)
	if interval < time.Millisecond {
		interval = time.Millisecond
	}
	t := time.NewTicker(interval)
	go func() {
		for range t.C {
			select {
			case ch <- struct{}{}:
			default:
			}
		}
	}()
	return ch
}

func hydrateBatch(
	ctx context.Context,
	rt *runtime.Runtime,
	batch []tasks.Task,
) (docsByType map[string]map[string]map[string]string, assetsByType map[string]map[string][]vl.AssetURL, err error) {
	// docsByType[entity_type][language][entity_id] = doc
	docsByType = map[string]map[string]map[string]string{}
	assetsByType = map[string]map[string][]vl.AssetURL{}

	idsByTypeLang := map[string]map[string]map[string]struct{}{}
	assetsNeededByType := map[string]map[string]struct{}{}

	for _, t := range batch {
		if strings.TrimSpace(t.EntityType) == "" || strings.TrimSpace(t.EntityID) == "" || strings.TrimSpace(t.Language) == "" {
			continue
		}
		if _, ok := idsByTypeLang[t.EntityType]; !ok {
			idsByTypeLang[t.EntityType] = map[string]map[string]struct{}{}
		}
		if _, ok := idsByTypeLang[t.EntityType][t.Language]; !ok {
			idsByTypeLang[t.EntityType][t.Language] = map[string]struct{}{}
		}
		idsByTypeLang[t.EntityType][t.Language][t.EntityID] = struct{}{}

		if rt.IsVLModel(t.Model) {
			if _, ok := assetsNeededByType[t.EntityType]; !ok {
				assetsNeededByType[t.EntityType] = map[string]struct{}{}
			}
			assetsNeededByType[t.EntityType][t.EntityID] = struct{}{}
		}
	}

	for et, byLang := range idsByTypeLang {
		for lang, set := range byLang {
			ids := make([]string, 0, len(set))
			for id := range set {
				ids = append(ids, id)
			}
			if len(ids) == 0 {
				continue
			}
			m, err := rt.BuildSemanticDocument(ctx, et, lang, ids)
			if err != nil {
				return nil, nil, err
			}
			if _, ok := docsByType[et]; !ok {
				docsByType[et] = map[string]map[string]string{}
			}
			docsByType[et][lang] = m
		}
	}

	for et, set := range assetsNeededByType {
		ids := make([]string, 0, len(set))
		for id := range set {
			ids = append(ids, id)
		}
		if len(ids) == 0 {
			continue
		}
		m, err := rt.ListAssetURLs(ctx, et, ids)
		if err != nil {
			return nil, nil, err
		}
		assetsByType[et] = m
	}

	return docsByType, assetsByType, nil
}

func handleTaskResult(
	ctx context.Context,
	repo *tasks.Repo,
	cfg Options,
	rng *rand.Rand,
	task tasks.Task,
	err error,
) {
	if err == nil || errors.Is(err, runtime.ErrEntityNotFound) {
		_ = repo.Complete(ctx, task.EntityType, task.EntityID, task.Model, task.Language, task.NextRunAt)
		return
	}

	log.Printf(
		"searchkit: task failed entity_type=%s entity_id=%s model=%s language=%s attempts=%d err=%T %v",
		task.EntityType,
		task.EntityID,
		task.Model,
		task.Language,
		task.Attempts,
		err,
		err,
	)

	// This failure counts as the next attempt (tasks.Attempts is prior failures).
	task.Attempts = task.Attempts + 1

	// Attempt cap: move to dead-letter queue.
	if task.Attempts >= cfg.MaxAttempts {
		_ = repo.DeadLetter(ctx, task, task.NextRunAt, err)
		return
	}

	// Permanent errors: move to dead-letter queue.
	if !isRetryable(err) {
		_ = repo.DeadLetter(ctx, task, task.NextRunAt, err)
		return
	}

	attempt := task.Attempts
	base := cfg.BackoffBase
	if isRateLimit(err) {
		base = cfg.BackoffBase
	}
	backoff := expBackoff(base, attempt, cfg.BackoffMax)
	backoff = addJitter(rng, backoff)
	_ = repo.Fail(ctx, task.EntityType, task.EntityID, task.Model, task.Language, task.NextRunAt, backoff)
}

func processBatch(ctx context.Context, rt *runtime.Runtime, repo *tasks.Repo, cfg Options, batch []tasks.Task, docsByType map[string]map[string]map[string]string, assetsByType map[string]map[string][]vl.AssetURL, sem chan struct{}, tokens <-chan struct{}, rng *rand.Rand) {
	type textWorkItem struct {
		task tasks.Task
		doc  string
	}
	type vlWorkItem struct {
		task   tasks.Task
		doc    string
		assets []vl.AssetURL
	}

	textByModel := map[string][]textWorkItem{}
	vlItems := make([]vlWorkItem, 0)

	for _, task := range batch {
		doc := ""
		if byLang, ok := docsByType[task.EntityType]; ok {
			if m, ok := byLang[task.Language]; ok {
				doc = m[task.EntityID]
			}
		}
		if strings.TrimSpace(doc) == "" {
			_ = repo.Complete(ctx, task.EntityType, task.EntityID, task.Model, task.Language, task.NextRunAt)
			continue
		}

		if rt.IsVLModel(task.Model) {
			var assets []vl.AssetURL
			if m, ok := assetsByType[task.EntityType]; ok {
				assets = m[task.EntityID]
			}
			if len(assets) == 0 {
				_ = repo.Complete(ctx, task.EntityType, task.EntityID, task.Model, task.Language, task.NextRunAt)
				continue
			}
			vlItems = append(vlItems, vlWorkItem{task: task, doc: doc, assets: assets})
			continue
		}

		textByModel[task.Model] = append(textByModel[task.Model], textWorkItem{task: task, doc: doc})
	}

	var wg sync.WaitGroup

	// Text tasks are batched per model into providerEmbedBatchSize requests.
	for model, items := range textByModel {
		model := model
		items := items
		for start := 0; start < len(items); start += providerEmbedBatchSize {
			end := start + providerEmbedBatchSize
			if end > len(items) {
				end = len(items)
			}
			chunk := items[start:end]

			sem <- struct{}{}
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()

				if tokens != nil {
					select {
					case <-ctx.Done():
						return
					case <-tokens:
					}
				}

				embedItems := make([]runtime.TextEmbeddingItem, len(chunk))
				for i, it := range chunk {
					embedItems[i] = runtime.TextEmbeddingItem{
						EntityType: it.task.EntityType,
						EntityID:   it.task.EntityID,
						Language:   it.task.Language,
						Document:   it.doc,
					}
				}

				perItemErrs, batchErr := rt.GenerateAndStoreTextEmbeddingsWithDocuments(ctx, model, embedItems)
				if perItemErrs == nil {
					perItemErrs = make([]error, len(chunk))
				}

				for i, it := range chunk {
					err := perItemErrs[i]
					if err == nil && batchErr != nil {
						err = batchErr
					}
					handleTaskResult(ctx, repo, cfg, rng, it.task, err)
				}
			}()
		}
	}

	// VL tasks remain one request per task.
	for _, it := range vlItems {
		it := it
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer func() {
				<-sem
				wg.Done()
			}()

			if tokens != nil {
				select {
				case <-ctx.Done():
					return
				case <-tokens:
				}
			}

			err := rt.GenerateAndStoreVLEmbeddingWithInputs(ctx, it.task.EntityType, it.task.EntityID, it.task.Model, it.task.Language, it.doc, it.assets)
			handleTaskResult(ctx, repo, cfg, rng, it.task, err)
		}()
	}

	wg.Wait()
}

// DrainOnce fetches and processes a single batch of ready tasks, then returns.
//
// This is useful for integrating searchkit into an external job runner (e.g.
// River/Cron) where you do not want an internal infinite polling loop.
func DrainOnce(ctx context.Context, rt *runtime.Runtime, repo *tasks.Repo, opts Options) error {
	if rt == nil {
		return fmt.Errorf("runtime is required")
	}
	if repo == nil {
		return fmt.Errorf("repo is required")
	}
	cfg := opts.withDefaults()

	batch, err := repo.FetchReady(ctx, cfg.BatchSize, cfg.LockAhead)
	if err != nil {
		return err
	}
	if len(batch) == 0 {
		return nil
	}

	docsByType, assetsByType, err := hydrateBatch(ctx, rt, batch)
	if err != nil {
		return err
	}

	sem := make(chan struct{}, cfg.MaxConcurrentEmbeds)
	var tokens <-chan struct{}
	if cfg.MaxRequestsPerSecond > 0 {
		tokens = makeTokenBucket(cfg.MaxRequestsPerSecond, cfg.MaxConcurrentEmbeds)
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	processBatch(ctx, rt, repo, cfg, batch, docsByType, assetsByType, sem, tokens, rng)
	return nil
}

// Run drains embedding tasks using the provided runtime and repository.
//
// This helper is optional; host apps can implement their own runner in River/Cron/etc.
func Run(ctx context.Context, rt *runtime.Runtime, repo *tasks.Repo, opts Options) error {
	if rt == nil {
		return fmt.Errorf("runtime is required")
	}
	if repo == nil {
		return fmt.Errorf("repo is required")
	}
	cfg := opts.withDefaults()

	sem := make(chan struct{}, cfg.MaxConcurrentEmbeds)
	var tokens <-chan struct{}
	if cfg.MaxRequestsPerSecond > 0 {
		tokens = makeTokenBucket(cfg.MaxRequestsPerSecond, cfg.MaxConcurrentEmbeds)
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	ticker := time.NewTicker(cfg.PollEvery)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			batch, err := repo.FetchReady(ctx, cfg.BatchSize, cfg.LockAhead)
			if err != nil {
				return err
			}

			docsByType, assetsByType, err := hydrateBatch(ctx, rt, batch)
			if err != nil {
				return err
			}

			processBatch(ctx, rt, repo, cfg, batch, docsByType, assetsByType, sem, tokens, rng)
		}
	}
}
