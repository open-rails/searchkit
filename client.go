package searchkit

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	querynorm "github.com/open-rails/searchkit/internal/normalize"
	"github.com/open-rails/searchkit/search"
)

type Embedder interface {
	EmbedQueryText(ctx context.Context, model string, text string) ([]float32, error)
}

type SearchMode string

const (
	SearchModeLexical  SearchMode = "lexical"
	SearchModeSemantic SearchMode = "semantic"
	SearchModeDual     SearchMode = "dual"
)

type LanguageMode string

const (
	// LanguageModeExact uses only the requested language.
	LanguageModeExact LanguageMode = "exact"
	// LanguageModeFallbackEnglish uses requested language first, then English.
	LanguageModeFallbackEnglish LanguageMode = "fallback_en"
)

type ClientConfig struct {
	Pool   *pgxpool.Pool
	Schema string

	Embedder Embedder

	// Defaults.
	DefaultLanguage  string
	DefaultModel     string
	DefaultLimit     int
	DefaultRRFK      int
	TwoStage         bool
	OversampleFactor int
}

type Client struct {
	pool     *pgxpool.Pool
	schema   string
	embedder Embedder

	defaultLanguage   string
	defaultModel      string
	defaultLimit      int
	defaultRRFK       int
	defaultTwoStage   bool
	defaultOversample int
}

func NewClient(cfg ClientConfig) (*Client, error) {
	if cfg.Pool == nil {
		return nil, fmt.Errorf("Pool is required")
	}
	if strings.TrimSpace(cfg.Schema) == "" {
		return nil, fmt.Errorf("Schema is required")
	}
	c := &Client{
		pool:              cfg.Pool,
		schema:            strings.TrimSpace(cfg.Schema),
		embedder:          cfg.Embedder,
		defaultLanguage:   strings.TrimSpace(cfg.DefaultLanguage),
		defaultModel:      strings.TrimSpace(cfg.DefaultModel),
		defaultLimit:      cfg.DefaultLimit,
		defaultRRFK:       cfg.DefaultRRFK,
		defaultTwoStage:   cfg.TwoStage,
		defaultOversample: cfg.OversampleFactor,
	}
	if c.defaultLanguage == "" {
		c.defaultLanguage = "en"
	}
	if c.defaultLimit <= 0 {
		c.defaultLimit = 20
	}
	if c.defaultRRFK <= 0 {
		c.defaultRRFK = 60
	}
	if c.defaultOversample < 0 {
		c.defaultOversample = 0
	}
	return c, nil
}

type SearchOptions struct {
	Language string
	// Defaults to LanguageModeExact when omitted.
	LanguageMode LanguageMode
	Mode         SearchMode

	// If set, applied to both lexical + semantic entity types unless explicitly overridden.
	EntityTypes []string

	LexicalEntityTypes  []string
	SemanticEntityTypes []string

	Limit int
	// CandidateLimit is the maximum number requested from each retrieval source
	// before RRF. It defaults to Limit and is clamped to at least Limit.
	CandidateLimit int

	// Semantic model override (defaults to client).
	Model string

	TwoStage         *bool
	OversampleFactor int
	RRFK             int
	// SemanticMinSimilarity drops semantic candidates below this cosine
	// similarity before RRF. Values <= 0 disable the additional floor.
	SemanticMinSimilarity float32

	FilterSQL  string
	FilterArgs map[string]any
}

type SearchHit struct {
	EntityType string
	EntityID   string
	Language   string
	Score      float32
}

type SimilarOptions struct {
	Language string
	Model    string
	Limit    int

	EntityTypes []string
	ExcludeIDs  []string

	MinSimilarity float32

	FilterSQL  string
	FilterArgs map[string]any
}

type SimilarHit struct {
	EntityType string
	EntityID   string
	Model      string
	Language   string
	Score      float32
}

func (c *Client) Search(ctx context.Context, userText string, opts SearchOptions) ([]SearchHit, error) {
	return c.search(ctx, userText, opts, nil)
}

// SearchWithTrace executes Search and returns opt-in retrieval provenance. On
// failure, the returned trace contains all work completed before the error.
func (c *Client) SearchWithTrace(ctx context.Context, userText string, opts SearchOptions) ([]SearchHit, SearchTrace, error) {
	var trace SearchTrace
	hits, err := c.search(ctx, userText, opts, &trace)
	return hits, trace, err
}

func (c *Client) search(ctx context.Context, userText string, opts SearchOptions, trace *SearchTrace) ([]SearchHit, error) {
	qEmbed := querynorm.QueryForEmbedding(userText)
	if trace != nil {
		*trace = initializeSearchTrace(c, qEmbed, opts)
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = c.defaultLimit
	}
	candidateLimit := opts.CandidateLimit
	if candidateLimit <= 0 {
		candidateLimit = limit
	}
	if candidateLimit < limit {
		candidateLimit = limit
	}
	if candidateLimit > search.MaxCandidateLimit {
		if trace != nil {
			trace.ErrorCategory = "validation"
		}
		return nil, fmt.Errorf("effective CandidateLimit must not exceed %d", search.MaxCandidateLimit)
	}
	if math.IsNaN(float64(opts.SemanticMinSimilarity)) || math.IsInf(float64(opts.SemanticMinSimilarity), 0) {
		if trace != nil {
			trace.ErrorCategory = "validation"
		}
		return nil, fmt.Errorf("SemanticMinSimilarity must be finite")
	}
	if qEmbed == "" || !hasAnyLetterOrNumber(qEmbed) {
		if trace != nil {
			trace.EmptyReason = EmptyReasonNormalizedQuery
		}
		return []SearchHit{}, nil
	}

	language := strings.TrimSpace(opts.Language)
	if language == "" {
		language = c.defaultLanguage
	}
	languages, err := resolveLanguageModes(language, opts.LanguageMode)
	if err != nil {
		if trace != nil {
			trace.ErrorCategory = "validation"
		}
		return nil, fmt.Errorf("invalid SearchOptions.LanguageMode %q", opts.LanguageMode)
	}
	mode := opts.Mode
	if mode == "" {
		mode = SearchModeDual
	}
	switch mode {
	case SearchModeLexical, SearchModeSemantic, SearchModeDual:
	default:
		if trace != nil {
			trace.ErrorCategory = "validation"
		}
		return nil, fmt.Errorf("invalid SearchOptions.Mode %q", mode)
	}

	semanticMinSimilarity := opts.SemanticMinSimilarity
	if semanticMinSimilarity <= 0 {
		semanticMinSimilarity = 0
	}

	rrfk := opts.RRFK
	if rrfk <= 0 {
		rrfk = c.defaultRRFK
	}
	if trace != nil {
		trace.Mode = mode
		trace.Languages = append([]string(nil), languages...)
		trace.ResultLimit = limit
		trace.CandidateLimit = candidateLimit
		trace.RRFK = rrfk
		trace.SemanticMinSimilarity = semanticMinSimilarity
	}

	lexTypes := cloneAndTrim(opts.LexicalEntityTypes)
	semTypes := cloneAndTrim(opts.SemanticEntityTypes)
	if len(opts.EntityTypes) > 0 {
		all := cloneAndTrim(opts.EntityTypes)
		if len(lexTypes) == 0 {
			lexTypes = all
		}
		if len(semTypes) == 0 {
			semTypes = all
		}
	}

	if mode != SearchModeSemantic && len(lexTypes) == 0 {
		if trace != nil {
			trace.ErrorCategory = "validation"
		}
		return nil, fmt.Errorf("LexicalEntityTypes is required for lexical/dual search")
	}
	if mode != SearchModeLexical && len(semTypes) == 0 {
		if trace != nil {
			trace.ErrorCategory = "validation"
		}
		return nil, fmt.Errorf("SemanticEntityTypes is required for semantic/dual search")
	}

	lists := make([][]search.RRFKey, 0, 3)

	if mode == SearchModeLexical || mode == SearchModeDual {
		for _, lang := range languages {
			lexLists, err := c.searchLexical(ctx, qEmbed, lang, candidateLimit, lexTypes, opts.FilterSQL, opts.FilterArgs, trace)
			if err != nil {
				return nil, err
			}
			lists = append(lists, lexLists...)
		}
	}

	if mode == SearchModeSemantic || mode == SearchModeDual {
		if c.embedder == nil {
			if trace != nil {
				trace.ErrorCategory = "embedder_required"
			}
			return nil, fmt.Errorf("Embedder is required for semantic search")
		}
		model := strings.TrimSpace(opts.Model)
		if model == "" {
			model = c.defaultModel
		}
		if strings.TrimSpace(model) == "" {
			if trace != nil {
				trace.ErrorCategory = "model_required"
			}
			return nil, fmt.Errorf("Model is required for semantic search")
		}

		twoStage := c.defaultTwoStage
		if opts.TwoStage != nil {
			twoStage = *opts.TwoStage
		}
		oversample := opts.OversampleFactor
		if oversample <= 0 {
			oversample = c.defaultOversample
		}
		oversample = search.EffectiveOversampleFactor(oversample)
		if trace != nil {
			trace.Model = model
			trace.TwoStage = twoStage
			trace.OversampleFactor = oversample
		}

		vec, err := c.embedder.EmbedQueryText(ctx, model, qEmbed)
		if err != nil {
			if trace != nil {
				trace.ErrorCategory = "embedding"
			}
			return nil, err
		}
		if len(vec) == 0 {
			if trace != nil {
				trace.EmptyReason = EmptyReasonEmbedding
			}
			return []SearchHit{}, nil
		}

		for _, lang := range languages {
			semKeys, err := c.searchSemantic(ctx, lang, model, vec, candidateLimit, semTypes, twoStage, oversample, semanticMinSimilarity, opts.FilterSQL, opts.FilterArgs, trace)
			if err != nil {
				return nil, err
			}
			lists = append(lists, semKeys)
		}
	}

	if len(lists) == 0 {
		if trace != nil {
			trace.EmptyReason = EmptyReasonNoRoute
		}
		return []SearchHit{}, nil
	}

	var fused []search.RRFHit
	if trace == nil {
		fused = search.FuseRRF(lists, search.RRFOptions{K: rrfk})
	} else {
		traced, err := search.FuseRRFWithTrace(lists, search.RRFOptions{K: rrfk})
		if err != nil {
			trace.ErrorCategory = "rrf"
			return nil, fmt.Errorf("fusing traced search results: %w", err)
		}
		fused = make([]search.RRFHit, 0, len(traced))
		trace.Results = make([]ResultTrace, 0, minInt(limit, len(traced)))
		for i, hit := range traced {
			fused = append(fused, hit.Hit)
			if i < limit {
				trace.Results = append(trace.Results, resultTraceFromRRF(i+1, hit))
			}
		}
	}
	if len(fused) == 0 && trace != nil {
		trace.EmptyReason = EmptyReasonNoCandidates
	}
	out := make([]SearchHit, 0, minInt(limit, len(fused)))
	for _, h := range fused {
		out = append(out, SearchHit{
			EntityType: h.EntityType,
			EntityID:   h.EntityID,
			Language:   h.Language,
			Score:      h.Score,
		})
		if len(out) >= limit {
			break
		}
	}
	return out, nil
}

func (c *Client) SimilarTo(ctx context.Context, entityType string, entityID string, opts SimilarOptions) ([]SimilarHit, error) {
	lang := strings.TrimSpace(opts.Language)
	if lang == "" {
		lang = c.defaultLanguage
	}
	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = c.defaultModel
	}
	if model == "" {
		return nil, fmt.Errorf("Model is required for similarity search")
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = c.defaultLimit
	}
	if strings.TrimSpace(entityType) == "" || strings.TrimSpace(entityID) == "" {
		return nil, fmt.Errorf("entityType and entityID are required")
	}

	rows, err := search.SimilarTo(ctx, c.pool, c.schema, entityType, entityID, model, lang, limit, search.Options{
		EntityTypes:   cloneAndTrim(opts.EntityTypes),
		ExcludeIDs:    cloneAndTrim(opts.ExcludeIDs),
		MinSimilarity: opts.MinSimilarity,
		FilterSQL:     opts.FilterSQL,
		FilterArgs:    opts.FilterArgs,
	})
	if err != nil {
		return nil, err
	}

	out := make([]SimilarHit, 0, len(rows))
	for _, row := range rows {
		out = append(out, SimilarHit{
			EntityType: row.EntityType,
			EntityID:   row.EntityID,
			Model:      row.Model,
			Language:   row.Language,
			Score:      row.Similarity,
		})
	}
	return out, nil
}

func (c *Client) searchLexical(ctx context.Context, q string, language string, limit int, entityTypes []string, filterSQL string, filterArgs map[string]any, trace *SearchTrace) ([][]search.RRFKey, error) {
	route := lexicalRouting(language, q, false)
	out := make([][]search.RRFKey, 0, 2)

	if route.useFTS {
		traceIndex := beginSourceTrace(trace, BackendFTS, language, "", ScoreFTSRank, limit)
		lex, err := search.FTSSearch(ctx, c.pool, q, search.FTSOptions{
			Schema:      c.schema,
			Language:    language,
			EntityTypes: entityTypes,
			Limit:       limit,
			FilterSQL:   filterSQL,
			FilterArgs:  filterArgs,
		})
		if err != nil {
			failSourceTrace(trace, traceIndex, "fts")
			return nil, err
		}
		keys := make([]search.RRFKey, 0, len(lex))
		var candidates []CandidateTrace
		if trace != nil {
			candidates = make([]CandidateTrace, 0, len(lex))
		}
		for i, h := range lex {
			keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
			if trace != nil {
				candidates = append(candidates, CandidateTrace{
					Key:  TraceKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language},
					Rank: i + 1, Score: h.Score,
				})
			}
		}
		completeSourceTrace(trace, traceIndex, candidates)
		out = append(out, keys)
	}

	if route.useTrigram {
		traceIndex := beginSourceTrace(trace, BackendTrigram, language, "", ScoreTrigramSimilarity, limit)
		lex, err := search.LexicalSearch(ctx, c.pool, q, search.LexicalOptions{
			Schema:        c.schema,
			Language:      language,
			EntityTypes:   entityTypes,
			Limit:         limit,
			MinSimilarity: 0.1,
			FilterSQL:     filterSQL,
			FilterArgs:    filterArgs,
		})
		if err != nil {
			failSourceTrace(trace, traceIndex, "trigram")
			return nil, err
		}
		keys := make([]search.RRFKey, 0, len(lex))
		var candidates []CandidateTrace
		if trace != nil {
			candidates = make([]CandidateTrace, 0, len(lex))
		}
		for i, h := range lex {
			keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
			if trace != nil {
				candidates = append(candidates, CandidateTrace{
					Key:  TraceKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language},
					Rank: i + 1, Score: h.Score,
				})
			}
		}
		completeSourceTrace(trace, traceIndex, candidates)
		out = append(out, keys)
	}

	if route.usePGroonga {
		traceIndex := beginSourceTrace(trace, BackendPGroonga, language, "", ScorePGroongaRaw, limit)
		lex, err := search.PGroongaSearch(ctx, c.pool, q, search.PGroongaOptions{
			Schema:      c.schema,
			Language:    language,
			EntityTypes: entityTypes,
			Limit:       limit,
			Prefix:      false,
			ScoreK:      1,
			FilterSQL:   filterSQL,
			FilterArgs:  filterArgs,
		})
		if err != nil {
			failSourceTrace(trace, traceIndex, "pgroonga")
			return nil, err
		}
		keys := make([]search.RRFKey, 0, len(lex))
		var candidates []CandidateTrace
		if trace != nil {
			candidates = make([]CandidateTrace, 0, len(lex))
		}
		for i, h := range lex {
			keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
			if trace != nil {
				normalizedScore := h.Score
				candidates = append(candidates, CandidateTrace{
					Key:             TraceKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language},
					Rank:            i + 1,
					Score:           h.RawScore,
					NormalizedScore: &normalizedScore,
				})
			}
		}
		completeSourceTrace(trace, traceIndex, candidates)
		out = append(out, keys)
	}

	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func (c *Client) searchSemantic(
	ctx context.Context,
	language string,
	model string,
	queryVec []float32,
	limit int,
	entityTypes []string,
	twoStage bool,
	oversampleFactor int,
	minSimilarity float32,
	filterSQL string,
	filterArgs map[string]any,
	trace *SearchTrace,
) ([]search.RRFKey, error) {
	traceIndex := beginSourceTrace(trace, BackendSemantic, language, model, ScoreCosineSimilarity, limit)
	sem, err := search.SemanticSearch(ctx, c.pool, search.Query{
		Schema:     c.schema,
		Model:      model,
		Language:   language,
		QueryVec:   queryVec,
		Limit:      limit,
		Dimensions: len(queryVec),
		Options: search.Options{
			EntityTypes:      entityTypes,
			MinSimilarity:    minSimilarity,
			TwoStage:         twoStage,
			OversampleFactor: oversampleFactor,
			FilterSQL:        filterSQL,
			FilterArgs:       filterArgs,
		},
	})
	if err != nil {
		failSourceTrace(trace, traceIndex, "semantic")
		return nil, err
	}
	keys := make([]search.RRFKey, 0, len(sem))
	var candidates []CandidateTrace
	if trace != nil {
		candidates = make([]CandidateTrace, 0, len(sem))
	}
	for i, h := range sem {
		keys = append(keys, search.RRFKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language})
		if trace != nil {
			candidates = append(candidates, CandidateTrace{
				Key:  TraceKey{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language},
				Rank: i + 1, Score: h.Similarity,
			})
		}
	}
	completeSourceTrace(trace, traceIndex, candidates)
	return keys, nil
}

type TypeaheadOptions struct {
	Language string
	// Defaults to LanguageModeExact when omitted.
	LanguageMode  LanguageMode
	EntityTypes   []string
	Limit         int
	MinSimilarity float32
	FilterSQL     string
	FilterArgs    map[string]any
}

type TypeaheadHit struct {
	EntityType string
	EntityID   string
	Language   string
	Score      float32
}

// Typeahead returns suggestions while a user is typing (typos/substring matching).
func (c *Client) Typeahead(ctx context.Context, userText string, opts TypeaheadOptions) ([]TypeaheadHit, error) {
	q := querynorm.QueryForEmbedding(userText)
	if q == "" || !hasAnyLetterOrNumber(q) {
		return []TypeaheadHit{}, nil
	}

	language := strings.TrimSpace(opts.Language)
	if language == "" {
		language = c.defaultLanguage
	}
	languages, err := resolveLanguageModes(language, opts.LanguageMode)
	if err != nil {
		return nil, fmt.Errorf("invalid TypeaheadOptions.LanguageMode %q", opts.LanguageMode)
	}
	entityTypes := cloneAndTrim(opts.EntityTypes)
	if len(entityTypes) == 0 {
		return nil, fmt.Errorf("EntityTypes is required")
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	minSim := opts.MinSimilarity

	type key struct {
		t string
		i string
		l string
	}
	merged := make(map[key]TypeaheadHit)
	add := func(h TypeaheadHit) {
		k := key{t: h.EntityType, i: h.EntityID, l: h.Language}
		if prev, ok := merged[k]; !ok || h.Score > prev.Score {
			merged[k] = h
		}
	}

	for _, lang := range languages {
		route := lexicalRouting(lang, q, true)

		if route.useTrigram {
			hits, err := search.LexicalSearch(ctx, c.pool, q, search.LexicalOptions{
				Schema:        c.schema,
				Language:      lang,
				EntityTypes:   entityTypes,
				Limit:         limit,
				MinSimilarity: minSim,
				FilterSQL:     opts.FilterSQL,
				FilterArgs:    opts.FilterArgs,
			})
			if err != nil {
				return nil, err
			}
			for _, h := range hits {
				add(TypeaheadHit{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language, Score: h.Score})
			}
		}

		if route.usePGroonga {
			hits, err := search.PGroongaSearch(ctx, c.pool, q, search.PGroongaOptions{
				Schema:      c.schema,
				Language:    lang,
				EntityTypes: entityTypes,
				Limit:       limit,
				Prefix:      true,
				ScoreK:      1,
				FilterSQL:   opts.FilterSQL,
				FilterArgs:  opts.FilterArgs,
			})
			if err != nil {
				return nil, err
			}
			for _, h := range hits {
				if minSim > 0 && h.Score < minSim {
					continue
				}
				add(TypeaheadHit{EntityType: h.EntityType, EntityID: h.EntityID, Language: h.Language, Score: h.Score})
			}
		}
	}

	out := make([]TypeaheadHit, 0, len(merged))
	for _, h := range merged {
		out = append(out, h)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if a.EntityType != b.EntityType {
			return a.EntityType < b.EntityType
		}
		if a.EntityID != b.EntityID {
			return a.EntityID < b.EntityID
		}
		return a.Language < b.Language
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func isCJKLanguage(lang string) bool {
	switch strings.ToLower(strings.TrimSpace(lang)) {
	case "ja", "zh", "ko":
		return true
	default:
		return false
	}
}

type lexicalRoute struct {
	useFTS      bool
	useTrigram  bool
	usePGroonga bool
}

func lexicalRouting(language string, q string, typeahead bool) lexicalRoute {
	if !isCJKLanguage(language) {
		if typeahead {
			return lexicalRoute{useTrigram: true}
		}
		return lexicalRoute{useFTS: true}
	}

	return lexicalRoute{
		useTrigram:  containsASCIIAlphaNum(q),
		usePGroonga: containsCJKScript(q),
	}
}

func resolveLanguageModes(language string, mode LanguageMode) ([]string, error) {
	lang := strings.ToLower(strings.TrimSpace(language))
	if lang == "" {
		lang = "en"
	}

	switch mode {
	case "", LanguageModeExact:
		return []string{lang}, nil
	case LanguageModeFallbackEnglish:
		if lang == "en" {
			return []string{"en"}, nil
		}
		return []string{lang, "en"}, nil
	default:
		return nil, fmt.Errorf("unsupported language mode")
	}
}

func cloneAndTrim(in []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
