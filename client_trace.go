package searchkit

import (
	"math"
	"strings"

	"github.com/open-rails/searchkit/search"
)

// RetrievalBackend identifies one candidate source.
type RetrievalBackend string

const (
	BackendFTS      RetrievalBackend = "fts"
	BackendTrigram  RetrievalBackend = "trigram"
	BackendPGroonga RetrievalBackend = "pgroonga"
	BackendSemantic RetrievalBackend = "semantic"
)

// ScoreKind identifies the numeric domain of a candidate or result score.
type ScoreKind string

const (
	ScoreFTSRank           ScoreKind = "fts_rank"
	ScoreTrigramSimilarity ScoreKind = "trigram_similarity"
	ScorePGroongaRaw       ScoreKind = "pgroonga_score"
	ScoreCosineSimilarity  ScoreKind = "cosine_similarity"
	ScoreRRF               ScoreKind = "rrf"
)

// SourceStatus records whether an attempted retrieval source succeeded.
type SourceStatus string

const (
	SourceStatusSucceeded SourceStatus = "succeeded"
	SourceStatusFailed    SourceStatus = "failed"
)

// EmptyReason explains a successful empty SearchKit response.
type EmptyReason string

const (
	EmptyReasonNormalizedQuery EmptyReason = "normalized_query_empty"
	EmptyReasonEmbedding       EmptyReason = "embedding_empty"
	EmptyReasonNoRoute         EmptyReason = "no_route"
	EmptyReasonNoCandidates    EmptyReason = "no_candidates"
)

// TraceKey identifies an entity in retrieval provenance.
type TraceKey struct {
	EntityType string `json:"entity_type"`
	EntityID   string `json:"entity_id"`
	Language   string `json:"language"`
}

// CandidateTrace records one source candidate at its raw source rank.
type CandidateTrace struct {
	Key             TraceKey `json:"key"`
	Rank            int      `json:"rank"`
	Score           float32  `json:"score"`
	NormalizedScore *float32 `json:"normalized_score,omitempty"`
}

// SourceTrace records one routed backend execution and its candidates.
type SourceTrace struct {
	Backend       RetrievalBackend `json:"backend"`
	Language      string           `json:"language"`
	Model         string           `json:"model,omitempty"`
	ScoreKind     ScoreKind        `json:"score_kind"`
	Limit         int              `json:"limit"`
	Status        SourceStatus     `json:"status"`
	ErrorCategory string           `json:"error_category,omitempty"`
	Candidates    []CandidateTrace `json:"candidates,omitempty"`
}

// ContributionTrace records one exact RRF source contribution.
type ContributionTrace struct {
	SourceIndex  int     `json:"source_index"`
	SourceRank   int     `json:"source_rank"`
	Weight       float32 `json:"weight"`
	Contribution float32 `json:"contribution"`
}

// ResultTrace records one fused result and its source contributions.
type ResultTrace struct {
	Key           TraceKey            `json:"key"`
	Rank          int                 `json:"rank"`
	Score         float32             `json:"score"`
	ScoreKind     ScoreKind           `json:"score_kind"`
	Contributions []ContributionTrace `json:"contributions"`
}

// SearchTrace contains opt-in effective configuration and retrieval provenance.
type SearchTrace struct {
	NormalizedQuery                       string        `json:"normalized_query"`
	RequestedMode                         SearchMode    `json:"requested_mode,omitempty"`
	Mode                                  SearchMode    `json:"mode"`
	RequestedLanguage                     string        `json:"requested_language,omitempty"`
	RequestedLanguageMode                 LanguageMode  `json:"requested_language_mode,omitempty"`
	Languages                             []string      `json:"languages,omitempty"`
	RequestedModel                        string        `json:"requested_model,omitempty"`
	Model                                 string        `json:"model,omitempty"`
	RequestedResultLimit                  int           `json:"requested_result_limit"`
	ResultLimit                           int           `json:"result_limit"`
	RequestedCandidateLimit               int           `json:"requested_candidate_limit"`
	CandidateLimit                        int           `json:"candidate_limit"`
	RequestedRRFK                         int           `json:"requested_rrf_k"`
	RRFK                                  int           `json:"rrf_k"`
	RequestedTwoStage                     *bool         `json:"requested_two_stage,omitempty"`
	TwoStage                              bool          `json:"two_stage"`
	RequestedOversampleFactor             int           `json:"requested_oversample_factor"`
	OversampleFactor                      int           `json:"oversample_factor"`
	RequestedSemanticMinSimilarity        *float32      `json:"requested_semantic_min_similarity,omitempty"`
	SemanticMinSimilarity                 float32       `json:"semantic_min_similarity"`
	RequestedSemanticMinSimilarityEnabled bool          `json:"requested_semantic_min_similarity_enabled"`
	SemanticMinSimilarityEnabled          bool          `json:"semantic_min_similarity_enabled"`
	Sources                               []SourceTrace `json:"sources,omitempty"`
	Results                               []ResultTrace `json:"results,omitempty"`
	EmptyReason                           EmptyReason   `json:"empty_reason,omitempty"`
	ErrorCategory                         string        `json:"error_category,omitempty"`
}

func initializeSearchTrace(client *Client, normalizedQuery string, opts SearchOptions) SearchTrace {
	mode := opts.Mode
	if mode == "" {
		mode = SearchModeDual
	}
	language := strings.TrimSpace(opts.Language)
	if language == "" {
		language = client.defaultLanguage
	}
	languages, _ := resolveLanguageModes(language, opts.LanguageMode)
	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = client.defaultModel
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = client.defaultLimit
	}
	candidateLimit := opts.CandidateLimit
	if candidateLimit <= 0 {
		candidateLimit = limit
	}
	if candidateLimit < limit {
		candidateLimit = limit
	}
	semanticMinSimilarity := opts.SemanticMinSimilarity
	finiteSemanticMinSimilarity := !math.IsNaN(float64(semanticMinSimilarity)) && !math.IsInf(float64(semanticMinSimilarity), 0)
	semanticMinSimilarityEnabled := finiteSemanticMinSimilarity && (opts.SemanticMinSimilarityEnabled || semanticMinSimilarity > 0)
	var requestedSemanticMinSimilarity *float32
	if finiteSemanticMinSimilarity {
		value := semanticMinSimilarity
		requestedSemanticMinSimilarity = &value
	}
	if (semanticMinSimilarity <= 0 && !semanticMinSimilarityEnabled) || requestedSemanticMinSimilarity == nil {
		semanticMinSimilarity = 0
	}
	rrfk := opts.RRFK
	if rrfk <= 0 {
		rrfk = client.defaultRRFK
	}
	twoStage := client.defaultTwoStage
	var requestedTwoStage *bool
	if opts.TwoStage != nil {
		twoStage = *opts.TwoStage
		value := *opts.TwoStage
		requestedTwoStage = &value
	}
	oversample := opts.OversampleFactor
	if oversample <= 0 {
		oversample = client.defaultOversample
	}
	oversample = search.EffectiveOversampleFactor(oversample)
	return SearchTrace{
		NormalizedQuery: normalizedQuery,
		RequestedMode:   opts.Mode, Mode: mode,
		RequestedLanguage: opts.Language, RequestedLanguageMode: opts.LanguageMode,
		Languages:      append([]string(nil), languages...),
		RequestedModel: opts.Model, Model: model,
		RequestedResultLimit: opts.Limit, ResultLimit: limit,
		RequestedCandidateLimit: opts.CandidateLimit, CandidateLimit: candidateLimit,
		RequestedRRFK: opts.RRFK, RRFK: rrfk,
		RequestedTwoStage: requestedTwoStage, TwoStage: twoStage,
		RequestedOversampleFactor: opts.OversampleFactor, OversampleFactor: oversample,
		RequestedSemanticMinSimilarity:        requestedSemanticMinSimilarity,
		SemanticMinSimilarity:                 semanticMinSimilarity,
		RequestedSemanticMinSimilarityEnabled: opts.SemanticMinSimilarityEnabled,
		SemanticMinSimilarityEnabled:          semanticMinSimilarityEnabled,
	}
}

func beginSourceTrace(trace *SearchTrace, backend RetrievalBackend, language string, model string, scoreKind ScoreKind, limit int) int {
	if trace == nil {
		return -1
	}
	trace.Sources = append(trace.Sources, SourceTrace{
		Backend: backend, Language: language, Model: model, ScoreKind: scoreKind, Limit: limit,
	})
	return len(trace.Sources) - 1
}

func failSourceTrace(trace *SearchTrace, index int, category string) {
	if trace == nil || index < 0 {
		return
	}
	trace.Sources[index].Status = SourceStatusFailed
	trace.Sources[index].ErrorCategory = category
	trace.ErrorCategory = category
}

func completeSourceTrace(trace *SearchTrace, index int, candidates []CandidateTrace) {
	if trace == nil || index < 0 {
		return
	}
	trace.Sources[index].Status = SourceStatusSucceeded
	trace.Sources[index].Candidates = candidates
}

func resultTraceFromRRF(rank int, hit search.RRFTraceHit) ResultTrace {
	contributions := make([]ContributionTrace, 0, len(hit.Contributions))
	for _, contribution := range hit.Contributions {
		contributions = append(contributions, ContributionTrace{
			SourceIndex: contribution.ListIndex, SourceRank: contribution.Rank,
			Weight: contribution.Weight, Contribution: contribution.Contribution,
		})
	}
	return ResultTrace{
		Key: TraceKey{
			EntityType: hit.Hit.EntityType,
			EntityID:   hit.Hit.EntityID,
			Language:   hit.Hit.Language,
		},
		Rank: rank, Score: hit.Hit.Score, ScoreKind: ScoreRRF, Contributions: contributions,
	}
}
