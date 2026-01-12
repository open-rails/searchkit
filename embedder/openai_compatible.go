package embedder

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"

	"github.com/doujins-org/embeddingkit/internal/normalize"
)

type OpenAICompatibleConfig struct {
	BaseURL    string
	APIKey     string
	Model      string // canonical model name used by the host app
	Dimensions int    // optional; 0 means provider default
	Timeout    time.Duration
	Provider   string // advisory (deepinfra|dashscope|modelscope|...)
}

type OpenAICompatibleEmbedder struct {
	client     *openai.Client
	model      string
	dimensions int
	provider   string
}

func NewOpenAICompatible(cfg OpenAICompatibleConfig) (*OpenAICompatibleEmbedder, error) {
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, fmt.Errorf("base URL is required")
	}
	openaiCfg := openai.DefaultConfig(cfg.APIKey)
	openaiCfg.BaseURL = cfg.BaseURL
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	openaiCfg.HTTPClient = &http.Client{Timeout: timeout}
	return &OpenAICompatibleEmbedder{
		client:     openai.NewClientWithConfig(openaiCfg),
		model:      cfg.Model,
		dimensions: cfg.Dimensions,
		provider:   cfg.Provider,
	}, nil
}

func (e *OpenAICompatibleEmbedder) Model() string { return e.model }
func (e *OpenAICompatibleEmbedder) Dimensions() int {
	return e.dimensions
}

// mapCanonicalModel maps a canonical model name to a provider-specific model id.
// This intentionally mirrors the current doujins behavior and can be expanded later.
func (e *OpenAICompatibleEmbedder) mapCanonicalModel(canonical string) string {
	hint := strings.ToLower(strings.TrimSpace(e.provider))
	name := strings.ToLower(strings.TrimSpace(canonical))
	switch name {
	case "qwen-3-embedding-4b":
		if hint == "deepinfra" {
			return "Qwen/Qwen3-Embedding-4B"
		}
		if hint == "dashscope" {
			return "text-embedding-v4"
		}
		return canonical
	default:
		return canonical
	}
}

func (e *OpenAICompatibleEmbedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.EmbedTexts(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("expected 1 embedding, got %d", len(vecs))
	}
	return vecs[0], nil
}

func (e *OpenAICompatibleEmbedder) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	req := openai.EmbeddingRequest{
		Input: texts,
		Model: openai.EmbeddingModel(e.mapCanonicalModel(e.model)),
	}
	if e.dimensions > 0 {
		req.Dimensions = e.dimensions
	}

	resp, err := e.client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(resp.Data))
	}

	out := make([][]float32, len(resp.Data))
	for i, row := range resp.Data {
		vec := make([]float32, len(row.Embedding))
		for j, v := range row.Embedding {
			vec[j] = float32(v)
		}
		normalize.L2NormalizeInPlace(vec)
		out[i] = vec
	}
	return out, nil
}
