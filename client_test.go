package searchkit

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/open-rails/searchkit/search"
)

type recordingEmbedder struct {
	called bool
	model  string
	text   string
	vec    []float32
	err    error
}

func TestClientSearchWithTrace_NormalizedEmptyMatchesSearch(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientConfig{
		Pool: newTestPool(t), Schema: "test", DefaultModel: "model",
		Embedder: &recordingEmbedder{vec: []float32{1}},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	opts := SearchOptions{Mode: SearchModeLexical, LexicalEntityTypes: []string{"gallery"}}
	want, err := client.Search(context.Background(), "!!!", opts)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	got, trace, err := client.SearchWithTrace(context.Background(), "!!!", opts)
	if err != nil {
		t.Fatalf("SearchWithTrace: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("traced results differ: got %#v, want %#v", got, want)
	}
	if trace.EmptyReason != EmptyReasonNormalizedQuery || trace.ErrorCategory != "" || len(trace.Sources) != 0 {
		t.Fatalf("unexpected trace: %#v", trace)
	}
	if trace.RequestedMode != SearchModeLexical || trace.Mode != SearchModeLexical || trace.ResultLimit != 20 || trace.RRFK != 60 || trace.OversampleFactor != 5 {
		t.Fatalf("effective defaults missing from early trace: %#v", trace)
	}
	if trace.CandidateLimit != trace.ResultLimit || trace.SemanticMinSimilarity != 0 {
		t.Fatalf("effective controls missing from early trace: %#v", trace)
	}
}

func TestClientSearchWithTrace_ClampsCandidateLimit(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientConfig{Pool: newTestPool(t), Schema: "test"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	_, trace, err := client.SearchWithTrace(context.Background(), "!!!", SearchOptions{
		Mode: SearchModeLexical, LexicalEntityTypes: []string{"gallery"}, Limit: 10, CandidateLimit: 2,
	})
	if err != nil {
		t.Fatalf("SearchWithTrace: %v", err)
	}
	if trace.RequestedCandidateLimit != 2 || trace.CandidateLimit != 10 {
		t.Fatalf("candidate limit was not traced/clamped: %#v", trace)
	}
}

func TestClientSearchWithTrace_RejectsNonfiniteSemanticFloor(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientConfig{Pool: newTestPool(t), Schema: "test"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	for _, tt := range []struct {
		name  string
		query string
		floor float32
	}{
		{name: "nan", query: "query", floor: float32(math.NaN())},
		{name: "positive infinity", query: "query", floor: float32(math.Inf(1))},
		{name: "negative infinity on normalized empty", query: "!!!", floor: float32(math.Inf(-1))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, trace, err := client.SearchWithTrace(context.Background(), tt.query, SearchOptions{
				Mode: SearchModeLexical, LexicalEntityTypes: []string{"gallery"},
				SemanticMinSimilarity: tt.floor,
			})
			if err == nil || trace.ErrorCategory != "validation" || trace.EmptyReason != "" {
				t.Fatalf("SearchWithTrace() error/trace = %v/%#v, want validation failure", err, trace)
			}
			if trace.RequestedSemanticMinSimilarity != nil || trace.SemanticMinSimilarity != 0 {
				t.Fatalf("nonfinite floor leaked into trace: %#v", trace)
			}
			if _, err := json.Marshal(trace); err != nil {
				t.Fatalf("json.Marshal(trace) error = %v", err)
			}
		})
	}
}

func TestClientSearchWithTrace_RejectsUnboundedCandidateLimit(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientConfig{
		Pool: newTestPool(t), Schema: "test", DefaultModel: "model",
		Embedder: &recordingEmbedder{vec: []float32{1}},
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	tests := []SearchOptions{
		{
			Mode: SearchModeLexical, LexicalEntityTypes: []string{"gallery"},
			CandidateLimit: search.MaxCandidateLimit + 1,
		},
		{
			Mode: SearchModeLexical, LexicalEntityTypes: []string{"gallery"},
			Limit: search.MaxCandidateLimit + 1,
		},
	}
	for _, opts := range tests {
		for _, query := range []string{"query", "!!!"} {
			_, trace, err := client.SearchWithTrace(context.Background(), query, opts)
			if err == nil || trace.ErrorCategory != "validation" {
				t.Fatalf("SearchWithTrace(%q) error/trace = %v/%#v, want validation failure", query, err, trace)
			}
		}
	}
}

func TestClientSearchWithTrace_ReturnsEmbeddingErrorTrace(t *testing.T) {
	t.Parallel()

	emb := &recordingEmbedder{err: errors.New("provider unavailable")}
	client, err := NewClient(ClientConfig{
		Pool: newTestPool(t), Schema: "test", Embedder: emb, DefaultModel: "model",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	_, trace, err := client.SearchWithTrace(context.Background(), "query", SearchOptions{
		Mode: SearchModeSemantic, SemanticEntityTypes: []string{"gallery"}, Limit: 7,
	})
	if err == nil {
		t.Fatal("SearchWithTrace() error = nil, want embedding error")
	}
	if trace.ErrorCategory != "embedding" || trace.Mode != SearchModeSemantic || trace.Model != "model" || trace.ResultLimit != 7 {
		t.Fatalf("unexpected partial trace: %#v", trace)
	}
	if trace.RequestedResultLimit != 7 || trace.RequestedOversampleFactor != 0 || trace.OversampleFactor != 5 {
		t.Fatalf("requested/effective settings missing: %#v", trace)
	}
}

func TestClientSearchWithTrace_ReturnsFailedSourceTrace(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientConfig{Pool: newTestPool(t), Schema: "test"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, trace, err := client.SearchWithTrace(ctx, "query", SearchOptions{
		Mode: SearchModeLexical, Language: "en", LexicalEntityTypes: []string{"gallery"},
	})
	if err == nil {
		t.Fatal("SearchWithTrace() error = nil, want source error")
	}
	if trace.ErrorCategory != "fts" || len(trace.Sources) != 1 {
		t.Fatalf("unexpected failed trace: %#v", trace)
	}
	source := trace.Sources[0]
	if source.Backend != BackendFTS || source.ScoreKind != ScoreFTSRank || source.Status != SourceStatusFailed || source.ErrorCategory != "fts" {
		t.Fatalf("unexpected failed source: %#v", source)
	}
}

func (r *recordingEmbedder) EmbedQueryText(_ context.Context, model string, text string) ([]float32, error) {
	r.called = true
	r.model = model
	r.text = text
	if r.err != nil {
		return nil, r.err
	}
	return r.vec, nil
}

func newTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()

	// Connection is established lazily; tests that don't hit the DB won't connect.
	// Port 1 should refuse quickly if a query is attempted.
	pool, err := pgxpool.New(context.Background(), "postgres://user:pass@127.0.0.1:1/db?sslmode=disable")
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func TestClientSearch_SemanticRequiresEmbedder(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientConfig{
		Pool:         newTestPool(t),
		Schema:       "test",
		DefaultModel: "model",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = client.Search(context.Background(), "two factor", SearchOptions{
		Mode:                SearchModeSemantic,
		SemanticEntityTypes: []string{"gallery"},
	})
	if err == nil || !strings.Contains(err.Error(), "Embedder is required") {
		t.Fatalf("expected embedder-required error, got: %v", err)
	}
}

func TestClientSearch_SemanticEmbedsNormalizedText(t *testing.T) {
	t.Parallel()

	emb := &recordingEmbedder{vec: []float32{1, 0, 0}}
	client, err := NewClient(ClientConfig{
		Pool:         newTestPool(t),
		Schema:       "test",
		Embedder:     emb,
		DefaultModel: "model",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, _ = client.Search(context.Background(), "two-factor", SearchOptions{
		Mode:                SearchModeSemantic,
		SemanticEntityTypes: []string{"gallery"},
	})
	if !emb.called {
		t.Fatalf("expected embedder to be called")
	}
	if emb.text != "two factor" {
		t.Fatalf("expected normalized query text %q, got %q", "two factor", emb.text)
	}
}

func TestClientSearch_LexicalDoesNotCallEmbedder(t *testing.T) {
	t.Parallel()

	emb := &recordingEmbedder{vec: []float32{1, 0, 0}}
	client, err := NewClient(ClientConfig{
		Pool:         newTestPool(t),
		Schema:       "test",
		Embedder:     emb,
		DefaultModel: "model",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, _ = client.Search(context.Background(), "two-factor", SearchOptions{
		Mode:               SearchModeLexical,
		LexicalEntityTypes: []string{"gallery"},
	})
	if emb.called {
		t.Fatalf("expected embedder not to be called in lexical mode")
	}
}
