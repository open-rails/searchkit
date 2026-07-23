package runtime

import (
	"context"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/open-rails/searchkit/embedder"
)

// fakeTextEmbedder records the text it was asked to embed so tests can assert
// exactly what reached the provider.
type fakeTextEmbedder struct {
	model    string
	dims     int
	lastText string
}

func (f *fakeTextEmbedder) Model() string   { return f.model }
func (f *fakeTextEmbedder) Dimensions() int { return f.dims }

func (f *fakeTextEmbedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	f.lastText = text
	return unitVec(f.dims), nil
}

func (f *fakeTextEmbedder) EmbedTexts(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = unitVec(f.dims)
	}
	return out, nil
}

// unitVec returns a non-zero vector so L2 normalization stays finite.
func unitVec(dims int) []float32 {
	vec := make([]float32, dims)
	if dims > 0 {
		vec[0] = 1
	}
	return vec
}

// lazyPool builds a pool object without connecting; the runtime constructor only
// stores it (no I/O), and these tests never acquire a connection.
func lazyPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), "postgres://u:p@127.0.0.1:1/db")
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func newRuntime(t *testing.T, fake embedder.Embedder, instructions map[string]string) *Runtime {
	t.Helper()
	rt, err := New(Options{
		Pool:                  lazyPool(t),
		Schema:                "test",
		TextEmbedders:         []embedder.Embedder{fake},
		BuildSemanticDocument: func(context.Context, string, string, []string) (map[string]string, error) { return nil, nil },
		QueryInstructions:     instructions,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return rt
}

func TestEmbedQueryText_AppliesInstructionPrefix(t *testing.T) {
	fake := &fakeTextEmbedder{model: "m", dims: 3}
	rt := newRuntime(t, fake, map[string]string{"m": "Given a search query, retrieve matching galleries"})

	if _, err := rt.EmbedQueryText(context.Background(), "m", "blonde vampire"); err != nil {
		t.Fatalf("EmbedQueryText: %v", err)
	}
	want := "Instruct: Given a search query, retrieve matching galleries\nQuery: blonde vampire"
	if fake.lastText != want {
		t.Fatalf("query text = %q, want %q", fake.lastText, want)
	}
}

func TestEmbedQueryText_NoInstructionLeavesTextBare(t *testing.T) {
	fake := &fakeTextEmbedder{model: "m", dims: 3}
	rt := newRuntime(t, fake, nil)

	if _, err := rt.EmbedQueryText(context.Background(), "m", "blonde vampire"); err != nil {
		t.Fatalf("EmbedQueryText: %v", err)
	}
	if fake.lastText != "blonde vampire" {
		t.Fatalf("query text = %q, want unmodified", fake.lastText)
	}
}

func TestEmbedQueryText_BlankInstructionSkipped(t *testing.T) {
	fake := &fakeTextEmbedder{model: "m", dims: 3}
	rt := newRuntime(t, fake, map[string]string{"m": "   "})

	if _, err := rt.EmbedQueryText(context.Background(), "m", "q"); err != nil {
		t.Fatalf("EmbedQueryText: %v", err)
	}
	if fake.lastText != "q" {
		t.Fatalf("blank instruction should be skipped; query text = %q", fake.lastText)
	}
}

func TestNew_RejectsInstructionForUnknownModel(t *testing.T) {
	fake := &fakeTextEmbedder{model: "m", dims: 3}
	_, err := New(Options{
		Pool:                  lazyPool(t),
		Schema:                "test",
		TextEmbedders:         []embedder.Embedder{fake},
		BuildSemanticDocument: func(context.Context, string, string, []string) (map[string]string, error) { return nil, nil },
		QueryInstructions:     map[string]string{"nope": "some task"},
	})
	if err == nil || !strings.Contains(err.Error(), "unknown text model") {
		t.Fatalf("New() error = %v, want unknown text model error", err)
	}
}
