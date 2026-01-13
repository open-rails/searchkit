package search

import (
	"context"
	"testing"
)

func TestFTSSearch_Validation(t *testing.T) {
	ctx := context.Background()

	if _, err := FTSSearch(ctx, nil, "x", FTSOptions{Schema: "", Language: "en", Limit: 10}); err == nil {
		t.Fatalf("expected error for empty schema")
	}
	if _, err := FTSSearch(ctx, nil, "x", FTSOptions{Schema: "s", Language: "", Limit: 10}); err == nil {
		t.Fatalf("expected error for empty language")
	}
	if _, err := FTSSearch(ctx, nil, "x", FTSOptions{Schema: "s", Language: "en", Limit: 10}); err == nil {
		t.Fatalf("expected error for nil pool")
	}
}
