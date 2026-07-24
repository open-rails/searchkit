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

func TestPrefixTSQuery(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"single term", "turning", "turning:*"},
		{"multi word", "erika ch", "erika & ch:*"},
		{"trailing partial word", "that qui", "that & qui:*"},
		{"punctuation stripped", "erika ch!", "erika & ch:*"},
		{"negation skipped", "cats -dogs", ""},
		{"leading negation skipped", "-factor", ""},
		{"empty", "", ""},
		{"only punctuation", "!!! ???", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := prefixTSQuery(tt.in); got != tt.want {
				t.Fatalf("prefixTSQuery(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestNormalizeFTSScore(t *testing.T) {
	if got := NormalizeFTSScore(0); got != 0 {
		t.Fatalf("expected 0, got %v", got)
	}
	if got := NormalizeFTSScore(-1); got != 0 {
		t.Fatalf("expected 0, got %v", got)
	}
	got := NormalizeFTSScore(1)
	if got < 0.49 || got > 0.51 {
		t.Fatalf("expected ~0.5, got %v", got)
	}
	got = NormalizeFTSScore(9)
	if got <= 0.8 || got >= 1 {
		t.Fatalf("expected in (0.8,1), got %v", got)
	}
}
