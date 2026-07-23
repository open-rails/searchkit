package searchkit

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestFuseSources_UnequalWeightsReorderEqualRankHits(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{
		{
			ID:     "lower-weight",
			Weight: 1,
			Hits:   []SearchHit{{EntityType: "gallery", EntityID: "a", Language: "en"}},
		},
		{
			ID:     "higher-weight",
			Weight: 2,
			Hits:   []SearchHit{{EntityType: "gallery", EntityID: "z", Language: "en"}},
		},
	}

	got, err := FuseSources(sources, FusionOptions{RRFK: 10})
	if err != nil {
		t.Fatalf("FuseSources() error = %v", err)
	}
	want := []SearchHit{
		{EntityType: "gallery", EntityID: "z", Language: "en", Score: float32(2) / 11},
		{EntityType: "gallery", EntityID: "a", Language: "en", Score: float32(1) / 11},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("FuseSources() = %#v, want %#v", got, want)
	}
}

func TestFuseSourcesWithTrace_MatchesFuseSources(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{
		{
			ID:     "lexical-v1",
			Weight: 1,
			Hits: []SearchHit{
				{EntityType: "gallery", EntityID: "shared", Language: "en", Score: 0.9},
				{EntityType: "gallery", EntityID: "lexical-only", Language: "en", Score: 0.8},
			},
		},
		{
			ID:     "semantic-v2",
			Weight: 2,
			Hits: []SearchHit{
				{EntityType: "gallery", EntityID: "semantic-only", Language: "en", Score: 0.7},
				{EntityType: "gallery", EntityID: "shared", Language: "en", Score: 0.6},
			},
		},
	}
	opts := FusionOptions{RRFK: 10}

	plain, err := FuseSources(sources, opts)
	if err != nil {
		t.Fatalf("FuseSources() error = %v", err)
	}
	traced, err := FuseSourcesWithTrace(sources, opts)
	if err != nil {
		t.Fatalf("FuseSourcesWithTrace() error = %v", err)
	}

	tracedHits := make([]SearchHit, 0, len(traced))
	for _, hit := range traced {
		tracedHits = append(tracedHits, hit.Hit)
	}
	if !reflect.DeepEqual(tracedHits, plain) {
		t.Fatalf("traced hits = %#v, want %#v", tracedHits, plain)
	}

	want := []FusionTraceHit{
		{
			Hit: SearchHit{
				EntityType: "gallery",
				EntityID:   "shared",
				Language:   "en",
				Score:      float32(1)/11 + float32(2)/12,
			},
			Contributions: []FusionContribution{
				{
					SourceID:     "lexical-v1",
					SourceRank:   1,
					Weight:       1,
					Contribution: float32(1) / 11,
				},
				{
					SourceID:     "semantic-v2",
					SourceRank:   2,
					Weight:       2,
					Contribution: float32(2) / 12,
				},
			},
		},
		{
			Hit: SearchHit{
				EntityType: "gallery",
				EntityID:   "semantic-only",
				Language:   "en",
				Score:      float32(2) / 11,
			},
			Contributions: []FusionContribution{
				{
					SourceID:     "semantic-v2",
					SourceRank:   1,
					Weight:       2,
					Contribution: float32(2) / 11,
				},
			},
		},
		{
			Hit: SearchHit{
				EntityType: "gallery",
				EntityID:   "lexical-only",
				Language:   "en",
				Score:      float32(1) / 12,
			},
			Contributions: []FusionContribution{
				{
					SourceID:     "lexical-v1",
					SourceRank:   2,
					Weight:       1,
					Contribution: float32(1) / 12,
				},
			},
		},
	}
	if !reflect.DeepEqual(traced, want) {
		t.Fatalf("FuseSourcesWithTrace() = %#v, want %#v", traced, want)
	}
}

func TestFuseSources_RejectsFiniteWeightScoreOverflow(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{
		{
			ID:     "first",
			Weight: math.MaxFloat32,
			Hits:   []SearchHit{{EntityType: "gallery", EntityID: "shared"}},
		},
		{
			ID:     "second",
			Weight: math.MaxFloat32,
			Hits:   []SearchHit{{EntityType: "gallery", EntityID: "shared"}},
		},
		{
			ID:     "third",
			Weight: math.MaxFloat32,
			Hits:   []SearchHit{{EntityType: "gallery", EntityID: "shared"}},
		},
	}
	opts := FusionOptions{RRFK: 1}

	if _, err := FuseSources(sources, opts); err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("FuseSources() error = %v, want score overflow error", err)
	}
	if _, err := FuseSourcesWithTrace(sources, opts); err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("FuseSourcesWithTrace() error = %v, want score overflow error", err)
	}
}

func TestFuseSources_LimitTruncatesCompletedFusion(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{
		{
			ID:     "first",
			Weight: 1,
			Hits: []SearchHit{
				{EntityType: "gallery", EntityID: "a"},
				{EntityType: "gallery", EntityID: "shared"},
			},
		},
		{
			ID:     "second",
			Weight: 1,
			Hits: []SearchHit{
				{EntityType: "gallery", EntityID: "z"},
				{EntityType: "gallery", EntityID: "shared"},
			},
		},
	}

	full, err := FuseSources(sources, FusionOptions{})
	if err != nil {
		t.Fatalf("FuseSources() error = %v", err)
	}
	if len(full) != 3 {
		t.Fatalf("len(FuseSources()) = %d, want all 3 results", len(full))
	}
	fullTrace, err := FuseSourcesWithTrace(sources, FusionOptions{})
	if err != nil {
		t.Fatalf("FuseSourcesWithTrace() error = %v", err)
	}
	if len(fullTrace) != 3 {
		t.Fatalf("len(FuseSourcesWithTrace()) = %d, want all 3 results", len(fullTrace))
	}
	for _, tt := range []struct {
		name  string
		limit int
	}{
		{name: "zero", limit: 0},
		{name: "negative", limit: -1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FuseSources(sources, FusionOptions{Limit: tt.limit})
			if err != nil {
				t.Fatalf("FuseSources(Limit: %d) error = %v", tt.limit, err)
			}
			if !reflect.DeepEqual(got, full) {
				t.Fatalf("FuseSources(Limit: %d) = %#v, want all results %#v", tt.limit, got, full)
			}

			traced, err := FuseSourcesWithTrace(sources, FusionOptions{Limit: tt.limit})
			if err != nil {
				t.Fatalf("FuseSourcesWithTrace(Limit: %d) error = %v", tt.limit, err)
			}
			if !reflect.DeepEqual(traced, fullTrace) {
				t.Fatalf(
					"FuseSourcesWithTrace(Limit: %d) = %#v, want all results %#v",
					tt.limit,
					traced,
					fullTrace,
				)
			}
		})
	}

	limited, err := FuseSources(sources, FusionOptions{Limit: 1})
	if err != nil {
		t.Fatalf("FuseSources(Limit: 1) error = %v", err)
	}
	if len(limited) != 1 || limited[0].EntityID != "shared" || !reflect.DeepEqual(limited, full[:1]) {
		t.Fatalf("FuseSources(Limit: 1) = %#v, want completed fusion prefix %#v", limited, full[:1])
	}

	limitedTrace, err := FuseSourcesWithTrace(sources, FusionOptions{Limit: 1})
	if err != nil {
		t.Fatalf("FuseSourcesWithTrace(Limit: 1) error = %v", err)
	}
	if len(limitedTrace) != 1 || !reflect.DeepEqual(limitedTrace, fullTrace[:1]) {
		t.Fatalf(
			"FuseSourcesWithTrace(Limit: 1) = %#v, want completed fusion prefix %#v",
			limitedTrace,
			fullTrace[:1],
		)
	}
}

func TestFuseSources_NonPositiveRRFKUsesDefault(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{{
		ID:     "source",
		Weight: 2,
		Hits:   []SearchHit{{EntityType: "gallery", EntityID: "1", Language: "en"}},
	}}
	wantHit := SearchHit{
		EntityType: "gallery",
		EntityID:   "1",
		Language:   "en",
		Score:      float32(2) / 61,
	}
	wantTrace := []FusionTraceHit{{
		Hit: wantHit,
		Contributions: []FusionContribution{{
			SourceID:     "source",
			SourceRank:   1,
			Weight:       2,
			Contribution: float32(2) / 61,
		}},
	}}

	for _, tt := range []struct {
		name string
		rrfk int
	}{
		{name: "zero", rrfk: 0},
		{name: "negative", rrfk: -1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FuseSources(sources, FusionOptions{RRFK: tt.rrfk})
			if err != nil {
				t.Fatalf("FuseSources(RRFK: %d) error = %v", tt.rrfk, err)
			}
			if !reflect.DeepEqual(got, []SearchHit{wantHit}) {
				t.Fatalf("FuseSources(RRFK: %d) = %#v, want K=60 result %#v", tt.rrfk, got, wantHit)
			}

			traced, err := FuseSourcesWithTrace(sources, FusionOptions{RRFK: tt.rrfk})
			if err != nil {
				t.Fatalf("FuseSourcesWithTrace(RRFK: %d) error = %v", tt.rrfk, err)
			}
			if !reflect.DeepEqual(traced, wantTrace) {
				t.Fatalf(
					"FuseSourcesWithTrace(RRFK: %d) = %#v, want K=60 result %#v",
					tt.rrfk,
					traced,
					wantTrace,
				)
			}
		})
	}
}

func TestFuseSources_DifferentLanguagesRemainDistinct(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{{
		ID:     "source",
		Weight: 1,
		Hits: []SearchHit{
			{EntityType: "gallery", EntityID: "1", Language: "en"},
			{EntityType: "gallery", EntityID: "1", Language: "ja"},
		},
	}}
	want := []SearchHit{
		{EntityType: "gallery", EntityID: "1", Language: "en", Score: float32(1) / 11},
		{EntityType: "gallery", EntityID: "1", Language: "ja", Score: float32(1) / 12},
	}

	got, err := FuseSources(sources, FusionOptions{RRFK: 10})
	if err != nil {
		t.Fatalf("FuseSources() error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("FuseSources() = %#v, want distinct language hits %#v", got, want)
	}

	traced, err := FuseSourcesWithTrace(sources, FusionOptions{RRFK: 10})
	if err != nil {
		t.Fatalf("FuseSourcesWithTrace() error = %v", err)
	}
	tracedHits := make([]SearchHit, 0, len(traced))
	for _, hit := range traced {
		tracedHits = append(tracedHits, hit.Hit)
	}
	if !reflect.DeepEqual(tracedHits, want) {
		t.Fatalf("traced hits = %#v, want distinct language hits %#v", tracedHits, want)
	}
}

func TestFuseSources_EmptyInputReturnsInitializedSlices(t *testing.T) {
	t.Parallel()

	got, err := FuseSources(nil, FusionOptions{})
	if err != nil {
		t.Fatalf("FuseSources() error = %v", err)
	}
	if got == nil || len(got) != 0 {
		t.Fatalf("FuseSources() = %#v, want non-nil empty slice", got)
	}

	traced, err := FuseSourcesWithTrace(nil, FusionOptions{})
	if err != nil {
		t.Fatalf("FuseSourcesWithTrace() error = %v", err)
	}
	if traced == nil || len(traced) != 0 {
		t.Fatalf("FuseSourcesWithTrace() = %#v, want non-nil empty slice", traced)
	}
}

func TestFuseSources_DoesNotMutateInputs(t *testing.T) {
	t.Parallel()

	sources := []FusionSource{
		{
			ID:     " lexical ",
			Weight: 1.5,
			Hits: []SearchHit{
				{EntityType: " gallery ", EntityID: " 1 ", Language: " en ", Score: 0.75},
			},
		},
	}
	wantSources := make([]FusionSource, len(sources))
	for i, source := range sources {
		wantSources[i] = source
		wantSources[i].Hits = append([]SearchHit(nil), source.Hits...)
	}
	opts := FusionOptions{RRFK: 20, Limit: 1}
	wantOpts := opts

	if _, err := FuseSources(sources, opts); err != nil {
		t.Fatalf("FuseSources() error = %v", err)
	}
	if _, err := FuseSourcesWithTrace(sources, opts); err != nil {
		t.Fatalf("FuseSourcesWithTrace() error = %v", err)
	}
	if !reflect.DeepEqual(sources, wantSources) {
		t.Fatalf("sources mutated: got %#v, want %#v", sources, wantSources)
	}
	if opts != wantOpts {
		t.Fatalf("options mutated: got %#v, want %#v", opts, wantOpts)
	}
}

func TestFuseSources_Validation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		sources []FusionSource
		wantErr string
	}{
		{
			name:    "blank source id",
			sources: []FusionSource{{ID: " \t", Weight: 1}},
			wantErr: "id must be non-empty",
		},
		{
			name: "duplicate source id",
			sources: []FusionSource{
				{ID: "source", Weight: 1},
				{ID: "source", Weight: 1},
			},
			wantErr: "must be unique",
		},
		{
			name: "duplicate source id after trimming",
			sources: []FusionSource{
				{ID: "source", Weight: 1},
				{ID: " source ", Weight: 1},
			},
			wantErr: "must be unique",
		},
		{
			name:    "zero weight",
			sources: []FusionSource{{ID: "source", Weight: 0}},
			wantErr: "greater than zero",
		},
		{
			name:    "negative weight",
			sources: []FusionSource{{ID: "source", Weight: -1}},
			wantErr: "greater than zero",
		},
		{
			name:    "nan weight",
			sources: []FusionSource{{ID: "source", Weight: float32(math.NaN())}},
			wantErr: "must be finite",
		},
		{
			name:    "positive infinite weight",
			sources: []FusionSource{{ID: "source", Weight: float32(math.Inf(1))}},
			wantErr: "must be finite",
		},
		{
			name:    "negative infinite weight",
			sources: []FusionSource{{ID: "source", Weight: float32(math.Inf(-1))}},
			wantErr: "must be finite",
		},
		{
			name: "blank entity type",
			sources: []FusionSource{{
				ID:     "source",
				Weight: 1,
				Hits:   []SearchHit{{EntityType: " ", EntityID: "1"}},
			}},
			wantErr: "entity type must be non-empty",
		},
		{
			name: "blank entity id",
			sources: []FusionSource{{
				ID:     "source",
				Weight: 1,
				Hits:   []SearchHit{{EntityType: "gallery", EntityID: " \t"}},
			}},
			wantErr: "entity id must be non-empty",
		},
		{
			name: "duplicate full identity within source",
			sources: []FusionSource{{
				ID:     "source",
				Weight: 1,
				Hits: []SearchHit{
					{EntityType: "gallery", EntityID: "1", Language: "en"},
					{EntityType: " gallery ", EntityID: " 1 ", Language: " en "},
				},
			}},
			wantErr: "duplicates identity",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if _, err := FuseSources(tt.sources, FusionOptions{}); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("FuseSources() error = %v, want error containing %q", err, tt.wantErr)
			}
			if _, err := FuseSourcesWithTrace(tt.sources, FusionOptions{}); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("FuseSourcesWithTrace() error = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}
