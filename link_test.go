package searchkit

import "testing"

func TestPlanFromHits(t *testing.T) {
	hits := []SearchHit{
		{EntityType: "tag", EntityID: "15", Score: 0.9},
		{EntityType: "tag", EntityID: "48", Score: 0.7},
		{EntityType: "artist", EntityID: "3", Score: 0.6},
	}
	plan := planFromHits("dominant woman", hits, 5, SearchTrace{})
	if plan.Query != "dominant woman" || plan.Residual != "dominant woman" {
		t.Fatalf("query/residual not preserved: %+v", plan)
	}
	if len(plan.LinkedEntities) != 3 {
		t.Fatalf("linked = %d, want 3", len(plan.LinkedEntities))
	}
	if plan.LinkedEntities[0].EntityType != "tag" || plan.LinkedEntities[0].EntityID != "15" || plan.LinkedEntities[0].Score != 0.9 {
		t.Fatalf("first linked entity wrong: %+v", plan.LinkedEntities[0])
	}
}

func TestPlanFromHits_LimitAndEmpty(t *testing.T) {
	hits := []SearchHit{
		{EntityType: "tag", EntityID: "1", Score: 0.9},
		{EntityType: "tag", EntityID: "2", Score: 0.8},
		{EntityType: "tag", EntityID: "3", Score: 0.7},
	}
	if got := planFromHits("q", hits, 2, SearchTrace{}); len(got.LinkedEntities) != 2 {
		t.Fatalf("limit not applied: got %d, want 2", len(got.LinkedEntities))
	}
	empty := planFromHits("q", nil, 5, SearchTrace{})
	if len(empty.LinkedEntities) != 0 || empty.Residual != "q" {
		t.Fatalf("empty hits should yield no links but keep residual: %+v", empty)
	}
}

func TestPlanFromHits_Provenance(t *testing.T) {
	hits := []SearchHit{
		{EntityType: "tag", EntityID: "7", Score: 0.032},  // lexical + semantic
		{EntityType: "tag", EntityID: "9", Score: 0.016},  // semantic only
		{EntityType: "tag", EntityID: "12", Score: 0.015}, // lexical only
	}
	trace := SearchTrace{Sources: []SourceTrace{
		{
			Backend: BackendFTS, Status: SourceStatusSucceeded,
			Candidates: []CandidateTrace{
				{Key: TraceKey{EntityType: "tag", EntityID: "7"}, Rank: 1, Score: 0.9},
				{Key: TraceKey{EntityType: "tag", EntityID: "12"}, Rank: 2, Score: 0.5},
			},
		},
		{
			Backend: BackendSemantic, Status: SourceStatusSucceeded,
			Candidates: []CandidateTrace{
				{Key: TraceKey{EntityType: "tag", EntityID: "7"}, Rank: 1, Score: 0.81},
				{Key: TraceKey{EntityType: "tag", EntityID: "9"}, Rank: 2, Score: 0.72},
			},
		},
	}}

	plan := planFromHits("ntr", hits, 5, trace)
	if len(plan.LinkedEntities) != 3 {
		t.Fatalf("linked = %d, want 3", len(plan.LinkedEntities))
	}
	both := plan.LinkedEntities[0]
	if !both.Lexical || both.SemanticSimilarity != 0.81 {
		t.Fatalf("dual-branch link wrong: %+v", both)
	}
	semOnly := plan.LinkedEntities[1]
	if semOnly.Lexical || semOnly.SemanticSimilarity != 0.72 {
		t.Fatalf("semantic-only link wrong: %+v", semOnly)
	}
	lexOnly := plan.LinkedEntities[2]
	if !lexOnly.Lexical || lexOnly.SemanticSimilarity != 0 {
		t.Fatalf("lexical-only link wrong: %+v", lexOnly)
	}
}

func TestProvenanceFromTrace_FailedAndMultiSource(t *testing.T) {
	trace := SearchTrace{Sources: []SourceTrace{
		{
			// Failed sources contribute nothing even with candidates recorded.
			Backend: BackendTrigram, Status: SourceStatusFailed,
			Candidates: []CandidateTrace{
				{Key: TraceKey{EntityType: "tag", EntityID: "1"}, Score: 0.9},
			},
		},
		{
			// Two semantic sources (e.g. multi-language): max cosine wins.
			Backend: BackendSemantic, Status: SourceStatusSucceeded,
			Candidates: []CandidateTrace{
				{Key: TraceKey{EntityType: "tag", EntityID: "1"}, Score: 0.61},
			},
		},
		{
			Backend: BackendSemantic, Status: SourceStatusSucceeded,
			Candidates: []CandidateTrace{
				{Key: TraceKey{EntityType: "tag", EntityID: "1"}, Score: 0.74},
			},
		},
	}}
	prov := provenanceFromTrace(trace)
	p := prov["tag\x001"]
	if p.lexical {
		t.Fatalf("failed lexical source must not mark lexical: %+v", p)
	}
	if p.semanticSimilarity != 0.74 {
		t.Fatalf("max semantic similarity = %v, want 0.74", p.semanticSimilarity)
	}
}
