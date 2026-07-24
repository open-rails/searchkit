package searchkit

import "testing"

func TestPlanFromHits(t *testing.T) {
	hits := []SearchHit{
		{EntityType: "tag", EntityID: "15", Score: 0.9},
		{EntityType: "tag", EntityID: "48", Score: 0.7},
		{EntityType: "artist", EntityID: "3", Score: 0.6},
	}
	plan := planFromHits("dominant woman", hits, 5)
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
	if got := planFromHits("q", hits, 2); len(got.LinkedEntities) != 2 {
		t.Fatalf("limit not applied: got %d, want 2", len(got.LinkedEntities))
	}
	empty := planFromHits("q", nil, 5)
	if len(empty.LinkedEntities) != 0 || empty.Residual != "q" {
		t.Fatalf("empty hits should yield no links but keep residual: %+v", empty)
	}
}
