package signal

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestSubject(t *testing.T) {
	u := Subject{UserID: "u1"}
	if u.Kind() != SubjectKindUser || u.Key() != "u1" {
		t.Fatalf("user subject: kind=%q key=%q", u.Kind(), u.Key())
	}
	a := Subject{AnonKey: "h1"}
	if a.Kind() != SubjectKindAnon || a.Key() != "h1" {
		t.Fatalf("anon subject: kind=%q key=%q", a.Kind(), a.Key())
	}
	if err := (Subject{}).Validate(); err == nil {
		t.Fatal("empty subject should be invalid")
	}
	if err := (Subject{UserID: "u", AnonKey: "a"}).Validate(); err == nil {
		t.Fatal("both-set subject should be invalid")
	}
	if err := u.Validate(); err != nil {
		t.Fatalf("user subject should be valid: %v", err)
	}
}

func TestEventIDDeterministic(t *testing.T) {
	at := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	s1 := Signal{
		EntityRef:  EntityRef{EntityType: "blog_post", EntityID: "42"},
		Subject:    Subject{UserID: "u1"},
		Type:       "view",
		OccurredAt: at,
	}
	s2 := s1
	if s1.eventID() != s2.eventID() {
		t.Fatal("same signal must produce the same event id")
	}
	s2.OccurredAt = at.Add(time.Second)
	if s1.eventID() == s2.eventID() {
		t.Fatal("different occurred_at must produce different event ids")
	}
	s3 := s1
	s3.EventID = "explicit"
	if s3.eventID() != "explicit" {
		t.Fatal("explicit EventID must win")
	}
}

func TestWindowDayAligned(t *testing.T) {
	if !(AllTime()).dayAligned() {
		t.Fatal("all-time window is day aligned")
	}
	midnight := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	if !Between(midnight, midnight.AddDate(0, 0, 7)).dayAligned() {
		t.Fatal("midnight-to-midnight window is day aligned")
	}
	if Between(midnight.Add(3*time.Hour), midnight.AddDate(0, 0, 1)).dayAligned() {
		t.Fatal("sub-day window must not be day aligned")
	}
	if !LastDays(30).dayAligned() {
		t.Fatal("LastDays must be day aligned so Popular stays on the rollup")
	}
}

func TestReplaceWord(t *testing.T) {
	got := replaceWord("subjects + scored_signals + signals", "signals", "X")
	want := "subjects + scored_signals + X"
	if got != want {
		t.Fatalf("replaceWord: got %q want %q", got, want)
	}
	got = replaceWord("log10(1 + toFloat64(subjects))", "subjects", "uniqExactMerge(subjects)")
	want = "log10(1 + toFloat64(uniqExactMerge(subjects)))"
	if got != want {
		t.Fatalf("replaceWord: got %q want %q", got, want)
	}
}

func TestRecordSignalBuildsEventAndState(t *testing.T) {
	fc := &fakeConn{}
	st, err := NewStore(fc, "hub")
	if err != nil {
		t.Fatal(err)
	}
	at := time.Date(2026, 6, 1, 10, 0, 0, 0, time.UTC)
	err = st.RecordSignal(context.Background(), "doujins", Signal{
		EntityRef:   EntityRef{EntityType: "blog_post", EntityID: "42"},
		Subject:     Subject{UserID: "u1"},
		Type:        "view",
		OccurredAt:  at,
		Progress:    80,
		ProgressMax: 100,
		Score:       65,
		Completed:   false,
		Resume:      "scroll:80",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(fc.execs) != 2 {
		t.Fatalf("expected 2 execs (event insert + state reproject), got %d", len(fc.execs))
	}
	ins := fc.execs[0]
	if !strings.Contains(ins.query, "INSERT INTO hub.signal_events") {
		t.Fatalf("first exec must insert the event, got: %s", ins.query)
	}
	// tenant, type, id, kind, subject, signal_type, event_id, occurred_at, ...
	if ins.args[0] != "doujins" || ins.args[3] != "user" || ins.args[4] != "u1" {
		t.Fatalf("unexpected insert args: %v", ins.args)
	}
	if w := ins.args[13]; w != float64(1) {
		t.Fatalf("weight must default to 1, got %v", w)
	}
	proj := fc.execs[1]
	if !strings.Contains(proj.query, "INSERT INTO hub.signal_state") ||
		!strings.Contains(proj.query, "uniqExact(event_id)") {
		t.Fatalf("second exec must reproject state, got: %s", proj.query)
	}
}

func TestRecordSignalValidation(t *testing.T) {
	st, _ := NewStore(&fakeConn{}, "hub")
	ctx := context.Background()
	if err := st.RecordSignal(ctx, "", Signal{}); err == nil {
		t.Fatal("missing tenant must error")
	}
	if err := st.RecordSignal(ctx, "t", Signal{}); err == nil {
		t.Fatal("missing entity must error")
	}
	if err := st.RecordSignal(ctx, "t", Signal{
		EntityRef: EntityRef{EntityType: "a", EntityID: "1"},
		Subject:   Subject{UserID: "u"},
	}); err == nil {
		t.Fatal("missing type must error")
	}
}

func TestHistoryStatusFilters(t *testing.T) {
	fc := &fakeConn{}
	st, _ := NewStore(fc, "hub")
	ctx := context.Background()
	sub := Subject{UserID: "u1"}

	for _, tc := range []struct {
		status HistoryStatus
		expect string
	}{
		{HistorySeen, "max_progress > 0"},
		{HistoryInProgress, "max_progress > 0 AND NOT completed"},
		{HistoryCompleted, "AND completed"},
	} {
		fc.queries = nil
		if _, err := st.History(ctx, "t", sub, HistoryOptions{Status: tc.status}); err != nil {
			t.Fatal(err)
		}
		if q := fc.queries[0].query; !strings.Contains(q, tc.expect) {
			t.Fatalf("status %q: query missing %q:\n%s", tc.status, tc.expect, q)
		}
	}
	if _, err := st.History(ctx, "t", sub, HistoryOptions{Status: "bogus"}); err == nil {
		t.Fatal("invalid status must error")
	}
}

func TestStatesGroupsRefsByType(t *testing.T) {
	now := time.Now().UTC()
	fc := &fakeConn{rowsFor: map[string][][]any{
		"signal_state": {
			{"gallery", "7", now, now, uint32(2), uint32(5), uint32(20), false, "p:5", true, int16(40)},
		},
	}}
	st, _ := NewStore(fc, "hub")
	got, err := st.States(context.Background(), "t", Subject{UserID: "u1"}, []EntityRef{
		{EntityType: "gallery", EntityID: "7"},
		{EntityType: "gallery", EntityID: "8"},
		{EntityType: "blog_post", EntityID: "1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	q := fc.queries[0].query
	if c := strings.Count(q, "entity_type = ? AND entity_id IN ?"); c != 2 {
		t.Fatalf("expected 2 per-type clauses, got %d:\n%s", c, q)
	}
	s, ok := got[EntityRef{EntityType: "gallery", EntityID: "7"}]
	if !ok {
		t.Fatalf("missing state for gallery/7: %v", got)
	}
	if !s.Seen || s.MaxProgress != 5 || s.ProgressMax != 20 || !s.HasInteracted || s.LastScore != 40 {
		t.Fatalf("unexpected state: %+v", s)
	}
	if _, ok := got[EntityRef{EntityType: "gallery", EntityID: "8"}]; ok {
		t.Fatal("no state row for gallery/8 should mean absent from map")
	}
}

func TestStatesEmptyRefs(t *testing.T) {
	st, _ := NewStore(&fakeConn{}, "hub")
	got, err := st.States(context.Background(), "t", Subject{UserID: "u"}, nil)
	if err != nil || len(got) != 0 {
		t.Fatalf("empty refs: got %v err %v", got, err)
	}
}

func TestPopularPathSelection(t *testing.T) {
	ctx := context.Background()
	midnight := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)

	// Day-aligned -> rollup.
	fc := &fakeConn{}
	st, _ := NewStore(fc, "hub")
	if _, err := st.Popular(ctx, "t", "gallery", PopularOptions{
		Window: Between(midnight, midnight.AddDate(0, 0, 30)),
	}); err != nil {
		t.Fatal(err)
	}
	if q := fc.queries[0].query; !strings.Contains(q, "entity_daily") {
		t.Fatalf("day-aligned window must use the rollup:\n%s", q)
	}

	// Sub-day -> events.
	fc.queries = nil
	if _, err := st.Popular(ctx, "t", "gallery", PopularOptions{
		Window: Between(midnight.Add(time.Hour), midnight.Add(26*time.Hour)),
	}); err != nil {
		t.Fatal(err)
	}
	if q := fc.queries[0].query; !strings.Contains(q, "signal_events") {
		t.Fatalf("sub-day window must scan events:\n%s", q)
	}

	// All-time -> rollup, no day predicates.
	fc.queries = nil
	if _, err := st.Popular(ctx, "t", "gallery", PopularOptions{}); err != nil {
		t.Fatal(err)
	}
	if q := fc.queries[0].query; !strings.Contains(q, "entity_daily") || strings.Contains(q, "day >=") {
		t.Fatalf("all-time must use the rollup with no day bounds:\n%s", q)
	}

	// Decay -> nested day-bucket query.
	fc.queries = nil
	if _, err := st.Popular(ctx, "t", "gallery", PopularOptions{
		Weights: RankWeights{HalfLifeDays: 14},
	}); err != nil {
		t.Fatal(err)
	}
	if q := fc.queries[0].query; !strings.Contains(q, "exp2(") || !strings.Contains(q, "GROUP BY entity_id, day") {
		t.Fatalf("half-life must produce decayed day buckets:\n%s", q)
	}

	// Host RankExpr replaces the default ranking and is alias-rewritten.
	fc.queries = nil
	if _, err := st.Popular(ctx, "t", "gallery", PopularOptions{
		RankExpr: "toFloat64(subjects) + toFloat64(completions)",
	}); err != nil {
		t.Fatal(err)
	}
	if q := fc.queries[0].query; !strings.Contains(q, "toFloat64(uniqExactMerge(subjects)) + toFloat64(toUInt64(sum(completions)))") {
		t.Fatalf("rank aliases must be rewritten to aggregates:\n%s", q)
	}
}

func TestEnsureSchemaValidation(t *testing.T) {
	ctx := context.Background()
	if err := EnsureSchema(ctx, &fakeConn{}, SchemaOptions{}); err == nil {
		t.Fatal("missing database must error")
	}
	if err := EnsureSchema(ctx, &fakeConn{}, SchemaOptions{Database: "bad-name"}); err == nil {
		t.Fatal("invalid database name must error")
	}
	fc := &fakeConn{}
	if err := EnsureSchema(ctx, fc, SchemaOptions{Database: "hub"}); err != nil {
		t.Fatal(err)
	}
	if len(fc.execs) != 5 { // db + 3 tables + mv
		t.Fatalf("expected 5 DDL statements, got %d", len(fc.execs))
	}
	for _, e := range fc.execs {
		if strings.Contains(e.query, "ON CLUSTER") || strings.Contains(e.query, "Replicated") {
			t.Fatalf("non-cluster DDL must not be replicated:\n%s", e.query)
		}
	}

	fc = &fakeConn{}
	if err := EnsureSchema(ctx, fc, SchemaOptions{Database: "hub", Cluster: "main"}); err != nil {
		t.Fatal(err)
	}
	for i, e := range fc.execs {
		if !strings.Contains(e.query, "ON CLUSTER 'main'") {
			t.Fatalf("cluster DDL %d must be ON CLUSTER:\n%s", i, e.query)
		}
	}
	if !strings.Contains(fc.execs[1].query, "ReplicatedReplacingMergeTree") {
		t.Fatalf("cluster tables must use Replicated engines:\n%s", fc.execs[1].query)
	}
}

func TestCoEngagedExcludesAnchor(t *testing.T) {
	fc := &fakeConn{}
	st, _ := NewStore(fc, "hub")
	if _, err := st.CoEngaged(context.Background(), "t", EntityRef{EntityType: "gallery", EntityID: "9"}, CoEngagedOptions{}); err != nil {
		t.Fatal(err)
	}
	q := fc.queries[0].query
	if !strings.Contains(q, "NOT (entity_type = ? AND entity_id = ?)") {
		t.Fatalf("co-engaged must exclude the anchor:\n%s", q)
	}
}
