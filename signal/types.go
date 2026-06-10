// Package signal implements the searchkit signal plane: an append-only event
// stream of host-defined interaction signals plus a durable per-(subject,
// entity) current-state projection, both stored in ClickHouse.
//
// The signal plane records *what each subject did and thought about each
// entity* and projects that into fast reads for history, unseen, engagement,
// popularity, and (above this package) personalized search + recommendations.
//
// Mechanism vs meaning: this package owns storage, aggregation, and queries.
// Signal types, entity types, scoring weights, and completion rules are
// host-defined data — no business noun appears in the schema.
package signal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

// EntityRef identifies one entity. The tenant dimension is supplied separately
// by the caller (the embedded hub pins a single tenant at construction).
type EntityRef struct {
	EntityType string
	EntityID   string
}

func (r EntityRef) validate() error {
	if strings.TrimSpace(r.EntityType) == "" || strings.TrimSpace(r.EntityID) == "" {
		return fmt.Errorf("signal: EntityType and EntityID are required")
	}
	return nil
}

// Subject is who acted: a resolved user id (e.g. from authkit) or, for
// anonymous traffic, a session-key hash. Exactly one of the two must be set.
//
// History/unseen are meaningful only for logged-in subjects; anonymous signals
// still feed popularity/engagement aggregates.
type Subject struct {
	UserID  string
	AnonKey string
}

// SubjectKind values stored in the subject_kind column.
const (
	SubjectKindUser = "user"
	SubjectKindAnon = "anon"
)

// Kind returns "user" or "anon".
func (s Subject) Kind() string {
	if strings.TrimSpace(s.UserID) != "" {
		return SubjectKindUser
	}
	return SubjectKindAnon
}

// Key returns the stored subject identifier.
func (s Subject) Key() string {
	if strings.TrimSpace(s.UserID) != "" {
		return strings.TrimSpace(s.UserID)
	}
	return strings.TrimSpace(s.AnonKey)
}

// Validate checks that exactly one of UserID / AnonKey is set.
func (s Subject) Validate() error {
	user := strings.TrimSpace(s.UserID) != ""
	anon := strings.TrimSpace(s.AnonKey) != ""
	if user == anon { // neither or both
		return fmt.Errorf("signal: exactly one of Subject.UserID / Subject.AnonKey must be set")
	}
	return nil
}

// Signal is one summarized interaction: a view-session, reaction, rating,
// purchase, etc. One row per session/interaction — never one row per
// read/scroll tick; the host reports a single summarized event when a session
// ends (exit beacon style).
type Signal struct {
	EntityRef
	Subject Subject

	// Type is HOST-DEFINED: "view" | "rate" | "purchase" | "listen" | ...
	Type string

	// EventID is an optional idempotency key. When empty, a deterministic
	// hash of (entity, subject, type, occurred_at) is used, so re-delivering
	// the same event deduplicates instead of double-counting.
	EventID string

	OccurredAt time.Time

	// Implicit consumption (all optional).
	DurationS   uint32 // active time
	Progress    uint32 // consumption numerator (pages / scroll % / watched s)
	ProgressMax uint32 // consumption denominator (page count / 100 / duration)

	// Explicit feedback (all optional).
	Value float64 // rating / vote / price / score
	Label string  // categorical: reaction kind / variant

	// Weight is host/scorer-assigned importance. Defaults to 1.
	Weight float64

	// Score is the engagement score from the entity type's Scorer. When a
	// Scorer is registered for the entity type, the hub fills this (and
	// Progress/ProgressMax/Completed) before recording.
	Score int16

	// Completed per the entity type's completion rule.
	Completed bool

	// Resume is an opaque host pointer for "pick up where you left off":
	// last page / scroll offset / timestamp. Carried into current-state.
	Resume string

	// Payload holds anything else, JSON-encoded at rest, for re-scoring later.
	Payload map[string]any
}

func (s Signal) validate() error {
	if err := s.EntityRef.validate(); err != nil {
		return err
	}
	if err := s.Subject.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(s.Type) == "" {
		return fmt.Errorf("signal: Type is required")
	}
	return nil
}

// eventID returns the explicit EventID or a deterministic content hash.
func (s Signal) eventID() string {
	if id := strings.TrimSpace(s.EventID); id != "" {
		return id
	}
	h := sha256.Sum256([]byte(strings.Join([]string{
		s.EntityType, s.EntityID, s.Subject.Kind(), s.Subject.Key(),
		s.Type, fmt.Sprintf("%d", s.OccurredAt.UTC().Unix()),
	}, "\x1f")))
	return hex.EncodeToString(h[:16])
}

// Scored is the result of an entity type's Scorer.
type Scored struct {
	Score       int16
	Progress    uint32
	ProgressMax uint32
	Completed   bool
}

// Scorer maps a raw signal (the "session") to a normalized engagement score,
// generic progress, and whether it counts as "completed". Host-provided per
// entity type; this is where entity types differ while the hub stays generic.
//
// Examples: gallery — progress = max page reached / page count, completed at
// ≥90%; blog post — read-time + scroll depth; video — watch %.
type Scorer interface {
	Score(ctx context.Context, s Signal) (Scored, error)
}

// ScorerFunc adapts a function to the Scorer interface.
type ScorerFunc func(ctx context.Context, s Signal) (Scored, error)

func (f ScorerFunc) Score(ctx context.Context, s Signal) (Scored, error) { return f(ctx, s) }

// State is one subject's standing with one entity — the row UI annotation is
// built from (seen?, progress bar, completed?, resume pointer).
type State struct {
	Seen          bool // MaxProgress > 0
	FirstSeenAt   time.Time
	LastSignalAt  time.Time
	TotalEvents   uint32
	MaxProgress   uint32 // progress bar = MaxProgress / ProgressMax
	ProgressMax   uint32
	Completed     bool
	Resume        string
	HasInteracted bool // any explicit feedback (Value/Label) recorded
	LastScore     int16
}

// StateRow is a State with its entity, as returned by History.
type StateRow struct {
	EntityRef
	State
}

// HistoryStatus filters History results.
type HistoryStatus string

const (
	// HistoryAny returns every entity the subject has any signal for.
	HistoryAny HistoryStatus = ""
	// HistorySeen returns entities with MaxProgress > 0.
	HistorySeen HistoryStatus = "seen"
	// HistoryInProgress returns seen-but-not-completed entities.
	HistoryInProgress HistoryStatus = "in_progress"
	// HistoryCompleted returns completed entities.
	HistoryCompleted HistoryStatus = "completed"
)

// HistoryOptions controls History reads.
type HistoryOptions struct {
	// EntityType limits results to one entity type. Empty = all types.
	EntityType string
	Status     HistoryStatus
	Limit      int // default 50
	Offset     int
}

// EntityEngagement aggregates signal quality for one entity —
// interaction-weighted, not raw view volume.
type EntityEngagement struct {
	UniqueUsers    uint64
	UniqueAnon     uint64
	Signals        uint64 // deduplicated events
	Completions    uint64 // subjects who completed
	CompletionRate float64
	// AvgScore averages over scored events (score != 0).
	AvgScore float64
	// SignalCounts counts events per host-defined signal type.
	SignalCounts map[string]uint64
}

// Window selects a time range for Popular. Zero value = all time.
type Window struct {
	From time.Time
	To   time.Time
}

// LastDays returns a window covering the last n days (UTC day-aligned,
// including today). Day alignment keeps Popular on the entity_daily rollup
// instead of scanning raw events.
func LastDays(n int) Window {
	today := time.Now().UTC().Truncate(24 * time.Hour)
	return Window{From: today.AddDate(0, 0, -n)}
}

// Between returns an arbitrary date-slice window.
func Between(from, to time.Time) Window { return Window{From: from, To: to} }

// AllTime returns the unbounded window.
func AllTime() Window { return Window{} }

func (w Window) allTime() bool { return w.From.IsZero() && w.To.IsZero() }

// dayAligned reports whether the window can be served from the daily rollup
// (both endpoints at UTC midnight, or unbounded). Sub-day windows scan the
// raw event stream instead.
func (w Window) dayAligned() bool {
	aligned := func(t time.Time) bool {
		if t.IsZero() {
			return true
		}
		u := t.UTC()
		return u.Hour() == 0 && u.Minute() == 0 && u.Second() == 0 && u.Nanosecond() == 0
	}
	return aligned(w.From) && aligned(w.To)
}

// RankWeights tunes the default popularity ranking:
//
//	volume  = log10(1 + unique subjects)            (log-scaled volume)
//	quality = (Σ score + PriorScore·PriorWeight) /
//	          (scored events + PriorWeight)          (Bayesian-smoothed avg)
//	rank    = volume × max(QualityFloor, quality)
//
// With HalfLifeDays > 0, each day-bucket's subject count is decayed by
// 2^(-age_days/half_life) before the log (time-decayed trending).
type RankWeights struct {
	// PriorWeight is the strength of the Bayesian prior in pseudo-events.
	// Defaults to 10. Prevents tiny-sample entities from outranking
	// high-volume ones.
	PriorWeight float64
	// PriorScore is the prior mean engagement score. Defaults to 0.
	PriorScore float64
	// QualityFloor keeps pure-volume ranking meaningful when hosts record no
	// scores (quality term would be ~0). Defaults to 1.
	QualityFloor float64
	// HalfLifeDays applies exponential time decay to volume. 0 = no decay.
	HalfLifeDays float64
}

func (w RankWeights) withDefaults() RankWeights {
	if w.PriorWeight <= 0 {
		w.PriorWeight = 10
	}
	if w.QualityFloor <= 0 {
		w.QualityFloor = 1
	}
	return w
}

// PopularOptions controls Popular reads.
type PopularOptions struct {
	Window Window
	Limit  int // default 20

	// Weights tunes the default ranking formula.
	Weights RankWeights

	// RankExpr, when set, REPLACES the default ranking with a host-supplied
	// ClickHouse expression (trusted SQL, mechanism vs meaning). It may
	// reference the aggregate aliases: subjects, signals, engagement_sum,
	// scored_signals, completions.
	RankExpr string
}

// PopularHit is one ranked entity from Popular.
type PopularHit struct {
	EntityRef
	Subjects    uint64
	Signals     uint64
	Completions uint64
	Score       float64
}

// CoEngagedOptions controls co-engagement queries ("subjects who engaged with
// X also engaged with Y").
type CoEngagedOptions struct {
	// EntityTypes limits result entity types. Empty = all.
	EntityTypes []string
	// Window bounds the event scan. Zero = all time.
	Window Window
	// MaxSubjects caps the engaged-subject set read from the anchor
	// (defaults to 10000).
	MaxSubjects int
	Limit       int // default 50
}

// CoEngagedHit is one co-engaged entity; Strength = co-engaged subjects.
type CoEngagedHit struct {
	EntityRef
	Strength uint64
}

// TopStatesOptions controls TopStates (recommendation seeds).
type TopStatesOptions struct {
	EntityTypes []string
	Limit       int // default 10
}
