package eval

import "testing"

var _ = Case{"name", "query", []Key{{EntityType: "gallery", EntityID: "1"}}}

func TestRecallAtK_CurrentBehavior(t *testing.T) {
	t.Parallel()

	a := Key{EntityType: "gallery", EntityID: "1"}
	b := Key{EntityType: "gallery", EntityID: "2"}
	c := Key{EntityType: "gallery", EntityID: "3"}

	tests := []struct {
		name     string
		got      []Key
		expected []Key
		k        int
		want     float64
	}{
		{name: "empty expectation is complete recall", got: nil, expected: nil, k: 10, want: 1},
		{name: "nonpositive k misses nonempty expectation", got: []Key{a}, expected: []Key{a}, k: 0, want: 0},
		{name: "k is capped to returned length", got: []Key{a}, expected: []Key{a, b}, k: 10, want: 0.5},
		{name: "only top k contributes", got: []Key{a, b, c}, expected: []Key{b, c}, k: 2, want: 0.5},
		{name: "legacy duplicate result counts twice", got: []Key{a, a}, expected: []Key{a}, k: 2, want: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := RecallAtK(tt.got, tt.expected, tt.k); got != tt.want {
				t.Fatalf("RecallAtK() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMRR_CurrentBehavior(t *testing.T) {
	t.Parallel()

	a := Key{EntityType: "gallery", EntityID: "1"}
	b := Key{EntityType: "gallery", EntityID: "2"}
	c := Key{EntityType: "gallery", EntityID: "3"}

	tests := []struct {
		name     string
		got      []Key
		expected []Key
		want     float64
	}{
		{name: "empty expectation is complete reciprocal rank", expected: nil, want: 1},
		{name: "first relevant result wins", got: []Key{a, b, c}, expected: []Key{b, c}, want: 0.5},
		{name: "miss is zero", got: []Key{a}, expected: []Key{b}, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := MRR(tt.got, tt.expected); got != tt.want {
				t.Fatalf("MRR() = %v, want %v", got, tt.want)
			}
		})
	}
}
