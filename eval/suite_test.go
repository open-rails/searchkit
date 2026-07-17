package eval

import (
	"strings"
	"testing"
)

func TestParseSuite(t *testing.T) {
	t.Parallel()

	valid := `{
  "id": "gallery-v1",
  "cases": [{
    "id": "title",
    "query": "known title",
    "language": "en",
    "entity_types": ["gallery"],
    "k": 5,
    "judgments": [{"key":{"entity_type":"gallery","entity_id":"42"},"relevance":3}],
    "labels": {"suite":"manual"}
  }]
}`

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{name: "valid", input: valid},
		{name: "unknown field", input: strings.Replace(valid, `"query":`, `"unknown":true,"query":`, 1), wantErr: true},
		{name: "missing suite id", input: strings.Replace(valid, `"gallery-v1"`, `""`, 1), wantErr: true},
		{name: "no cases", input: `{"id":"empty","cases":[]}`, wantErr: true},
		{name: "missing case id", input: strings.Replace(valid, `"title"`, `""`, 1), wantErr: true},
		{name: "trailing value", input: valid + ` {}`, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			suite, err := ParseSuite(strings.NewReader(tt.input))
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseSuite() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && (suite.ID != "gallery-v1" || len(suite.Cases) != 1) {
				t.Fatalf("unexpected suite: %#v", suite)
			}
		})
	}
}

func TestParseSuite_DuplicateCaseID(t *testing.T) {
	t.Parallel()

	input := `{
  "id":"duplicates",
  "cases":[
    {"id":"same","query":"one","k":1},
    {"id":"same","query":"two","k":1}
  ]
}`
	if _, err := ParseSuite(strings.NewReader(input)); err == nil {
		t.Fatal("ParseSuite() error = nil, want duplicate case id error")
	}
}

func TestParseSuite_RejectsWhitespaceEquivalentCaseIDs(t *testing.T) {
	t.Parallel()

	input := `{
  "id":"duplicates",
  "cases":[
    {"id":"same","query":"one","k":1},
    {"id":" same ","query":"two","k":1}
  ]
}`
	if _, err := ParseSuite(strings.NewReader(input)); err == nil {
		t.Fatal("ParseSuite() error = nil, want canonical duplicate case id error")
	}
}
