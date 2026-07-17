package eval

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Suite is a portable golden-query fixture. ID is a human-managed version;
// reports independently hash the validated contents.
type Suite struct {
	ID    string       `json:"id"`
	Cases []GoldenCase `json:"cases"`
}

// ParseSuite decodes and validates one golden-query suite.
func ParseSuite(r io.Reader) (Suite, error) {
	if r == nil {
		return Suite{}, fmt.Errorf("suite reader is required")
	}
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()

	var suite Suite
	if err := decoder.Decode(&suite); err != nil {
		return Suite{}, fmt.Errorf("decode suite: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return Suite{}, fmt.Errorf("decode suite: multiple JSON values")
		}
		return Suite{}, fmt.Errorf("decode suite trailing data: %w", err)
	}
	suite.ID = strings.TrimSpace(suite.ID)
	if suite.ID == "" {
		return Suite{}, fmt.Errorf("suite id is required")
	}
	if len(suite.Cases) == 0 {
		return Suite{}, fmt.Errorf("suite %q has no cases", suite.ID)
	}
	seen := make(map[string]struct{}, len(suite.Cases))
	for i := range suite.Cases {
		suite.Cases[i].ID = strings.TrimSpace(suite.Cases[i].ID)
		if suite.Cases[i].ID == "" {
			return Suite{}, fmt.Errorf("suite %q case %d: id is required", suite.ID, i)
		}
		id := caseID(suite.Cases[i])
		if _, ok := seen[id]; ok {
			return Suite{}, fmt.Errorf("suite %q: duplicate case id %q", suite.ID, suite.Cases[i].ID)
		}
		seen[id] = struct{}{}
		if err := ValidateCase(suite.Cases[i]); err != nil {
			return Suite{}, err
		}
		suite.Cases[i] = normalizeCase(suite.Cases[i])
	}
	return suite, nil
}
