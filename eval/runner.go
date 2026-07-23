package eval

import (
	"context"
	"fmt"
)

// CaseRunner executes one golden case against a retrieval backend and returns
// its ordered results. The eval package stays dependency-free: callers wire
// their own client (which owns query text, mode, and filtering) behind this
// interface and map their hit type to Result.
//
// A non-nil error reports an execution failure for that single case; the
// returned errCategory is a stable, sanitized identifier (e.g. "timeout",
// "search") recorded on the failed outcome. Returning an error does not abort
// the suite — the case is recorded as failed and the run continues.
type CaseRunner interface {
	Run(ctx context.Context, c GoldenCase) (results []Result, errCategory string, err error)
}

// RunSuite executes every case in the suite through the runner, evaluates each
// against its expectation, and builds one deterministic report. Per-case
// execution failures are captured as failed outcomes rather than aborting the
// whole run; a nil runner, an invalid suite, or a report-build error aborts.
func RunSuite(ctx context.Context, s Suite, runner CaseRunner, identity ReportIdentity, groupLabels ...string) (Report, error) {
	if ctx == nil {
		return Report{}, fmt.Errorf("context is required")
	}
	if runner == nil {
		return Report{}, fmt.Errorf("runner is required")
	}
	if err := validateIdentity(identity); err != nil {
		return Report{}, err
	}
	if len(s.Cases) == 0 {
		return Report{}, fmt.Errorf("suite %q has no cases", s.ID)
	}

	outcomes := make([]Outcome, 0, len(s.Cases))
	for _, c := range s.Cases {
		if err := ctx.Err(); err != nil {
			return Report{}, fmt.Errorf("run suite: %w", err)
		}
		if err := ValidateCase(c); err != nil {
			return Report{}, err
		}
		results, category, runErr := runner.Run(ctx, c)
		if runErr != nil {
			outcomes = append(outcomes, Failed(c, category))
			continue
		}
		outcome, err := Evaluate(c, results)
		if err != nil {
			return Report{}, fmt.Errorf("evaluate case %q: %w", caseID(c), err)
		}
		outcomes = append(outcomes, outcome)
	}

	report, err := BuildReport(identity, outcomes, groupLabels...)
	if err != nil {
		return Report{}, fmt.Errorf("build report: %w", err)
	}
	return report, nil
}
