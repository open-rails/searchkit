package signal

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
)

// fakeConn captures Exec/Query calls and serves canned rows, keyed by a
// substring of the query.
type fakeConn struct {
	execs   []capturedCall
	queries []capturedCall
	// rowsFor maps a query substring -> rows to return.
	rowsFor map[string][][]any
	execErr error
}

type capturedCall struct {
	query string
	args  []any
}

func (f *fakeConn) Exec(_ context.Context, query string, args ...any) error {
	f.execs = append(f.execs, capturedCall{query: query, args: args})
	return f.execErr
}

func (f *fakeConn) Query(_ context.Context, query string, args ...any) (driver.Rows, error) {
	f.queries = append(f.queries, capturedCall{query: query, args: args})
	for sub, rows := range f.rowsFor {
		if strings.Contains(query, sub) {
			return &fakeRows{rows: rows}, nil
		}
	}
	return &fakeRows{}, nil
}

type fakeRows struct {
	rows [][]any
	idx  int
	cur  []any
}

func (r *fakeRows) Next() bool {
	if r.idx >= len(r.rows) {
		return false
	}
	r.cur = r.rows[r.idx]
	r.idx++
	return true
}

func (r *fakeRows) Scan(dest ...any) error {
	if len(dest) != len(r.cur) {
		return fmt.Errorf("fakeRows: scan %d dests, row has %d values", len(dest), len(r.cur))
	}
	for i, d := range dest {
		dv := reflect.ValueOf(d)
		if dv.Kind() != reflect.Ptr {
			return fmt.Errorf("fakeRows: dest %d is not a pointer", i)
		}
		sv := reflect.ValueOf(r.cur[i])
		if !sv.Type().AssignableTo(dv.Elem().Type()) {
			return fmt.Errorf("fakeRows: dest %d: cannot assign %s to %s", i, sv.Type(), dv.Elem().Type())
		}
		dv.Elem().Set(sv)
	}
	return nil
}

func (r *fakeRows) HasData() bool                    { return len(r.rows) > 0 }
func (r *fakeRows) ScanStruct(any) error             { return fmt.Errorf("not implemented") }
func (r *fakeRows) ColumnTypes() []driver.ColumnType { return nil }
func (r *fakeRows) Totals(...any) error              { return fmt.Errorf("not implemented") }
func (r *fakeRows) Columns() []string                { return nil }
func (r *fakeRows) Close() error                     { return nil }
func (r *fakeRows) Err() error                       { return nil }
