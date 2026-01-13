package pg

// QuoteSchema validates and safely quotes a schema identifier for embedding in SQL.
func QuoteSchema(schema string) (string, error) {
	return quoteIdent(schema)
}
