package tasks

import "time"

type Task struct {
	EntityType string
	EntityID   string
	Model      string
	Reason     string
	Attempts   int
	NextRunAt  time.Time
	StartedAt  *time.Time
	CreatedAt  time.Time
	UpdatedAt  time.Time
}
