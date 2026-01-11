package river

import (
	"context"
	"time"

	"github.com/riverqueue/river"

	"github.com/doujins-org/embeddingkit/runtime"
	"github.com/doujins-org/embeddingkit/tasks"
)

type TaskBatchWorker struct {
	river.WorkerDefaults[EmbeddingTaskBatchArgs]

	Runtime   *runtime.Runtime
	TaskRepo  *tasks.Repo
	LockAhead time.Duration
	Backoff   time.Duration
}

func (w *TaskBatchWorker) Work(ctx context.Context, job *river.Job[EmbeddingTaskBatchArgs]) error {
	if w.Runtime == nil || w.TaskRepo == nil {
		return nil
	}

	limit := job.Args.Limit
	if limit <= 0 {
		limit = 250
	}

	lockAhead := w.LockAhead
	if lockAhead <= 0 {
		lockAhead = 30 * time.Second
	}

	backoff := w.Backoff
	if backoff <= 0 {
		backoff = 30 * time.Second
	}

	tasksToRun, err := w.TaskRepo.FetchReady(ctx, limit, lockAhead)
	if err != nil {
		return err
	}

	for _, t := range tasksToRun {
		if err := w.Runtime.GenerateAndStoreEmbedding(ctx, t.EntityType, t.EntityID, t.Model); err != nil {
			_ = w.TaskRepo.Fail(ctx, t.ID, backoff)
			continue
		}
		_ = w.TaskRepo.Complete(ctx, t.ID)
	}

	return nil
}
