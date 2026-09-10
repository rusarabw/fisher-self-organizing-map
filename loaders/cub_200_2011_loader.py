import os
import numpy as np

class CUB200Loader:
    """
    Loader for pre-extracted CUB-200-2011 features.
    Expects two files in `feat_dir`:
      - cub200_train_feats.npz (X, y, ids)
      - cub200_test_feats.npz  (X, y, ids)
    Splits into tasks by grouping classes.
    """
    def __init__(self, feat_dir, tasks=20, classes_per_task=10):
        """
        :param feat_dir: directory containing .npz feature files
        :param tasks: number of tasks (default 20 → 10 classes each)
        :param classes_per_task: classes per task
        """
        self.feat_dir = feat_dir
        self.T = tasks
        self.C = classes_per_task

        train = np.load(os.path.join(feat_dir, 'cub200_train_feats.npz'))
        test  = np.load(os.path.join(feat_dir, 'cub200_test_feats.npz'))

        X_tr, y_tr, ids_tr = train['X'], train['y'], train['ids']
        X_te, y_te, ids_te = test ['X'], test ['y'], test ['ids']

        # Group by task
        self.train_task_data = {}
        self.test_task_data  = {}
        for t in range(self.T):
            cls_start = t * self.C
            cls_end   = cls_start + self.C

            mask_tr = (y_tr >= cls_start) & (y_tr < cls_end)
            mask_te = (y_te >= cls_start) & (y_te < cls_end)

            self.train_task_data[t] = (X_tr[mask_tr], y_tr[mask_tr])
            self.test_task_data[t] = (X_te[mask_te], y_te[mask_te])

    def stream(self):
        """
        Yields (task_id, X_train, y_train, X_test, y_test) for each task.
        """
        for t in range(self.T):
            X_tr, _ = self.train_task_data[t]
            yield t, X_tr