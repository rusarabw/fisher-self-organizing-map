import os
import numpy as np

class Core50NCFeatureLoader:
    """
    CORe50 New Classes (NC) protocol loader.
    Splits 50 classes into tasks (default: 10 tasks * 5 classes each).
    """
    def __init__(self, feat_dir, tasks=10, classes_per_task=5):
        self.T = tasks
        self.C = classes_per_task

        train = np.load(os.path.join(feat_dir, "core50_train_feats.npz"))
        test  = np.load(os.path.join(feat_dir, "core50_test_feats.npz"))

        X_tr, y_tr = train["X"], train["y"]
        X_te, y_te = test["X"], test["y"]

        self.train_task_data = {}
        self.test_task_data  = {}

        for t in range(self.T):
            lo, hi = t * self.C, (t + 1) * self.C
            mask_tr = (y_tr >= lo) & (y_tr < hi)
            mask_te = (y_te >= lo) & (y_te < hi)

            self.train_task_data[t] = (X_tr[mask_tr], y_tr[mask_tr])
            self.test_task_data[t]  = (X_te[mask_te], y_te[mask_te])

    def stream(self):
        """Yields (task_id, X_train)."""
        for t in range(self.T):
            X_tr, _ = self.train_task_data[t]
            yield t, X_tr