import os
import numpy as np

class SplitCIFAR100FeatureLoader:
    """
    Loader for pre-extracted CIFAR-100 features saved as .npz.
    Splits into 10 tasks of 10 classes each.
    """
    def __init__(self, feat_dir, tasks=10, classes_per_task=10):
        self.T = tasks
        self.C = classes_per_task

        train = np.load(os.path.join(feat_dir, 'cifar100_train_feats.npz'))
        test  = np.load(os.path.join(feat_dir, 'cifar100_test_feats.npz'))

        X_tr, y_tr = train['X'], train['y']
        X_te, y_te = test ['X'], test ['y']

        # Build per-task dictionaries
        self.train_task_data = {}
        self.test_task_data  = {}
        for t in range(self.T):
            lo = t * self.C
            hi = lo + self.C
            mask_tr = (y_tr >= lo) & (y_tr < hi)
            mask_te = (y_te >= lo) & (y_te < hi)

            self.train_task_data[t] = (X_tr[mask_tr], y_tr[mask_tr])
            self.test_task_data [t] = (X_te[mask_te], y_te[mask_te])

    def stream(self):
        """
        Yields (task_id, X_train, y_train, X_test, y_test) for t=0..T-1
        """
        for t in range(self.T):
            X_tr, _ = self.train_task_data[t]
            yield t, X_tr