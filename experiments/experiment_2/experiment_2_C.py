import unittest
import matplotlib.pyplot as plt
import numpy as np

from algorithms.fsom import FSOM
from algorithms.non_ewc_fsom import NON_EWC_FSOM
from loaders.cub_200_2011_loader import CUB200Loader
from util.helper import find_r_metrices_combined, compute_combined_metrics

class Experiment02_C(unittest.TestCase):
    def test_experiment_02_c(self):
        loader = CUB200Loader(feat_dir="features\\cub_200_2011", tasks=20, classes_per_task=10)
        T = 20  # number of tasks

        fsom = FSOM(m=20, n=20, dim=2048, learning_rate=0.2, seed=42, ewc_lambda=1.0, fisher_sigma=5.0)
        print("FSOM initialized!")
        non_ewc_fsom = NON_EWC_FSOM(m=20, n=20, dim=2048, learning_rate=0.2, seed=42)
        print("NON_EWC_FSOM initialized!")

        R_fsom_bmu, R_fsom_topo, R_fsom, R_fsom_comb = find_r_metrices_combined(fsom, loader, T)
        R_non_ewc_fsom_bmu, R_non_ewc_fsom_topo, R_non_ewc_fsom, R_non_ewc_fsom_comb = find_r_metrices_combined(non_ewc_fsom, loader, T)

        # --- Plotting ---
        AIA_fsom_bmu, AIA_fsom_topo, AIA_fsom, AIA_fsom_comb = compute_combined_metrics(T, R_fsom_bmu, R_fsom_topo, R_fsom, R_fsom_comb)
        AIA_non_ewc_fsom_bmu, AIA_non_ewc_fsom_topo, AIA_non_ewc_fsom, AIA_non_ewc_fsom_comb = compute_combined_metrics(T, R_non_ewc_fsom_bmu, R_non_ewc_fsom_topo, R_non_ewc_fsom, R_non_ewc_fsom_comb)

        tasks = np.arange(1, T + 1)

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Performance Metrics FSOM vs NON_EWC_FSOM on CUB-200-2011\nm=20, n=20, dim=2048, learning_rate=0.2, seed=42', fontsize=12)

        # Average Incremental BMU Weight Shift Error
        axes[0, 0].plot(tasks, AIA_non_ewc_fsom_bmu, marker='o', color='blue', label='NON-EWC')
        axes[0, 0].plot(tasks, AIA_fsom_bmu, marker='x', color='red', label='EWC (λ=0.5)')
        axes[0, 0].set_title('Average Incremental BMU Weight Shift Error')
        axes[0, 0].set_xlabel('Task')
        axes[0, 0].set_ylabel('BMU Weight Shift Error')
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # Average Incremental Topographic Preservation Error
        axes[0, 1].plot(tasks, AIA_non_ewc_fsom_topo, marker='o', color='blue', label='NON-EWC')
        axes[0, 1].plot(tasks, AIA_fsom_topo, marker='x', color='red', label='EWC (λ=0.5)')
        axes[0, 1].set_title('Average Incremental Topography Preservation Error')
        axes[0, 1].set_xlabel('Task')
        axes[0, 1].set_ylabel('Topography Preservation Error')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # Average Incremental Accuracy
        axes[1, 0].plot(tasks, AIA_non_ewc_fsom, marker='o', color='blue', label='NON-EWC')
        axes[1, 0].plot(tasks, AIA_fsom, marker='x', color='red', label='EWC (λ=0.5)')
        axes[1, 0].set_title('Average Incremental Accuracy (AIA)')
        axes[1, 0].set_xlabel('Task')
        axes[1, 0].set_ylabel('AIA (%)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # Average Incremental Combined Metric
        axes[1, 1].plot(tasks, AIA_non_ewc_fsom_comb, marker='o', color='blue', label='NON-EWC')
        axes[1, 1].plot(tasks, AIA_fsom_comb, marker='x', color='red', label='EWC (λ=0.5)')
        axes[1, 1].set_title('Average Incremental Combined Metric')
        axes[1, 1].set_xlabel('Task')
        axes[1, 1].set_ylabel('Combined Metric')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('results\\experiment_2\\experiment_2_cub_200_2011.png')

if __name__ == '__main__':
    unittest.main()