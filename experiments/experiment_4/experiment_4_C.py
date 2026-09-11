import unittest
import matplotlib.pyplot as plt
import numpy as np

from algorithms.fsom import FSOM
from loaders.cub_200_2011_loader import CUB200Loader
from util.helper import find_r_metrices_combined, compute_combined_metrics

class Experiment04_C(unittest.TestCase):
    def test_experiment_04_c(self):
        loader = CUB200Loader(feat_dir="features\\cub_200_2011", tasks=20, classes_per_task=10)
        T = 20  # number of tasks
        
        acc_list = []
        
        ewc_lambda_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for ewc_lambda in ewc_lambda_list:
            fsom = FSOM(m=20, n=20, dim=2048, learning_rate=0.2, seed=42, ewc_lambda=ewc_lambda, fisher_sigma=5.0)
            print(f"FSOM (lambda={ewc_lambda}) initialized!")

            R_bmu, R_topo, R, R_sci = find_r_metrices_combined(fsom, loader, T)

            _, _, AIA, _ = compute_combined_metrics(T, R_bmu, R_topo, R, R_sci)

            acc_list.append(np.mean(AIA))

            print(acc_list)

        # Common settings
        save_dir = 'results\\experiment_4'
        title_base = r'(m=20, n=20, dim=2048, learning_rate=0.2, seed=42, $\sigma_{fisher}=5$)'

        plt.figure(figsize=(10, 6))
        plt.plot(ewc_lambda_list, acc_list, marker='o', color='blue')
        plt.title(f'FSOM ewc_lambda Variation on CUB-200-2011\n{title_base}')
        plt.xlabel('ewc_lambda')
        plt.ylabel('Average Average Incremental Accuracy (AAIA)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}\\experiment_4_cub_200_2011.png')
        plt.close() 

if __name__ == '__main__':
    unittest.main()