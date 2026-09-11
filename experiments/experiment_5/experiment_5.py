import unittest
import matplotlib.pyplot as plt
import numpy as np

from algorithms.fsom import FSOM
from loaders.split_cifar100_feature_loader import SplitCIFAR100FeatureLoader
from util.helper import find_r_metrices_combined, compute_combined_metrics

class Experiment05(unittest.TestCase):
    def test_experiment_05(self):
        loader = SplitCIFAR100FeatureLoader(feat_dir="features\\cifar_100", tasks=10, classes_per_task=10)
        T = 10  # number of tasks
        
        fisher_sigma_list = [0.0, 0.5, 1.0, 5.0, 10.0]

        # --- IEEE Style Global Settings ---
        plt.rc('font', family='serif', size=10)
        plt.rc('axes', labelsize=10)
        plt.rc('legend', fontsize=9)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)

        # Initialize figures before the loop
        fig_acc, ax_acc = plt.subplots(figsize=(7, 5))
        fig_bmu, ax_bmu = plt.subplots(figsize=(7, 5))
        fig_topo, ax_topo = plt.subplots(figsize=(7, 5))

        # X-axis representing Tasks 1 to T
        tasks_x = np.arange(1, T + 1)
        
        # Distinct markers and colors for each fisher_sigma line
        markers = ['o', 's', '^', 'D', 'v']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, fisher_sigma in enumerate(fisher_sigma_list):
            fsom = FSOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42, ewc_lambda=0.5, fisher_sigma=fisher_sigma)
            print(f"FSOM (σ={fisher_sigma}) initialized!")

            R_fsom_bmu, R_fsom_topo, R_fsom, R_fsom_comb = find_r_metrices_combined(fsom, loader, T)

            # Extract metrics across T tasks
            _, AIA_fsom_bmu, _, AIA_fsom_topo, _, _, _, _ = compute_combined_metrics(T, R_fsom_bmu, R_fsom_topo, R_fsom, R_fsom_comb)

            # --- Plotting inside loop: Add one line per fisher_sigma ---
            label_str = f'$\sigma$ = {fisher_sigma}'
            
            ax_bmu.plot(tasks_x, AIA_fsom_bmu, marker=markers[idx], color=colors[idx], label=label_str, linewidth=1.5, markersize=6)
            ax_topo.plot(tasks_x, AIA_fsom_topo, marker=markers[idx], color=colors[idx], label=label_str, linewidth=1.5, markersize=6)


        # --- Finalizing and Saving Plots ---
        save_dir = 'results\\experiment_5'
        title_base = f'(m=20, n=20, dim=512, lr=0.2, seed=42, $\lambda$=0.5)'

        # BMU Plot
        ax_bmu.set_title(f'Fisher $\sigma$ Variation on Split CIFAR-100\n{title_base}')
        ax_bmu.set_xlabel('Task Number')
        ax_bmu.set_ylabel('AI. BMU Weight Shift Error')
        ax_bmu.set_xticks(tasks_x)
        ax_bmu.grid(True, linestyle='--', alpha=0.7)
        ax_bmu.legend(loc='best', frameon=True, edgecolor='black')
        fig_bmu.tight_layout()
        fig_bmu.savefig(f'{save_dir}\\experiment_5_split_cifar100_bwse.png', dpi=300)
        plt.close(fig_bmu)

        # Topo Plot
        ax_topo.set_title(f'Fisher $\sigma$ Variation on Split CIFAR-100\n{title_base}')
        ax_topo.set_xlabel('Task Number')
        ax_topo.set_ylabel('AI. Topography Preservation Error')
        ax_topo.set_xticks(tasks_x)
        ax_topo.grid(True, linestyle='--', alpha=0.7)
        ax_topo.legend(loc='best', frameon=True, edgecolor='black')
        fig_topo.tight_layout()
        fig_topo.savefig(f'{save_dir}\\experiment_5_split_cifar100_tpe.png', dpi=300)
        plt.close(fig_topo)

if __name__ == '__main__':
    unittest.main()