import unittest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from algorithms.som import SOM
from algorithms.ewc_som import EWC_SOM
from loaders.core50_nc_feature_loader import Core50NCFeatureLoader
from util.helper import find_r_metrices_combined, compute_combined_metrics

class Experiment07_C(unittest.TestCase):
    def test_experiment_07_c(self):
        loader = Core50NCFeatureLoader(feat_dir="features\\core50")
        T = 10  # number of tasks

        ewc_som = EWC_SOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42, ewc_lambda=100, fisher_sigma=2.5)
        print("EWCSOM initialized!")
        som = SOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42)
        print("SOM initialized!")

        R_ewc_som_bmu, R_ewc_som_topo, R_ewc_som, R_ewc_som_comb = find_r_metrices_combined(ewc_som, loader, T)
        R_bmu, R_topo, R, R_comb = find_r_metrices_combined(som, loader, T)

        # --- Plotting ---
        AIA_ewc_som_bmu, AIA_ewc_som_topo, AIA_ewc_som, AIA_ewc_som_comb = compute_combined_metrics(T, R_ewc_som_bmu, R_ewc_som_topo, R_ewc_som, R_ewc_som_comb)
        AIA_bmu, AIA_topo, AIA, AIA_comb = compute_combined_metrics(T, R_bmu, R_topo, R, R_comb)

        tasks = np.arange(1, T + 1)

        # --- Config ---
        mpl.rcParams.update({
            'font.family':        'serif',
            'font.serif':         ['Times New Roman', 'DejaVu Serif'],
            'font.size':          10,
            'axes.titlesize':     10,
            'axes.labelsize':     9,
            'xtick.labelsize':    8,
            'ytick.labelsize':    8,
            'legend.fontsize':    8,
            'figure.titlesize':   11,
            'axes.linewidth':     0.8,
            'grid.linewidth':     0.5,
            'lines.linewidth':    1.5,
            'lines.markersize':   5,
            'axes.spines.top':    False,
            'axes.spines.right':  False,
            'grid.alpha':         0.4,
            'grid.color':         '#cccccc',
        })

        # --- Colors (colorblind-safe) ---
        BLUE  = '#0072B2'
        RED   = '#D55E00'

        # --- Plot Titles & Labels ---
        panels = [
            {
                'ax_idx':  (0, 0),
                'som':   AIA_bmu,
                'ewc':     AIA_ewc_som_bmu,
                'title':   'BMU Weight Shift Error',
                'ylabel':  'Avg. Incremental BWSE',
            },
            {
                'ax_idx':  (0, 1),
                'som':   AIA_topo,
                'ewc':     AIA_ewc_som_topo,
                'title':   'Topographic Preservation Error',
                'ylabel':  'Avg. Incremental TPE',
            },
            {
                'ax_idx':  (1, 0),
                'som':   AIA,
                'ewc':     AIA_ewc_som,
                'title':   'Average Incremental Accuracy (AIA)',
                'ylabel':  'AIA (%)',
            },
            {
                'ax_idx':  (1, 1),
                'som':   AIA_comb,
                'ewc':     AIA_ewc_som_comb,
                'title':   'Combined Metric',
                'ylabel':  'Avg. Incremental Combined Metric',
            },
        ]

        # --- Figure ---
        fig, axes = plt.subplots(
            2, 2,
            figsize=(7.16, 5.0),  
            dpi=300,
        )

        for p in panels:
            ax = axes[p['ax_idx']]

            ax.plot(tasks, p['som'],
                    color=BLUE, marker='o', markersize=5,
                    linestyle='-',  linewidth=1.5, label='SOM')

            ax.plot(tasks, p['ewc'],
                    color=RED,  marker='s', markersize=5,
                    linestyle='--', linewidth=1.5, label=r'EWCSOM ($\lambda=100$)')

            ax.set_title(p['title'], pad=6, fontweight='bold')
            ax.set_xlabel('Task', labelpad=4)
            ax.set_ylabel(p['ylabel'], labelpad=4)
            ax.set_xticks(tasks)
            ax.grid(True, axis='y')
            ax.legend(frameon=False)

        # --- Shared caption-style suptitle ---
        fig.suptitle(
            r'EWCSOM vs. SOM on CORe50'
            '\n'
            r'($m{=}20,\ n{=}20,\ d{=}512,\ \eta{=}0.2,\ \mathrm{seed}{=}42$)',
            fontsize=10,
            y=1.02,
        )

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(
            r'results\\experiment_7\\experiment_7_core50.pdf',
            dpi=600,
            bbox_inches='tight',
            format='pdf',
        )
        plt.savefig(
            r'results\\experiment_7\\experiment_7_core50.png',
            dpi=600,
            bbox_inches='tight',
        )

if __name__ == '__main__':
    unittest.main()