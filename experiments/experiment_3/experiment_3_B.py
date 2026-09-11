import unittest
import matplotlib.pyplot as plt
import numpy as np

from algorithms.fsom import FSOM
from algorithms.non_ewc_fsom import NON_EWC_FSOM
from loaders.cifar100_superclass_loader import CIFAR100SuperclassPythonLoader
from util.helper import find_stability_plasticity

class Experiment03_B(unittest.TestCase):
    def test_experiment_03_b(self):
        loader = CIFAR100SuperclassPythonLoader(data_path='datasets\\cifar100', batch_size=128, device='cpu')
        T = 20  # number of tasks

        fsom = FSOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42, ewc_lambda=0.5, fisher_sigma=5.0)
        print("FSOM initialized!")
        non_ewc_fsom = NON_EWC_FSOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42)
        print("NON_EWC_FSOM initialized!")

        acc_fsom_S, acc_fsom_P, _, _, _, _ = find_stability_plasticity(fsom, loader, T)
        acc_non_ewc_fsom_S, acc_non_ewc_fsom_P, _, _, _, _ = find_stability_plasticity(non_ewc_fsom, loader, T)

        # Filter data by ignoring task 1
        acc_fsom_S, acc_fsom_P = acc_fsom_S[1:], acc_fsom_P
        acc_non_ewc_fsom_S, acc_non_ewc_fsom_P = acc_non_ewc_fsom_S[1:], acc_non_ewc_fsom_P

        tasks = np.arange(1, T + 1)

        # Common settings
        save_dir = 'results\\experiment_3'
        title_base = 'FSOM vs NON_EWC_FSOM on CIFAR-100 Superclass\n(m=20, n=20, dim=512, lr=0.2, seed=42)'

        plt.figure(figsize=(10, 6))
        plt.plot(tasks[1:], acc_fsom_S, marker='o', color='blue', label='EWC (λ=0.5)')
        plt.plot(tasks[1:], acc_non_ewc_fsom_S, marker='x', color='red', label='NON-EWC')
        plt.title(f'Stability (Accuracy) Comparison\n{title_base}')
        plt.xlabel('Task')
        plt.ylabel('Stability (Accuracy)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}\\experiment_3_stability_cifar100_superclass.png')
        plt.close() 
        
        plt.figure(figsize=(10, 6))
        plt.plot(tasks, acc_fsom_P, marker='o', color='blue', label='EWC (λ=0.5)')
        plt.plot(tasks, acc_non_ewc_fsom_P, marker='x', color='red', label='NON-EWC')
        plt.title(f'Plasticity (Accuracy) Comparison\n{title_base}')
        plt.xlabel('Task')
        plt.ylabel('Plasticity (Accuracy)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}\\experiment_3_plasticity_cifar100_superclass.png')
        plt.close()

if __name__ == '__main__':
    unittest.main()