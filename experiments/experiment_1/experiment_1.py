import unittest
from algorithms.som import SOM
from loaders.split_cifar100_feature_loader import SplitCIFAR100FeatureLoader
from loaders.cub_200_2011_loader import CUB200Loader
from loaders.core50_nc_feature_loader import Core50NCFeatureLoader
from util.force_forget import run_experiment1
from evaluation.accuracy import evaluate_accuracy

class FinalExperiment01(unittest.TestCase):
    def test_final_experiment_01(self):
        loader_cifar = SplitCIFAR100FeatureLoader(feat_dir="features\\cifar_100", tasks=10, classes_per_task=10)
        loader_cub = CUB200Loader(feat_dir="features\\cub_200_2011", tasks=20, classes_per_task=10)
        loader_core50 = Core50NCFeatureLoader(feat_dir="features\\core50")

        som_cifar_100 = SOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42)
        som_cub = SOM(m=20, n=20, dim=2048, learning_rate=0.2, seed=42)
        som_core50 = SOM(m=20, n=20, dim=512, learning_rate=0.2, seed=42)

        table_cifar = run_experiment1(
                som                  = som_cifar_100,
                loader               = loader_cifar,
                evaluate_accuracy_fn = evaluate_accuracy,
                dataset_name         = "Split CIFAR-100",
                epsilons             = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0],
                n_repeats            = 5,
                random_seed          = 42,
                save_path            = "results\\experiment_1\\experiment_1_split_cifar100.png",
            )
        
        table_cub = run_experiment1(
                som                  = som_cub,
                loader               = loader_cub,
                evaluate_accuracy_fn = evaluate_accuracy,
                dataset_name         = "CUB-200-2011",
                epsilons             = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0],
                n_repeats            = 5,
                random_seed          = 42,
                save_path            = "results\\experiment_1\\experiment_1_cub_200_2011.png",
            )
        
        table_core50 = run_experiment1(
                som                  = som_core50,
                loader               = loader_core50,
                evaluate_accuracy_fn = evaluate_accuracy,
                dataset_name         = "CORe50",
                epsilons             = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0],
                n_repeats            = 5,
                random_seed          = 42,
                save_path            = "results\\experiment_1\\experiment_1_core50.png",
            )

if __name__ == '__main__':
    unittest.main()