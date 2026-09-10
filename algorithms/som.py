import numpy as np
import sys

class SOM:
    def __init__(self, m, n, dim, learning_rate=0.5, seed=None):
        """
        Initialize a rectangular SOM.
        Args:
            m: number of rows
            n: number of columns
            dim: dimensionality of input vectors
            learning_rate: initial learning rate
            sigma: initial neighborhood radius (defaults to max(m,n)/2)
            seed: random seed for reproducibility
        Returns:
            none
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.lr = learning_rate
        self.sigma = max(m, n) / 2.0
        self._rng = np.random.default_rng(seed)
        # Weight vectors: shape (m, n, dim)
        self.weights = self._rng.random((m, n, dim))
        # Location of each neuron in grid for distance computations
        self.neurons = np.array([[np.array([i, j]) for j in range(n)] for i in range(m)])

    def find_bmu(self, x):
        """
        Find Best Matching Unit (BMU) for input x.
        Args:
            x: input vector of shape (dim,)
        Returns:
            tuple (bmu_location (i,j), bmu_distance)
        """
        # Compute Euclidean distance to all weights
        diff = x - self.weights
        dist = np.linalg.norm(diff, axis=2)
        idx = np.unravel_index(np.argmin(dist), (self.m, self.n))
        return idx, dist[idx]

    def neighborhood(self, bmu_idx, sigma):
        """
        Compute Gaussian neighborhood function around BMU.
        Args:
            bmu_idx: tuple (i,j)
            sigma: neighborhood radius
        Returns:
            2D array of shape (m, n)
        """
        sq_dist = np.sum((self.neurons - bmu_idx) ** 2, axis=2)
        return np.exp(-np.clip(sq_dist / (2 * sigma ** 2), 0, 100))
    
    def step(self, x, global_iter_counter, total_iterations):
        """
        Perform a single training step.
        Args:
            x: input vector
            global_iter_counter: global iteration counter
            T: total number of iterations
        """
        lr_t = self.lr * (1 - global_iter_counter / total_iterations)
        sigma_t = self.sigma * (1 - global_iter_counter / total_iterations)
        (i, j), _ = self.find_bmu(x)
        h = self.neighborhood((i, j), sigma_t)
        delta = lr_t * h[..., np.newaxis] * (x - self.weights)
        self.weights += delta

    def fit(self, data, epochs=10, shuffle=True, random_state=42):
        """
        Train the SOM over multiple epochs with terminal progress display.
        Args:
            data: array of shape (N, dim)
            epochs: number of passes over the dataset
            shuffle: whether to shuffle the data each epoch
            random_state: seed for reproducibility
        """
        n_samples = data.shape[0]
        self.total_iterations = epochs * n_samples
        self.global_iter_counter = 0

        bar_width = 40

        for epoch in range(epochs):
            if shuffle:
                rng = np.random.default_rng(random_state)
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            print(f"\nEpoch {epoch+1}/{epochs}")
            for i, idx in enumerate(indices):
                x = data[idx]
                self.step(x, self.global_iter_counter, self.total_iterations)
                self.global_iter_counter += 1

                # Display progress bar
                progress = (i + 1) / n_samples
                filled = int(bar_width * progress)
                bar = '#' * filled + '-' * (bar_width - filled)
                percent = int(progress * 100)
                sys.stdout.write(f'\r[{bar}] {percent}%')
                sys.stdout.flush()
            print()


    def map_vects(self, data):
        """
        Map each input to its BMU coordinate.
        Args:
            data: array of shape (N, dim)
        Returns:
            array of BMU coordinates shape (N, 2)
        """
        mapped = []
        for x in data:
            idx, _ = self.find_bmu(x)
            mapped.append(idx)
        return np.array(mapped)