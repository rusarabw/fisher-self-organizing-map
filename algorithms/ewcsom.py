import numpy as np
import sys
import copy

class EWCSOM:
    def __init__(self, m, n, dim, learning_rate=0.5, seed=None, ewc_lambda=0.5, fisher_sigma=5.0):
        self.m = m
        self.n = n
        self.dim = dim
        self.lr = learning_rate
        self.sigma = max(m, n) / 2.0
        self.fisher_sigma = fisher_sigma
        
        self.ewc_lambda = ewc_lambda
        self._rng = np.random.default_rng(seed)
        
        # Initialize weights
        self.weights = self._rng.random((m, n, dim))
        
        # Grid coordinates
        self.neurons = np.array([[np.array([i, j]) for j in range(n)] for i in range(m)])
        
        # History
        self.previous_tasks = []

    def find_bmu(self, x):
        diff = x - self.weights
        # Robustness: Check for NaNs/Infs in case of previous overflow
        if not np.all(np.isfinite(diff)):
            # Fallback to 0 if weights are corrupted, preventing crash
            return (0, 0), 0.0
            
        dist = np.linalg.norm(diff, axis=2)
        idx = np.unravel_index(np.argmin(dist), (self.m, self.n))
        return idx, dist[idx]

    def neighborhood(self, bmu_idx, sigma):
        sq_dist = np.sum((self.neurons - bmu_idx) ** 2, axis=2)
        # Numerical stability: limit the exponent to prevent underflow warnings
        return np.exp(-np.clip(sq_dist / (2 * sigma ** 2), 0, 100))

    def compute_fisher(self, data):
        """
        Compute Fisher Information diagonal.
        """
        fisher_accum = np.zeros_like(self.weights)
        n_samples = data.shape[0]
        if n_samples == 0: return fisher_accum

        for x in data:
            (i, j), _ = self.find_bmu(x)
            h = self.neighborhood((i, j), self.fisher_sigma)
            
            grad = h[..., np.newaxis] * (x - self.weights)
            fisher_accum += grad ** 2

        raw_fisher = fisher_accum / n_samples
        return raw_fisher

    def consolidate(self, data):
        """
        Consolidate with internal normalization to handle varying data scales.
        """
        print(f"Consolidating knowledge for Task {len(self.previous_tasks) + 1}...")
        
        raw_fisher = self.compute_fisher(data)

        # Scale Fisher values to [0, 1]
        # from multiplying with lambda and causing overflows.
        max_f = np.max(raw_fisher)
        if max_f > 1e-9:
            normalized_fisher = raw_fisher / max_f
        else:
            normalized_fisher = raw_fisher # Avoid div by zero
            
        star_weights = copy.deepcopy(self.weights)
        
        # Store the normalized fisher
        self.previous_tasks.append((star_weights, normalized_fisher))

    def step(self, x, global_iter_counter, total_iterations):
        lr_t = self.lr * (1 - global_iter_counter / total_iterations)
        sigma_t = self.sigma * (1 - global_iter_counter / total_iterations)
        
        (i, j), _ = self.find_bmu(x)
        h = self.neighborhood((i, j), sigma_t)
        
        # Plasticity (Learning from new data)
        plasticity_delta = lr_t * h[..., np.newaxis] * (x - self.weights)
        
        # Stability (Remembering old tasks)
        stability_delta = np.zeros_like(self.weights)
        
        if self.previous_tasks:
            for star_weights, fisher in self.previous_tasks:
                # Calculate the difference
                diff = self.weights - star_weights
                
                # If weights drifted extremely far, this term could still be huge.
                diff = np.clip(diff, -50.0, 50.0)
                
                anchor_force = fisher * diff
                stability_delta += (self.ewc_lambda * anchor_force)
            
            stability_delta *= lr_t

        # Combine
        total_delta = plasticity_delta - stability_delta

        self.weights += total_delta

    def fit(self, data, epochs=10, shuffle=True, random_state=42):
        n_samples = data.shape[0]
        if n_samples == 0: return 

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

                # Progress bar
                if i % 10 == 0 or i == n_samples - 1:
                    progress = (i + 1) / n_samples
                    filled = int(bar_width * progress)
                    bar = '#' * filled + '-' * (bar_width - filled)
                    percent = int(progress * 100)
                    sys.stdout.write(f'\r[{bar}] {percent}%')
                    sys.stdout.flush()
            print()
            
        self.consolidate(data)
    
    def map_vects(self, data):
        mapped = []
        for x in data:
            idx, _ = self.find_bmu(x)
            mapped.append(idx)
        return np.array(mapped)