import numpy as np
import sys
import copy

class FSOM:
    def __init__(self, m, n, dim, learning_rate=0.2, eta_min=0.01, seed=None, ewc_lambda=0.5, fisher_sigma=5.0):
        """
        Initialize Fisher Self-Organizing Map (FSOM).
        
        Args:
            m, n: Map dimensions.
            dim: Input dimensionality.
            learning_rate: Initial/Maximum learning rate (eta_max).
            sigma: Initial neighborhood radius (sigma_0).
            eta_min: Minimum learning rate for TASOM adaptation.
            seed: Random seed.
            ewc_lambda: Importance weight for the EWC penalty (higher = more stability).
            fisher_sigma: Fixed sigma used specifically for Fisher Information calculation.
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.eta_max = learning_rate
        self.eta_min = eta_min
        self.sigma_0 = max(m, n) / 2.0
        self.fisher_sigma = fisher_sigma
        self.ewc_lambda = ewc_lambda
        
        self._rng = np.random.default_rng(seed)
        
        # Weight vectors: shape (m, n, dim)
        self.weights = self._rng.random((m, n, dim))
        
        # Location of each neuron in grid
        self.neurons = np.array([[np.array([i, j]) for j in range(n)] for i in range(m)])
        
        # FSOM state: win counts
        self.win_counts = np.zeros((m, n), dtype=int)
        
        # EWC state: list of (star_weights, fisher_matrix)
        self.previous_tasks = []
        
        # Training state
        self.global_iter_counter = 0
        self.total_iterations = 0

    def find_bmu(self, x):
        """
        Find Best Matching Unit (BMU) robustly.
        """
        diff = x - self.weights
        # Robustness: Check for NaNs/Infs
        if not np.all(np.isfinite(diff)):
            return (0, 0), 0.0
            
        dist = np.linalg.norm(diff, axis=2)
        idx = np.unravel_index(np.argmin(dist), (self.m, self.n))
        return idx, dist[idx]

    def neighborhood(self, bmu_idx, sigma_t):
        """
        Compute Gaussian neighborhood function with numerical safety.
        """
        sq_dist = np.sum((self.neurons - bmu_idx) ** 2, axis=2)
        
        # Epsilon to prevent div by zero if sigma_t decays to 0
        epsilon_sigma = 1e-9
        denominator = 2 * sigma_t ** 2 + epsilon_sigma
        
        exponent = sq_dist / denominator
        
        return np.exp(-exponent)

    def compute_fisher(self, data):
        """
        Compute Fisher Information diagonal matrix for the current task data.
        """
        fisher_accum = np.zeros_like(self.weights)
        n_samples = data.shape[0]
        if n_samples == 0: return fisher_accum

        for x in data:
            (i, j), _ = self.find_bmu(x)
            # Use fixed fisher_sigma for sensitivity analysis
            h = self.neighborhood((i, j), self.fisher_sigma)
            
            # Gradient approximation for SOM: h * (x - w)
            grad = h[..., np.newaxis] * (x - self.weights)
            fisher_accum += grad ** 2

        raw_fisher = fisher_accum / n_samples
        return raw_fisher

    def consolidate(self, data):
        """
        Consolidate knowledge after training a task.
        Calculates Fisher Information and stores the current weights as 'star_weights'.
        """
        task_id = len(self.previous_tasks) + 1
        print(f"Consolidating knowledge for Task {task_id} (EWC)...")
        
        raw_fisher = self.compute_fisher(data)

        # Normalize Fisher to [0, 1] to keep lambda consistent across varying gradients
        max_f = np.max(raw_fisher)
        if max_f > 1e-9:
            normalized_fisher = raw_fisher / max_f
        else:
            normalized_fisher = raw_fisher
            
        star_weights = copy.deepcopy(self.weights)
        self.previous_tasks.append((star_weights, normalized_fisher))
        print(f"Task {task_id} consolidated.")

    def step(self, x, global_iter_counter, total_iterations):
        """
        Perform a single training step combining TASOM adaptation and EWC regularization.
        """
        self.global_iter_counter = global_iter_counter
        self.total_iterations = total_iterations
        
        # Find BMU and Update Win Counts
        (i, j), _ = self.find_bmu(x)
        self.win_counts[i, j] += 1
        
        t = self.global_iter_counter + 1
        
        # Calculate Time-Decreasing Sigma
        if self.sigma_0 is None:
            max_dim = max(self.m, self.n)
            self.sigma_0 = max_dim / 2.0
            
        sigma_t = self.sigma_0 * (1 - t / self.total_iterations)
        
        # Calculate Neighborhood
        h = self.neighborhood((i, j), sigma_t)
        
        # Calculate Adaptive Learning Rate Matrix (eta_matrix)
        # eta_j(t) = freq_j(t) * (eta_max - eta_min) + eta_min
        epsilon_t = 1e-6
        freq_matrix = self.win_counts / (t + epsilon_t)
        eta_matrix = freq_matrix * (self.eta_max - self.eta_min) + self.eta_min
        
        # Calculate Forces
        
        # Plasticity Force (Learning from current data)
        # Gradient direction: (x - w) * h
        plasticity_force = h[..., np.newaxis] * (x - self.weights)
        
        # Stability Force (EWC Anchor)
        stability_force = np.zeros_like(self.weights)
        
        if self.previous_tasks:
            for star_weights, fisher in self.previous_tasks:
                # Distance from anchor
                diff = self.weights - star_weights
                
                
                # Spring force: k * x (where k = lambda * fisher)
                anchor_force = fisher * diff
                stability_force += (self.ewc_lambda * anchor_force)
        
        # Apply Update
        # Update rule: w_new = w + eta * (Plasticity - Stability)
        total_delta = eta_matrix[..., np.newaxis] * (plasticity_force - stability_force)
        
        self.weights += total_delta

    def fit(self, data, epochs=10, shuffle=True, random_state=42):
        """
        Train the FSOM on a specific task/dataset.
        Call this method sequentially for each task in a CL scenario.
        """
        n_samples = data.shape[0]
        if n_samples == 0: return

        # Reset FSOM counters for the new task phase
        self.total_iterations = epochs * n_samples
        self.global_iter_counter = 0
        
        # Reset win counts for the new distribution adaptation
        self.win_counts = np.zeros((self.m, self.n), dtype=int)
        
        # Reset sigma logic for the new task
        self.sigma_0 = max(self.m, self.n) / 2.0

        bar_width = 40

        for epoch in range(epochs):
            rng = np.random.default_rng(random_state)
            if shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            print(f"\nEpoch {epoch+1}/{epochs}")
            for i, idx in enumerate(indices):
                x = data[idx]
                self.step(x, self.global_iter_counter, self.total_iterations)
                self.global_iter_counter += 1

                # Progress Bar
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
        """
        Map inputs to BMU coordinates.
        """
        mapped = []
        for x in data:
            idx, _ = self.find_bmu(x)
            mapped.append(idx)
        return np.array(mapped)