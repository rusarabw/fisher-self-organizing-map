import numpy as np
import sys

class NON_EWC_FSOM:
    """
    Non-weight Consolidating Counterpart of the FSOM Algorithm
    """
    def __init__(self, m, n, dim, learning_rate=0.2, eta_min=0.01, seed=None):
        """
        Initialize NON_EWC_FSOM.
        
        Args:
            m: number of rows
            n: number of columns
            dim: dimensionality of input vectors
            learning_rate (eta_max): initial (maximum) learning rate. This acts as eta_max.
            sigma: initial neighborhood radius (sigma_0).
            eta_min: minimum learning rate for the adaptive calculation.
            seed: random seed for reproducibility
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.eta_max = learning_rate
        self.eta_min = eta_min
        self.sigma_0 = max(m, n) / 2.0
        self._rng = np.random.default_rng(seed)
        
        # Weight vectors: shape (m, n, dim)
        self.weights = self._rng.random((m, n, dim))
        
        # Location of each neuron in grid for distance computations
        self.neurons = np.array([[np.array([i, j]) for j in range(n)] for i in range(m)])
        
        # win counts
        self.win_counts = np.zeros((m, n), dtype=int)
        
        # Training state variables
        self.global_iter_counter = 0
        self.total_iterations = 0

    def find_bmu(self, x):
        """
        Find Best Matching Unit (BMU) for input x using Euclidean distance.
        
        Args:
            x: input vector of shape (dim,)
        Returns:
            tuple (bmu_location (i,j), bmu_distance)
        """
        # Compute Euclidean distance (L2 norm) to all weights
        diff = x - self.weights
        dist = np.linalg.norm(diff, axis=2)
        # Get the index of the minimum distance
        idx = np.unravel_index(np.argmin(dist), (self.m, self.n))
        return idx, dist[idx]

    def neighborhood(self, bmu_idx, sigma_t):
        """
        Compute Gaussian neighborhood function around BMU.
        
        Args:
            bmu_idx: tuple (i,j)
            sigma_t: current neighborhood radius
        Returns:
            2D array of shape (m, n) containing the influence of the BMU on all neurons
        """
        # Calculate squared Euclidean distance between neuron locations and BMU location
        sq_dist = np.sum((self.neurons - bmu_idx) ** 2, axis=2)
        
        # Gaussian function.
        epsilon_sigma = 1e-9
        return np.exp(-sq_dist / (2 * sigma_t ** 2 + epsilon_sigma))
    
    def step(self, x, global_iter_counter, total_iterations):
        """
        Perform a single training step using the update rule.
        
        Args:
            x: input vector
            global_iter_counter: global iteration counter (t)
            total_iterations: total number of planned iterations (T)
        """
        self.global_iter_counter = global_iter_counter
        self.total_iterations = total_iterations
        
        # Find BMU
        (i, j), _ = self.find_bmu(x)
        
        # Update Win Count for the BMU
        self.win_counts[i, j] += 1

        t = self.global_iter_counter + 1
        
        # Calculate Time-Decreasing Sigma (Neighborhood Radius)
        if self.sigma_0 is None:
            max_dim = max(self.m, self.n)
            self.sigma_0 = max_dim / 2.0
            
        sigma_t = self.sigma_0 * (1 - t / self.total_iterations)
        
        # Calculate Neighborhood Function
        h = self.neighborhood((i, j), sigma_t)

        # Calculate Adaptive Learning Rate Matrix (eta_matrix)
        # adaptive learning rate eta_j(t) = freq_j(t) * (eta_max - eta_min) + eta_min
        
        # Add a small epsilon to the denominator to prevent division by zero at t=0
        epsilon = 1e-6
        freq_matrix = self.win_counts / (t + epsilon) 
        
        # Calculate the adaptive learning rate for every neuron
        eta_matrix = freq_matrix * (self.eta_max - self.eta_min) + self.eta_min
        
        # Update Weights
        # Delta_w = eta_matrix * h_matrix * (x - w)
        delta = eta_matrix[..., np.newaxis] * h[..., np.newaxis] * (x - self.weights)
        self.weights += delta

    def fit(self, data, epochs=10, shuffle=True, random_state=42):
        """
        Train the NON_EWC_FSOM over multiple epochs with terminal progress display.
        
        Args:
            data: array of shape (N, dim)
            epochs: number of passes over the dataset
            shuffle: whether to shuffle the data each epoch
            random_state: seed for reproducibility
        """
        n_samples = data.shape[0]
        self.total_iterations = epochs * n_samples
        self.global_iter_counter = 0
        
        # Set default sigma_0 if not provided
        if self.sigma_0 is None:
            self.sigma_0 = max(self.m, self.n) / 2.0

        # Reset win counts before starting new training
        self.win_counts = np.zeros((self.m, self.n), dtype=int)
        
        bar_width = 40

        print(f"Starting NON_EWC_FSOM training for {self.total_iterations} iterations...")

        for epoch in range(epochs):
            rng = np.random.default_rng(random_state)
            
            if shuffle:
                # Ensure the same random state is used for shuffling if provided
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            sys.stdout.write(f"\nEpoch {epoch+1}/{epochs}...")
            
            for i, idx in enumerate(indices):
                x = data[idx]
                
                # Perform the update step
                self.step(x, self.global_iter_counter, self.total_iterations)
                
                # Increment the global counter after the step
                self.global_iter_counter += 1

                # Display progress bar (optional, but requested in original code)
                progress = (i + 1) / n_samples
                filled = int(bar_width * progress)
                bar = '#' * filled + '-' * (bar_width - filled)
                percent = int(progress * 100)
                sys.stdout.write(f'\rEpoch {epoch+1}/{epochs}: [{bar}] {percent}%')
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        print("Training complete.")


    def map_vects(self, data):
        """
        Map each input vector to its BMU coordinate.
        
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