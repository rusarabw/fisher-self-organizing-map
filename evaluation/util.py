import numpy as np

def find_first_second_bmu(som, X_test):
    weights_j = []
    adj_j = []
    for x in X_test:
        # Find BMU and second BMU indices
        weights_flat = som.weights.reshape(-1, som.dim)
        dists = np.linalg.norm(weights_flat - x, axis=1)
        bmu_idx = np.argmin(dists)
        dists[bmu_idx] = np.inf
        second_idx = np.argmin(dists)

        # Record BMU weight
        w_bmu = weights_flat[bmu_idx]
        weights_j.append(w_bmu)

        # Check if first and second BMU are adjacent on the grid
        i1, j1 = divmod(bmu_idx, som.n)
        i2, j2 = divmod(second_idx, som.n)
        adjacent = (abs(i1 - i2) + abs(j1 - j2) == 1)
        adj_j.append(adjacent)

    return weights_j, adj_j