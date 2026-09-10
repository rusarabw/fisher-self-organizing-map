import numpy as np

def find_quantization_error(som, X):
    quantization_errors = []

    for x in X:
        _, bmu_weight = som.find_bmu(x)
        quantization_error = np.linalg.norm(x - bmu_weight)
        quantization_errors.append(quantization_error)
    
    quantization_errors = np.array(quantization_errors)
    quantization_error_max = np.max(quantization_errors)

    # return normalized QE, 1e-8 is used to avoid division by 0
    return np.mean(quantization_errors / (quantization_error_max + 1e-8))

def find_topographic_error(som, X):
    topo_violations = 0

    for x in X:
        bmu_distances = np.linalg.norm(som.weights - x, axis=2)
        flat = bmu_distances.reshape(-1)
        sorted_indices = np.argsort(flat)
        bmu1 = np.unravel_index(sorted_indices[0], som.weights.shape[:2])
        bmu2 = np.unravel_index(sorted_indices[1], som.weights.shape[:2])
        is_adjacent = np.sum(np.abs(np.array(bmu1) - np.array(bmu2))) == 1
        if not is_adjacent:
            topo_violations += 1

    return topo_violations / len(X)

def find_qe_te(som, X):
    quantization_errors = []
    topo_violations = 0

    for x in X:
        _, bmu_weight = som.find_bmu(x)
        quantization_error = np.linalg.norm(x - bmu_weight)
        quantization_errors.append(quantization_error)

        bmu_distances = np.linalg.norm(som.weights - x, axis=2)
        flat = bmu_distances.reshape(-1)
        sorted_indices = np.argsort(flat)
        bmu1 = np.unravel_index(sorted_indices[0], som.weights.shape[:2])
        bmu2 = np.unravel_index(sorted_indices[1], som.weights.shape[:2])
        is_adjacent = np.sum(np.abs(np.array(bmu1) - np.array(bmu2))) == 1
        if not is_adjacent:
            topo_violations += 1
    
    quantization_errors = np.array(quantization_errors)
    quantization_error_max = np.max(quantization_errors)

    # return normalized QE, 1e-8 is used to avoid division by 0
    return np.mean(quantization_errors / (quantization_error_max + 1e-8)), topo_violations / len(X)

def find_bmu_and_topo_error(som, j, X_test_j, baseline_info):
    baseline_weights = baseline_info[j]['weights']
    baseline_adj = baseline_info[j]['adj']

    bmu_errors = []
    topo_errors = []
    for idx, x in enumerate(X_test_j):
        # BMU and second BMU after training up to task_id
        weights_flat = som.weights.reshape(-1, som.dim)
        dists = np.linalg.norm(weights_flat - x, axis=1)
        bmu_idx = np.argmin(dists)
        dists[bmu_idx] = np.inf
        second_idx = np.argmin(dists)
        w_bmu_current = weights_flat[bmu_idx]

        # BMU weight-difference error for this sample
        err = np.linalg.norm(w_bmu_current - baseline_weights[idx])
        bmu_errors.append(err)

        # Topography preservation error for this sample
        i1, j1 = divmod(bmu_idx, som.n)
        i2, j2 = divmod(second_idx, som.n)
        adj_current = (abs(i1 - i2) + abs(j1 - j2) == 1)
        # Count an error if originally adjacent but now not
        if baseline_adj[idx] and not adj_current:
            topo_errors.append(1.0)
        else:
            topo_errors.append(0.0)

    return bmu_errors, topo_errors