import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from evaluation.accuracy import evaluate_accuracy
from evaluation.error import find_qe_te

def find_r_metrices_combined(som, loader, T, beta=0.5, plot=False, out_path=None):
    # Store the BMU weight and adjacency for each sample in each task after training that task
    baseline_info = {}  
    
    # Matrices to record metrics
    R_bmu = []   # R_bmu[i][j] = average BMU-weight-difference for task j after training up to task i
    R_topo = []  # R_topo[i][j] = fraction of samples that lost/gained BMU adjacency erroneously
    R = []       # R[i][j] = Accuracy

    for task_id, X_task in loader.stream():
        print(f"Training SOM on task {task_id+1}/{T}...")
        som.fit(X_task, epochs=5, shuffle=True)
        print(f"Training SOM completed on task {task_id+1}/{T}!")

        # --- Compute and store baseline BMU info for this task ---
        X_test, _ = loader.test_task_data[task_id]
        X_test = np.stack(X_test)
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

        baseline_info[task_id] = {
            'weights': np.stack(weights_j),
            'adj': np.array(adj_j, dtype=bool)
        }

        # Evaluate errors for all tasks up to task_id
        row_bmu = []
        row_topo = []
        row = []
        
        for j in range(task_id + 1):
            X_test_j, y_test_j = loader.test_task_data[j]
            X_test_j = np.stack(X_test_j)
            y_test_j = np.array(y_test_j)

            # Evaluate functional accuracy
            acc = evaluate_accuracy(som, X_test_j, y_test_j)
            row.append(acc)

            if (plot):
                # X, y: numpy arrays (N, d) and (N,)
                plot_som_map(som, X_test_j, y_test_j, task_id, j, out_path=out_path, dr_method="pca")

            # If evaluating the current task, structural errors are 0 by definition
            if j == task_id:
                row_bmu.append(0.0)
                row_topo.append(0.0)
                continue

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

                # BMU weight-difference error for this sample (BWSE)
                err = np.linalg.norm(w_bmu_current - baseline_weights[idx])
                bmu_errors.append(err)

                # Topography preservation error for this sample (TPE)
                i1, j1 = divmod(bmu_idx, som.n)
                i2, j2 = divmod(second_idx, som.n)
                adj_current = (abs(i1 - i2) + abs(j1 - j2) == 1)
                
                # Error if adjacency status has changed from baseline
                if baseline_adj[idx] != adj_current:
                    topo_errors.append(1.0)
                else:
                    topo_errors.append(0.0)

            # Average over all samples of task j
            avg_bmu_err = np.mean(bmu_errors)
            avg_topo_err = np.mean(topo_errors)
            
            row_bmu.append(avg_bmu_err)
            row_topo.append(avg_topo_err)

        R_bmu.append(row_bmu)
        R_topo.append(row_topo)
        R.append(row)

    # Post-Processing: Vectorized Padding and Normalization
    
    # Pad matrices to full T×T with NaNs for unused entries to form lower triangular matrices
    R_bmu = np.array([np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in R_bmu])
    R_topo = np.array([np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in R_topo])
    R = np.array([np.pad(r, (0, T - len(r)), constant_values=np.nan) for r in R])

    # Determine maximums for normalization.
    max_bmu = np.nanmax(R_bmu) if np.nanmax(R_bmu) > 0 else 1.0
    max_topo = np.nanmax(R_topo) if np.nanmax(R_topo) > 0 else 1.0
    max_acc = np.nanmax(R) if np.nanmax(R) > 0 else 1.0

    # Normalize the error matrices
    R_bmu_norm = R_bmu / max_bmu
    # R_topo_norm = R_topo / max_topo
    # R_norm = R / max_acc

    # Vectorized calculation of the SOM-CL Integrity Score (SCI)
    R_sci = R / (1 + beta * R_bmu_norm + (1-beta) * R_topo)
    print(R, R, max_acc)
    print(R_bmu, R_bmu_norm, max_bmu)
    print(R_topo, R_topo, max_topo)

    return R_bmu, R_topo, R, R_sci

def compute_metrics(T, R_bmu, R_topo, R, R_sil):
    AA_bmu = [np.nanmean(R_bmu[i, :i+1]) for i in range(T)]
    AIA_bmu = [np.mean(AA_bmu[:i+1]) for i in range(T)]
    AA_topo = [np.nanmean(R_topo[i, :i+1]) for i in range(T)]
    AIA_topo = [np.mean(AA_topo[:i+1]) for i in range(T)]
    AA = [100 * np.nanmean(R[i, :i+1]) for i in range(T)]
    AIA = [np.mean(AA[:i+1]) for i in range(T)]
    AA_sil = [np.nanmean(R_sil[i, :i+1]) for i in range(T)]
    AIA_sil = [np.mean(AA_sil[:i+1]) for i in range(T)]
    return AA_bmu, AIA_bmu, AA_topo, AIA_topo, AA, AIA, AA_sil, AIA_sil

def compute_error_metrics(T, R_qe, R_te):
    AA_qe = [np.nanmean(R_qe[i, :i+1]) for i in range(T)]
    AIA_qe = [np.mean(AA_qe[:i+1]) for i in range(T)]
    AA_te = [np.nanmean(R_te[i, :i+1]) for i in range(T)]
    AIA_te = [np.mean(AA_te[:i+1]) for i in range(T)]
    return AA_qe, AIA_qe, AA_te, AIA_te

def compute_combined_metrics(T, R_bmu, R_topo, R, R_comb):
    AA_bmu = [np.nanmean(R_bmu[i, :i+1]) for i in range(T)]
    AIA_bmu = [np.mean(AA_bmu[:i+1]) for i in range(T)]
    AA_topo = [np.nanmean(R_topo[i, :i+1]) for i in range(T)]
    AIA_topo = [np.mean(AA_topo[:i+1]) for i in range(T)]
    AA = [100 * np.nanmean(R[i, :i+1]) for i in range(T)]
    AIA = [np.mean(AA[:i+1]) for i in range(T)]
    AA_comb = [np.nanmean(R_comb[i, :i+1]) for i in range(T)]
    AIA_comb = [np.mean(AA_comb[:i+1]) for i in range(T)]
    return AIA_bmu, AIA_topo, AIA, AIA_comb

def plot_metrics_columns(R, T, metric="Average Accuracy"):
    plt.figure(figsize=(10, 6))
    
    for j in range(T):
        # Take column j starting from row j downwards (valid part of lower triangle)
        col_values = R[j:, j]  
        x = np.arange(j+1, T+1)
        plt.plot(x, col_values, marker='o', label=f"Task {j+1}")
    
    plt.title(metric)
    plt.xlabel("Task")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_som_map(
    som,
    X,
    y,
    task_id,
    test_task_id,
    out_path="som_map.png",
    dr_method="pca",
    dr_random_state=0,
    tsne_perplexity=30,
    plot_data_alpha=0.6,
    neuron_edge_width=0.5,
    neuron_marker="o",
    neuron_base_size=80,
    cmap="tab20",
    show_figure=False
):
    """
    Visualize SOM neurons and data on a 2D plane using dimensionality reduction.
    - Projects both neuron weight vectors and the input X to the same 2D space.
    - Draws lattice connections between adjacent neurons (4-neighbors).
    - Plots data points (colored by y) on top of the neuron map.
    - Computes majority-vote label per neuron and purity (fraction of majority).
    - Colors neuron markers by majority label and scales marker size by the number of assigned points.
    - Also saves a right-hand heatmap showing majority label grid and purity grid (original lattice coordinates).
    - Saves the figure to `out_path`.

    Parameters
    ----------
    som : BaseSOM-like object
        Must provide attributes:
            som.weights -> np.ndarray (m, n, dim)
            som.m, som.n (optional: can be inferred from shape)
        and method:
            som.map_vects(X) -> array of BMU coords (N,2)
    X : np.ndarray, shape (N, d)
        Data vectors to plot (same d as som.weights last dim).
    y : array-like, shape (N,)
        Integer class labels to color the data points.
    out_path : str
        Path to save the resulting PNG (or other ext supported by matplotlib).
    dr_method : {'pca','tsne'}
        Which dimensionality reduction to use for visualization.
    """
    # --- prepare SOM dimensions and weights ---
    W = np.asarray(som.weights)  # shape (m,n,dim)
    if W.ndim != 3:
        raise ValueError("som.weights must be shape (m, n, dim)")

    m, n, dim = W.shape
    M = m * n
    W_flat = W.reshape(M, dim)

    # --- compute BMU assignments for X (if som provides map_vects) ---
    try:
        mapped = som.map_vects(X)  # (N,2) coordinates
    except Exception:
        # fallback: user might have find_bmu per-sample; attempt to call per sample
        mapped = np.array([som._find_bmu(x)[0] for x in X])  # uses internal method if available

    # flattened BMU index for each data sample
    bmu_flat_idx = mapped[:, 0] * n + mapped[:, 1]

    # --- majority vote and purity per neuron ---
    labels_per_neuron = defaultdict(list)
    for idx_flat, lbl in zip(bmu_flat_idx, y):
        labels_per_neuron[int(idx_flat)].append(int(lbl))

    majority_label = np.full(M, -1, dtype=int)   # -1 => empty neuron
    counts = np.zeros(M, dtype=int)
    purity = np.zeros(M, dtype=float)            # fraction of majority label among assigned points
    for i in range(M):
        labs = labels_per_neuron.get(i, [])
        if len(labs) == 0:
            majority_label[i] = -1
            counts[i] = 0
            purity[i] = 0.0
        else:
            counts[i] = len(labs)
            c = Counter(labs)
            maj, cnt = c.most_common(1)[0]
            majority_label[i] = int(maj)
            purity[i] = cnt / len(labs)

    # --- dimensionality reduction: fit on stacked (weights + data) so coordinates are comparable ---
    stacked = np.vstack([W_flat, X])
    if dr_method.lower() == "pca":
        dr = PCA(n_components=2, random_state=dr_random_state)
        proj = dr.fit_transform(stacked)
    elif dr_method.lower() == "tsne":
        # TSNE is slower; fit only on a sample if huge
        dr = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=dr_random_state, init="pca")
        proj = dr.fit_transform(stacked)
    else:
        raise ValueError("dr_method must be 'pca' or 'tsne'")

    W2 = proj[:M]                # neuron 2D positions (M,2)
    X2 = proj[M:]                # data 2D positions (N,2)

    # reshape neuron 2D positions to grid index arrays for mapping back to lattice
    neuron_pos = W2.reshape(m, n, 2)  # (m,n,2)

    # --- plotting ---
    fig = plt.figure(figsize=(14, 6))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_heat = fig.add_subplot(1, 2, 2)

    # 1) Draw lattice connections (4-neighbors) in the embedding space
    for i in range(m):
        for j in range(n):
            p = neuron_pos[i, j]
            # right neighbor
            if j + 1 < n:
                q = neuron_pos[i, j+1]
                ax_map.plot([p[0], q[0]], [p[1], q[1]], color="#bbbbbb", linewidth=0.7, zorder=0)
            # down neighbor
            if i + 1 < m:
                q = neuron_pos[i+1, j]
                ax_map.plot([p[0], q[0]], [p[1], q[1]], color="#bbbbbb", linewidth=0.7, zorder=0)

    # 2) plot data points (colored by label)
    labels = np.asarray(y)
    unique_labels = np.unique(labels)
    # use categorical colormap
    cmap_obj = plt.get_cmap(cmap)
    num_colors = max(20, unique_labels.max() + 1)  # ensure enough colors
    colors = [cmap_obj(i % cmap_obj.N) for i in range(num_colors)]

    for lbl in unique_labels:
        mask = labels == lbl
        ax_map.scatter(X2[mask, 0], X2[mask, 1], s=10, alpha=plot_data_alpha, color=colors[int(lbl) % len(colors)], label=f"label {int(lbl)}", linewidths=0)

    # 3) plot neurons colored by majority label, size by count, and edge by purity
    # marker sizes scaling
    max_count = max(1, counts.max())
    sizes = neuron_base_size * (0.3 + 0.7 * (counts / max_count))  # between 0.3 and 1.0 * base
    edge_alphas = 0.25 + 0.75 * purity  # scale border alpha by purity

    for idx_flat in range(M):
        pos = W2[idx_flat]
        lab = majority_label[idx_flat]
        size = sizes[idx_flat]
        if lab == -1:
            # empty neuron: small gray marker
            ax_map.scatter(pos[0], pos[1], s=12, color="#f0f0f0", edgecolor="#bbbbbb", linewidths=0.4, zorder=3, marker=neuron_marker)
        else:
            face = colors[lab % len(colors)]
            # darker border for high purity
            edgecol = (0, 0, 0, edge_alphas[idx_flat])
            ax_map.scatter(pos[0], pos[1], s=size, color=face, edgecolor=edgecol, linewidths=neuron_edge_width, zorder=4, marker=neuron_marker)

    ax_map.set_title("SOM Neuron Map (2D embedding) with Test Data Points")
    ax_map.axis("off")
    # show legend only if few labels
    if len(unique_labels) <= 12:
        ax_map.legend(loc="best", fontsize="small", markerscale=1, ncol=2)

    # --- Right-hand: heatmap of majority-vote labels ---
    maj_grid = majority_label.reshape(m, n)
    maj_grid_vis = np.where(maj_grid >= 0, maj_grid, np.nan)

    im1 = ax_heat.imshow(
        maj_grid_vis,
        origin="lower",
        cmap=cmap,
        interpolation="nearest"
    )

    ax_heat.set_title("Neuron Majority Label (grid view)")
    ax_heat.set_xlabel("grid column")
    ax_heat.set_ylabel("grid row")

    cbar1 = plt.colorbar(im1, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar1.set_label("label id (NaN = empty)")

    plt.suptitle(f"SOM Visualization\nAfter Training on TaskID {task_id} When Testing on TaskID {test_task_id}", fontsize=12)

    # save figure
    fig.savefig(f"{out_path}\\TaskID_{task_id}_TestTaskID_{test_task_id}.png", dpi=200)
    if show_figure:
        plt.show()
    plt.close(fig)

def find_stability_plasticity(som, loader, T):
    """
    Calculates Stability and Plasticity for Accuracy, Quantization Error, and Topographic Error.
    
    Stability = Average metric on all previously seen tasks (0 to task_id-1).
    Plasticity = Metric on the current task (task_id).
    """
    
    acc_S, acc_P = [], []
    qe_S, qe_P = [], []
    te_S, te_P = [], []

    for task_id, X_task in loader.stream():
        print(f"Training SOM on task {task_id+1}/{T}...")
        som.fit(X_task, epochs=5, shuffle=True)
        print(f"Training SOM completed on task {task_id+1}/{T}!")

        # --- Evaluate metrics for all tasks seen so far (0 to task_id) ---
        row_acc = []
        row_qe = []
        row_te = []

        for j in range(task_id + 1):
            X_test_j, y_test_j = loader.test_task_data[j]
            X_test_j = np.stack(X_test_j)
            y_test_j = np.array(y_test_j)

            # 1. Accuracy
            acc = evaluate_accuracy(som, X_test_j, y_test_j)
            row_acc.append(acc)

            # 2. Quantization Error & Topographic Error
            nqe, te = find_qe_te(som, X_test_j)
            row_qe.append(nqe)
            row_te.append(te)

        # Plasticity: Performance on the CURRENT task
        acc_P.append(row_acc[-1])
        qe_P.append(row_qe[-1])
        te_P.append(row_te[-1])

        # Stability: Average performance on PREVIOUS tasks
        if task_id == 0:
            # Undefined for the first task
            acc_S.append(0.0)
            qe_S.append(0.0)
            te_S.append(0.0)
        else:
            # Average of all tasks up to the current one
            acc_S.append(np.mean(row_acc[:-1]))
            qe_S.append(np.mean(row_qe[:-1]))
            te_S.append(np.mean(row_te[:-1]))
            
    return acc_S, acc_P, qe_S, qe_P, te_S, te_P