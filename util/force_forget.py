import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats

# HELPER: Per-sample BWSE and TPE given a perturbed SOM and frozen baseline
def _compute_bwse_tpe_acc(som_perturbed, X_test, y_test, baseline_bmu_weights, baseline_adj, evaluate_accuracy_fn):
    """
    For every sample in X_test, compare the current SOM state against the
    baseline (recorded right after training on task 1) and compute:
      - Per-sample BWSE  : ||w_bmu_baseline  -  w_bmu_current||
      - Per-sample TPE   : 1 if BMU/2nd-BMU adjacency changed, else 0
      - Classification accuracy using majority-vote labels
    """
    weights_flat = som_perturbed.weights.reshape(-1, som_perturbed.dim)
    n = som_perturbed.n  # grid side length

    bwse_vals, tpe_vals = [], []

    for idx, x in enumerate(X_test):
        dists = np.linalg.norm(weights_flat - x, axis=1)

        bmu_idx = np.argmin(dists)
        dists_copy = dists.copy()
        dists_copy[bmu_idx] = np.inf
        second_idx = np.argmin(dists_copy)

        # BWSE
        w_current = weights_flat[bmu_idx]
        bwse_vals.append(np.linalg.norm(w_current - baseline_bmu_weights[idx]))

        # TPE
        i1, j1 = divmod(bmu_idx, n)
        i2, j2 = divmod(second_idx, n)
        adj_current = (abs(i1 - i2) + abs(j1 - j2) == 1)
        tpe_vals.append(0.0 if (baseline_adj[idx] == adj_current) else 1.0)

    acc = evaluate_accuracy_fn(som_perturbed, X_test, y_test)

    return np.mean(bwse_vals), np.mean(tpe_vals), acc

# Train on task 1, record baseline, return a clean copy of the SOM
def train_task1_and_record_baseline(som, loader, epochs=5):
    """
    Trains `som` on the first task only, then records:
      - baseline_bmu_weights  : weight vector of each sample's BMU  (N * dim)
      - baseline_adj          : bool array  - was BMU adjacent to 2nd-BMU?  (N,)
      - X_test_t1, y_test_t1  : test split for task 1

    Returns
    -------
    som_trained       : the SOM after training on task 1
    baseline_info     : dict with keys 'weights', 'adj'
    X_test_t1         : np.ndarray  (N * dim)
    y_test_t1         : np.ndarray  (N,)
    """
    # Train on task 1
    task_id = 0
    for tid, X_task in loader.stream():
        print(f"Training SOM on task {tid + 1} ...")
        som.fit(X_task, epochs=epochs, shuffle=True)
        if tid == task_id:
            break

    # Record test split
    X_test_list, y_test_list = loader.test_task_data[task_id]
    X_test_t1 = np.stack(X_test_list)
    y_test_t1 = np.array(y_test_list)

    # Record per-sample baseline BMU info
    weights_flat = som.weights.reshape(-1, som.dim)
    n = som.n

    baseline_bmu_weights, baseline_adj = [], []
    for x in X_test_t1:
        dists = np.linalg.norm(weights_flat - x, axis=1)

        bmu_idx = np.argmin(dists)
        dists_copy = dists.copy()
        dists_copy[bmu_idx] = np.inf
        second_idx = np.argmin(dists_copy)

        baseline_bmu_weights.append(weights_flat[bmu_idx].copy())

        i1, j1 = divmod(bmu_idx, n)
        i2, j2 = divmod(second_idx, n)
        baseline_adj.append(abs(i1 - i2) + abs(j1 - j2) == 1)

    baseline_info = {
        'weights': np.stack(baseline_bmu_weights),   # (N, dim)
        'adj':     np.array(baseline_adj, dtype=bool) # (N,)
    }

    print("Baseline recorded.")
    return som, baseline_info, X_test_t1, y_test_t1

# Perturbation sweep
def run_controlled_forgetting_experiment(som_trained, baseline_info, X_test_t1, y_test_t1, evaluate_accuracy_fn, epsilons=None, n_repeats=5, random_seed=42):
    """
    Injects synthetic Gaussian noise of increasing magnitude ε into the
    trained SOM's weight matrix and measures BWSE, TPE, and ACC at each level.

    The SOM passed in is NEVER modified — we work on deep copies.

    Parameters
    ----------
    som_trained          : trained SOM (after task 1)
    baseline_info        : dict returned by train_task1_and_record_baseline()
    X_test_t1, y_test_t1 : task-1 test split
    evaluate_accuracy_fn : callable(som, X, y) → float  (0-100 or 0-1, consistent)
    epsilons             : list of noise std-devs to sweep (default 0→10)
    n_repeats            : number of random noise realisations per ε (for error bars)
    random_seed          : base seed for reproducibility

    Returns
    -------
    results : dict with keys 'epsilons', 'bwse', 'tpe', 'acc', each containing arrays of shape (len(epsilons), n_repeats)
    """
    if epsilons is None:
        epsilons = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]

    rng = np.random.default_rng(random_seed)

    bwse_grid = np.zeros((len(epsilons), n_repeats))
    tpe_grid  = np.zeros((len(epsilons), n_repeats))
    acc_grid  = np.zeros((len(epsilons), n_repeats))

    original_weights = som_trained.weights.copy()

    for i, eps in enumerate(epsilons):
        for r in range(n_repeats):
            # Deep-copy so the original SOM is never mutated
            som_perturbed = copy.deepcopy(som_trained)

            if eps == 0.0:
                # Identity: baseline = current, all errors must be 0
                noise = np.zeros_like(original_weights)
            else:
                noise = rng.normal(loc=0.0, scale=eps, size=original_weights.shape)

            som_perturbed.weights = original_weights + noise

            bwse, tpe, acc = _compute_bwse_tpe_acc(
                som_perturbed,
                X_test_t1,
                y_test_t1,
                baseline_info['weights'],
                baseline_info['adj'],
                evaluate_accuracy_fn
            )

            bwse_grid[i, r] = bwse
            tpe_grid[i, r]  = tpe
            acc_grid[i, r]  = acc

        print(f"  ε={eps:.2f} | "
              f"BWSE={bwse_grid[i].mean():.4f} | "
              f"TPE={tpe_grid[i].mean():.4f}  | "
              f"ACC={acc_grid[i].mean():.2f}")

    results = {
        'epsilons': np.array(epsilons),
        'bwse':     bwse_grid,
        'tpe':      tpe_grid,
        'acc':      acc_grid,
    }
    return results

# Plot + correlation table
def plot_and_report_metric_validation(results, dataset_name="Dataset",
                                       save_path=None):
    """
    Produces:
      Figure A - line plots (mean ± std) of BWSE, TPE, and ACC vs ε,
                 all normalised to [0, 1] on a shared axis to check
                 they rise together.
      Figure B - three scatter plots: BWSE vs ACC-drop, TPE vs ACC-drop,
                 BWSE vs TPE.
      Console  - Pearson, Spearman, and p-values for each metric pair.
    """
    epsilons = results['epsilons']
    bwse_mean = results['bwse'].mean(axis=1)
    bwse_std  = results['bwse'].std(axis=1)
    tpe_mean  = results['tpe'].mean(axis=1)
    tpe_std   = results['tpe'].std(axis=1)
    acc_mean  = results['acc'].mean(axis=1)
    acc_std   = results['acc'].std(axis=1)

    # ACC degradation: how much accuracy dropped relative to ε=0
    acc_drop      = acc_mean[0] - acc_mean          # positive = degradation
    acc_drop_std  = acc_std                          # std unchanged

    # Normalise each metric to [0, 1] for overlay plot
    def _norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-12 else np.zeros_like(v)

    bwse_n = _norm(bwse_mean)
    tpe_n  = _norm(tpe_mean)
    drop_n = _norm(acc_drop)   # acc degradation (↑ = more forgetting)

    # Colour scheme
    C_BWSE = "#185FA5"   # blue
    C_TPE  = "#D85A30"   # coral
    C_ACC  = "#1D9E75"   # teal

    # Figure A: normalised overlay
    fig_a, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(epsilons, bwse_n, 'o-', color=C_BWSE, lw=2,   label='BWSE (norm.)')
    ax.plot(epsilons, tpe_n,  'x-', color=C_TPE,  lw=2,   label='TPE (norm.)')
    ax.plot(epsilons, drop_n, 's-', color=C_ACC,  lw=2,   label='ACC drop (norm.)')

    ax.set_xlabel('Perturbation magnitude ε (noise Standard Deviation)', fontsize=12)
    ax.set_ylabel('Normalised value', fontsize=12)
    ax.set_title(f'Metric Validation via Controlled Forgetting\n'
                 f'{dataset_name} — Task 1', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig_a.tight_layout()

    if save_path:
        fig_a.savefig(save_path.replace('.png', '_overlay.png'), dpi=150, bbox_inches='tight')

    # Figure B: scatter + regression
    fig_b, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [
        (bwse_mean, acc_drop, 'BWSE', 'ACC drop', C_BWSE, C_ACC),
        (tpe_mean,  acc_drop, 'TPE',  'ACC drop', C_TPE,  C_ACC),
        (bwse_mean, tpe_mean, 'BWSE', 'TPE',      C_BWSE, C_TPE),
    ]

    for ax_s, (x, y, xlabel, ylabel, cx, cy) in zip(axes, pairs):
        ax_s.scatter(x, y, c=cx, s=60, zorder=3)
        ax_s.set_xlabel(xlabel, fontsize=11)
        ax_s.set_ylabel(ylabel, fontsize=11)
        ax_s.set_title(f'{xlabel} vs {ylabel}', fontsize=11)
        ax_s.grid(True, linestyle='--', alpha=0.3)

    fig_b.suptitle(f'Pairwise Metric Correlations — {dataset_name}', fontsize=13)
    fig_b.tight_layout()

    if save_path:
        fig_b.savefig(save_path.replace('.png', '_scatter.png'), dpi=150, bbox_inches='tight')

    plt.show()

    # Correlation table (console + return)
    metric_pairs = [
        ('BWSE',  'ACC degradation', bwse_mean,  acc_drop),
        ('TPE',   'ACC degradation', tpe_mean,   acc_drop),
        ('BWSE',  'TPE',             bwse_mean,  tpe_mean),
        ('BWSE',  'BWT proxy',       bwse_mean,  -acc_drop),  # BWT ≈ –drop
        ('TPE',   'BWT proxy',       tpe_mean,   -acc_drop),
    ]

    header = (f"\n{'─'*72}\n"
              f"{'Metric pair':<28} {'Pearson r':>10} {'Spearman ρ':>12} "
              f"{'p-value (P)':>14} {'p-value (S)':>14}\n"
              f"{'─'*72}")
    print(header)

    table_rows = []
    for (m1, m2, v1, v2) in metric_pairs:
        pr, pp = stats.pearsonr(v1, v2)
        sr, sp = stats.spearmanr(v1, v2)
        pair_label = f"{m1} vs {m2}"
        row_str = (f"{pair_label:<28} {pr:>+10.4f} {sr:>+12.4f} "
                   f"{pp:>14.4e} {sp:>14.4e}")
        print(row_str)
        table_rows.append({
            'pair':      pair_label,
            'pearson_r': pr, 'pearson_p': pp,
            'spearman_r': sr, 'spearman_p': sp,
        })
    print('─' * 72)

    # Boundary-condition sanity check
    print("\nBoundary condition check (ε = 0.0):")
    idx_zero = np.argmin(np.abs(results['epsilons']))
    print(f"  BWSE at ε=0: {results['bwse'][idx_zero].mean():.6f}  (expect ≈ 0)")
    print(f"  TPE  at ε=0: {results['tpe'][idx_zero].mean():.6f}  (expect ≈ 0)")
    print(f"  ACC  at ε=0: {results['acc'][idx_zero].mean():.4f}")

    return table_rows

# run the full experiment in one call
def run_experiment1(som, loader, evaluate_accuracy_fn, dataset_name="Dataset", epsilons=None, n_repeats=5, epochs=5, random_seed=42, save_path=None):
    """
    End-to-end runner for Experiment 1.

    Usage
    -----
    table = run_experiment1(
        som               = my_som,
        loader            = my_loader,
        evaluate_accuracy_fn = evaluate_accuracy,
        dataset_name      = "CORe50 NC",
        epsilons          = [0, 0.5, 1, 2, 3, 5, 7.5, 10],
        n_repeats         = 5,
        save_path         = "exp1_core50.png",
    )

    Parameters
    ----------
    som                  : an untrained (or freshly initialised) SOM instance
    loader               : your dataset loader (must support .stream() and
                           .test_task_data[task_id])
    evaluate_accuracy_fn : callable(som, X, y) → float
    dataset_name         : string used in plot titles
    epsilons             : perturbation levels to sweep
    n_repeats            : number of random realisations per ε
    epochs               : training epochs on task 1
    random_seed          : reproducibility seed
    save_path            : if provided, saves both figures (overlay + scatter)

    Returns
    -------
    table_rows : list of dicts with correlation statistics
    """
    print(f"\n{'='*60}")
    print(f" Experiment 1: Metric Validation — {dataset_name}")
    print(f"{'='*60}\n")

    # Train on task 1 only and capture the baseline
    som_trained, baseline_info, X_test_t1, y_test_t1 = \
        train_task1_and_record_baseline(som, loader, epochs=epochs)

    # Sweep perturbation levels
    print("\nRunning perturbation sweep ...")
    results = run_controlled_forgetting_experiment(
        som_trained          = som_trained,
        baseline_info        = baseline_info,
        X_test_t1            = X_test_t1,
        y_test_t1            = y_test_t1,
        evaluate_accuracy_fn = evaluate_accuracy_fn,
        epsilons             = epsilons,
        n_repeats            = n_repeats,
        random_seed          = random_seed,
    )

    # Plot and print the correlation table
    print("\nGenerating plots and correlation table ...")
    table_rows = plot_and_report_metric_validation(
        results      = results,
        dataset_name = dataset_name,
        save_path    = save_path,
    )

    return table_rows
