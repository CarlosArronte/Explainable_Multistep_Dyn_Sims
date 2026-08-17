import argparse
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# GP exacto -> coste aproximadamente cúbico con N.
# 2500 es un punto de partida razonable para validar el pipeline.
MAX_GP_TRAIN_SAMPLES = 2000

BASE_FREQUENCY_HZ = 100.0

HORIZONS = [
    1,   # 10 ms
    2,   # 20 ms
    5,   # 50 ms
    10,  # 100 ms
]

# Limitar la evaluación evita predicciones excesivamente pesadas.
MAX_EVALUATION_SAMPLES = 12000

# Estados del modelo
STATE_COLUMNS = [
    "v_x",
    "v_y",
    "r",
    "omega_wheels",
]

# Entradas de control
CONTROL_COLUMNS = [
    "delta",
    "Iq",
]

INPUT_COLUMNS = STATE_COLUMNS + CONTROL_COLUMNS

REQUIRED_COLUMNS = INPUT_COLUMNS + ["run_id"]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD CSV
# ============================================================

def load_dataset(path: Path):
    """
    Lee uno de los CSV procesados por los autores.

    La primera columna del CSV corresponde al tiempo y se usa
    como índice, exactamente como hace OptitrackDataset en
    el repositorio original.
    """

    logger.info("Loading %s", path)

    df = pd.read_csv(
        path,
        index_col=0,
    )

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}"
        )

    # Convertir explícitamente el índice temporal.
    df.index = pd.to_numeric(
        df.index,
        errors="coerce",
    )

    # Solo exigimos que sean válidas las variables que
    # realmente usaremos.
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    logger.info(
        "%s | samples=%d | runs=%d",
        path.name,
        len(df),
        df["run_id"].nunique(),
    )

    logger.info(
        "%s | run IDs=%s",
        path.name,
        sorted(df["run_id"].unique().tolist()),
    )

    return df


# ============================================================
# CREATE TRANSITIONS
# ============================================================

def create_transitions(df, horizon=1):
    """
    Construye transiciones a un horizonte H:

        X_k = [x_k, u_k]

        Δx_k^(H) = x_(k+H) - x_k

    horizon = 1  -> 10 ms
    horizon = 2  -> 20 ms
    horizon = 5  -> 50 ms
    horizon = 10 -> 100 ms

    Las transiciones siempre se crean dentro de cada run_id.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    X_list = []
    delta_list = []

    x_current_list = []
    x_next_list = []

    time_list = []
    dt_list = []
    run_list = []

    for run_id, run_df in df.groupby(
        "run_id",
        sort=False,
    ):

        run_df = run_df.sort_index()

        if len(run_df) <= horizon:
            logger.warning(
                "Skipping run %s: not enough samples for H=%d",
                run_id,
                horizon,
            )
            continue

        x = (
            run_df[STATE_COLUMNS]
            .to_numpy(dtype=float)
        )

        u = (
            run_df[CONTROL_COLUMNS]
            .to_numpy(dtype=float)
        )

        t = (
            run_df.index
            .to_numpy(dtype=float)
        )

        # ----------------------------------------------------
        # Current and future states
        # ----------------------------------------------------

        x_current = x[:-horizon]

        x_next = x[horizon:]

        # Por ahora usamos solamente u_k
        u_current = u[:-horizon]

        # ----------------------------------------------------
        # GP input
        # ----------------------------------------------------

        X_run = np.hstack([
            x_current,
            u_current,
        ])

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        delta_run = (
            x_next
            - x_current
        )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        time_current = t[:-horizon]

        dt_run = (
            t[horizon:]
            - t[:-horizon]
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X_list.append(X_run)
        delta_list.append(delta_run)

        x_current_list.append(
            x_current
        )

        x_next_list.append(
            x_next
        )

        time_list.append(
            time_current
        )

        dt_list.append(
            dt_run
        )

        run_list.append(
            np.full(
                len(X_run),
                run_id,
            )
        )

    if not X_list:
        raise RuntimeError(
            f"No valid transitions for horizon={horizon}"
        )

    return {
        "X": np.vstack(X_list),

        "delta": np.vstack(delta_list),

        "x_current": np.vstack(
            x_current_list
        ),

        "x_next": np.vstack(
            x_next_list
        ),

        "time": np.concatenate(
            time_list
        ),

        "dt": np.concatenate(
            dt_list
        ),

        "run_id": np.concatenate(
            run_list
        ),
    }

# ============================================================
# BALANCED SUBSAMPLING BY RUN
# ============================================================

def balanced_subset_indices(
    run_ids,
    max_samples,
):
    """
    Selecciona muestras distribuidas temporalmente dentro
    de cada run.

    De esta forma un run muy largo no domina automáticamente
    el conjunto del GP.
    """

    n_total = len(run_ids)

    if n_total <= max_samples:
        return np.arange(n_total)

    unique_runs = np.unique(
        run_ids
    )

    samples_per_run = max(
        1,
        max_samples // len(unique_runs),
    )

    selected = []

    for run in unique_runs:

        indices = np.where(
            run_ids == run
        )[0]

        if len(indices) <= samples_per_run:

            selected.extend(
                indices.tolist()
            )

        else:

            local = np.linspace(
                0,
                len(indices) - 1,
                samples_per_run,
                dtype=int,
            )

            selected.extend(
                indices[local].tolist()
            )

    selected = np.array(
        sorted(selected),
        dtype=int,
    )

    # En caso de redondeo, respetar límite.
    if len(selected) > max_samples:

        keep = np.linspace(
            0,
            len(selected) - 1,
            max_samples,
            dtype=int,
        )

        selected = selected[keep]

    return selected


# ============================================================
# METRICS
# ============================================================

def evaluate_predictions(
    label,
    delta_true,
    delta_pred,
    sigma,
    x_current,
    x_next_true,
):
    """
    Calcula métricas tanto sobre Δx como sobre x(k+1).
    """

    # --------------------------------------------------------
    # Reconstruct next state
    # --------------------------------------------------------

    x_next_pred = (
        x_current
        + delta_pred
    )

    # Persistence baseline:
    #
    # x_hat(k+1) = x(k)
    #
    x_next_baseline = (
        x_current.copy()
    )

    # --------------------------------------------------------
    # Delta metrics
    # --------------------------------------------------------

    delta_rmse = np.sqrt(
        mean_squared_error(
            delta_true,
            delta_pred,
            multioutput="raw_values",
        )
    )

    delta_std = np.std(
    delta_true,
    axis=0,
    )   

    delta_nrmse = (
        delta_rmse
        / np.maximum(
            delta_std,
            1e-12,
        )
    )

    delta_mae = mean_absolute_error(
        delta_true,
        delta_pred,
        multioutput="raw_values",
    )

    delta_r2 = r2_score(
        delta_true,
        delta_pred,
        multioutput="raw_values",
    )

    # --------------------------------------------------------
    # Next-state metrics
    # --------------------------------------------------------

    next_rmse = np.sqrt(
        mean_squared_error(
            x_next_true,
            x_next_pred,
            multioutput="raw_values",
        )
    )

    next_mae = mean_absolute_error(
        x_next_true,
        x_next_pred,
        multioutput="raw_values",
    )

    next_r2 = r2_score(
        x_next_true,
        x_next_pred,
        multioutput="raw_values",
    )

    # --------------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------------

    persistence_rmse = np.sqrt(
        mean_squared_error(
            x_next_true,
            x_next_baseline,
            multioutput="raw_values",
        )
    )

    improvement = np.where(
        persistence_rmse > 0,
        100.0
        * (
            persistence_rmse
            - next_rmse
        )
        / persistence_rmse,
        np.nan,
    )

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    lower = (
        delta_pred
        - 1.96 * sigma
    )

    upper = (
        delta_pred
        + 1.96 * sigma
    )

    coverage = (
        (
            (delta_true >= lower)
            &
            (delta_true <= upper)
        )
        .mean(axis=0)
        * 100.0
    )

    mean_std = sigma.mean(
        axis=0
    )

    # --------------------------------------------------------
    # Error-vs-uncertainty relation
    # --------------------------------------------------------

    abs_error = np.abs(
        delta_true
        - delta_pred
    )

    error_sigma_spearman = []

    for i in range(
        len(STATE_COLUMNS)
    ):

        rho, _ = spearmanr(
            abs_error[:, i],
            sigma[:, i],
        )

        error_sigma_spearman.append(
            rho
        )

    # --------------------------------------------------------
    # Gaussian negative log likelihood
    # --------------------------------------------------------

    sigma_safe = np.maximum(
        sigma,
        1e-12,
    )

    gaussian_nll = (
        0.5
        * np.log(
            2.0
            * np.pi
            * sigma_safe**2
        )
        +
        0.5
        * (
            (delta_true - delta_pred)
            / sigma_safe
        ) ** 2
    )

    mean_nll = gaussian_nll.mean(
        axis=0
    )

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    metrics = pd.DataFrame({

        "dataset":
            label,

        "state":
            STATE_COLUMNS,

        "delta_rmse":
            delta_rmse,

        "delta_std":
            delta_std,

        "delta_nrmse":
            delta_nrmse,

        "delta_mae":
            delta_mae,

        "delta_r2":
            delta_r2,

        "next_state_rmse":
            next_rmse,

        "next_state_mae":
            next_mae,

        "next_state_r2":
            next_r2,

        "persistence_rmse":
            persistence_rmse,

        "improvement_vs_persistence_pct":
            improvement,

        "mean_predictive_std":
            mean_std,

        "coverage_95_pct":
            coverage,

        "spearman_abs_error_vs_sigma":
            error_sigma_spearman,

        "gaussian_nll":
            mean_nll,
    })

    return (
        metrics,
        x_next_pred,
        x_next_baseline,
    )


# ============================================================
# PER-RUN METRICS
# ============================================================

def evaluate_per_run(
    dataset_label,
    run_ids,
    delta_true,
    delta_pred,
    sigma,
    x_current,
    x_next,
):
    """
    Calcula las mismas métricas para cada run por separado.
    """

    results = []

    for run in np.unique(
        run_ids
    ):

        mask = (
            run_ids == run
        )

        metrics, _, _ = evaluate_predictions(

            f"{dataset_label}_run_{run}",

            delta_true[mask],
            delta_pred[mask],
            sigma[mask],

            x_current[mask],
            x_next[mask],
        )

        metrics.insert(
            1,
            "run_id",
            run,
        )

        results.append(
            metrics
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================
# PLOTS
# ============================================================

def plot_delta_predictions(
    dataset_label,
    time_values,
    run_ids,
    delta_true,
    delta_pred,
    sigma,
    output_dir,
):
    """
    Muestra un único run para evitar conectar gráficamente
    trayectorias independientes.
    """

    unique_runs = np.unique(
        run_ids
    )

    plot_run = unique_runs[0]

    mask = (
        run_ids == plot_run
    )

    indices = np.where(
        mask
    )[0]

    # Limitar visualización.
    if len(indices) > 1500:

        local = np.linspace(
            0,
            len(indices) - 1,
            1500,
            dtype=int,
        )

        indices = indices[
            local
        ]

    t = time_values[
        indices
    ]

    # Tiempo relativo del run.
    t = t - t[0]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        sharex=True,
    )

    axes = axes.ravel()

    for i, ax in enumerate(
        axes
    ):

        real = delta_true[
            indices,
            i
        ]

        pred = delta_pred[
            indices,
            i
        ]

        std = sigma[
            indices,
            i
        ]

        ax.plot(
            t,
            real,
            label="Real Δx",
            linewidth=1.0,
        )

        ax.plot(
            t,
            pred,
            label="GP",
            linewidth=1.0,
        )

        ax.fill_between(
            t,
            pred - 1.96 * std,
            pred + 1.96 * std,
            alpha=0.2,
            label="95% interval",
        )

        ax.axhline(
            0,
            linewidth=0.7,
            linestyle="--",
        )

        ax.set_title(
            f"Δ{STATE_COLUMNS[i]}"
        )

        ax.grid(
            alpha=0.25
        )

    axes[0].legend()

    axes[-2].set_xlabel(
        "Time [s]"
    )

    axes[-1].set_xlabel(
        "Time [s]"
    )

    fig.suptitle(
        f"{dataset_label} — run {plot_run}"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{dataset_label.lower()}_delta_prediction.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("code/opti_test"),
        help=(
            "Directory containing the processed CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gp_results_horizon_sweep"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        data_dir
        / "hoons_all_train_and_val.csv"
    )

    id_test_path = (
        data_dir
        / "hoons_all_test.csv"
    )

    ood_test_path = (
        data_dir
        / "different_tires.csv"
    )

    for path in [
        train_path,
        id_test_path,
        ood_test_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD DATASETS ONCE
    # ========================================================

    train_df = load_dataset(
        train_path
    )

    id_df = load_dataset(
        id_test_path
    )

    ood_df = load_dataset(
        ood_test_path
    )

    # These collect results across all horizons.
    all_global_metrics = []
    all_per_run_metrics = []

    # ========================================================
    # HORIZON SWEEP
    # ========================================================

    for horizon in HORIZONS:

        nominal_dt = (
            horizon
            / BASE_FREQUENCY_HZ
        )

        equivalent_frequency = (
            BASE_FREQUENCY_HZ
            / horizon
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"HORIZON H={horizon}"
        )

        print(
            f"Nominal prediction horizon: "
            f"{nominal_dt:.3f} s "
            f"({nominal_dt * 1000:.0f} ms)"
        )

        print(
            f"Equivalent frequency: "
            f"{equivalent_frequency:.1f} Hz"
        )

        print(
            "=" * 80
        )

        logger.info(
            "Starting horizon H=%d (%.3f s)",
            horizon,
            nominal_dt,
        )

        # ----------------------------------------------------
        # Separate output directory for each H
        # ----------------------------------------------------

        horizon_output_dir = (
            output_dir
            / f"H{horizon:02d}"
        )

        horizon_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # CREATE H-STEP TRANSITIONS
        # ====================================================

        train = create_transitions(
            train_df,
            horizon=horizon,
        )

        id_test = create_transitions(
            id_df,
            horizon=horizon,
        )

        ood_test = create_transitions(
            ood_df,
            horizon=horizon,
        )

        logger.info(
            "H=%d | Train transitions: %d",
            horizon,
            len(train["X"]),
        )

        logger.info(
            "H=%d | ID transitions: %d",
            horizon,
            len(id_test["X"]),
        )

        logger.info(
            "H=%d | OOD transitions: %d",
            horizon,
            len(ood_test["X"]),
        )

        logger.info(
            "H=%d | median actual dt train = %.6f s",
            horizon,
            float(np.median(train["dt"])),
        )

        logger.info(
            "H=%d | median actual dt ID = %.6f s",
            horizon,
            float(np.median(id_test["dt"])),
        )

        logger.info(
            "H=%d | median actual dt OOD = %.6f s",
            horizon,
            float(np.median(ood_test["dt"])),
        )

        # ====================================================
        # SCALE — FIT ONLY ON TRAIN, FOR THIS HORIZON
        # ====================================================

        scaler_X = StandardScaler()

        scaler_delta = StandardScaler()

        X_train_full = (
            scaler_X.fit_transform(
                train["X"]
            )
        )

        delta_train_full = (
            scaler_delta.fit_transform(
                train["delta"]
            )
        )

        X_id_full = (
            scaler_X.transform(
                id_test["X"]
            )
        )

        X_ood_full = (
            scaler_X.transform(
                ood_test["X"]
            )
        )

        joblib.dump(
            scaler_X,
            horizon_output_dir
            / "scaler_X.pkl",
        )

        joblib.dump(
            scaler_delta,
            horizon_output_dir
            / "scaler_delta.pkl",
        )

        # ====================================================
        # BALANCED TRAIN SUBSET
        # ====================================================

        train_idx = balanced_subset_indices(
            train["run_id"],
            MAX_GP_TRAIN_SAMPLES,
        )

        X_train = (
            X_train_full[
                train_idx
            ]
        )

        delta_train = (
            delta_train_full[
                train_idx
            ]
        )

        logger.info(
            "H=%d | GP training samples used: %d",
            horizon,
            len(X_train),
        )

        # ====================================================
        # TEST SUBSETS
        # ====================================================

        id_idx = balanced_subset_indices(
            id_test["run_id"],
            MAX_EVALUATION_SAMPLES,
        )

        ood_idx = balanced_subset_indices(
            ood_test["run_id"],
            MAX_EVALUATION_SAMPLES,
        )

        X_id = X_id_full[
            id_idx
        ]

        X_ood = X_ood_full[
            ood_idx
        ]

        # Keep all metadata aligned with the selected samples.
        id_eval = {
            key: value[id_idx]
            for key, value
            in id_test.items()
            if key != "X"
        }

        ood_eval = {
            key: value[ood_idx]
            for key, value
            in ood_test.items()
            if key != "X"
        }

        # ====================================================
        # TRAIN ONE GP PER OUTPUT
        # ====================================================

        n_inputs = len(
            INPUT_COLUMNS
        )

        n_outputs = len(
            STATE_COLUMNS
        )

        id_pred_scaled = np.zeros(
            (
                len(X_id),
                n_outputs,
            )
        )

        id_sigma_scaled = np.zeros_like(
            id_pred_scaled
        )

        ood_pred_scaled = np.zeros(
            (
                len(X_ood),
                n_outputs,
            )
        )

        ood_sigma_scaled = np.zeros_like(
            ood_pred_scaled
        )

        models = {}

        logger.info(
            "========================================"
        )

        logger.info(
            "Starting GP training for H=%d",
            horizon,
        )

        logger.info(
            "========================================"
        )

        for output_idx, state in enumerate(
            STATE_COLUMNS
        ):

            logger.info(
                "H=%d | Training GP for Δ%s",
                horizon,
                state,
            )

            # ARD RBF:
            # one length scale per input.
            kernel = (

                C(
                    1.0,
                    (1e-3, 1e3),
                )

                *

                RBF(
                    length_scale=np.ones(
                        n_inputs
                    ),
                    length_scale_bounds=(
                        1e-2,
                        1e2,
                    ),
                )

                +

                WhiteKernel(
                    noise_level=1e-3,
                    noise_level_bounds=(
                        1e-6,
                        1e0,
                    ),
                )
            )

            gp = GaussianProcessRegressor(

                kernel=kernel,

                # Small numerical jitter.
                alpha=1e-6,

                # Keep zero restarts for this initial sweep.
                n_restarts_optimizer=0,

                normalize_y=False,

                random_state=RANDOM_STATE,
            )

            start = time.time()

            gp.fit(
                X_train,
                delta_train[:, output_idx],
            )

            logger.info(
                "H=%d | Δ%s training time: %.2f s",
                horizon,
                state,
                time.time() - start,
            )

            logger.info(
                "H=%d | Δ%s optimized kernel: %s",
                horizon,
                state,
                gp.kernel_,
            )

            # -----------------------------------------------
            # ID prediction
            # -----------------------------------------------

            pred, std = gp.predict(
                X_id,
                return_std=True,
            )

            id_pred_scaled[
                :,
                output_idx
            ] = pred

            id_sigma_scaled[
                :,
                output_idx
            ] = std

            # -----------------------------------------------
            # OOD prediction
            # -----------------------------------------------

            pred, std = gp.predict(
                X_ood,
                return_std=True,
            )

            ood_pred_scaled[
                :,
                output_idx
            ] = pred

            ood_sigma_scaled[
                :,
                output_idx
            ] = std

            models[state] = gp

        joblib.dump(
            models,
            horizon_output_dir
            / "gp_models.pkl",
        )

        # ====================================================
        # BACK TO PHYSICAL UNITS
        # ====================================================

        id_delta_pred = (
            scaler_delta.inverse_transform(
                id_pred_scaled
            )
        )

        ood_delta_pred = (
            scaler_delta.inverse_transform(
                ood_pred_scaled
            )
        )

        # Standard deviation:
        # inverse scaling only requires multiplication
        # by the target standard deviation.
        id_sigma = (
            id_sigma_scaled
            * scaler_delta.scale_
        )

        ood_sigma = (
            ood_sigma_scaled
            * scaler_delta.scale_
        )

        # ====================================================
        # EVALUATION — ID
        # ====================================================

        id_metrics, _, _ = evaluate_predictions(

            "ID",

            id_eval["delta"],

            id_delta_pred,

            id_sigma,

            id_eval["x_current"],

            id_eval["x_next"],
        )

        id_metrics.insert(
            1,
            "horizon",
            horizon,
        )

        id_metrics.insert(
            2,
            "horizon_s",
            nominal_dt,
        )

        id_metrics.insert(
            3,
            "equivalent_frequency_hz",
            equivalent_frequency,
        )

        # ====================================================
        # EVALUATION — OOD
        # ====================================================

        ood_metrics, _, _ = evaluate_predictions(

            "OOD",

            ood_eval["delta"],

            ood_delta_pred,

            ood_sigma,

            ood_eval["x_current"],

            ood_eval["x_next"],
        )

        ood_metrics.insert(
            1,
            "horizon",
            horizon,
        )

        ood_metrics.insert(
            2,
            "horizon_s",
            nominal_dt,
        )

        ood_metrics.insert(
            3,
            "equivalent_frequency_hz",
            equivalent_frequency,
        )

        # ====================================================
        # GLOBAL RESULTS FOR THIS HORIZON
        # ====================================================

        global_metrics = pd.concat(
            [
                id_metrics,
                ood_metrics,
            ],
            ignore_index=True,
        )

        global_metrics.to_csv(
            horizon_output_dir
            / "gp_global_metrics.csv",
            index=False,
        )

        all_global_metrics.append(
            global_metrics
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"GLOBAL RESULTS — H={horizon} "
            f"({nominal_dt * 1000:.0f} ms)"
        )

        print(
            "=" * 80
        )

        print(
            global_metrics.to_string(
                index=False
            )
        )

        # ====================================================
        # PER-RUN RESULTS
        # ====================================================

        id_per_run = evaluate_per_run(

            "ID",

            id_eval["run_id"],

            id_eval["delta"],

            id_delta_pred,

            id_sigma,

            id_eval["x_current"],

            id_eval["x_next"],
        )

        ood_per_run = evaluate_per_run(

            "OOD",

            ood_eval["run_id"],

            ood_eval["delta"],

            ood_delta_pred,

            ood_sigma,

            ood_eval["x_current"],

            ood_eval["x_next"],
        )

        per_run_metrics = pd.concat(
            [
                id_per_run,
                ood_per_run,
            ],
            ignore_index=True,
        )

        per_run_metrics.insert(
            1,
            "horizon",
            horizon,
        )

        per_run_metrics.insert(
            2,
            "horizon_s",
            nominal_dt,
        )

        per_run_metrics.insert(
            3,
            "equivalent_frequency_hz",
            equivalent_frequency,
        )

        per_run_metrics.to_csv(
            horizon_output_dir
            / "gp_per_run_metrics.csv",
            index=False,
        )

        all_per_run_metrics.append(
            per_run_metrics
        )

        # ====================================================
        # PLOTS
        # ====================================================

        plot_delta_predictions(

            "ID",

            id_eval["time"],

            id_eval["run_id"],

            id_eval["delta"],

            id_delta_pred,

            id_sigma,

            horizon_output_dir,
        )

        plot_delta_predictions(

            "OOD",

            ood_eval["time"],

            ood_eval["run_id"],

            ood_eval["delta"],

            ood_delta_pred,

            ood_sigma,

            horizon_output_dir,
        )

        # ====================================================
        # ID VS OOD COMPARISON FOR THIS HORIZON
        # ====================================================

        comparison = global_metrics.pivot(
            index="state",
            columns="dataset",
            values=[
                "delta_rmse",
                "delta_std",
                "delta_nrmse",
                "delta_r2",
                "improvement_vs_persistence_pct",
                "mean_predictive_std",
                "coverage_95_pct",
                "spearman_abs_error_vs_sigma",
                "gaussian_nll",
            ],
        )

        comparison.to_csv(
            horizon_output_dir
            / "id_vs_ood_comparison.csv"
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"ID vs OOD — H={horizon}"
        )

        print(
            "-" * 80
        )

        print(
            comparison
        )

        logger.info(
            "Finished horizon H=%d. Results saved in %s",
            horizon,
            horizon_output_dir,
        )

    # ========================================================
    # FINAL COMPARISON ACROSS ALL HORIZONS
    # ========================================================

    horizon_comparison = pd.concat(
        all_global_metrics,
        ignore_index=True,
    )

    horizon_comparison = horizon_comparison.sort_values(
        by=[
            "state",
            "dataset",
            "horizon",
        ]
    ).reset_index(
        drop=True
    )

    horizon_comparison.to_csv(
        output_dir
        / "horizon_comparison.csv",
        index=False,
    )

    per_run_horizon_comparison = pd.concat(
        all_per_run_metrics,
        ignore_index=True,
    )

    per_run_horizon_comparison = (
        per_run_horizon_comparison
        .sort_values(
            by=[
                "state",
                "dataset",
                "run_id",
                "horizon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    per_run_horizon_comparison.to_csv(
        output_dir
        / "horizon_per_run_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compact table for terminal inspection
    # --------------------------------------------------------

    columns_to_show = [
        "dataset",
        "state",
        "horizon",
        "horizon_s",
        "equivalent_frequency_hz",
        "delta_std",
        "delta_rmse",
        "delta_nrmse",
        "delta_r2",
        "improvement_vs_persistence_pct",
        "mean_predictive_std",
        "coverage_95_pct",
        "spearman_abs_error_vs_sigma",
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL HORIZON COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        horizon_comparison[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Additional pivot focused on the main prediction metrics
    # --------------------------------------------------------

    main_metric_pivot = horizon_comparison.pivot_table(
        index=[
            "state",
            "dataset",
        ],
        columns="horizon",
        values=[
            "delta_r2",
            "delta_nrmse",
            "improvement_vs_persistence_pct",
            "mean_predictive_std",
            "coverage_95_pct",
        ],
    )

    main_metric_pivot.to_csv(
        output_dir
        / "horizon_main_metrics_pivot.csv"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Horizon sweep finished."
    )

    logger.info(
        "Main comparison: %s",
        output_dir
        / "horizon_comparison.csv",
    )

    logger.info(
        "Per-run comparison: %s",
        output_dir
        / "horizon_per_run_comparison.csv",
    )


if __name__ == "__main__":
    main()
