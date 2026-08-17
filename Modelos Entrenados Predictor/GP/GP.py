import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import preprocessing
import joblib
import logging
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# #Join the csv of the ds
# # 1. Define la ruta a la carpeta donde están tus CSVs
# carpeta = Path("../DS/Train")

# # 2. Obtén la lista de todos los archivos .csv
# archivos_csv = sorted(carpeta.glob("*.txt"))

# # 3. Lee cada CSV y conéctalos en un único DataFrame
# df_lista = [pd.read_csv(f) for f in archivos_csv]
# df_completo = pd.concat(df_lista, ignore_index=True)

# # 4. Guarda el resultado en un archivo final
# df_completo.to_csv("dataset_completo.csv", index=False)

# print(
#     f"Proceso finalizado: {len(archivos_csv)} archivos unidos ({len(df_completo)} filas)."
# )

#Loading DS
df = pd.read_csv('dataset_completo.csv') #dataset gigante (se necesita hacer un resampling para llevarlo de 2000Hz (original) a 100Hz, pe)


# ============================================================
# RESAMPLING
# ============================================================

DT_ORIGINAL = 0.0005   # 2000 Hz
DT_NEW = 0.05         # 20 Hz

# GaussianProcessRegressor de scikit-learn es un GP exacto: necesita una
# matriz de covarianza N x N. Por tanto no se puede entrenar con millones de
# muestras. Este límite mantiene el entrenamiento y la predicción manejables.
MAX_GP_TRAIN_SAMPLES = 8000
MAX_EVALUATION_SAMPLES = 20000

step = int(DT_NEW / DT_ORIGINAL)

df_resampled = df.iloc[::step].copy()
df_resampled.reset_index(drop=True, inplace=True)

print(f"Original samples:   {len(df)}")
print(f"Resampled samples:  {len(df_resampled)}")
print(f"Original dt:        {DT_ORIGINAL} s")
print(f"New dt:             {DT_NEW} s")
print(f"New frequency:      {1 / DT_NEW} Hz")

#Inspect the DS
print(df_resampled.head())
print(df_resampled.columns)
print(df_resampled.dtypes)
print(df_resampled.isnull().sum())
df_resampled.dropna(inplace=True) #Clean the NaN data

state_columns = [    
    "Vy",
    "AVz",
    "Beta",
    "Ax",
    "Ay",
    "AVx",
    "Roll"
]

X = df_resampled[state_columns]
y = df_resampled[state_columns].shift(-1)

valid = y.notna().all(axis=1) #Se elimina el ultimo elemento de x (ya que x(fin)->y(fin+1 = NULL))

X = X.loc[valid]
y = y.loc[valid]
time_values = df_resampled.loc[valid, "time"].to_numpy()

# Train/test split temporal: evita que muestras consecutivas aparezcan a la
# vez en train y test. Si el CSV concatena trayectorias independientes,
# conviene hacer esta separación por identificador de trayectoria.
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X.values,
    y.values,
    test_size=0.10,
    shuffle=False,
)

# Ajustar los escaladores únicamente con train evita filtrar información del
# conjunto de evaluación al entrenamiento.
scaler_X = preprocessing.StandardScaler()
scaler_Y = preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
y_train = scaler_Y.fit_transform(y_train_raw)
X_test = scaler_X.transform(X_test_raw)
y_test = scaler_Y.transform(y_test_raw)
time_test = time_values[-len(X_test):]

# Usar esto luego para las predicciones
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_Y, "scaler_Y.pkl")


def evenly_spaced_subset(X_data, y_data, max_samples):
    """Devuelve como máximo ``max_samples`` muestras repartidas en el tiempo."""
    if len(X_data) <= max_samples:
        indices = np.arange(len(X_data))
        return X_data, y_data, indices

    indices = np.linspace(0, len(X_data) - 1, max_samples, dtype=int)
    return X_data[indices], y_data[indices], indices


X_train, y_train, _ = evenly_spaced_subset(
    X_train, y_train, MAX_GP_TRAIN_SAMPLES
)
X_test, y_test, test_indices = evenly_spaced_subset(
    X_test, y_test, MAX_EVALUATION_SAMPLES
)
time_test = time_test[test_indices]


# kernel = C(1.0) * RBF(length_scale=1.0)
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# gp.fit(X_train, y_train)

# # Predicción con desviación estándar (incertidumbre)
# y_pred, sigma = gp.predict(X_test, return_std=True)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# GP
# ============================================================

kernel = C(1.0) * RBF(length_scale=1.0)

logger.info("========================================")
logger.info("Starting Gaussian Process training")
logger.info("========================================")

logger.info("Initial kernel: %s", kernel)
logger.info("X_train shape: %s", X_train.shape)
logger.info("y_train shape: %s", y_train.shape)

start_time = time.time()

gp = GaussianProcessRegressor(
    kernel=kernel,
    # Cada reinicio vuelve a evaluar matrices N x N; primero comprueba el
    # ajuste con cero reinicios y auméntalo solo si el tiempo lo permite.
    n_restarts_optimizer=0,
    random_state=42,
)

logger.info("Fitting GP...")

gp.fit(X_train, y_train)

training_time = time.time() - start_time

logger.info("GP training finished")
logger.info("Training time: %.3f s", training_time)

logger.info("Optimized kernel: %s", gp.kernel_)
logger.info("Log marginal likelihood: %.6f",
            gp.log_marginal_likelihood(gp.kernel_.theta))


# ============================================================
# PREDICTION
# ============================================================

logger.info("Starting prediction...")

start_time = time.time()

y_pred, sigma = gp.predict(
    X_test,
    return_std=True
)

prediction_time = time.time() - start_time

logger.info("Prediction finished")
logger.info("Prediction time: %.6f s", prediction_time)

logger.info("y_pred shape: %s", y_pred.shape)
logger.info("sigma shape: %s", sigma.shape)

logger.info(
    "Prediction statistics | min=%.4f | max=%.4f | mean=%.4f",
    y_pred.min(),
    y_pred.max(),
    y_pred.mean()
)

logger.info(
    "Uncertainty statistics | min=%.4f | max=%.4f | mean=%.4f",
    sigma.min(),
    sigma.max(),
    sigma.mean()
)


# ============================================================
# EVALUATION AND VISUALIZATION (physical units)
# ============================================================

y_pred_real = scaler_Y.inverse_transform(y_pred)
y_test_real = scaler_Y.inverse_transform(y_test)
# La desviación estándar se transforma multiplicando por la escala de cada
# variable; no se aplica inverse_transform porque no es una observación.
sigma_real = sigma * scaler_Y.scale_

metrics_df = pd.DataFrame({
    "state": state_columns,
    "rmse": np.sqrt(mean_squared_error(y_test_real, y_pred_real,
                                        multioutput="raw_values")),
    "mae": mean_absolute_error(y_test_real, y_pred_real,
                                multioutput="raw_values"),
    "r2": r2_score(y_test_real, y_pred_real, multioutput="raw_values"),
    "mean_predictive_std": sigma_real.mean(axis=0),
})
metrics_df.to_csv("gp_metrics.csv", index=False)

logger.info("Metrics in physical units:\n%s", metrics_df.to_string(index=False))

# Una selección visual de puntos evita figuras ilegibles si se eleva el límite
# de evaluación. Las métricas anteriores siempre usan todas las muestras.
plot_indices = np.linspace(0, len(time_test) - 1,
                           min(1_000, len(time_test)), dtype=int)
fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
axes = axes.ravel()

for column, ax in enumerate(axes[:len(state_columns)]):
    name = state_columns[column]
    ax.plot(time_test[plot_indices], y_test_real[plot_indices, column],
            color="black", linewidth=1.0, label="Real")
    ax.plot(time_test[plot_indices], y_pred_real[plot_indices, column],
            color="tab:blue", linewidth=1.0, label="Predicción GP")
    ax.fill_between(
        time_test[plot_indices],
        y_pred_real[plot_indices, column] - 1.96 * sigma_real[plot_indices, column],
        y_pred_real[plot_indices, column] + 1.96 * sigma_real[plot_indices, column],
        color="tab:blue", alpha=0.18, label="IC 95 %",
    )
    ax.set_title(f"{name} | RMSE={metrics_df.loc[column, 'rmse']:.4g}, "
                 f"R²={metrics_df.loc[column, 'r2']:.3f}")
    ax.set_ylabel(name)
    ax.grid(alpha=0.25)

axes[len(state_columns)].set_visible(False)
axes[0].legend(loc="best")
for ax in axes[-2:]:
    if ax.get_visible():
        ax.set_xlabel("Tiempo [s]")
fig.suptitle("GP: predicción a un paso e intervalo de confianza", fontsize=16)
fig.tight_layout()
fig.savefig("gp_evaluation.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(metrics_df["state"], metrics_df["rmse"], color="tab:red")
axes[0].set_title("RMSE por variable")
axes[0].set_ylabel("Unidades físicas")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(metrics_df["state"], metrics_df["r2"], color="tab:green")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("R² por variable")
axes[1].set_ylabel("R²")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("gp_error_summary.png", dpi=180, bbox_inches="tight")
plt.close(fig)

logger.info("Saved gp_metrics.csv, gp_evaluation.png and gp_error_summary.png")
