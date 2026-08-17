# Explainable and Uncertainty-Aware Dynamic Feasibility Prediction for Adaptive Speed Scaling in Autonomous Racing (tentativo)

## Partes:

### Control base:

Pure Pursuit recibe trayectoria y devuelve:
 $$
 u_{nom} = [\delta , V_x]
 $$

### Predictor dinaámica

Red Neural entrenada para predecir:

$$
x(t+1:t+H) + I(t:t+H) + C(t:t+H) = f_{NN}(x(t:t-N),u(t:t-N)) 
$$

Donde:
- $I(t:t+H)$: Incertidumbre para cada rolout
- $C(t:t+H)$: Matriz de Covarianza de las variables 

### XAI:

Usado para caracterizar las predicciones (SHAP tentativamenet):

```
¿Qué estados/acciones pasadas provocaron esta desviación lateral futura?
```

(Quizás una idea futura sea entrenar un modelo para predecir lo que dice SHAP)

### Supervisor:

Toma las informaciones y decide:

$$ \Delta Vx  = f(infos) $$

### CBF (opcional):
Filtro por si el sistema falla

### OOD (futuro):

Determinar si estoy operando dentro/fuera del dominio

### DAgger (opcional):

Reentrenar con lo observado en OOD