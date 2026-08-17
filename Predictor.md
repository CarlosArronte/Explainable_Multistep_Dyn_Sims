# Matriz de seleccion del predictor

Se evaluaran la stecnologias que pueden ser usadas para este predictor

## Entrada 

Se parte de:

$$ 
x_t = [\psi, V_x, V_y, r...] 
$$

y del historial:

$$
X_{t-L:t}
$$
$$
U_{t-L:t}
$$

donde:
$$
u_t = [V_x^{PP}, \delta^{PP}]
$$

## Salida

El Predictor entrega:

$$
\hat{X}_{t+1:t+H},  \Sigma_{t+1:t+H}
$$

Donde $\Sigma$ es la incertidumbre de prediccion


## Tecnologías candidatas

5-Excelente  
.  
.  
.  
1-Poco Recomendable

| Predictor                      | Multistep | Incertidumbre |  SHAP | Tiempo real | Datos | Interpretabilidad |   OOD | Publicación |
| ------------------------------ | --------: | ------------: | ----: | ----------: | ----: | ----------------: | ----: | ----------: |
| **GP**                         |         4 |         **5** | **5** |           2 | **5** |             **5** |     3 |           4 |
| **Prob. NN**                   |         4 |             4 |     4 |           4 |     3 |                 2 |     3 |           4 |
| **Deep Ensemble**              |     **5** |         **5** |     4 |           3 |     2 |                 2 | **5** |       **5** |
| **LSTM/GRU probabilístico**    |     **5** |             4 |     4 |       **5** |     3 |                 2 |     3 |       **5** |
| **Transformer probabilístico** |     **5** |             4 |     3 |           2 |     1 |                 1 |     4 |           5 |
| **Neural State-Space**         |     **5** |             4 |     4 |           4 |     3 |             **4** |     4 |       **5** |
| **Physics + Neural**           |     **5** |         **5** |     4 |           4 | **4** |             **5** | **5** |       **5** |


**Conclusiones de la tabla:**

- Es interesante usar GP como baseline (no como opcion principal debido al coste computacional)
- Prob NN es buena pero puede dar incertidumbre epistemicamente incorrecta si se trabaja OOD (interesante usar como baseline para comparar el efecto del OOD)
- Deep Ensembles: muy interesante (considerar usar)
- GRU o LSTM (probar con GRU)
- Transformer: Usar solamente si necesitamos trabajar con horizontes demasiado largos.
- Neural State Space: Muy interesante (rollout sobre f(x,u)) (incertidumbre?)
- Physics + Neural: Opcion 1 (Neural aprende el error del modelo y physics puede ser un modelo simple)


## Métricas de evaluacion

- RMSE (x_{t+1})
- RMSE (H): Calcula la propagacion de incertidumbre en los rollouts

**Tecnicas relacionadas (estudiar)**:   
- NLL;
- coverage;
- calibration curves;
- prediction interval coverage probability;
- CRPS, si utilizamos predicciones probabilísticas adecuadas.