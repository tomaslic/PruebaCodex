# Pokémon Card Pricing ML + Oportunidades

Pipeline end-to-end para estimar precio "justo" de cartas Pokémon y detectar oportunidades comparando vs. precio publicado (estilo Limitless/TCGplayer).

## Contenido
- Datos **demo** sintéticos (reproducibles) con features: rareza, uso en mazos, win rate, HP, oferta (listados), región, precio publicado.
- EDA: histogramas, boxplots por rareza, matriz de correlación.
- Modelado: LinearRegression (sklearn), OLS (statsmodels), DecisionTree, RandomForest.
- Evaluación: MAE, RMSE, R², MAPE + boxplot de errores y barras de R².
- “Plataforma”: predicciones para todas las cartas, `opportunity_diff` y `opportunity_ratio`, top 25 y mapa Folium.
- Artefactos exportados en `figures/` y `outputs/`.

## Requisitos
```bash
pip install -r requirements.txt
