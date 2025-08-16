# -*- coding: utf-8 -*-
"""
Proyecto: Modelado de precios de cartas Pokémon + "plataforma" de oportunidades

Incluye:
- Generación de datos demo (sintéticos) con features relevantes.
- EDA: histogramas, boxplots por rareza, matriz de correlación.
- Pipelines con ColumnTransformer (OneHot + StandardScaler) y train/test split (80/20).
- Modelos: LinearRegression (sklearn), OLS (statsmodels), DecisionTree, RandomForest.
- Evaluación: MAE, RMSE, R², MAPE + boxplot de errores y barras de R².
- "Plataforma": predicción vs precio publicado para detectar oportunidades (gap y ratio).
- Salidas: CSVs, figuras PNG/HTML, resumen OLS, mapa Folium.

Requisitos: ver requirements.txt
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import MarkerCluster

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm

# ---------------------------
# Utilidades
# ---------------------------

def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def save_fig(name: str):
    """Guarda la figura actual en figures/<name>.png sin estilos extra."""
    path = os.path.join("figures", name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig] {path}")

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}

# ---------------------------
# 1) Armar/Cargar Base (demo reproducible)
# ---------------------------

def make_synthetic_data(n_cards=220, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    names = [f"Card_{i:03d}" for i in range(n_cards)]
    sets = rng.choice(["DRI", "PAR", "TEF", "TWM", "SVI", "OBF", "PGO", "CRZ"], size=n_cards)
    rarities = rng.choice(
        ["Common", "Uncommon", "Rare", "Ultra Rare", "Secret Rare"],
        p=[0.35, 0.28, 0.20, 0.12, 0.05], size=n_cards
    )
    types_ = rng.choice(
        ["Grass", "Fire", "Water", "Electric", "Psychic", "Fighting", "Metal", "Dark", "Dragon", "Colorless"],
        size=n_cards
    )
    stage = rng.choice(["Basic", "Stage 1", "Stage 2"], p=[0.55, 0.30, 0.15], size=n_cards)
    hp = rng.integers(40, 330, size=n_cards)

    # Stats competitivos (simulados)
    usage_rate = np.clip(rng.normal(0.06, 0.04, size=n_cards), 0, 0.35)  # 0–35%
    win_rate   = np.clip(rng.normal(0.52, 0.05, size=n_cards), 0.35, 0.65) # 35–65%

    # Oferta/stock (aprox. cantidad de listados)
    supply_count = rng.integers(3, 120, size=n_cards)

    # Región y precios (publicados) por mercado
    region = rng.choice(["NA", "EU"], size=n_cards, p=[0.6, 0.4])

    # Señal latente de "valor intrínseco" para generar el target (precio "real")
    rarity_weight = {"Common": 1.0, "Uncommon": 1.15, "Rare": 1.4, "Ultra Rare": 2.1, "Secret Rare": 3.2}
    stage_weight  = {"Basic": 1.0, "Stage 1": 1.05, "Stage 2": 1.15}

    intrinsic_score = (
        10
        + 20 * np.vectorize(rarity_weight.get)(rarities)
        + 0.04 * hp
        + 90  * usage_rate
        + 40  * (win_rate - 0.5)
        - 0.15 * supply_count
        + 5 * np.vectorize(stage_weight.get)(stage)
        + rng.normal(0, 8, size=n_cards)
    )
    usd_price_true = np.clip(intrinsic_score, 0.5, None)

    market_noise = rng.normal(0, 4.5, size=n_cards)
    region_factor = np.where(region == "EU", 0.97, 1.0)  # leve diferencia
    usd_price_published = np.clip(usd_price_true * region_factor + market_noise, 0.2, None)

    eur_fx = 0.9
    eur_price_published = usd_price_published * eur_fx

    df = pd.DataFrame({
        "name": names,
        "set": sets,
        "rarity": rarities,
        "type": types_,
        "stage": stage,
        "hp": hp,
        "usage_rate": usage_rate,
        "win_rate": win_rate,
        "supply_count": supply_count,
        "region": region,
        "usd_price_published": usd_price_published,
        "eur_price_published": eur_price_published,
        "usd_price_true": usd_price_true,
    })

    # Limpieza base
    df.drop_duplicates(subset=["name", "set"], inplace=True)
    df["usage_rate"].fillna(0, inplace=True)
    df["win_rate"].fillna(df["win_rate"].median(), inplace=True)
    df["supply_count"].fillna(df["supply_count"].median(), inplace=True)

    return df

# ---------------------------
# 2) Análisis exploratorio (EDA)
# ---------------------------

def run_eda(df: pd.DataFrame):
    avg_price = df["usd_price_published"].mean()
    print(f"[EDA] Precio promedio publicado (USD): {avg_price:.2f}")
    # Histograma de precios
    plt.figure()
    plt.hist(df["usd_price_published"], bins=30)
    plt.title("Distribución de precios publicados (USD)")
    plt.xlabel("USD"); plt.ylabel("Frecuencia")
    save_fig("01_hist_precios_publicados.png")

    # Boxplot por rareza
    rarity_order = ["Common", "Uncommon", "Rare", "Ultra Rare", "Secret Rare"]
    plt.figure()
    data_to_plot = [df.loc[df["rarity"] == r, "usd_price_published"] for r in rarity_order]
    plt.boxplot(data_to_plot, labels=rarity_order)
    plt.title("Precios publicados por rareza (USD)")
    plt.ylabel("USD")
    save_fig("02_boxplot_precio_por_rareza.png")

    # Correlación numérica
    numeric_cols = ["hp", "usage_rate", "win_rate", "supply_count",
                    "usd_price_published", "usd_price_true"]
    corr = df[numeric_cols].corr()
    plt.figure()
    plt.imshow(corr, interpolation='nearest')
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Matriz de correlación (numérica)")
    plt.colorbar()
    save_fig("03_matriz_correlacion.png")

# ---------------------------
# 3) Train/Test Split + Pipelines
# ---------------------------

def make_splits_and_preprocess(df: pd.DataFrame, target="usd_price_true", seed=123):
    categorical = ["set", "rarity", "type", "stage", "region"]
    numeric = ["hp", "usage_rate", "win_rate", "supply_count", "usd_price_published"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    X = df[categorical + numeric]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return (categorical, numeric, preprocess, X_train, X_test, y_train, y_test)

# ---------------------------
# 4) Modelos
# ---------------------------

def train_models(preprocess, X_train, y_train, X, y, categorical, numeric, seed=123):
    # 4.1 LinearRegression (sklearn)
    linreg_model = Pipeline(steps=[("prep", preprocess), ("model", LinearRegression())])
    linreg_model.fit(X_train, y_train)

    # 4.2 OLS (statsmodels) para estadísticas
    X_ohe_full = pd.get_dummies(X[categorical + numeric], drop_first=True)
    X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(
        X_ohe_full, y, test_size=0.2, random_state=seed
    )
    X_train_sm = sm.add_constant(X_train_ohe)
    X_test_sm = sm.add_constant(X_test_ohe, has_constant='add')
    ols_model = sm.OLS(y_train_ohe, X_train_sm).fit()

    # 4.3 Árbol
    tree_model = Pipeline(steps=[
        ("prep", preprocess),
        ("model", DecisionTreeRegressor(random_state=seed, max_depth=8, min_samples_leaf=5))
    ])
    tree_model.fit(X_train, y_train)

    # 4.4 RandomForest
    rf_model = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=250, random_state=seed, max_depth=12, min_samples_leaf=3, n_jobs=-1
        ))
    ])
    rf_model.fit(X_train, y_train)

    return linreg_model, ols_model, X_test_sm, y_test_ohe, tree_model, rf_model

# ---------------------------
# 5) Evaluación y comparación
# ---------------------------

def compare_models(X_test, y_test, linreg_model, ols_model, X_test_sm, tree_model, rf_model):
    # Predicciones
    y_pred_lin = linreg_model.predict(X_test)
    y_pred_sm  = ols_model.predict(X_test_sm)
    y_pred_tree = tree_model.predict(X_test)
    y_pred_rf   = rf_model.predict(X_test)

    rows = []
    comps = [
        ("Linear (sklearn)", y_pred_lin),
        ("Linear (statsmodels)", y_pred_sm),
        ("Decision Tree", y_pred_tree),
        ("Random Forest", y_pred_rf),
    ]
    for name, preds in comps:
        m = metrics(y_test, preds)
        m["Modelo"] = name
        rows.append(m)

    results_df = pd.DataFrame(rows)[["Modelo","MAE","RMSE","R2","MAPE%"]]
    results_df.sort_values("R2", ascending=False, inplace=True)
    results_df.round(4).to_csv(os.path.join("outputs", "model_results.csv"), index=False)
    print("[out] outputs/model_results.csv")

    # Boxplot de errores absolutos
    plt.figure()
    errors = [np.abs(y_test - preds) for _, preds in comps]
    labels = [name for name, _ in comps]
    plt.boxplot(errors, labels=labels)
    plt.title("Distribución de errores absolutos por modelo")
    plt.ylabel("|Error|")
    plt.xticks(rotation=10, ha="right")
    save_fig("04_boxplot_errores_modelos.png")

    # Barras de R²
    plt.figure()
    plt.bar(results_df["Modelo"], results_df["R2"])
    plt.title("Coeficiente de determinación (R²) por modelo")
    plt.ylabel("R²")
    plt.xticks(rotation=10, ha="right")
    save_fig("05_barras_r2.png")

    # Scatter Predicho vs Real (mejor R²)
    best_name = results_df.iloc[0]["Modelo"]
    best_pred = dict(comps)[best_name]

    plt.figure()
    plt.scatter(y_test, best_pred, alpha=0.7)
    plt.title(f"Predicho vs Real • {best_name}")
    plt.xlabel("Precio real (USD)"); plt.ylabel("Predicción (USD)")
    lims = [0, max(float(np.max(y_test)), float(np.max(best_pred))) * 1.05]
    plt.plot(lims, lims)
    plt.xlim(lims); plt.ylim(lims)
    save_fig("06_scatter_pred_vs_real.png")

    return results_df, best_name

# ---------------------------
# 6) “Plataforma” de oportunidades
# ---------------------------

def build_opportunities(df, best_name, linreg_model, tree_model, rf_model,
                        categorical, numeric):
    # Elegimos pipeline servible
    best_pipeline = {
        "Linear (sklearn)": linreg_model,
        "Linear (statsmodels)": None,  # si ganara, usamos RF por robustez
        "Decision Tree": tree_model,
        "Random Forest": rf_model,
    }.get(best_name)

    if best_pipeline is None:
        best_pipeline = rf_model
        best_name_for_platform = "Random Forest"
    else:
        best_name_for_platform = best_name

    pred_all = best_pipeline.predict(df[categorical + numeric])
    df = df.copy()
    df["predicted_usd"] = pred_all

    df["opportunity_diff"]  = df["predicted_usd"] - df["usd_price_published"]
    df["opportunity_ratio"] = df["opportunity_diff"] / np.clip(df["usd_price_published"], 1e-9, None)

    # Heurística de filtro (ajustable)
    opps = df[
        (df["opportunity_ratio"] >= 0.2) &
        (df["usage_rate"] >= 0.05) &
        (df["win_rate"] >= 0.50) &
        (df["supply_count"] <= 35)
    ].copy()
    opps.sort_values(["opportunity_ratio","usage_rate","win_rate"],
                     ascending=[False, False, False], inplace=True)
    opps_top = opps.head(25).reset_index(drop=True)

    opps_cols = ["name","set","rarity","type","usage_rate","win_rate","supply_count",
                 "usd_price_published","predicted_usd","opportunity_diff","opportunity_ratio","region"]
    opps_top[opps_cols].round(4).to_csv(os.path.join("outputs", "top_opportunities_demo.csv"), index=False)
    print("[out] outputs/top_opportunities_demo.csv")

    # Scatter interactivo: Predicho vs Publicado (Plotly)
    fig = px.scatter(
        df, x="usd_price_published", y="predicted_usd",
        hover_data=["name","set","rarity","usage_rate","win_rate"],
        title=f"Predicción vs Precio publicado • Modelo: {best_name_for_platform}"
    )
    # Línea y=x
    x1 = float(df["usd_price_published"].max())
    y1 = float(df["predicted_usd"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=x1, y1=y1)
    fig_path = os.path.join("figures", "07_scatter_plotly_pred_vs_pub.html")
    fig.write_html(fig_path)
    print(f"[fig] {fig_path}")

    # Mapa Folium (NA/EU) con oportunidades
    coords = {"NA": (39.5, -98.35), "EU": (54.5, 15.0)}  # centroides aproximados
    m = folium.Map(location=[45, -10], zoom_start=3)
    cluster = MarkerCluster().add_to(m)

    rng = np.random.default_rng(12345)
    for _, row in opps_top.iterrows():
        lat, lon = coords.get(row["region"], (45, 0))
        popup = folium.Popup(html=(
            f"<b>{row['name']}</b> ({row['set']})<br>"
            f"Rarity: {row['rarity']}<br>"
            f"Uso: {row['usage_rate']:.1%} | Win: {row['win_rate']:.1%}<br>"
            f"Stock: {int(row['supply_count'])}<br>"
            f"Publicado: ${row['usd_price_published']:.2f} | Pred: ${row['predicted_usd']:.2f}<br>"
            f"Dif: ${row['opportunity_diff']:.2f} ({row['opportunity_ratio']:.1%})<br>"
            f"Mercado: {row['region']}"
        ), max_width=300)
        folium.CircleMarker(
            location=(lat + rng.normal(0, 0.3), lon + rng.normal(0, 0.3)),
            radius=7,
            popup=popup
        ).add_to(cluster)

    map_path = os.path.join("outputs", "opportunities_map.html")
    m.save(map_path)
    print(f"[map] {map_path}")

    return df, opps_top, best_name_for_platform

# ---------------------------
# 7) Guardar resumen OLS y reporte JSON
# ---------------------------

def save_ols(ols_model):
    txt = ols_model.summary().as_text()
    path = os.path.join("outputs", "ols_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[out] {path}")

def save_report(results_df, best_name, best_name_for_platform):
    out = {
        "results": results_df.to_dict(orient="records"),
        "best_by_r2": best_name,
        "platform_model": best_name_for_platform,
    }
    path = os.path.join("outputs", "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[out] {path}")

# ---------------------------
# 8) Main
# ---------------------------

def main():
    ensure_dirs()
    # Datos (demo)
    df = make_synthetic_data(n_cards=220, seed=42)

    # EDA
    run_eda(df)

    # Split + preprocess
    categorical, numeric, preprocess, X_train, X_test, y_train, y_test = make_splits_and_preprocess(df)

    # Entrenar
    linreg_model, ols_model, X_test_sm, y_test_ohe, tree_model, rf_model = train_models(
        preprocess, X_train, y_train, df, df["usd_price_true"].values, categorical, numeric
    )

    # Evaluación
    results_df, best_name = compare_models(
        X_test, y_test, linreg_model, ols_model, X_test_sm, tree_model, rf_model
    )
    print("\n[Resultados]\n", results_df.round(4))

    # Guardar OLS summary
    save_ols(ols_model)

    # Plataforma
    full_df, opps_top, best_name_for_platform = build_opportunities(
        df, best_name, linreg_model, tree_model, rf_model, categorical, numeric
    )

    # Resumen JSON
    save_report(results_df, best_name, best_name_for_platform)

    # Extras: CSVs base
    df.to_csv(os.path.join("outputs", "dataset_demo.csv"), index=False)
    print("[out] outputs/dataset_demo.csv")
    full_df.to_csv(os.path.join("outputs", "dataset_with_predictions.csv"), index=False)
    print("[out] outputs/dataset_with_predictions.csv")

if __name__ == "__main__":
    main()
