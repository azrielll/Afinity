"""Pipeline data mining (identik dengan notebook): clustering K-Means + forecasting 2026-2029."""
import json
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             mean_absolute_error, mean_squared_error, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.tsa.api import Holt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
BASE = ROOT / "data"
FITUR = ["Penduduk_Akses_Internet", "Penduduk_Telepon_Seluler", "Penduduk_Menggunakan_Komputer",
         "Rumah_Tangga_Komputer", "Rumah_Tangga_Akses_Internet"]
NAMA = {"Penduduk_Akses_Internet": "Penduduk akses internet",
        "Penduduk_Telepon_Seluler": "Penduduk telepon seluler",
        "Penduduk_Menggunakan_Komputer": "Penduduk pakai komputer",
        "Rumah_Tangga_Komputer": "Rumah tangga punya komputer",
        "Rumah_Tangga_Akses_Internet": "Rumah tangga akses internet"}
NAMA_GEO = {"Kep. Bangka Belitung": "Kepulauan Bangka Belitung", "Kep. Riau": "Kepulauan Riau",
            "DI Yogyakarta": "Daerah Istimewa Yogyakarta"}
LABELS = ["Digital Rendah", "Digital Sedang", "Digital Tinggi"]
MODELS = ["Linear Regression", "Polynomial Regression", "Holt Exponential Smoothing"]


@st.cache_data
def load_master():
    return pd.read_csv(BASE / "dataset_BPS_telekomunikasi_2022_2025_lengkap.csv")


@st.cache_data
def load_geo():
    return json.load(open(BASE / "provinsi_38.geojson", encoding="utf-8"))


def _muat_model_notebook():
    """Cari scaler.joblib & kmeans_model.joblib hasil notebook (folder models/ atau root)."""
    for folder in (ROOT / "models", ROOT):
        try:
            sc = joblib.load(folder / "scaler.joblib")
            km = joblib.load(folder / "kmeans_model.joblib")
            if sc.n_features_in_ == len(FITUR) and km.n_clusters == 3:
                return sc, km
        except Exception:
            continue
    return None, None


@st.cache_data
def clustering():
    d = load_master().query("Tahun == 2025").reset_index(drop=True).copy()
    sc, km = _muat_model_notebook()
    sumber = "Model dimuat dari notebook (file joblib)"
    if sc is None:
        sc = StandardScaler().fit(d[FITUR].values)
        km = KMeans(3, random_state=42, n_init=10).fit(sc.transform(d[FITUR].values))
        sumber = "Model dilatih ulang dengan konfigurasi notebook"
    X = sc.transform(d[FITUR].values)
    rows = []
    for k in range(2, 11):
        km_k = KMeans(k, random_state=42, n_init=10)
        l = km_k.fit_predict(X)
        rows.append({"K": k, "Inertia": km_k.inertia_, "Silhouette": silhouette_score(X, l),
                     "Davies-Bouldin": davies_bouldin_score(X, l),
                     "Calinski-Harabasz": calinski_harabasz_score(X, l),
                     "Ukuran Klaster Min": int(np.bincount(l).min())})
    raw = km.predict(X)
    urut = d.groupby(raw)["Penduduk_Akses_Internet"].mean().sort_values().index
    cmap = {c: LABELS[i] for i, c in enumerate(urut)}
    d["Kategori"] = pd.Series(raw).map(cmap).values
    d["Cluster_KMeans"] = raw
    d["Provinsi_Geo"] = d["Provinsi"].replace(NAMA_GEO)
    agg = AgglomerativeClustering(3).fit_predict(X)
    gmm = GaussianMixture(3, random_state=42).fit_predict(X)
    algo = pd.DataFrame([{"Algoritma": n, "Silhouette": silhouette_score(X, l),
                          "Davies-Bouldin": davies_bouldin_score(X, l),
                          "Calinski-Harabasz": calinski_harabasz_score(X, l),
                          "ARI vs K-Means": adjusted_rand_score(raw, l)}
                         for n, l in [("K-Means", raw), ("Agglomerative", agg), ("Gaussian Mixture", gmm)]])
    pca = PCA(2).fit(X)
    c = pca.transform(X)
    d["PCA1"], d["PCA2"] = c[:, 0], c[:, 1]
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=FITUR)
    cent["Kategori"] = [cmap[i] for i in range(3)]
    return dict(sumber=sumber, d=d, X=X, scaler=sc, km=km, cmap=cmap, k_eval=pd.DataFrame(rows), algo=algo,
                var=pca.explained_variance_ratio_, centroid=cent.set_index("Kategori").reindex(LABELS))


def prediksi_kategori(M, values):
    """values: array (n,5) dalam skala asli -> list kategori."""
    raw = M["km"].predict(M["scaler"].transform(np.asarray(values, float)))
    return [M["cmap"][r] for r in raw]


def _fit_pred(m, x, y, xf):
    x, xf = np.asarray(x).reshape(-1, 1), np.asarray(xf).reshape(-1, 1)
    if m == "Linear Regression":
        p = LinearRegression().fit(x, y).predict(xf)
    elif m == "Polynomial Regression":
        pf = PolynomialFeatures(2)
        p = LinearRegression().fit(pf.fit_transform(x), y).predict(pf.transform(xf))
    else:
        try:
            p = Holt(y, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2).forecast(len(xf))
        except Exception:
            p = LinearRegression().fit(x, y).predict(xf)
    return np.clip(p, 0, 100)


@st.cache_data
def forecasting():
    df = load_master()
    ok = df.groupby("Provinsi")["Penduduk_Akses_Internet"].count().loc[lambda s: s == 4].index
    f = df[df["Provinsi"].isin(ok)][["Provinsi", "Tahun"] + FITUR]
    provs = f["Provinsi"].unique()
    ev, best = [], {}
    for ind in FITUR:
        sc_ = []
        for m in MODELS:
            yt, yp = [], []
            for p in provs:
                dp = f[f.Provinsi == p].sort_values("Tahun")
                tr, te = dp[dp.Tahun <= 2024], dp[dp.Tahun == 2025]
                yp.append(_fit_pred(m, tr.Tahun, tr[ind].values, te.Tahun)[0])
                yt.append(te[ind].values[0])
            yt, yp = np.array(yt), np.array(yp)
            r = {"Indikator": ind, "Model": m, "MAE": mean_absolute_error(yt, yp),
                 "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
                 "MAPE (%)": float(np.mean(np.abs((yt - yp) / yt)) * 100)}
            ev.append(r)
            sc_.append((r["MAPE (%)"], m))
        best[ind] = min(sc_)[1]
    fut = np.array([2026, 2027, 2028, 2029])
    rec = []
    for p in provs:
        dp = f[f.Provinsi == p].sort_values("Tahun")
        for ind in FITUR:
            for _, r in dp.iterrows():
                rec.append((p, int(r.Tahun), ind, float(r[ind]), "Aktual"))
            for y, v in zip(fut, _fit_pred(best[ind], dp.Tahun, dp[ind].values, fut)):
                rec.append((p, int(y), ind, float(v), "Forecast"))
    long = pd.DataFrame(rec, columns=["Provinsi", "Tahun", "Indikator", "Nilai", "Tipe_Data"])
    return pd.DataFrame(ev), best, long
