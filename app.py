"""
Dashboard Pemetaan Tipologi & Proyeksi Kesenjangan Digital Antarprovinsi Indonesia
Tim Afinity - Lomba Data Mining.   Jalankan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as P
from pipeline import FITUR, LABELS, NAMA

st.set_page_config(page_title="Afinity", page_icon="logo.png", layout="wide")
WARNA = {"Digital Rendah": "#c0392b", "Digital Sedang": "#2c6fad", "Digital Tinggi": "#27714b"}
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Reset & base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header title style — clean text header without container shape */
.hero2 {
  padding: 0 0 16px 0;
  margin-bottom: 20px;
  border-bottom: 1px solid #e5e7eb;
}
.hero2 h1 {
  color: #111827;
  margin: 0 0 6px;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.3px;
  line-height: 1.3;
}
.hero2 p { margin: 0 0 6px; color: #4b5563; font-size: .95rem; }
.hero2 small { color: #6b7280; font-size: .8rem; font-weight: 500; }

/* KPI — number-first, label below */
.kpi {
  border-top: 2px solid #e5e7eb;
  padding: 14px 4px 10px;
  background: transparent;
}
.kpi b { font-size: 1.85rem; display: block; font-weight: 700; color: #111827; line-height: 1; }
.kpi span { opacity: .6; font-size: .78rem; display: block; margin-top: 4px; }

/* Section header — plain, no border gimmicks */
.sec {
  font-size: 1rem;
  font-weight: 600;
  color: #111827;
  margin: 28px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid #e5e7eb;
  letter-spacing: -.1px;
}

/* Cluster cards — accent line left only, no card container box */
.card {
  border: none;
  border-left: 3px solid;
  padding: 4px 0 8px 12px;
  margin-bottom: 16px;
  background: transparent;
}
.card h4 { margin: 0 0 2px; font-size: .95rem; font-weight: 600; }
.card .big { font-size: 1.5rem; font-weight: 700; color: #111827; }
.small { font-size: .82rem; color: #374151; }
.muted { color: #6b7280; margin-top: 4px; font-size: .78rem; line-height: 1.4; }

/* CRISP-DM steps — clean top-line list without box shapes */
.steps { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
.step {
  border: none;
  border-top: 2px solid #111827;
  padding: 8px 0 0 0;
  background: transparent;
}
.step i {
  display: inline-block;
  font-weight: 700;
  font-size: .82rem;
  color: #111827;
  font-style: normal;
}
.step b { display: block; margin: 4px 0 2px; font-size: .85rem; color: #111827; }
.step span { font-size: .75rem; color: #6b7280; }

/* Guide items — clean grid without card container boxes */
.guide { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px 12px; }
.g {
  border: none;
  border-top: 1px solid #e5e7eb;
  padding: 8px 0 0 0;
  background: transparent;
}
.g b { display: block; font-size: .85rem; color: #111827; margin-bottom: 2px; }
.g span { font-size: .78rem; color: #6b7280; }

</style>""", unsafe_allow_html=True)

M = P.clustering()
D, GEO = M["d"], P.load_geo()
EV, BEST, LONG = P.forecasting()
MASTER = P.load_master()


def kpi(col, nilai, ket):
    col.markdown(f'<div class="kpi"><b>{nilai}</b><span>{ket}</span></div>', unsafe_allow_html=True)


def hero(judul, sub):
    st.markdown(f'<div class="hero2"><h1>{judul}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def _layout(fig, judul, h=540):
    fig.update_layout(map=dict(style="white-bg", center=dict(lat=-2.5, lon=118), zoom=3.4),
                      title=dict(text=judul, x=0.5), margin=dict(l=0, r=0, t=50, b=0), height=h,
                      legend=dict(title="", x=0.99, y=0.99, xanchor="right", bgcolor="rgba(255,255,255,.85)",
                                  font=dict(color="#222")))
    return fig


def _abu(fig, sisa):
    if len(sisa):
        a = px.choropleth_map(sisa.assign(v=1), geojson=GEO, locations="Provinsi_Geo",
                              featureidkey="properties.PROVINSI", color="v")
        a.update_traces(coloraxis=None, colorscale=[[0, "#d1d5db"], [1, "#d1d5db"]], showscale=False,
                        hoverinfo="skip", showlegend=False, marker_line_color="white", marker_line_width=.6)
        for t in a.data:
            fig.add_trace(t)


def peta_kategori(df, judul, tampil=LABELS):
    hd = {"Provinsi_Geo": False, "Kategori": True} | {f: ":.2f" for f in FITUR if f in df}
    fig = px.choropleth_map(df[df.Kategori.isin(tampil)], geojson=GEO, locations="Provinsi_Geo",
                            featureidkey="properties.PROVINSI", color="Kategori", color_discrete_map=WARNA,
                            category_orders={"Kategori": LABELS}, hover_name="Provinsi", hover_data=hd,
                            labels=NAMA)
    fig.update_traces(marker_line_color="white", marker_line_width=.6, selector=dict(type="choroplethmap"))
    _abu(fig, df[~df.Kategori.isin(tampil)])
    return _layout(fig, judul)


# ============================================================ HALAMAN
def p_beranda():
    st.markdown("""<div class="hero2">
<h1>Pemetaan Tipologi dan Proyeksi Kesenjangan Digital Antarprovinsi Indonesia</h1>
<p>Mengelompokkan 38 provinsi berdasarkan 5 indikator TIK BPS tahun 2025, lalu memproyeksikan
tren adopsi digital hingga 2029.</p></div>""",
                unsafe_allow_html=True)

    ke = M["k_eval"].set_index("K")
    mape = EV["MAPE (%)"].min()
    c = st.columns(6)
    for col, (n, k) in zip(c, [("38", "provinsi dianalisis"), ("5", "indikator TIK"), ("3", "klaster (K optimum)"),
                               (f"{ke.Silhouette[3]:.2f}", "Silhouette Score"), ("4 tahun", "data historis 2022-2025"),
                               (f"{mape:.2f}%", "MAPE forecasting terbaik")]):
        kpi(col, n, k)

    st.markdown('<div class="sec">Peta dan ringkasan klaster</div>', unsafe_allow_html=True)
    a, b = st.columns([1.7, 1])
    a.plotly_chart(peta_kategori(D, "Peta Tipologi Kesenjangan Digital Provinsi (2025)"), width="stretch")
    for lab in LABELS[::-1]:
        s = D[D.Kategori == lab]
        nama = ", ".join(s.Provinsi) if len(s) <= 6 else f"{len(s)} provinsi, antara lain {', '.join(s.Provinsi[:3])}"
        m1 = round(s.Penduduk_Akses_Internet.mean(), 1)
        m2 = round(s.Rumah_Tangga_Akses_Internet.mean(), 1)
        b.markdown(f"""<div class="card" style="border-left-color:{WARNA[lab]}">
<h4 style="color:{WARNA[lab]}">{lab}</h4>
<div class="big">{len(s)} provinsi</div>
<div class="small">Akses internet {m1}% &nbsp;|&nbsp; Rumah tangga akses internet {m2}%</div>
<div class="small muted">{nama}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Kesenjangan dalam angka</div>', unsafe_allow_html=True)
    hi, lo = D.loc[D.Penduduk_Akses_Internet.idxmax()], D.loc[D.Penduduk_Akses_Internet.idxmin()]
    c = st.columns(4)
    kpi(c[0], f"{hi.Penduduk_Akses_Internet:.1f}%", f"akses internet tertinggi: {hi.Provinsi}")
    kpi(c[1], f"{lo.Penduduk_Akses_Internet:.1f}%", f"akses internet terendah: {lo.Provinsi}")
    kpi(c[2], f"{hi.Penduduk_Akses_Internet - lo.Penduduk_Akses_Internet:.1f} poin", "selisih antarprovinsi")
    kpi(c[3], f"{hi.Penduduk_Akses_Internet / lo.Penduduk_Akses_Internet:.1f}x", "rasio tertinggi terhadap terendah")
    st.write("")
    a, b = st.columns(2)
    ext = pd.concat([D.nlargest(5, "Penduduk_Akses_Internet"), D.nsmallest(5, "Penduduk_Akses_Internet")]
                    ).sort_values("Penduduk_Akses_Internet")
    f = px.bar(ext, x="Penduduk_Akses_Internet", y="Provinsi", orientation="h", color="Kategori",
               color_discrete_map=WARNA, text="Penduduk_Akses_Internet", title="Lima tertinggi dan lima terendah (akses internet, %)")
    f.update_traces(texttemplate="%{text:.1f}")
    f.update_layout(height=400, legend_title="", yaxis_title="", xaxis_title="")
    a.plotly_chart(f, width="stretch")
    g = LONG[LONG.Indikator == "Penduduk_Akses_Internet"].groupby("Tahun").Nilai.mean().reset_index()
    f = px.area(g, x="Tahun", y="Nilai", markers=True, title="Rata-rata akses internet: aktual 2022-2025 dan proyeksi hingga 2029")
    f.add_vrect(x0=2025.5, x1=2029.5, fillcolor="orange", opacity=.15, line_width=0,
                annotation_text="Proyeksi", annotation_position="top left")
    f.update_layout(height=400, xaxis=dict(dtick=1), yaxis=dict(range=[0, 100], title="%"))
    f.update_traces(line_color="#2c6fad", fillcolor="rgba(44,111,173,.2)")
    b.plotly_chart(f, width="stretch")

    st.markdown('<div class="sec">Alur CRISP-DM</div>', unsafe_allow_html=True)
    steps = [("Business Understanding", "Rumusan masalah dan tujuan"), ("Data Understanding", "EDA, outlier, korelasi, urban-rural"),
             ("Data Preparation", "StandardScaler dan split time-series"), ("Modelling", "clustering dan forecasting"),
             ("Evaluation", "Silhouette, DB, CH, Elbow, MAPE"), ("Deployment", "Model joblib dan dashboard Streamlit")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><i>{i}</i><b>{t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(steps, 1)) + "</div>",
        unsafe_allow_html=True)

    st.markdown('<div class="sec">2. Eksplorasi Dashboard</div>', unsafe_allow_html=True)
    guide = [("Peta Kesenjangan", "Peta interaktif dengan filter kategori"), ("EDA", "Statistik, boxplot, ranking, korelasi, tren"),
             ("Evaluasi Clustering", "Pemilihan K, perbandingan algoritma, PCA"), ("Profil Klaster", "Karakteristik dan radar tiap klaster"),
             ("Forecasting", "Evaluasi model dan proyeksi 2026-2029"), ("Urban vs Rural", "Disparitas perkotaan dan perdesaan"),
             ("Simulasi", "Uji nilai indikator, lihat kategorinya"), ("Kesimpulan", "Ringkasan temuan otomatis")]
    st.markdown('<div class="guide">' + "".join(f'<div class="g"><b>{t}</b><span>{d}</span></div>' for t, d in guide) + "</div>",
                unsafe_allow_html=True)



def p_peta():
    hero("Peta Kesenjangan Digital 2025", "Rendah · Sedang · Tinggi — hasil K-Means (K=3)")
    tampil = st.multiselect("Kategori yang ditampilkan", LABELS, default=LABELS)
    if tampil:
        st.plotly_chart(peta_kategori(D, "Peta Tipologi Kesenjangan Digital Provinsi (2025)", tampil), width="stretch")
    t = D[["Provinsi", "Kategori"] + FITUR].sort_values("Penduduk_Akses_Internet", ascending=False)
    st.dataframe(t.rename(columns=NAMA), width="stretch", height=380, hide_index=True)
    st.download_button("Unduh hasil klaster (CSV)", D.drop(columns=["Provinsi_Geo"]).to_csv(index=False).encode(),
                       "dataset_clustering_2025_dengan_cluster.csv", "text/csv")


def p_eda():
    hero("Data Understanding & EDA", "Statistik deskriptif, outlier, ranking, korelasi, dan tren 2022–2025")
    t1, t2, t3, t4, t5 = st.tabs(["Statistik", "Boxplot", "Ranking", "Korelasi", "Tren"])
    with t1:
        c = st.columns(3)
        kpi(c[0], f"{len(MASTER)}", "baris data master"); kpi(c[1], f"{MASTER.Provinsi.nunique()}", "provinsi")
        kpi(c[2], f"{int(D[FITUR].isna().sum().sum())}", "missing value (fitur klaster 2025)")
        st.write("")
        st.dataframe(D[FITUR].describe().T[["mean", "std", "min", "50%", "max"]].round(2).rename(index=NAMA),
                     width="stretch")
    with t2:
        m = D.melt(value_vars=FITUR, var_name="Indikator", value_name="Persen")
        m["Indikator"] = m.Indikator.map(NAMA)
        f = px.box(m, x="Indikator", y="Persen", color="Indikator", points="all", title="Distribusi & outlier (2025)")
        f.update_layout(showlegend=False, height=480)
        st.plotly_chart(f, width="stretch")
    with t3:
        ind = st.selectbox("Indikator", FITUR, format_func=NAMA.get)
        s = D.sort_values(ind)
        f = px.bar(s, x=ind, y="Provinsi", color="Kategori", color_discrete_map=WARNA, orientation="h",
                   height=820, title=f"Ranking {NAMA[ind]} (2025)")
        f.add_vline(x=s[ind].mean(), line_dash="dash", annotation_text=f"Rata-rata {s[ind].mean():.2f}%")
        st.plotly_chart(f, width="stretch")
    with t4:
        r = D[FITUR].corr().rename(index=NAMA, columns=NAMA)
        st.plotly_chart(px.imshow(r, text_auto=".2f", color_continuous_scale="Blues", height=480,
                                  title="Korelasi antar-fitur digital (2025)"), width="stretch")
    with t5:
        g = MASTER.groupby("Tahun")[FITUR].mean().reset_index().melt("Tahun", var_name="Indikator", value_name="Rata-rata (%)")
        g["Indikator"] = g.Indikator.map(NAMA)
        st.plotly_chart(px.line(g, x="Tahun", y="Rata-rata (%)", color="Indikator", markers=True, height=460,
                                title="Rata-rata nasional per indikator"), width="stretch")


def p_eval():
    hero("Evaluasi Clustering", "Pemilihan K objektif, perbandingan algoritma, dan visualisasi PCA")
    ke = M["k_eval"]
    st.dataframe(ke.round(3), width="stretch", hide_index=True)
    from plotly.subplots import make_subplots
    fig = make_subplots(2, 2, subplot_titles=("Elbow (Inertia)", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"))
    for i, (c, col) in enumerate([("Inertia", "#1f77b4"), ("Silhouette", "#d62728"),
                                  ("Davies-Bouldin", "#2ca02c"), ("Calinski-Harabasz", "#9467bd")]):
        fig.add_trace(go.Scatter(x=ke.K, y=ke[c], mode="lines+markers", line_color=col, showlegend=False),
                      row=i // 2 + 1, col=i % 2 + 1)
        fig.add_vline(x=3, line_dash="dot", line_color="gray", row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(height=560, title="Metrik evaluasi K = 2…10 (garis putus = K final = 3)")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Perbandingan algoritma (K = 3)")
    st.dataframe(M["algo"].round(3), width="stretch", hide_index=True)
    v = M["var"]
    f = px.scatter(D, x="PCA1", y="PCA2", color="Kategori", color_discrete_map=WARNA, text="Provinsi",
                   height=560, title=f"PCA 2D (PC1 {v[0]:.0%} + PC2 {v[1]:.0%} = {v.sum():.0%} varians)")
    f.update_traces(textposition="top center", marker_size=11, textfont_size=9)
    st.plotly_chart(f, width="stretch")


def p_profil():
    hero("Profil Klaster", "Karakteristik rata-rata, median, dan anggota tiap klaster")
    mean = D.groupby("Kategori")[FITUR].mean().reindex(LABELS)
    med = D.groupby("Kategori")[FITUR].median().reindex(LABELS)
    a, b = st.columns(2)
    a.subheader("Rata-rata (%)"); a.dataframe(mean.round(2).rename(columns=NAMA), width="stretch")
    b.subheader("Median (%)"); b.dataframe(med.round(2).rename(columns=NAMA), width="stretch")
    rad = go.Figure()
    for lab in LABELS:
        v = mean.loc[lab].tolist()
        rad.add_trace(go.Scatterpolar(r=v + v[:1], theta=[NAMA[f] for f in FITUR] + [NAMA[FITUR[0]]],
                                      name=lab, line_color=WARNA[lab], fill="toself", opacity=.55))
    rad.update_layout(height=520, polar=dict(radialaxis=dict(range=[0, 100])), title="Radar profil klaster")
    st.plotly_chart(rad, width="stretch")
    for lab in LABELS:
        s = D[D.Kategori == lab].Provinsi
        st.markdown(f"**{lab} ({len(s)} provinsi):** " + ", ".join(s))


def p_forecast():
    hero("Forecasting 2026–2029", "Linear Regression · Polynomial Regression · Holt (latih 2022–2024, uji 2025)")
    st.subheader("Evaluasi model (data uji 2025, 34 provinsi)")
    e = EV.copy(); e["Indikator"] = e.Indikator.map(NAMA)
    st.dataframe(e.round(2), width="stretch", hide_index=True, height=300)
    st.plotly_chart(px.bar(e, x="Indikator", y="MAPE (%)", color="Model", barmode="group", height=400,
                           title="MAPE per model (lebih rendah = lebih baik)"), width="stretch")
    st.markdown("**Model terpilih (MAPE terendah):** " + " · ".join(f"{NAMA[k]} → **{v}**" for k, v in BEST.items()))
    st.subheader("Aktual vs proyeksi")
    mode = st.radio("Tampilan", ["Rata-rata nasional (34 provinsi)", "Per provinsi"], horizontal=True)
    if mode.startswith("Rata"):
        g = LONG.groupby(["Tahun", "Indikator"]).Nilai.mean().reset_index()
        g["Indikator"] = g.Indikator.map(NAMA)
        f = px.line(g, x="Tahun", y="Nilai", color="Indikator", markers=True, height=480)
        st.dataframe(LONG[LONG.Tipe_Data == "Forecast"].groupby(["Tahun", "Indikator"]).Nilai.mean()
                     .unstack().rename(columns=NAMA).round(2), width="stretch")
    else:
        ind = st.selectbox("Indikator", FITUR, format_func=NAMA.get)
        ps = st.multiselect("Provinsi", sorted(LONG.Provinsi.unique()), default=["DKI Jakarta", "Jawa Barat", "Aceh"])
        f = px.line(LONG[(LONG.Indikator == ind) & LONG.Provinsi.isin(ps)], x="Tahun", y="Nilai",
                    color="Provinsi", markers=True, height=480, title=NAMA[ind])
    f.add_vrect(x0=2025.5, x1=2029.5, fillcolor="orange", opacity=.12, line_width=0,
                annotation_text="Proyeksi", annotation_position="top left")
    f.update_layout(xaxis=dict(dtick=1), yaxis_title="Persen (%)")
    st.plotly_chart(f, width="stretch")
    st.download_button("Unduh hasil forecasting (CSV)", LONG.to_csv(index=False).encode(),
                       "hasil_forecasting_2026_2029_34_provinsi.csv", "text/csv")


def p_urban():
    hero("Kesenjangan Urban vs Rural", "Selisih akses internet perkotaan − perdesaan")
    thn = st.select_slider("Tahun", [2022, 2023, 2024, 2025], value=2025)
    g = MASTER.dropna(subset=["Gap_Akses_Internet_Urban_Rural"])
    s = g[g.Tahun == thn].sort_values("Gap_Akses_Internet_Urban_Rural", ascending=False)
    n = st.slider("Tampilkan N provinsi teratas", 5, len(s), 10)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(s.head(n).iloc[::-1], x="Gap_Akses_Internet_Urban_Rural", y="Provinsi", orientation="h",
                          color="Gap_Akses_Internet_Urban_Rural", color_continuous_scale="Reds",
                          title=f"Disparitas terbesar {thn}", labels={"Gap_Akses_Internet_Urban_Rural": "Gap (%)"}),
                   width="stretch")
    b.plotly_chart(px.scatter(s, x="Penduduk_Akses_Internet_Perdesaan", y="Penduduk_Akses_Internet_Perkotaan",
                              hover_name="Provinsi", title="Perkotaan vs perdesaan").add_shape(
        type="line", x0=0, y0=0, x1=100, y1=100, line_dash="dash"), width="stretch")
    t = g.groupby("Tahun").Gap_Akses_Internet_Urban_Rural.mean().reset_index()
    st.plotly_chart(px.line(t, x="Tahun", y="Gap_Akses_Internet_Urban_Rural", markers=True,
                            title="Rata-rata gap nasional", labels={"Gap_Akses_Internet_Urban_Rural": "Gap (%)"}),
                    width="stretch")


def p_simulasi():
    hero("Simulasi Klaster", "Masukkan nilai indikator — model K-Means akan menentukan kategorinya")
    ref = D[FITUR].mean()
    cols = st.columns(5)
    v = [cols[i].slider(NAMA[f], 0.0, 100.0, float(round(ref[f], 1)), .5) for i, f in enumerate(FITUR)]
    kat = P.prediksi_kategori(M, [v])[0]
    st.markdown(f"### Hasil: <span style='color:{WARNA[kat]}'>{kat}</span>", unsafe_allow_html=True)
    z = M["scaler"].transform([v])
    dist = pd.Series({M["cmap"][i]: float(np.linalg.norm(z[0] - c)) for i, c in enumerate(M["km"].cluster_centers_)})
    a, b = st.columns(2)
    a.plotly_chart(px.bar(dist.reindex(LABELS).reset_index(), x="index", y=0, color="index", color_discrete_map=WARNA,
                          labels={"index": "", "0": "Jarak"}, title="Jarak ke centroid (lebih kecil = lebih dekat)"),
                   width="stretch")
    dp = ((D[FITUR].values - np.array(v)) / D[FITUR].std().values) ** 2
    D2 = D.assign(Jarak=np.sqrt(dp.sum(1))).nsmallest(5, "Jarak")[["Provinsi", "Kategori", "Jarak"]]
    b.subheader("Provinsi paling mirip"); b.dataframe(D2.round(2), hide_index=True, width="stretch")


def p_kesimpulan():
    hero("Kesimpulan", "Dihitung otomatis dari hasil model")
    ke = M["k_eval"].set_index("K")
    turun = (ke.Inertia[2] - ke.Inertia[3]) / ke.Inertia[2] * 100
    st.markdown(f"**1. Karakteristik 2025.** Disparitas TIK nyata: klaster tinggi rata-rata akses internet "
                f"{D[D.Kategori=='Digital Tinggi'].Penduduk_Akses_Internet.mean():.1f}%, sedang "
                f"{D[D.Kategori=='Digital Sedang'].Penduduk_Akses_Internet.mean():.1f}%, rendah "
                f"{D[D.Kategori=='Digital Rendah'].Penduduk_Akses_Internet.mean():.1f}%.")
    st.markdown(f"**2. K optimum = 3.** Inertia turun {turun:.1f}% dari K=2 ke K=3, Silhouette {ke.Silhouette[3]:.2f}, "
                f"Davies-Bouldin {ke['Davies-Bouldin'][3]:.2f}, Calinski-Harabasz {ke['Calinski-Harabasz'][3]:.2f}.")
    for lab in LABELS[::-1]:
        s = D[D.Kategori == lab]
        st.markdown(f"   - **{lab}** ({len(s)}): " + (", ".join(s.Provinsi) if len(s) <= 6 else "30 provinsi lainnya"
                    if lab == "Digital Sedang" else ", ".join(s.Provinsi)))
    st.markdown("**3. Model forecasting terbaik per indikator:**")
    for ind, m in BEST.items():
        r = EV[(EV.Indikator == ind) & (EV.Model == m)].iloc[0]
        st.markdown(f"   - {NAMA[ind]}: **{m}** (MAPE {r['MAPE (%)']:.2f}%, MAE {r.MAE:.2f})")
    ai = LONG[(LONG.Indikator == "Penduduk_Akses_Internet") & (LONG.Tipe_Data == "Forecast")].groupby("Tahun").Nilai.mean()
    st.markdown("**4. Proyeksi akses internet (rata-rata 34 provinsi):** " + ", ".join(f"{t}: {v:.2f}%" for t, v in ai.items()))
    st.info("Catatan: pemekaran Papua (data historis tidak lengkap) tidak diproyeksikan, dan hasil forecast bersifat "
            "ekstrapolasi tren dari hanya 4 titik tahun sehingga perlu dibaca sebagai indikasi arah, bukan angka pasti.")


PAGES = {"Beranda": p_beranda, "Peta Kesenjangan": p_peta, "EDA": p_eda, "Evaluasi Clustering": p_eval,
         "Profil Klaster": p_profil, "Forecasting": p_forecast,
         "Urban vs Rural": p_urban, "Simulasi": p_simulasi, "Kesimpulan": p_kesimpulan}
with st.sidebar:
    st.image("logo.png", width=80)
    halaman = st.radio("Navigasi", list(PAGES), label_visibility="collapsed")
PAGES[halaman]()
