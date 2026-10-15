import os
import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet

# -------------------------------------------------
# Genel Ayarlar & Konfigürasyon
# -------------------------------------------------
st.set_page_config(
    page_title="Küresel CO₂ Emisyon Dashboard – OWID + Prophet + Wikipedia",
    page_icon="🌍",
    layout="wide"
)

# Dosya yollarını dinamik olarak çöz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "owid-co2-data.csv")
WIKI_CACHE_PATH = os.path.join(BASE_DIR, "data", "wikipedia_co2_cached.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = "owid-co2-data.csv"

TARGET_LAST_YEAR = 2024   # OWID için modele kadar kullanılacak son gerçek yıl
BASELINE_YEAR = 2000      # Artış/azalış ve scatter için başlangıç yılı
FORECAST_HORIZON = 2050   # Prophet projeksiyon son yılı

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions_per_capita"
WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# -------------------------------------------------
# OWID Veri Yükleme
# -------------------------------------------------
@st.cache_data
def load_owid_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Sadece gerçek ülke satırları (World, Asia vb. hariç)
    df = df[df["iso_code"].notna()].copy()

    cols = [
        "country", "iso_code", "year",
        "co2", "co2_per_capita",
        "population", "gdp",
        "total_ghg", "total_ghg_excluding_lucf"
    ]
    existing = [c for c in cols if c in df.columns]
    df = df[existing]

    df = df.sort_values(["country", "year"])
    return df

try:
    df = load_owid_data(DATA_PATH)
    countries = sorted(df["country"].unique())
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    latest_year = min(TARGET_LAST_YEAR, max_year)
except Exception as e:
    st.error(f"OWID veri seti yüklenemedi: {e}")
    st.stop()

# -------------------------------------------------
# Wikipedia (Web Scraping) Veri Yükleme & Temizleme
# -------------------------------------------------
def clean_wikipedia_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = ['_'.join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col) for col in df.columns]

    rename_map = {
        'Location_Location': 'Country',
        'Emissions per capita (tons per year)_2023': 'Emissions_2023',
        'Emissions per capita (tons per year)_2000': 'Emissions_2000',
        '% change from 2000_% change from 2000': 'Pct_Change_2000_to_2023'
    }

    df = df.rename(columns=rename_map)
    cols_to_keep = [c for c in rename_map.values() if c in df.columns]
    df = df[cols_to_keep]

    # Ülke isimlerindeki dipnotları ([n 4], [note 1] vb.) temizle
    df['Country'] = df['Country'].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip()

    # Ülke olmayan bölgesel toplamları filtrele
    non_countries = ['World', 'European Union', 'Europe']
    df = df[~df['Country'].isin(non_countries)].copy()

    # Sayısal değerleri temizle
    for col in ['Emissions_2000', 'Emissions_2023', 'Pct_Change_2000_to_2023']:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace('−', '-', regex=False)
                .str.replace('–', '-', regex=False)
                .str.replace('—', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('+', '', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Emissions_2023']).copy()
    if 'Emissions_2000' in df.columns:
        df['Abs_Change'] = df['Emissions_2023'] - df['Emissions_2000']
        if 'Pct_Change_2000_to_2023' not in df.columns or df['Pct_Change_2000_to_2023'].isna().all():
            df['Pct_Change_2000_to_2023'] = np.where(
                df['Emissions_2000'] != 0,
                df['Abs_Change'] / df['Emissions_2000'] * 100,
                np.nan
            )

    return df

@st.cache_data(ttl=3600, show_spinner="Wikipedia'dan canlı kişi başı emisyon tablosu çekiliyor...")
def load_wikipedia_data(url: str):
    is_live = False
    try:
        r = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        r.raise_for_status()
        # pandas 2.2+ uyumu için io.StringIO kullanımı zorunludur
        tables = pd.read_html(io.StringIO(r.text))
        if tables:
            df_cleaned = clean_wikipedia_dataframe(tables[0])
            # Önbelleğe kaydet
            try:
                os.makedirs(os.path.dirname(WIKI_CACHE_PATH), exist_ok=True)
                df_cleaned.to_csv(WIKI_CACHE_PATH, index=False)
            except Exception:
                pass
            is_live = True
            return df_cleaned, 2000, 2023, is_live
    except Exception:
        pass

    # Canlı çekilemezse yerel yedekten yükle
    if os.path.exists(WIKI_CACHE_PATH):
        df_cached = pd.read_csv(WIKI_CACHE_PATH)
        return df_cached, 2000, 2023, False

    raise ValueError("Ne canlı Wikipedia verisine ne de yerel önbellek dosyasına ulaşılamadı.")

# -------------------------------------------------
# OWID & Prophet Yardımcı Fonksiyonlar
# -------------------------------------------------
def get_country_series(country: str) -> pd.DataFrame:
    cdf = df[df["country"] == country].copy()
    cdf = cdf.sort_values("year")
    cdf = cdf[cdf["co2_per_capita"].notna()]
    if cdf.empty:
        return cdf
    upper = min(TARGET_LAST_YEAR, int(cdf["year"].max()))
    cdf = cdf[cdf["year"] <= upper]
    return cdf

def fit_prophet_for_country(cdf: pd.DataFrame) -> Prophet:
    dfp = cdf[["year", "co2_per_capita"]].copy()
    dfp["ds"] = pd.to_datetime(dfp["year"].astype(str) + "-01-01")
    dfp["y"] = dfp["co2_per_capita"].astype(float)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(dfp[["ds", "y"]])
    return m

def forecast_to_year(model: Prophet, last_real_year: int, horizon_year: int) -> pd.DataFrame:
    periods = max(0, horizon_year - last_real_year)
    try:
        future = model.make_future_dataframe(periods=periods, freq="YE")
    except Exception:
        future = model.make_future_dataframe(periods=periods, freq="Y")
    forecast = model.predict(future)
    forecast["year"] = forecast["ds"].dt.year
    return forecast

# -------------------------------------------------
# Sidebar Menü & Bilgi Alanı
# -------------------------------------------------
st.sidebar.title("🌍 Menü")
menu = st.sidebar.selectbox(
    "Görünüm Seçiniz:",
    [
        "Ana Sayfa (OWID)",
        "Ülke Karşılaştırma (OWID - Gerçek Veriler)",
        "Top 10 Artış (OWID 2000–2024)",
        "Top 10 Azalış (OWID 2000–2024)",
        "Dağılım Grafiği (OWID 2000 vs 2024)",
        "Tahmin Modeli (OWID Tek Ülke – Prophet)",
        "Wikipedia Verileri (Web Scraping)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**OWID Veri Aralığı:** {min_year} – {max_year}")
st.sidebar.markdown(f"**Son Gerçek Yıl:** {latest_year}")
st.sidebar.markdown(f"**Prophet Projeksiyon Ufku:** {FORECAST_HORIZON}")
st.sidebar.info("💡 **Geliştirici:** Utku Aydın\nİzmir Bakırçay Üniversitesi YL Projesi")

# -------------------------------------------------
# 1) Ana Sayfa (OWID)
# -------------------------------------------------
if menu == "Ana Sayfa (OWID)":
    st.title("🌍 Küresel CO₂ Emisyon Analizi (Our World in Data)")
    st.markdown(
        "Bu panel, Our World in Data (OWID) veri tabanındaki tarihsel karbon ayak izi verilerini kullanarak "
        "ülkelerin kişi başına düşen CO₂ emisyonlarını incelemektedir."
    )

    latest = df[(df["year"] == latest_year) & (df["co2_per_capita"].notna())]

    col1, col2, col3 = st.columns(3)
    col1.metric("İncelenen Ülke Sayısı", f"{latest['country'].nunique()} Ülke")
    col2.metric("Küresel Ort. Kişi Başı CO₂", f"{latest['co2_per_capita'].mean():.2f} ton/yıl")
    total_co2_val = latest["co2"].sum() if "co2" in latest.columns else 0
    col3.metric("Toplam CO₂ Emisyonu", f"{total_co2_val:,.1f} Milyon Ton")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(f"⬆️ En Yüksek 10 Ülke ({latest_year})")
        top10_latest = latest.sort_values("co2_per_capita", ascending=False).head(10)
        st.dataframe(top10_latest[["country", "co2_per_capita"]].rename(columns={"country": "Ülke", "co2_per_capita": "Kişi Başı CO₂ (ton)"}), use_container_width=True)

    with col_b:
        st.subheader(f"⬇️ En Düşük 10 Ülke ({latest_year})")
        bottom10_latest = latest.sort_values("co2_per_capita", ascending=True).head(10)
        st.dataframe(bottom10_latest[["country", "co2_per_capita"]].rename(columns={"country": "Ülke", "co2_per_capita": "Kişi Başı CO₂ (ton)"}), use_container_width=True)

# -------------------------------------------------
# 2) Ülke Karşılaştırma (OWID - Gerçek Veriler)
# -------------------------------------------------
elif menu == "Ülke Karşılaştırma (OWID - Gerçek Veriler)":
    st.title("🏳️ Ülke Karşılaştırma Analizi (Gerçek Veriler)")

    selected_countries = st.multiselect(
        "Karşılaştırmak istediğiniz ülkeleri seçin:",
        countries,
        default=["Turkey", "Germany", "United States", "China"] if all(c in countries for c in ["Turkey", "Germany", "United States", "China"]) else countries[:3]
    )

    start_year, end_year = st.slider(
        "Zaman Aralığı Seçiniz:",
        min_year,
        latest_year,
        (max(min_year, 1980), latest_year)
    )

    if selected_countries:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for c in selected_countries:
            cdf = get_country_series(c)
            cdf = cdf[(cdf["year"] >= start_year) & (cdf["year"] <= end_year)]
            if cdf.empty:
                continue
            ax.plot(cdf["year"], cdf["co2_per_capita"], marker="o", markersize=3, linewidth=2, label=c)

        ax.set_xlabel("Yıl", fontsize=11)
        ax.set_ylabel("Kişi Başı CO₂ (ton/yıl)", fontsize=11)
        ax.set_title(f"Ülkelerin Kişi Başı CO₂ Emisyon Eğilimleri ({start_year} - {end_year})", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Lütfen en az bir ülke seçiniz.")

# -------------------------------------------------
# 3) Top 10 Artış (OWID 2000–2024)
# -------------------------------------------------
elif menu == "Top 10 Artış (OWID 2000–2024)":
    st.title("📈 2000–2024 Arası En Fazla Artış Gösteren 10 Ülke")

    subset = df[df["year"].isin([BASELINE_YEAR, latest_year])]
    pivot = subset.pivot(index="country", columns="year", values="co2_per_capita")
    pivot = pivot.dropna(subset=[BASELINE_YEAR, latest_year])

    pivot["abs_change"] = pivot[latest_year] - pivot[BASELINE_YEAR]
    pivot["pct_change"] = pivot["abs_change"] / pivot[BASELINE_YEAR] * 100

    top10 = pivot.sort_values("pct_change", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top10.index, top10["pct_change"], color="#e74c3c")
    ax.set_xlabel("Yüzde Artış (%)", fontsize=11)
    ax.set_title(f"{BASELINE_YEAR}–{latest_year} Kişi Başı CO₂ Yüzde Artışında İlk 10", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    st.pyplot(fig)

    st.dataframe(
        top10[["pct_change", "abs_change", BASELINE_YEAR, latest_year]]
        .rename(columns={BASELINE_YEAR: f"Değer ({BASELINE_YEAR})", latest_year: f"Değer ({latest_year})", "pct_change": "% Değişim", "abs_change": "Mutlak Fark"}),
        use_container_width=True
    )

# -------------------------------------------------
# 4) Top 10 Azalış (OWID 2000–2024)
# -------------------------------------------------
elif menu == "Top 10 Azalış (OWID 2000–2024)":
    st.title("📉 2000–2024 Arası En Fazla Azalış Gösteren 10 Ülke")

    subset = df[df["year"].isin([BASELINE_YEAR, latest_year])]
    pivot = subset.pivot(index="country", columns="year", values="co2_per_capita")
    pivot = pivot.dropna(subset=[BASELINE_YEAR, latest_year])

    pivot["abs_change"] = pivot[latest_year] - pivot[BASELINE_YEAR]
    pivot["pct_change"] = pivot["abs_change"] / pivot[BASELINE_YEAR] * 100

    bottom10 = pivot.sort_values("pct_change", ascending=True).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(bottom10.index, bottom10["pct_change"], color="#2ecc71")
    ax.set_xlabel("Yüzde Azalış (%)", fontsize=11)
    ax.set_title(f"{BASELINE_YEAR}–{latest_year} Kişi Başı CO₂ Yüzde Azalışında İlk 10", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    st.pyplot(fig)

    st.dataframe(
        bottom10[["pct_change", "abs_change", BASELINE_YEAR, latest_year]]
        .rename(columns={BASELINE_YEAR: f"Değer ({BASELINE_YEAR})", latest_year: f"Değer ({latest_year})", "pct_change": "% Değişim", "abs_change": "Mutlak Fark"}),
        use_container_width=True
    )

# -------------------------------------------------
# 5) Dağılım Grafiği (OWID 2000 vs 2024)
# -------------------------------------------------
elif menu == "Dağılım Grafiği (OWID 2000 vs 2024)":
    st.title("🟣 2000 vs 2024 Kişi Başı CO₂ Dağılımı")

    subset = df[df["year"].isin([BASELINE_YEAR, latest_year])]
    pivot = subset.pivot(index="country", columns="year", values="co2_per_capita")
    pivot = pivot.dropna(subset=[BASELINE_YEAR, latest_year])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pivot[BASELINE_YEAR], pivot[latest_year], alpha=0.7, color="#3498db", edgecolors="black")
    ax.set_xlabel(f"{BASELINE_YEAR} Kişi Başı CO₂ (ton)", fontsize=11)
    ax.set_ylabel(f"{latest_year} Kişi Başı CO₂ (ton)", fontsize=11)
    ax.set_title(f"{BASELINE_YEAR} ve {latest_year} Yılları Arasında Ülke Emisyon Dağılımı", fontsize=12, fontweight="bold")

    min_val = min(pivot[BASELINE_YEAR].min(), pivot[latest_year].min())
    max_val = max(pivot[BASELINE_YEAR].max(), pivot[latest_year].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Değişimsiz Çizgi (y = x)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

# -------------------------------------------------
# 6) Tahmin Modeli (OWID Tek Ülke – Prophet)
# -------------------------------------------------
elif menu == "Tahmin Modeli (OWID Tek Ülke – Prophet)":
    st.title("🔮 Prophet ile Zaman Serisi Tahmini (2050 Projeksiyonu)")

    country = st.selectbox(
        "Projeksiyon yapmak istediğiniz ülkeyi seçin:",
        countries,
        index=countries.index("Turkey") if "Turkey" in countries else 0
    )

    cdf = get_country_series(country)

    if cdf.shape[0] < 5:
        st.warning("Bu ülke için Prophet modelini çalıştırmak için yeterli veri bulunmamaktadır (en az 5 gözlem gereklidir).")
    else:
        with st.spinner(f"{country} için Prophet modeli eğitiliyor ve {FORECAST_HORIZON} yılına kadar tahmin üretiliyor..."):
            model = fit_prophet_for_country(cdf)
            last_real_year = int(cdf["year"].max())
            forecast = forecast_to_year(model, last_real_year, horizon_year=FORECAST_HORIZON)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.scatter(cdf["year"], cdf["co2_per_capita"], color="#c0392b", label="Gerçek Veri (OWID)", zorder=5)
        ax.plot(forecast["year"], forecast["yhat"], color="#2980b9", linewidth=2, label="Prophet Eğilim / Tahmin (yhat)")
        ax.fill_between(forecast["year"], forecast["yhat_lower"], forecast["yhat_upper"], color="#3498db", alpha=0.2, label="%80 Güven Aralığı")
        ax.axvline(x=last_real_year, color="black", linestyle="--", alpha=0.7, label=f"Son Gerçek Veri: {last_real_year}")
        ax.set_xlabel("Yıl", fontsize=11)
        ax.set_ylabel("Kişi Başı CO₂ (ton/yıl)", fontsize=11)
        ax.set_title(f"{country} – {FORECAST_HORIZON}'ye Kadar Kişi Başı CO₂ Projeksiyonu", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

        # 2030, 2040 ve 2050 dönüm noktaları
        years_to_check = [2030, 2040, FORECAST_HORIZON]
        future_df = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in years_to_check])})
        future_pred = model.predict(future_df)
        future_pred["year"] = future_pred["ds"].dt.year

        last_real_value = float(cdf[cdf["year"] == last_real_year]["co2_per_capita"].iloc[0])
        pred_2050 = float(future_pred.loc[future_pred["year"] == FORECAST_HORIZON, "yhat"].iloc[0])

        col1, col2, col3 = st.columns(3)
        col1.metric("Son Gerçek Yıl", f"{last_real_year}")
        col2.metric("Son Gerçek Değer", f"{last_real_value:.3f} ton")
        diff_2050 = pred_2050 - last_real_value
        col3.metric(f"{FORECAST_HORIZON} Tahmini", f"{pred_2050:.3f} ton", delta=f"{diff_2050:+.3f} ton")

        st.subheader("📌 Hedef Yıl Projeksiyon Değerleri")
        summary_rows = []
        for _, r in future_pred.iterrows():
            summary_rows.append({
                "Hedef Yıl": int(r['year']),
                "Beklenen Tahmin (yhat)": f"{r['yhat']:.3f} ton",
                "Alt Sınır (İyimser/Kötümser)": f"{r['yhat_lower']:.3f} ton",
                "Üst Sınır": f"{r['yhat_upper']:.3f} ton"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# -------------------------------------------------
# 7) Wikipedia Verileri (Web Scraping)
# -------------------------------------------------
elif menu == "Wikipedia Verileri (Web Scraping)":
    st.title("📚 Wikipedia Canlı Verileri (Web Scraping)")
    st.markdown("Wikipedia üzerinden web scraping yöntemiyle çekilen güncel ve temizlenmiş kişi başı karbon ayak izi analizleri.")

    try:
        wiki_df, wiki_base_year, wiki_latest_year, is_live = load_wikipedia_data(WIKI_URL)
    except Exception as e:
        st.error(f"Wikipedia verisi alınırken hata oluştu: {e}")
        st.stop()

    if is_live:
        st.success("🟢 Veriler Wikipedia üzerinden canlı (web scraping) olarak başarıyla çekildi.")
    else:
        st.info("ℹ️ Canlı bağlantı yerine son güncellenmiş yerel önbellek verisi gösteriliyor.")

    colA, colB, colC = st.columns(3)
    colA.metric("Kazınan Ülke Sayısı", f"{wiki_df['Country'].nunique()} Ülke")
    colB.metric("Referans Yıl", f"{wiki_base_year}")
    colC.metric("En Güncel Yıl", f"{wiki_latest_year}")

    st.markdown("---")

    # Sekmeli Analiz Görünümü
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top 10 En Yüksek / En Düşük",
        "📊 % Artış & Azalış Değişimi",
        "📈 Dağılım & Ülke Karşılaştırma",
        "📄 Veri Tablosu & Arama"
    ])

    # Tab 1: En Yüksek ve En Düşük
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"⬆️ En Yüksek 10 Ülke ({wiki_latest_year})")
            top10 = wiki_df.sort_values(f"Emissions_{wiki_latest_year}", ascending=False).head(10)
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.barh(top10["Country"], top10[f"Emissions_{wiki_latest_year}"], color="#e67e22")
            ax1.set_xlabel("Kişi Başı CO₂ (ton)", fontsize=11)
            ax1.set_title(f"En Yüksek 10 Ülke ({wiki_latest_year})", fontsize=12, fontweight="bold")
            ax1.invert_yaxis()
            ax1.grid(axis="x", linestyle="--", alpha=0.5)
            st.pyplot(fig1)

        with c2:
            st.subheader(f"⬇️ En Düşük 10 Ülke ({wiki_latest_year})")
            bot10 = wiki_df.sort_values(f"Emissions_{wiki_latest_year}", ascending=True).head(10)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(bot10["Country"], bot10[f"Emissions_{wiki_latest_year}"], color="#27ae60")
            ax2.set_xlabel("Kişi Başı CO₂ (ton)", fontsize=11)
            ax2.set_title(f"En Düşük 10 Ülke ({wiki_latest_year})", fontsize=12, fontweight="bold")
            ax2.invert_yaxis()
            ax2.grid(axis="x", linestyle="--", alpha=0.5)
            st.pyplot(fig2)

    # Tab 2: Yüzdelik Değişimler
    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader(f"📈 {wiki_base_year}–{wiki_latest_year} En Fazla Yüzdelik Artış Gösteren 10 Ülke")
            top_growth = wiki_df.sort_values("Pct_Change_2000_to_2023", ascending=False).head(10)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.barh(top_growth["Country"], top_growth["Pct_Change_2000_to_2023"], color="#c0392b")
            ax3.set_xlabel("% Değişim", fontsize=11)
            ax3.set_title("CO₂ Salınımında En Çok Artış (%)", fontsize=12, fontweight="bold")
            ax3.invert_yaxis()
            ax3.grid(axis="x", linestyle="--", alpha=0.5)
            st.pyplot(fig3)

        with c4:
            st.subheader(f"📉 {wiki_base_year}–{wiki_latest_year} En Fazla Yüzdelik Azalış Gösteren 10 Ülke")
            top_drop = wiki_df.sort_values("Pct_Change_2000_to_2023", ascending=True).head(10)
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.barh(top_drop["Country"], top_drop["Pct_Change_2000_to_2023"], color="#2980b9")
            ax4.set_xlabel("% Değişim", fontsize=11)
            ax4.set_title("CO₂ Salınımında En Çok Azalış (%)", fontsize=12, fontweight="bold")
            ax4.invert_yaxis()
            ax4.grid(axis="x", linestyle="--", alpha=0.5)
            st.pyplot(fig4)

    # Tab 3: Dağılım & Karşılaştırma
    with tab3:
        st.subheader(f"📌 {wiki_base_year} vs {wiki_latest_year} Kişi Başı CO₂ Dağılım Grafiği (Scatter)")
        fig5, ax5 = plt.subplots(figsize=(9, 5.5))
        ax5.scatter(wiki_df[f"Emissions_{wiki_base_year}"], wiki_df[f"Emissions_{wiki_latest_year}"], color="#f39c12", alpha=0.6, edgecolors="black")
        max_em = max(wiki_df[f"Emissions_{wiki_base_year}"].max(), wiki_df[f"Emissions_{wiki_latest_year}"].max())
        ax5.plot([0, max_em], [0, max_em], 'k--', lw=1, label="Değişimsiz Eksen (y = x)")
        ax5.set_xlabel(f"{wiki_base_year} Emisyonu (ton/kişi)", fontsize=11)
        ax5.set_ylabel(f"{wiki_latest_year} Emisyonu (ton/kişi)", fontsize=11)
        ax5.set_title("2000 vs 2023 Karşılaştırma Dağılımı", fontsize=12, fontweight="bold")
        ax5.legend()
        ax5.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig5)

        st.markdown("---")
        st.subheader("🔍 Seçili Ülkelerin 2000 → 2023 Eğilim Kıyaslaması")
        wiki_countries = sorted(wiki_df["Country"].unique())
        selected_wiki = st.multiselect(
            "Karşılaştırmak istediğiniz ülkeleri seçin:",
            wiki_countries,
            default=["Turkey", "Germany", "Qatar", "China"] if all(x in wiki_countries for x in ["Turkey", "Germany", "Qatar", "China"]) else wiki_countries[:3]
        )
        if selected_wiki:
            sel_df = wiki_df[wiki_df["Country"].isin(selected_wiki)]
            fig6, ax6 = plt.subplots(figsize=(10, 5))
            for _, r in sel_df.iterrows():
                ax6.plot([wiki_base_year, wiki_latest_year],
                         [r[f"Emissions_{wiki_base_year}"], r[f"Emissions_{wiki_latest_year}"]],
                         marker="o", markersize=6, linewidth=2, label=r["Country"])
            ax6.set_xlabel("Yıl", fontsize=11)
            ax6.set_ylabel("ton/kişi", fontsize=11)
            ax6.set_title(f"Seçilen Ülkelerde {wiki_base_year} → {wiki_latest_year} Değişimi", fontsize=12, fontweight="bold")
            ax6.legend()
            ax6.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig6)

    # Tab 4: Tablo & Filtreleme
    with tab4:
        st.subheader("📋 Tüm Temizlenmiş Wikipedia Veri Kümesi")
        search_query = st.text_input("Ülke Ara:", placeholder="Örn: Turkey, Germany...")
        filtered_df = wiki_df
        if search_query:
            filtered_df = wiki_df[wiki_df["Country"].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            filtered_df.rename(columns={
                "Country": "Ülke",
                f"Emissions_{wiki_base_year}": f"{wiki_base_year} Emisyonu (ton)",
                f"Emissions_{wiki_latest_year}": f"{wiki_latest_year} Emisyonu (ton)",
                "Pct_Change_2000_to_2023": "% Değişim",
                "Abs_Change": "Mutlak Fark (ton)"
            }),
            use_container_width=True
        )
