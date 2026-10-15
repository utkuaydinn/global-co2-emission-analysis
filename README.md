# 🌍 Global CO₂ Emission Analysis, Web Scraping & Prophet Time-Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1+-teal.svg)](https://facebook.github.io/prophet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data analytics and forecasting platform that consolidates historical carbon footprint data from **Our World in Data (OWID)** and real-time data via **Wikipedia Web Scraping**, delivering interactive cross-country trend analyses and **2050 projections** powered by Facebook's **Prophet** algorithm.

---

## 📌 Project Overview

This project bridges the gap between static climate reports and dynamic, user-driven decision support tools. It provides an intuitive, web-based dashboard built with **Streamlit** to explore country-based carbon dioxide emissions from the 1800s to the present, compare historical trajectories, and forecast future emissions up to **2050**.

### 🌟 Key Features

1. **Global Summary & Core Metrics (OWID):**
   - High-level KPIs: Total analyzed countries, global average CO₂ per capita, and cumulative global emissions.
   - Real-time Top 10 highest and lowest emitting countries for the latest year.

2. **Dynamic Multi-Country Comparison:**
   - Multi-country selector (e.g., Turkey, Germany, United States, China) with an interactive year-range slider (from 1800 to 2024).

3. **Top 10 Growth & Reduction Analysis (2000–2024):**
   - Absolute and percentage-based shifts over the last 25 years.
   - Identification of green transformation leaders vs. rapidly industrializing high-growth nations.

4. **Distribution & Scatter Analysis:**
   - 2000 vs. 2024 scatter plot with a `y = x` reference line to identify countries deviating from baseline emissions.

5. **Time-Series Forecasting to 2050 (Prophet):**
   - Country-specific additive regression model using Facebook Prophet.
   - Long-term trend decomposition with 80% uncertainty intervals (`yhat_lower`, `yhat_upper`).
   - Milestone projections for **2030, 2040, and 2050**.

6. **Live Web Scraping (Wikipedia):**
   - Real-time data scraping using `BeautifulSoup` & `pandas` with automatic cleaning (footnote stripping, non-country aggregate filtering).
   - Local caching mechanism ensuring zero downtime even under offline or rate-limited environments.
   - 4-tab interactive analysis: Top/Bottom 10, % change, scatter matrix, and search filter.

---

## 📁 Repository Structure

```
.
├── app.py                         # Streamlit Dashboard application
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── data/                          # Datasets
│   ├── owid-co2-data.csv          # Our World in Data (OWID) comprehensive dataset (~33 MB)
│   ├── wikipedia_co2_cached.csv   # Scraped and cleaned Wikipedia dataset
│   └── 1milyonyol.bas.co2_mik.ton.csv
├── notebooks/                     # Exploratory Data Analysis & Modeling
│   └── co2_analiz_ve_tahmin.ipynb # Unified end-to-end data pipeline, scraping & Prophet forecasting
└── raporlar_ve_gorseller/         # Academic Deliverables, Diagrams & Plots
    ├── Akademik_Makale_Raporu_CO2.docx  # Full-length academic publication paper draft
    ├── Vize_Makalesi_CO2.pdf            # Research term paper
    └── grafikler/                 # Clean, publication-ready diagrams, charts & methodology figures
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/utkuaydinn/global-co2-emission-analysis.git
cd global-co2-emission-analysis
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The application will automatically launch at `http://localhost:8501`.

---

## 📊 Methodology & Technologies

| Layer | Tools / Libraries | Purpose |
|---|---|---|
| **Data Extraction** | `requests`, `BeautifulSoup4`, `lxml` | Automated Wikipedia web scraping |
| **Data Processing** | `pandas`, `numpy` | Multi-index parsing, outlier filtering, data cleaning |
| **Forecasting** | `Prophet` | Additive non-linear trend estimation up to 2050 |
| **Visualization** | `matplotlib`, `seaborn`, `streamlit` | Interactive dashboards, time-series line charts, bar plots |

---

## 👤 Author

**Utku Aydın**  
*M.Sc. Computer Engineering Student & AI/Data Science Researcher*  
Izmir Bakircay University

- **LinkedIn:** [linkedin.com/in/utku-aydin](https://www.linkedin.com/in/utku-aydin/)
- **GitHub:** [github.com/utkuaydinn](https://github.com/utkuaydinn)
- **Medium:** [utkuaydinn.medium.com](https://utkuaydinn.medium.com/)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
