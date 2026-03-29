# 🧬 AI Data Cleaning & EDA Engine

An autonomous, AI-powered data cleaning and exploratory data analysis (EDA) tool built with **Streamlit** and **Google Gemini**. Upload a messy dataset, and the AI agent will clean it, generate insights, run SQL queries, and produce beautiful EDA visualizations — all in one pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=flat-square&logo=streamlit)
![Gemini](https://img.shields.io/badge/Google_Gemini-AI_Agent-4285F4?style=flat-square&logo=google)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📋 **Data Profiler** | Instant overview: rows, columns, missing values, duplicates, data types |
| 🤖 **Agentic Cleaning** | Gemini-powered AI writes & executes pandas code to clean your data autonomously |
| 🔁 **Reflection Loop** | Self-correcting agent — if code fails, the AI reflects on the error and retries |
| 🧠 **Insight Engine** | Outlier detection, skewness analysis, correlation mining, business recommendations |
| 🗄️ **SQL Explorer** | Auto-generated SQL query library + custom SQL editor on your cleaned data |
| 📊 **EDA & Visualizations** | Correlation heatmaps, numeric/categorical distributions via Plotly |
| 📄 **HTML Reports** | Downloadable, self-contained HTML report of your cleaned data |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-data-cleaning-engine.git
cd ai-data-cleaning-engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```
> Get a free API key at [https://aistudio.google.com/](https://aistudio.google.com/)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔧 How It Works

```
Upload CSV/Excel → Data Profiling → AI Agent Cleans Data → Insight Engine
                                                              ↓
                                          SQL Layer ← Cleaned DataFrame → EDA Charts
                                                              ↓
                                                      Download CSV + HTML Report
```

1. **Upload** a messy CSV or Excel file via the sidebar
2. **Profile** — the app instantly shows data quality metrics
3. **Clean** — the Gemini AI agent writes and executes pandas cleaning code in a sandboxed environment
4. **Reflect** — if the code fails, the agent self-corrects (up to N retries)
5. **Analyse** — the Insight Engine detects outliers, skewness, correlations, and generates business recommendations
6. **Query** — cleaned data is saved to SQLite with auto-generated queries
7. **Visualize** — Plotly charts for numeric/categorical distributions and correlation matrices
8. **Export** — download cleaned CSV and a full HTML EDA report

---

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── agent/
│   ├── __init__.py
│   └── cleaning_agent.py     # Gemini-powered agentic cleaning loop
├── utils/
│   ├── __init__.py
│   ├── data_profiler.py      # Dataset profiling utilities
│   ├── insight_engine.py     # Statistical insight generation
│   ├── reporting.py          # HTML report generator
│   ├── sql_layer.py          # SQLite storage & query engine
│   └── visualizer.py         # Plotly chart generators
├── reports/                  # Generated HTML reports (gitignored)
├── .env.example              # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛡️ Security

- AI-generated code runs in a **sandboxed namespace** — no access to `os`, `open()`, or other builtins
- Only `pandas` and the DataFrame are exposed to the execution environment
- API keys are never stored in code — loaded from `.env` or entered via the sidebar

---

## 🧰 Tech Stack

- **Frontend:** Streamlit with custom CSS (dark theme)
- **AI Agent:** Google Gemini (2.0 Flash / 2.5 Pro) via LangChain
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Database:** SQLite (built-in)
- **Reporting:** Jinja2 HTML templates

---
---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a PR.
