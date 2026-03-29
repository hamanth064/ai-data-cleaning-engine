import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from agent.cleaning_agent import run_cleaning_agent
from utils.data_profiler import get_data_profile, get_profile_dict
from utils.reporting import generate_report
from utils.visualizer import (
    get_numeric_distributions,
    get_categorical_distributions,
    get_correlation_heatmap,
    get_missing_values_bar,
    get_dtypes_pie,
)
from utils.insight_engine import generate_full_insights
from utils.sql_layer import save_to_sqlite, run_query, generate_auto_queries, get_schema

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Cleaning Engine",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root {
  --bg:#0f1117; --surface:#1a1d27; --surface2:#22263a;
  --accent:#6366f1; --accent2:#22d3ee; --accent3:#f59e0b;
  --text:#e2e8f0; --muted:#64748b;
  --success:#22c55e; --warn:#f59e0b; --danger:#ef4444;
  --border:rgba(255,255,255,0.07); --radius:14px;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--surface);border-radius:12px;padding:6px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;padding:8px 18px!important;font-weight:600!important;font-size:13px!important;color:var(--muted)!important;background:transparent!important;border:none!important;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:white!important;}
.stButton>button{width:100%;border-radius:10px!important;height:3em;background:linear-gradient(135deg,var(--accent),#818cf8)!important;color:white!important;font-weight:700!important;font-size:14px!important;border:none!important;transition:opacity 0.2s,transform 0.15s;}
.stButton>button:hover{opacity:.9;transform:translateY(-1px);}
.stDownloadButton>button{width:100%;border-radius:10px!important;height:3em;background:linear-gradient(135deg,#059669,#22c55e)!important;color:white!important;font-weight:700!important;font-size:14px!important;border:none!important;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;padding:20px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700!important;font-size:2rem!important;}
.stDataFrame{border-radius:var(--radius)!important;overflow:hidden;border:1px solid var(--border)!important;}
.stSuccess{background:rgba(34,197,94,.12)!important;border:1px solid rgba(34,197,94,.3)!important;border-radius:10px!important;}
.stError{background:rgba(239,68,68,.10)!important;border:1px solid rgba(239,68,68,.3)!important;border-radius:10px!important;}
.stWarning{background:rgba(245,158,11,.10)!important;border:1px solid rgba(245,158,11,.3)!important;border-radius:10px!important;}
.stInfo{background:rgba(99,102,241,.10)!important;border:1px solid rgba(99,102,241,.3)!important;border-radius:10px!important;}
hr{border-color:var(--border)!important;}
.stTextInput input,.stTextArea textarea{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}
.hero-title{font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#fff 20%,var(--accent2) 80%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;margin-bottom:6px;}
.hero-sub{color:var(--muted);font-size:1rem;margin-bottom:32px;}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:var(--accent2);margin-bottom:8px;}
.pill{display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;margin-right:4px;}
.pill-indigo{background:rgba(99,102,241,.15);color:#818cf8;}
.pill-cyan{background:rgba(34,211,238,.12);color:#67e8f9;}
.pill-green{background:rgba(34,197,94,.12);color:#4ade80;}
.pill-amber{background:rgba(245,158,11,.12);color:#fcd34d;}
.insight-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin:8px 0;font-size:14px;line-height:1.6;}
.insight-card.warn{border-left:3px solid var(--warn);}
.insight-card.info{border-left:3px solid var(--accent2);}
.insight-card.good{border-left:3px solid var(--success);}
.insight-card.danger{border-left:3px solid var(--danger);}
.rec-card{background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(34,211,238,.05));border:1px solid rgba(99,102,241,.2);border-radius:12px;padding:16px 20px;margin:8px 0;font-size:14px;}
.score-badge{display:inline-block;padding:8px 20px;border-radius:30px;font-size:1.4rem;font-weight:800;}
.score-good{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.3);}
.score-medium{background:rgba(245,158,11,.15);color:#fcd34d;border:1px solid rgba(245,158,11,.3);}
.score-bad{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.3);}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "df": None, "cleaned_df": None,
        "agent_logs": [], "generated_code": "",
        "cleaning_done": False, "insights": None,
        "sql_table": None, "auto_queries": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 AI Data Engine")
    st.markdown("<p style='color:#64748b;font-size:13px;margin-top:-8px;'>Powered by Gemini 1.5 Pro</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='section-label'>🔑 Authentication</div>", unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", value=os.getenv("GOOGLE_API_KEY", ""),
                            type="password", placeholder="AIza...",
                            help="Get your free key at https://aistudio.google.com/")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API key loaded ✓", icon="🔒")
    st.divider()
    st.markdown("<div class='section-label'>📂 Dataset</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a messy CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    if uploaded_file and st.button("🔄 Reload Original Data"):
        for k in ["df","cleaned_df","agent_logs","generated_code","cleaning_done","insights","sql_table","auto_queries"]:
            st.session_state[k] = None if k not in ["agent_logs","auto_queries"] else []
        st.session_state.cleaning_done = False
        st.rerun()
    st.divider()
    user_goal = st.text_area("Cleaning Goal",
        value="Clean missing values, remove duplicates, fix inconsistent data types, and standardize text columns.",
        height=100)
    model_name = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06"], index=0)
    max_retries = st.slider("Reflection Retries", 1, 5, 3)
    st.divider()
    st.markdown("""<div style='font-size:11px;color:#475569;line-height:1.8;'>
    <b>Full Pipeline</b><br>
    1. Upload CSV → Profile<br>
    2. AI Agent cleans data<br>
    3. Insight Engine analyses<br>
    4. SQL Layer stores & queries<br>
    5. EDA charts generated<br>
    6. Download CSV + HTML report
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='hero-title'>AI Data Cleaning & EDA Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Upload → Clean → Analyse → Query → Insight. Full data pipeline in one tool.</div>", unsafe_allow_html=True)
st.markdown("""
<span class='pill pill-indigo'>Gemini 1.5 Pro</span>
<span class='pill pill-cyan'>Reflection Loop</span>
<span class='pill pill-green'>SQL Layer</span>
<span class='pill pill-amber'>Insight Engine</span>
<span class='pill pill-indigo'>Auto EDA</span>
<span class='pill pill-cyan'>HTML Reports</span>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file and st.session_state.df is None:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

        # Auto-fix: if data loaded as a single column with commas, split it
        temp_df = st.session_state.df
        if len(temp_df.columns) == 1:
            col_name = str(temp_df.columns[0])
            if "," in col_name:
                import io
                header = col_name
                rows = [header] + temp_df.iloc[:, 0].astype(str).tolist()
                csv_text = "\n".join(rows)
                st.session_state.df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        st.error(f"Error loading file: {e}")

if st.session_state.df is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div style='text-align:center;background:#1a1d27;border:1px dashed rgba(99,102,241,0.4);
             border-radius:20px;padding:60px 40px;margin-top:20px;'>
          <div style='font-size:3rem;margin-bottom:16px;'>📂</div>
          <div style='font-size:1.2rem;font-weight:600;color:#e2e8f0;margin-bottom:8px;'>Upload a CSV to begin</div>
          <div style='color:#64748b;font-size:14px;'>Use the sidebar uploader to load your messy dataset.</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Data Profile",
    "🤖  Agentic Cleaning",
    "🧠  Insights & Analysis",
    "🗄️  SQL Explorer",
    "📊  EDA & Results",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    profile_dict = get_profile_dict(df)
    st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows",           profile_dict["rows"])
    c2.metric("Columns",        profile_dict["cols"])
    c3.metric("Missing Cells",  profile_dict["total_missing"])
    c4.metric("Missing %",      f"{profile_dict['missing_pct']}%")
    c5.metric("Duplicate Rows", profile_dict["duplicates"])
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown("<div class='section-label'>Raw Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=380)
    with right:
        mf = get_missing_values_bar(df)
        df2 = get_dtypes_pie(df)
        if mf:  st.plotly_chart(mf,  use_container_width=True)
        if df2: st.plotly_chart(df2, use_container_width=True)
    with st.expander("Show full text profile"):
        st.code(get_data_profile(df), language=None)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Autonomous AI Cleaning Agent</div>", unsafe_allow_html=True)
    col_btn, col_info = st.columns([1,3], gap="large")
    with col_btn:
        run_clicked = st.button("🚀 Run Cleaning Agent", use_container_width=True)
        if st.session_state.cleaning_done and st.session_state.cleaned_df is not None:
            st.success(f"✅ Cleaned!\n{df.shape} → {st.session_state.cleaned_df.shape}")
    with col_info:
        if not api_key:
            st.warning("⚠️ Add your **Gemini API key** in the sidebar.")
        else:
            st.info(f"Goal: *\"{user_goal[:80]}...\"* — up to **{max_retries} retries**.")

    if run_clicked:
        if not api_key:
            st.error("❌ Gemini API Key is required.")
        else:
            st.session_state.agent_logs = []
            st.session_state.cleaning_done = False
            progress = st.progress(0, text="Initialising agent...")
            with st.spinner(""):
                progress.progress(15, text="Profiling dataset...")
                new_df, code, logs = run_cleaning_agent(df, user_goal, max_retries, model_name=model_name)
                progress.progress(60, text="Running insight engine...")
                insights = generate_full_insights(new_df)
                progress.progress(80, text="Saving to SQLite...")
                fname = uploaded_file.name.replace(".csv","") if uploaded_file else "dataset"
                table = save_to_sqlite(new_df, fname)
                auto_queries = generate_auto_queries(new_df, fname)
                progress.progress(100, text="Pipeline complete!")
            st.session_state.cleaned_df    = new_df
            st.session_state.agent_logs    = logs
            st.session_state.generated_code = code
            st.session_state.cleaning_done  = True
            st.session_state.insights       = insights
            st.session_state.sql_table      = table
            st.session_state.auto_queries   = auto_queries
            st.rerun()

    if st.session_state.agent_logs:
        st.markdown("<br><div class='section-label'>Agent Execution Log</div>", unsafe_allow_html=True)
        for log in st.session_state.agent_logs:
            if "❌" in log:     st.error(log)
            elif "✅" in log:   st.success(log)
            elif "🔄" in log or "🔁" in log: st.info(log)
            elif "📋" in log:
                st.markdown(f"<div class='insight-card info'>{log}</div>", unsafe_allow_html=True)
            elif "💻" in log:
                st.markdown("**💻 Generated Code:**")
                if st.session_state.generated_code:
                    st.code(st.session_state.generated_code, language="python")
            else:
                st.markdown(log)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if st.session_state.insights is None:
        st.markdown("""<div style='text-align:center;background:#1a1d27;border:1px dashed rgba(34,211,238,0.3);
             border-radius:20px;padding:60px 40px;'>
          <div style='font-size:3rem;'>🧠</div><br>
          <div style='font-size:1.1rem;font-weight:600;color:#e2e8f0;'>No insights yet</div>
          <div style='color:#64748b;font-size:14px;'>Run the cleaning agent first.</div>
        </div>""", unsafe_allow_html=True)
    else:
        ins = st.session_state.insights

        # Health score
        st.markdown("<div class='section-label'>Data Health Score</div>", unsafe_allow_html=True)
        score = max(0, 100 - ins["total_issues"] * 8)
        sc = "score-good" if score>=75 else "score-medium" if score>=50 else "score-bad"
        sl = "Healthy ✅" if score>=75 else "Needs Attention ⚠️" if score>=50 else "Poor Quality ❌"
        sc1,sc2,sc3,sc4 = st.columns(4)
        sc1.markdown(f"<div style='text-align:center'><div class='score-badge {sc}'>{score}/100</div><br><small style='color:var(--muted)'>{sl}</small></div>", unsafe_allow_html=True)
        sc2.metric("Outlier Columns",     len(ins["outliers"]))
        sc3.metric("Skewed Columns",      len(ins["skewed"]))
        sc4.metric("Strong Correlations", len(ins["correlations"]))
        st.divider()

        # Business recommendations
        st.markdown("<div class='section-label'>💡 Business Recommendations</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>Analyst-level insights derived from statistical analysis — your interpretation of the data.</p>", unsafe_allow_html=True)
        for rec in ins["recommendations"]:
            st.markdown(f"<div class='rec-card'>{rec}</div>", unsafe_allow_html=True)
        st.divider()

        # Outliers
        if ins["outliers"]:
            st.markdown("<div class='section-label'>🔍 Outlier Analysis</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(ins["outliers"]), use_container_width=True)
            for o in ins["outliers"][:3]:
                sev = "danger" if o["pct"]>10 else "warn"
                st.markdown(f"<div class='insight-card {sev}'>⚠️ <b>{o['column']}</b>: {o['count']} outliers ({o['pct']}% of data) outside [{o['lower_bound']}, {o['upper_bound']}]</div>", unsafe_allow_html=True)
            st.divider()

        # Skewness
        if ins["skewed"]:
            st.markdown("<div class='section-label'>📐 Distribution Skewness</div>", unsafe_allow_html=True)
            for s in ins["skewed"]:
                st.markdown(f"<div class='insight-card warn'>📐 <b>{s['column']}</b> is {s['severity']} skewed {s['direction']} (skew={s['skew']}). {s['recommendation']}</div>", unsafe_allow_html=True)
            st.divider()

        # Correlations
        if ins["correlations"]:
            st.markdown("<div class='section-label'>🔗 Correlation Insights</div>", unsafe_allow_html=True)
            for c in ins["correlations"]:
                style = "danger" if abs(c["r"])>0.9 else "warn"
                st.markdown(f"<div class='insight-card {style}'>🔗 {c['insight']}</div>", unsafe_allow_html=True)
            st.divider()

        # Categorical
        if ins["categorical"]:
            st.markdown("<div class='section-label'>📝 Categorical Insights</div>", unsafe_allow_html=True)
            for ci in ins["categorical"]:
                st.markdown(f"<div class='insight-card info'>📝 {ci['insight']}</div>", unsafe_allow_html=True)
            st.divider()

        # Numeric summary
        if ins["numeric_summary"]:
            st.markdown("<div class='section-label'>📊 Numeric Deep-Dive</div>", unsafe_allow_html=True)
            ns = pd.DataFrame(ins["numeric_summary"])
            ns["mean≠median?"] = ns["mean_vs_median_gap"].apply(lambda x: "⚠️ Gap" if x>0.5 else "✅ OK")
            st.dataframe(ns, use_container_width=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    if st.session_state.sql_table is None:
        st.markdown("""<div style='text-align:center;background:#1a1d27;border:1px dashed rgba(245,158,11,0.3);
             border-radius:20px;padding:60px 40px;'>
          <div style='font-size:3rem;'>🗄️</div><br>
          <div style='font-size:1.1rem;font-weight:600;color:#e2e8f0;'>No SQL database yet</div>
          <div style='color:#64748b;font-size:14px;'>Run the cleaning agent — data is auto-saved to SQLite.</div>
        </div>""", unsafe_allow_html=True)
    else:
        table = st.session_state.sql_table
        cdf   = st.session_state.cleaned_df
        st.markdown("<div class='section-label'>SQLite Database</div>", unsafe_allow_html=True)
        db1,db2,db3 = st.columns(3)
        db1.metric("Table",       table)
        db2.metric("Rows Stored", len(cdf))
        db3.metric("Columns",     len(cdf.columns))
        with st.expander("📋 Table Schema"):
            st.code(get_schema(table), language=None)
        st.divider()

        # Auto query library
        st.markdown("<div class='section-label'>📚 Auto-Generated Query Library</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8;font-size:13px;margin-bottom:12px;'>Queries auto-built from your dataset's columns. Click any to run.</p>", unsafe_allow_html=True)
        queries = st.session_state.auto_queries
        if queries:
            cats = ["All"] + list(dict.fromkeys(q["category"] for q in queries))
            sel  = st.selectbox("Filter by category", cats)
            filt = queries if sel=="All" else [q for q in queries if q["category"]==sel]
            for i, q in enumerate(filt):
                with st.expander(f"**{q['title']}** — {q['description']}"):
                    st.code(q["sql"], language="sql")
                    if st.button("▶ Run", key=f"rq_{i}"):
                        res, err = run_query(q["sql"])
                        if err: st.error(err)
                        else:   st.dataframe(res, use_container_width=True)
        st.divider()

        # Custom SQL
        st.markdown("<div class='section-label'>⌨️ Custom SQL Editor</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#94a3b8;font-size:13px;'>Table: <code style='color:#22d3ee'>{table}</code></p>", unsafe_allow_html=True)
        default_sql = f"SELECT * FROM {table} LIMIT 10;"
        custom_sql  = st.text_area("SQL", value=default_sql, height=120, label_visibility="collapsed")
        if st.button("▶ Execute SQL"):
            res, err = run_query(custom_sql)
            if err: st.error(f"❌ {err}")
            else:
                st.success(f"✅ {len(res)} rows returned.")
                st.dataframe(res, use_container_width=True)
                st.download_button("📥 Download Result CSV", data=res.to_csv(index=False),
                                   file_name="query_result.csv", mime="text/csv")

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    if st.session_state.cleaned_df is None:
        st.markdown("""<div style='text-align:center;background:#1a1d27;border:1px dashed rgba(34,211,238,0.3);
             border-radius:20px;padding:60px 40px;'>
          <div style='font-size:3rem;'>📊</div><br>
          <div style='font-size:1.1rem;font-weight:600;color:#e2e8f0;'>No cleaned data yet</div>
          <div style='color:#64748b;'>Run the cleaning agent first.</div>
        </div>""", unsafe_allow_html=True)
    else:
        cdf = st.session_state.cleaned_df
        st.markdown("<div class='section-label'>Before vs After</div>", unsafe_allow_html=True)
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Rows",       cdf.shape[0],  delta=cdf.shape[0]-df.shape[0])
        b2.metric("Columns",    cdf.shape[1],  delta=cdf.shape[1]-df.shape[1])
        b3.metric("Missing",    int(cdf.isnull().sum().sum()), delta=int(cdf.isnull().sum().sum())-int(df.isnull().sum().sum()))
        b4.metric("Duplicates", int(cdf.duplicated().sum()),   delta=int(cdf.duplicated().sum())-int(df.duplicated().sum()))
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Downloads</div>", unsafe_allow_html=True)
        dl1,dl2 = st.columns(2, gap="large")
        with dl1:
            st.download_button("📥 Download Cleaned CSV", data=cdf.to_csv(index=False),
                               file_name="cleaned_data.csv", mime="text/csv", use_container_width=True)
        with dl2:
            try:
                rp = generate_report(cdf, "cleaning_report.html")
                with open(rp,"rb") as f:
                    st.download_button("📄 Download HTML EDA Report", data=f,
                                       file_name="cleaning_report.html", mime="text/html",
                                       use_container_width=True)
            except Exception as e:
                st.error(f"Report generation failed: {e}")

        st.divider()
        st.markdown("<div class='section-label'>Cleaned Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(cdf.head(20), use_container_width=True, height=300)
        st.divider()

        cf = get_correlation_heatmap(cdf)
        if cf:
            st.markdown("<div class='section-label'>Correlation Matrix</div>", unsafe_allow_html=True)
            st.plotly_chart(cf, use_container_width=True)
            st.divider()

        nc = get_numeric_distributions(cdf)
        cc = get_categorical_distributions(cdf)
        if nc:
            st.markdown("<div class='section-label'>Numeric Distributions</div>", unsafe_allow_html=True)
            for i in range(0, len(nc), 2):
                cols = st.columns(2, gap="large")
                cols[0].plotly_chart(nc[i], use_container_width=True)
                if i+1<len(nc): cols[1].plotly_chart(nc[i+1], use_container_width=True)
        if cc:
            st.markdown("<div class='section-label'>Categorical Distributions</div>", unsafe_allow_html=True)
            for i in range(0, len(cc), 2):
                cols = st.columns(2, gap="large")
                cols[0].plotly_chart(cc[i], use_container_width=True)
                if i+1<len(cc): cols[1].plotly_chart(cc[i+1], use_container_width=True)
