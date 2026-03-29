import os
import pandas as pd
from jinja2 import Template
from datetime import datetime

REPORT_DIR = "reports"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Data Quality & EDA Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');
    :root {
      --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
      --accent: #6366f1; --accent2: #22d3ee; --text: #e2e8f0;
      --muted: #64748b; --success: #22c55e; --warn: #f59e0b; --danger: #ef4444;
      --border: rgba(255,255,255,0.07);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; line-height: 1.6; }
    .container { max-width: 1100px; margin: 0 auto; padding: 48px 24px; }
    header { text-align: center; margin-bottom: 56px; }
    header .badge { display: inline-block; background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: white; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700;
      letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px; }
    header h1 { font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #fff 30%, var(--accent2));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    header p { color: var(--muted); margin-top: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 40px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 24px; }
    .card .label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 8px; }
    .card .value { font-size: 2rem; font-weight: 700; }
    .card.accent .value { color: var(--accent2); }
    .card.warn .value { color: var(--warn); }
    .card.success .value { color: var(--success); }
    .card.danger .value { color: var(--danger); }
    section { margin-bottom: 48px; }
    section h2 { font-size: 1.1rem; font-weight: 600; color: var(--text); border-left: 3px solid var(--accent);
      padding-left: 12px; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead tr { background: var(--surface2); }
    th { padding: 12px 16px; text-align: left; font-weight: 600; color: var(--muted);
      font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    td { padding: 10px 16px; border-bottom: 1px solid var(--border); font-family: 'DM Mono', monospace; }
    tr:hover td { background: var(--surface2); }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
    .pill.numeric { background: rgba(99,102,241,0.15); color: var(--accent); }
    .pill.object  { background: rgba(34,211,238,0.12); color: var(--accent2); }
    .pill.other   { background: rgba(100,116,139,0.2); color: var(--muted); }
    .bar-wrap { background: rgba(255,255,255,0.05); border-radius: 4px; height: 6px; width: 100%; min-width: 80px; }
    .bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--warn), var(--danger)); }
    footer { text-align: center; color: var(--muted); font-size: 12px; margin-top: 64px; border-top: 1px solid var(--border); padding-top: 24px; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">AI Data Cleaning Engine</div>
    <h1>Data Quality & EDA Report</h1>
    <p>Generated on {{ generated_at }} &nbsp;·&nbsp; Dataset: {{ filename }}</p>
  </header>

  <div class="grid">
    <div class="card accent"><div class="label">Total Rows</div><div class="value">{{ rows }}</div></div>
    <div class="card accent"><div class="label">Columns</div><div class="value">{{ cols }}</div></div>
    <div class="card {% if missing_pct > 10 %}danger{% elif missing_pct > 0 %}warn{% else %}success{% endif %}">
      <div class="label">Missing Cells</div><div class="value">{{ missing_pct }}%</div>
    </div>
    <div class="card {% if duplicates > 0 %}warn{% else %}success{% endif %}">
      <div class="label">Duplicates</div><div class="value">{{ duplicates }}</div>
    </div>
  </div>

  <section>
    <h2>Column Overview</h2>
    <table>
      <thead><tr><th>#</th><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th><th>Missing %</th></tr></thead>
      <tbody>
        {% for col in columns %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ col.name }}</td>
          <td><span class="pill {{ col.dtype_class }}">{{ col.dtype }}</span></td>
          <td>{{ col.unique }}</td>
          <td>{{ col.missing }}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="bar-wrap"><div class="bar-fill" style="width:{{ col.missing_pct }}%"></div></div>
              <span>{{ col.missing_pct }}%</span>
            </div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  {% if numeric_stats %}
  <section>
    <h2>Numeric Statistics</h2>
    <table>
      <thead><tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>Median</th><th>75%</th><th>Max</th></tr></thead>
      <tbody>
        {% for s in numeric_stats %}
        <tr>
          <td>{{ s.col }}</td><td>{{ s.mean }}</td><td>{{ s.std }}</td><td>{{ s.min }}</td>
          <td>{{ s.q25 }}</td><td>{{ s.median }}</td><td>{{ s.q75 }}</td><td>{{ s.max }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  <footer>
    Report auto-generated by <strong>AI Data Cleaning &amp; EDA Engine</strong> · Powered by Gemini 1.5 Pro
  </footer>
</div>
</body>
</html>
"""


def generate_report(df: pd.DataFrame, filename: str = "cleaning_report.html") -> str:
    """
    Generates a standalone HTML EDA report and saves it to the reports/ directory.
    Returns the path to the saved report file.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, filename)

    # Build column metadata
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / len(df) * 100, 1)
        unique = int(df[col].nunique())

        if "int" in dtype or "float" in dtype:
            dtype_class = "numeric"
        elif "object" in dtype or "category" in dtype:
            dtype_class = "object"
        else:
            dtype_class = "other"

        columns.append({
            "name": col,
            "dtype": dtype,
            "dtype_class": dtype_class,
            "unique": unique,
            "missing": missing,
            "missing_pct": missing_pct,
        })

    # Numeric stats
    numeric_stats = []
    for col in df.select_dtypes(include="number").columns:
        desc = df[col].describe()
        numeric_stats.append({
            "col": col,
            "mean": round(desc["mean"], 3),
            "std": round(desc["std"], 3),
            "min": round(desc["min"], 3),
            "q25": round(desc["25%"], 3),
            "median": round(desc["50%"], 3),
            "q75": round(desc["75%"], 3),
            "max": round(desc["max"], 3),
        })

    template = Template(HTML_TEMPLATE)
    html = template.render(
        generated_at=datetime.now().strftime("%B %d, %Y at %H:%M"),
        filename=filename.replace("_", " ").replace(".html", ""),
        rows=len(df),
        cols=len(df.columns),
        missing_pct=round(df.isnull().sum().sum() / df.size * 100, 1),
        duplicates=int(df.duplicated().sum()),
        columns=columns,
        numeric_stats=numeric_stats,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
