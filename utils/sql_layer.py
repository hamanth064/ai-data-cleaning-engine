"""
SQL Layer — stores cleaned DataFrames into a local SQLite database
and provides a library of analytical queries (trends, aggregations,
distributions) that a data analyst would run manually.
"""
import sqlite3
import os
import re
import pandas as pd
from typing import Optional

DB_PATH = "cleaned_data.db"


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _safe_table_name(name: str) -> str:
    """Sanitise a string into a valid SQLite table name."""
    name = re.sub(r"[^\w]", "_", name.strip().lower())
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "dataset"


# ── Write ─────────────────────────────────────────────────────────────────────

def save_to_sqlite(df: pd.DataFrame, table_name: str = "cleaned_data") -> str:
    """
    Saves a DataFrame to a local SQLite database.
    Returns the table name used.
    """
    table_name = _safe_table_name(table_name)
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return table_name


# ── Auto-query library ────────────────────────────────────────────────────────

def run_query(sql: str) -> tuple[pd.DataFrame, str]:
    """
    Execute a SQL query against the local DB.
    Returns (result_df, error_string). error_string is empty on success.
    """
    try:
        conn = get_connection()
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result, ""
    except Exception as e:
        return pd.DataFrame(), str(e)


def get_table_names() -> list[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_schema(table_name: str) -> str:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    rows = cursor.fetchall()
    conn.close()
    lines = ["Column Info:"]
    for row in rows:
        lines.append(f"  {row[1]} ({row[2]})")
    return "\n".join(lines)


def generate_auto_queries(df: pd.DataFrame, table_name: str) -> list[dict]:
    """
    Generates a library of meaningful SQL queries based on the DataFrame's
    actual columns — the kind a data analyst would write manually.
    Returns list of {title, description, sql, category}.
    """
    t = _safe_table_name(table_name)
    queries = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    # ── 1. Row count & overview ──────────────────────────────────────────────
    queries.append({
        "title": "Dataset Overview",
        "description": "Total row count and column count of the cleaned dataset.",
        "category": "Overview",
        "sql": f"SELECT COUNT(*) AS total_rows, {len(all_cols)} AS total_columns FROM {t};",
    })

    # ── 2. Null audit ────────────────────────────────────────────────────────
    null_checks = [f"SUM(CASE WHEN [{c}] IS NULL THEN 1 ELSE 0 END) AS null_{_safe_table_name(c)}" for c in all_cols[:10]]
    queries.append({
        "title": "Null Value Audit",
        "description": "Count of NULL values per column — data quality check.",
        "category": "Data Quality",
        "sql": f"SELECT {', '.join(null_checks)} FROM {t};",
    })

    # ── 3. Numeric aggregations ──────────────────────────────────────────────
    for col in numeric_cols[:4]:
        queries.append({
            "title": f"Stats — {col}",
            "description": f"Min, Max, Average, and Std deviation for '{col}'.",
            "category": "Aggregation",
            "sql": (
                f"SELECT "
                f"MIN([{col}]) AS min_val, "
                f"MAX([{col}]) AS max_val, "
                f"ROUND(AVG([{col}]), 2) AS avg_val, "
                f"COUNT([{col}]) AS count_non_null "
                f"FROM {t};"
            ),
        })

    # ── 4. Categorical distributions ────────────────────────────────────────
    for col in cat_cols[:3]:
        queries.append({
            "title": f"Category Breakdown — {col}",
            "description": f"Frequency distribution of values in '{col}' — top 10.",
            "category": "Distribution",
            "sql": (
                f"SELECT [{col}], COUNT(*) AS frequency, "
                f"ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {t}), 1) AS pct "
                f"FROM {t} "
                f"GROUP BY [{col}] "
                f"ORDER BY frequency DESC "
                f"LIMIT 10;"
            ),
        })

    # ── 5. Numeric × Categorical aggregation ────────────────────────────────
    if numeric_cols and cat_cols:
        num_col = numeric_cols[0]
        cat_col = cat_cols[0]
        queries.append({
            "title": f"Avg {num_col} by {cat_col}",
            "description": f"Average of '{num_col}' grouped by '{cat_col}' — reveals segment differences.",
            "category": "Trend Analysis",
            "sql": (
                f"SELECT [{cat_col}], "
                f"ROUND(AVG([{num_col}]), 2) AS avg_{_safe_table_name(num_col)}, "
                f"COUNT(*) AS row_count "
                f"FROM {t} "
                f"GROUP BY [{cat_col}] "
                f"ORDER BY avg_{_safe_table_name(num_col)} DESC;"
            ),
        })

    # ── 6. Outlier detection via SQL ─────────────────────────────────────────
    for col in numeric_cols[:2]:
        queries.append({
            "title": f"Outliers — {col}",
            "description": f"Rows where '{col}' is more than 3 standard deviations from the mean.",
            "category": "Outlier Detection",
            "sql": (
                f"SELECT * FROM {t} "
                f"WHERE ABS([{col}] - (SELECT AVG([{col}]) FROM {t})) "
                f"> 3 * (SELECT AVG(([{col}] - (SELECT AVG([{col}]) FROM {t})) * ([{col}] - (SELECT AVG([{col}]) FROM {t}))) FROM {t}) "
                f"LIMIT 20;"
            ),
        })

    # ── 7. Duplicate detection ───────────────────────────────────────────────
    if len(all_cols) >= 2:
        group_cols = ", ".join(f"[{c}]" for c in all_cols[:5])
        queries.append({
            "title": "Duplicate Row Detection",
            "description": "Find groups of rows that share identical values across key columns.",
            "category": "Data Quality",
            "sql": (
                f"SELECT {group_cols}, COUNT(*) AS occurrences "
                f"FROM {t} "
                f"GROUP BY {group_cols} "
                f"HAVING COUNT(*) > 1 "
                f"ORDER BY occurrences DESC "
                f"LIMIT 20;"
            ),
        })

    # ── 8. Percentile analysis ───────────────────────────────────────────────
    for col in numeric_cols[:2]:
        queries.append({
            "title": f"Percentile Buckets — {col}",
            "description": f"Distribute '{col}' into quartile buckets for segment analysis.",
            "category": "Distribution",
            "sql": (
                f"SELECT "
                f"CASE "
                f"  WHEN [{col}] <= (SELECT [{col}] FROM {t} ORDER BY [{col}] LIMIT 1 OFFSET CAST(COUNT(*)*0.25 AS INT)) THEN 'Q1 (Bottom 25%)' "
                f"  WHEN [{col}] <= (SELECT [{col}] FROM {t} ORDER BY [{col}] LIMIT 1 OFFSET CAST(COUNT(*)*0.50 AS INT)) THEN 'Q2 (25-50%)' "
                f"  WHEN [{col}] <= (SELECT [{col}] FROM {t} ORDER BY [{col}] LIMIT 1 OFFSET CAST(COUNT(*)*0.75 AS INT)) THEN 'Q3 (50-75%)' "
                f"  ELSE 'Q4 (Top 25%)' "
                f"END AS quartile, "
                f"COUNT(*) AS count "
                f"FROM {t} "
                f"GROUP BY quartile;"
            ),
        })

    return queries


def delete_db():
    """Remove the local SQLite database file."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
