"""
Insight Engine — generates human-style analytical observations,
business recommendations, and statistical flags from a cleaned DataFrame.
This is the "analyst thinking" layer — NOT the AI agent.
"""
import pandas as pd
import numpy as np
from typing import Any


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct(val: float) -> str:
    return f"{val:.1f}%"

def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


# ── Core insight functions ────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame) -> list[dict]:
    """IQR-based outlier detection for numeric columns."""
    results = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = s[(s < lower) | (s > upper)]
        if len(outliers) > 0:
            results.append({
                "column": col,
                "count": len(outliers),
                "pct": round(len(outliers) / len(s) * 100, 1),
                "min_outlier": round(float(outliers.min()), 3),
                "max_outlier": round(float(outliers.max()), 3),
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
            })
    return sorted(results, key=lambda x: x["pct"], reverse=True)


def detect_skewness(df: pd.DataFrame) -> list[dict]:
    """Flag highly skewed numeric columns."""
    results = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        skew = float(s.skew())
        if abs(skew) > 1.0:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            severity = "highly" if abs(skew) > 2 else "moderately"
            results.append({
                "column": col,
                "skew": round(skew, 3),
                "direction": direction,
                "severity": severity,
                "recommendation": (
                    f"Consider log-transform for '{col}' — {severity} skewed {direction}. "
                    "This can improve model performance significantly."
                ),
            })
    return sorted(results, key=lambda x: abs(x["skew"]), reverse=True)


def detect_correlations(df: pd.DataFrame, threshold: float = 0.7) -> list[dict]:
    """Find strong correlations between numeric columns."""
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) < 2:
        return []
    corr = numeric.corr()
    results = []
    seen = set()
    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            if i >= j:
                continue
            key = tuple(sorted([col1, col2]))
            if key in seen:
                continue
            seen.add(key)
            r = corr.loc[col1, col2]
            if abs(r) >= threshold and not np.isnan(r):
                strength = "very strong" if abs(r) > 0.9 else "strong"
                direction = "positive" if r > 0 else "negative"
                results.append({
                    "col1": col1,
                    "col2": col2,
                    "r": round(r, 3),
                    "strength": strength,
                    "direction": direction,
                    "insight": (
                        f"'{col1}' and '{col2}' have a {strength} {direction} correlation "
                        f"(r={r:.2f}). {'Consider removing one to avoid multicollinearity.' if abs(r)>0.9 else 'This relationship may be worth investigating further.'}"
                    ),
                })
    return sorted(results, key=lambda x: abs(x["r"]), reverse=True)


def detect_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> list[dict]:
    """Flag columns with near-zero variance — often useless for modeling."""
    results = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        cv = s.std() / s.mean() if s.mean() != 0 else 0
        if abs(cv) < threshold:
            results.append({
                "column": col,
                "cv": round(float(cv), 5),
                "insight": f"'{col}' has near-zero variance (CV={cv:.4f}). Likely a constant or near-constant — low predictive value.",
            })
    return results


def get_categorical_insights(df: pd.DataFrame) -> list[dict]:
    """Detect dominance, high cardinality, and rare categories."""
    results = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts()
        n = len(df[col].dropna())
        if n == 0:
            continue
        top_val = vc.index[0]
        top_pct = vc.iloc[0] / n * 100
        cardinality = df[col].nunique()

        if top_pct > 80:
            results.append({
                "column": col,
                "type": "dominant_category",
                "insight": f"'{col}': value '{top_val}' dominates at {top_pct:.1f}% of rows. Column may have low discriminative power.",
                "severity": "warn",
            })
        if cardinality > 50:
            results.append({
                "column": col,
                "type": "high_cardinality",
                "insight": f"'{col}' has {cardinality} unique values — high cardinality. Consider encoding or grouping rare categories.",
                "severity": "info",
            })

        # Rare categories (< 1%)
        rare = vc[vc / n < 0.01]
        if len(rare) > 0:
            results.append({
                "column": col,
                "type": "rare_categories",
                "insight": f"'{col}' has {len(rare)} rare categories (< 1% each). Consider grouping them into an 'Other' bucket.",
                "severity": "info",
            })
    return results


def generate_business_recommendations(df: pd.DataFrame, outliers: list, skewed: list, correlations: list) -> list[str]:
    """Generate plain-English business-level recommendations an analyst would make."""
    recs = []
    n_rows, n_cols = df.shape

    # Size
    if n_rows < 1000:
        recs.append(f"⚠️ Dataset is small ({n_rows:,} rows). Statistical conclusions may not generalise well — consider collecting more data before drawing firm business decisions.")
    elif n_rows > 100_000:
        recs.append(f"✅ Dataset is large ({n_rows:,} rows) — statistical patterns are likely reliable.")

    # Outliers
    severe = [o for o in outliers if o["pct"] > 5]
    if severe:
        cols = ", ".join(f"'{o['column']}'" for o in severe[:3])
        recs.append(f"🔍 Outlier alert in {cols}: over 5% of values are statistical outliers. Investigate whether these represent genuine extreme events or data entry errors before modelling.")

    # Skewness
    if skewed:
        cols = ", ".join(f"'{s['column']}'" for s in skewed[:3])
        recs.append(f"📐 Skewed distributions in {cols} suggest the data is not normally distributed. If using linear regression or similar models, apply log or Box-Cox transforms.")

    # Correlations
    if correlations:
        c = correlations[0]
        recs.append(f"🔗 Strong correlation detected: '{c['col1']}' ↔ '{c['col2']}' (r={c['r']}). In a predictive model, including both may cause multicollinearity — consider dropping one or using PCA.")

    # General
    numeric_ratio = len(df.select_dtypes(include="number").columns) / n_cols if n_cols > 0 else 0
    if numeric_ratio > 0.8:
        recs.append("📊 Dataset is heavily numeric — well-suited for regression, clustering, or anomaly detection tasks.")
    elif numeric_ratio < 0.3:
        recs.append("📝 Dataset is mostly categorical — NLP encoding (one-hot, target encoding) will be important for any ML pipeline.")

    if not recs:
        recs.append("✅ Dataset looks clean and well-structured. No major data quality concerns detected.")

    return recs


def generate_full_insights(df: pd.DataFrame) -> dict:
    """
    Master function — runs all insight generators and returns a structured dict.
    """
    outliers     = detect_outliers(df)
    skewed       = detect_skewness(df)
    correlations = detect_correlations(df)
    low_var      = detect_low_variance(df)
    cat_insights = get_categorical_insights(df)
    recommendations = generate_business_recommendations(df, outliers, skewed, correlations)

    # Summary stats per numeric col
    numeric_summary = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        numeric_summary.append({
            "column": col,
            "mean":   round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "std":    round(float(s.std()), 3),
            "min":    round(float(s.min()), 3),
            "max":    round(float(s.max()), 3),
            "mean_vs_median_gap": round(abs(float(s.mean()) - float(s.median())), 3),
        })

    return {
        "outliers":          outliers,
        "skewed":            skewed,
        "correlations":      correlations,
        "low_variance":      low_var,
        "categorical":       cat_insights,
        "recommendations":   recommendations,
        "numeric_summary":   numeric_summary,
        "total_issues":      len(outliers) + len(skewed) + len(low_var) + len(cat_insights),
    }
