import pandas as pd
import io


def get_data_profile(df: pd.DataFrame) -> str:
    """
    Returns a rich text summary of the dataframe for LLM context and UI display.
    Includes shape, dtypes, missing values, duplicates, and descriptive stats.
    """
    buffer = io.StringIO()

    buffer.write("=" * 60 + "\n")
    buffer.write("  DATA PROFILE REPORT\n")
    buffer.write("=" * 60 + "\n\n")

    # Shape
    buffer.write(f"Shape           : {df.shape[0]} rows × {df.shape[1]} columns\n")
    buffer.write(f"Duplicate Rows  : {df.duplicated().sum()}\n")
    buffer.write(f"Total Cells     : {df.size}\n")
    buffer.write(f"Total Missing   : {df.isnull().sum().sum()} "
                 f"({df.isnull().sum().sum() / df.size * 100:.1f}%)\n\n")

    # Column-level detail
    buffer.write("-" * 60 + "\n")
    buffer.write(f"{'Column':<25} {'DType':<15} {'Missing':<10} {'%Missing':<10} {'Unique'}\n")
    buffer.write("-" * 60 + "\n")

    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = missing / len(df) * 100
        unique = df[col].nunique()
        dtype = str(df[col].dtype)
        buffer.write(f"{col:<25} {dtype:<15} {missing:<10} {pct:<10.1f} {unique}\n")

    buffer.write("\n")

    # Descriptive stats for numeric columns
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        buffer.write("-" * 60 + "\n")
        buffer.write("  NUMERIC STATISTICS\n")
        buffer.write("-" * 60 + "\n")
        buffer.write(numeric_df.describe().round(2).to_string())
        buffer.write("\n\n")

    # Sample values for categorical columns
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        buffer.write("-" * 60 + "\n")
        buffer.write("  CATEGORICAL COLUMNS — TOP VALUES\n")
        buffer.write("-" * 60 + "\n")
        for col in cat_df.columns:
            top = df[col].value_counts().head(5)
            buffer.write(f"\n  [{col}]\n")
            for val, cnt in top.items():
                buffer.write(f"    {str(val):<30} → {cnt}\n")

    buffer.write("\n" + "=" * 60 + "\n")
    return buffer.getvalue()


def get_profile_dict(df: pd.DataFrame) -> dict:
    """
    Returns structured profile data as a dict for use in UI metrics.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_by_col = df.isnull().sum()
    missing_pct_by_col = (missing_by_col / len(df) * 100).round(1)

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "total_missing": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 1),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "missing_by_col": missing_by_col[missing_by_col > 0].to_dict(),
        "missing_pct_by_col": missing_pct_by_col[missing_pct_by_col > 0].to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
