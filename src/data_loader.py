"""
ROADGUARD AI - Data Loading & Schema Mapping Module
Handles multi-format dataset loading (CSV/Excel), encoding detection, intelligent fuzzy column mapping,
and comprehensive data profiling.
"""

import io
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from src.config import CANONICAL_COLUMNS, COLUMN_ALIASES

def read_uploaded_file(file_obj, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely reads CSV or Excel files from Streamlit's UploadedFile object with multi-encoding fallback.
    Returns (DataFrame, error_message).
    """
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".csv":
            # Multi-encoding attempt
            encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
            for enc in encodings:
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding=enc)
                    return df, None
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            return None, f"Could not parse CSV file '{filename}'. Encoding or formatting unsupported."

        elif ext in [".xlsx", ".xls"]:
            file_obj.seek(0)
            df = pd.read_excel(file_obj)
            return df, None
        else:
            return None, f"Unsupported file format '{ext}'. Please upload a .csv or .xlsx/.xls file."

    except Exception as e:
        return None, f"File reading failed: {str(e)}"


def load_sample_dataset() -> pd.DataFrame:
    """Loads the pre-packaged synthetic academic demonstration dataset."""
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_accidents.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        # Fallback inline generation if file not found
        from data.generate_sample import generate_synthetic_accidents
        df = generate_synthetic_accidents()
        return df


def auto_map_columns(raw_columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently maps raw column names to canonical schema columns using synonym dictionaries
    and fuzzy normalization.
    Returns a dictionary mapping: canonical_col -> raw_col (or None if unmapped).
    """
    mapping = {}
    normalized_raw = {
        col: col.strip().lower().replace("_", " ").replace("-", " ")
        for col in raw_columns
    }

    used_raw_cols = set()

    for canonical_col, aliases in COLUMN_ALIASES.items():
        matched_raw = None
        
        # 1. Exact match on canonical name
        for raw_col, norm in normalized_raw.items():
            if raw_col in used_raw_cols:
                continue
            if norm == canonical_col.lower().replace("_", " "):
                matched_raw = raw_col
                break

        # 2. Check aliases if not matched
        if not matched_raw:
            for raw_col, norm in normalized_raw.items():
                if raw_col in used_raw_cols:
                    continue
                for alias in aliases:
                    alias_norm = alias.strip().lower().replace("_", " ").replace("-", " ")
                    if norm == alias_norm:
                        matched_raw = raw_col
                        break
                if matched_raw:
                    break

        # 3. Substring containment match
        if not matched_raw:
            for raw_col, norm in normalized_raw.items():
                if raw_col in used_raw_cols:
                    continue
                for alias in aliases:
                    alias_norm = alias.strip().lower().replace("_", " ").replace("-", " ")
                    if len(alias_norm) > 3 and (alias_norm in norm or norm in alias_norm):
                        matched_raw = raw_col
                        break
                if matched_raw:
                    break

        if matched_raw:
            mapping[canonical_col] = matched_raw
            used_raw_cols.add(matched_raw)
        else:
            mapping[canonical_col] = None

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Renames columns in the raw dataframe to canonical schema names based on user/auto mapping.
    Unmapped raw columns are preserved for reference.
    """
    rename_dict = {raw_col: canonical_col for canonical_col, raw_col in mapping.items() if raw_col}
    mapped_df = df.rename(columns=rename_dict).copy()
    return mapped_df


def inspect_dataset_quality(df: pd.DataFrame) -> Dict:
    """
    Performs comprehensive data auditing:
    - Missing values count & percentage
    - Duplicate records
    - Numeric vs categorical feature breakdown
    - Geographic coordinate validity
    - Overall data health index (0 to 100%)
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_count = int(df.duplicated().sum())

    missing_series = df.isnull().sum()
    missing_pct_series = (missing_series / max(total_rows, 1)) * 100

    col_stats = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(missing_series[col])
        null_pct = round(float(missing_pct_series[col]), 2)
        unique_count = int(df[col].nunique())
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        col_stats.append({
            "Column": col,
            "Type": "Numerical" if is_numeric else "Categorical/Text",
            "Data Type": dtype,
            "Missing Count": null_count,
            "Missing %": null_pct,
            "Unique Values": unique_count,
        })

    col_df = pd.DataFrame(col_stats)

    # Coordinate check
    has_geo = False
    valid_geo_count = 0
    if "Latitude" in df.columns and "Longitude" in df.columns:
        valid_geo_mask = (
            pd.to_numeric(df["Latitude"], errors="coerce").between(-90, 90) &
            pd.to_numeric(df["Longitude"], errors="coerce").between(-180, 180)
        )
        valid_geo_count = int(valid_geo_mask.sum())
        has_geo = valid_geo_count > 0

    # Data health index computation
    completeness = 100.0 - float(df.isnull().sum().sum() / max(total_rows * total_cols, 1) * 100)
    uniqueness = 100.0 - (duplicate_count / max(total_rows, 1) * 100)
    health_score = round(0.6 * completeness + 0.4 * uniqueness, 1)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_count": duplicate_count,
        "completeness_pct": round(completeness, 1),
        "health_score": health_score,
        "has_geo": has_geo,
        "valid_geo_count": valid_geo_count,
        "column_summary_df": col_df,
    }
