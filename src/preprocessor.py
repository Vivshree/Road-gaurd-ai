"""
ROADGUARD AI - Preprocessing & Feature Engineering Pipeline
Handles missing value imputation, duplicate removal, temporal and spatial feature engineering,
categorical encoding, and standardisation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from src.config import SEVERITY_MAPPING, DEFAULT_CLUSTERING_FEATURES

class AccidentPreprocessor:
    def __init__(
        self,
        impute_strategy: str = "median",
        remove_duplicates: bool = True,
        scaler_type: str = "standard",
        selected_features: Optional[List[str]] = None,
    ):
        self.impute_strategy = impute_strategy
        self.remove_duplicates = remove_duplicates
        self.scaler_type = scaler_type
        self.selected_features = selected_features or DEFAULT_CLUSTERING_FEATURES
        self.scaler = None
        self.audit_log: List[str] = []
        self.feature_names_out: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """
        Executes full preprocessing and feature engineering.
        Returns:
            - processed_df: Full DataFrame with engineered columns added
            - scaled_features: Scaled 2D numpy array ready for clustering algorithms
            - summary: Dict of transformations applied and statistics
        """
        self.audit_log = []
        clean_df = df.copy()
        initial_rows = len(clean_df)

        # 1. Duplicate Removal
        if self.remove_duplicates:
            dup_count = clean_df.duplicated().sum()
            if dup_count > 0:
                clean_df = clean_df.drop_duplicates().reset_index(drop=True)
                self.audit_log.append(f"Removed {dup_count} duplicate record(s).")
            else:
                self.audit_log.append("No duplicate records found.")

        # 2. Date & Time Parsing & Temporal Engineering
        clean_df = self._engineer_temporal_features(clean_df)

        # 3. Categorical & Severity Encoding
        clean_df = self._encode_severity(clean_df)
        clean_df = self._engineer_environmental_flags(clean_df)

        # 4. Numerical Imputation
        clean_df = self._impute_numerical_columns(clean_df)

        # 5. Canonical Schema Harmonisation (Ensures all 8 views have valid attributes)
        clean_df = self._ensure_canonical_schema(clean_df)

        # 6. Feature Selection for Clustering
        available_features = [f for f in self.selected_features if f in clean_df.columns]
        if len(available_features) < 2:
            # Fallback to any numeric columns
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = [c for c in numeric_cols if not c.lower().endswith("id") and c not in ["Latitude", "Longitude"]]

        self.feature_names_out = available_features
        self.audit_log.append(f"Selected {len(available_features)} features for clustering: {', '.join(available_features)}")

        # 6. Feature Matrix Extraction & Scaling
        X = clean_df[available_features].values
        
        # Ensure no residual NaNs or infinities in X
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)
            self.audit_log.append("Imputed residual missing values in feature matrix with median.")

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
            self.audit_log.append("Applied RobustScaler (median and interquartile range scaling).")
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
            self.audit_log.append("Applied MinMaxScaler (range [0, 1]).")
        else:
            self.scaler = StandardScaler()
            self.audit_log.append("Applied StandardScaler (zero mean, unit variance standardisation).")

        X_scaled = self.scaler.fit_transform(X)

        summary = {
            "initial_rows": initial_rows,
            "final_rows": len(clean_df),
            "rows_removed": initial_rows - len(clean_df),
            "features_used": available_features,
            "scaler": self.scaler_type,
            "audit_log": self.audit_log,
        }

        return clean_df, X_scaled, summary

    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses Date and Time columns to produce Hour, Day_of_Week, Month, Is_Weekend, Is_Night."""
        # Hour extraction
        if "Hour" not in df.columns:
            if "Time" in df.columns:
                try:
                    time_series = df["Time"].astype(str)
                    df["Hour"] = time_series.str.extract(r"(\d{1,2})")[0].astype(float).fillna(12.0)
                    self.audit_log.append("Extracted 'Hour' from 'Time' string column.")
                except Exception:
                    df["Hour"] = 12.0
            elif "Date" in df.columns:
                try:
                    dt_series = pd.to_datetime(df["Date"], errors="coerce")
                    df["Hour"] = dt_series.dt.hour.fillna(12.0)
                    self.audit_log.append("Extracted 'Hour' from datetime 'Date'.")
                except Exception:
                    df["Hour"] = 12.0
            else:
                df["Hour"] = 12.0

        # Day of week and weekend
        if "Date" in df.columns:
            try:
                dt_series = pd.to_datetime(df["Date"], errors="coerce")
                if "Day_of_Week" not in df.columns:
                    df["Day_of_Week"] = dt_series.dt.day_name().fillna("Wednesday")
                df["Is_Weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                df["Month"] = dt_series.dt.month.fillna(6).astype(int)
                self.audit_log.append("Engineered 'Is_Weekend' and 'Month' from 'Date'.")
            except Exception:
                if "Day_of_Week" not in df.columns:
                    df["Day_of_Week"] = "Wednesday"
                df["Is_Weekend"] = 0
                df["Month"] = 6
        else:
            if "Day_of_Week" not in df.columns:
                df["Day_of_Week"] = "Wednesday"
            df["Is_Weekend"] = 0
            df["Month"] = 6

        # Is_Night flag (20:00 to 06:00)
        df["Is_Night"] = df["Hour"].apply(lambda h: 1 if (h >= 20.0 or h < 6.0) else 0)
        return df

    def _encode_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes Accident_Severity into an ordinal numeric score."""
        if "Accident_Severity" in df.columns:
            def map_sev(val):
                if pd.isna(val):
                    return 1
                v_str = str(val).strip().lower()
                for k, score in SEVERITY_MAPPING.items():
                    if k in v_str:
                        return score
                return 1

            df["Severity_Numeric"] = df["Accident_Severity"].apply(map_sev)
            self.audit_log.append("Encoded 'Accident_Severity' to 'Severity_Numeric' (1=Slight, 2=Serious, 3=Fatal).")
        else:
            # Fallback based on fatalities
            if "Number_of_Fatalities" in df.columns:
                df["Severity_Numeric"] = df["Number_of_Fatalities"].apply(lambda f: 3 if f > 0 else 1)
            else:
                df["Severity_Numeric"] = 1
            df["Accident_Severity"] = df["Severity_Numeric"].map({1: "Slight", 2: "Serious", 3: "Fatal"})

        return df

    def _engineer_environmental_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags adverse weather and unlit conditions."""
        if "Weather" in df.columns:
            adverse_keywords = ["rain", "fog", "mist", "storm", "snow", "sleet", "wind", "hail"]
            df["Is_Adverse_Weather"] = df["Weather"].astype(str).str.lower().apply(
                lambda w: 1 if any(k in w for k in adverse_keywords) else 0
            )
            self.audit_log.append("Engineered 'Is_Adverse_Weather' flag.")
        else:
            df["Is_Adverse_Weather"] = 0

        return df

    def _impute_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing numerical data safely with robust non-NaN fallbacks."""
        fallback_defaults = {
            "Number_of_Vehicles": 1.0,
            "Number_of_Casualties": 1.0,
            "Number_of_Fatalities": 0.0,
            "Number_of_Injuries": 0.0,
            "Speed_Limit": 50.0,
            "Hour": 12.0,
        }

        # Safe coordinate conversion if present
        if "Latitude" in df.columns:
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        if "Longitude" in df.columns:
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        target_numerics = [
            "Number_of_Vehicles", "Number_of_Casualties", 
            "Number_of_Fatalities", "Number_of_Injuries", 
            "Speed_Limit", "Hour"
        ]

        for col in target_numerics:
            default_val = fallback_defaults.get(col, 1.0)
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].isnull().any():
                    if self.impute_strategy == "mean":
                        calc_val = df[col].mean()
                    elif self.impute_strategy == "zero":
                        calc_val = 0.0
                    else:  # median
                        calc_val = df[col].median()

                    # Guard against all-NaN columns where mean/median is NaN
                    if pd.isna(calc_val):
                        calc_val = default_val

                    df[col] = df[col].fillna(calc_val)
                    self.audit_log.append(f"Imputed missing values in '{col}' using {self.impute_strategy} ({float(calc_val):.1f}).")
            else:
                # Default fallback values if column is missing from dataset
                df[col] = default_val

        return df

    def _ensure_canonical_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantees that all canonical schema columns are populated with valid, non-null values.
        If a column was missing from the user's uploaded dataset, an intelligent default is provided
        so that all dashboard pages (Overview, Maps, Clusters, Reports) execute seamlessly.
        """
        n_rows = len(df)
        if n_rows == 0:
            return df

        # 1. Identifier
        if "Accident_ID" not in df.columns or df["Accident_ID"].isnull().all():
            df["Accident_ID"] = [f"ACC-{i+1:05d}" for i in range(n_rows)]
            self.audit_log.append("Synthesized canonical 'Accident_ID' codes.")
        else:
            df["Accident_ID"] = df["Accident_ID"].fillna(pd.Series([f"ACC-{i+1:05d}" for i in range(n_rows)], index=df.index))

        # 2. Date & Time fallbacks if unmapped
        if "Date" not in df.columns:
            df["Date"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_rows, freq="h").strftime("%Y-%m-%d")
            self.audit_log.append("Populated default sequential 'Date' values.")
        else:
            df["Date"] = df["Date"].fillna(pd.Timestamp.today().strftime("%Y-%m-%d"))

        if "Time" not in df.columns:
            df["Time"] = "12:00"
        else:
            df["Time"] = df["Time"].fillna("12:00")

        # 3. Administrative / Spatial Jurisdictions
        if "State" not in df.columns or df["State"].isnull().all():
            df["State"] = "National Corridor"
        else:
            df["State"] = df["State"].fillna("General Region").astype(str).str.strip()

        if "City" not in df.columns or df["City"].isnull().all():
            if "State" in df.columns and not df["State"].isnull().all() and df["State"].nunique() > 1:
                df["City"] = df["State"]
            else:
                df["City"] = "Metro Node"
        else:
            df["City"] = df["City"].fillna("Urban District").astype(str).str.strip()

        # 4. Road Classification & Environment
        if "Road_Type" not in df.columns or df["Road_Type"].isnull().all():
            if "Speed_Limit" in df.columns:
                df["Road_Type"] = df["Speed_Limit"].apply(
                    lambda s: "Expressway / Highway" if s >= 80 else ("National Highway" if s >= 60 else "Urban Arterial")
                )
            else:
                df["Road_Type"] = "Urban Arterial"
        else:
            df["Road_Type"] = df["Road_Type"].fillna("Urban Arterial").astype(str).str.strip()

        if "Weather" not in df.columns or df["Weather"].isnull().all():
            df["Weather"] = df["Is_Adverse_Weather"].map({1: "Heavy Rain / Fog", 0: "Clear"}) if "Is_Adverse_Weather" in df.columns else "Clear"
        else:
            df["Weather"] = df["Weather"].fillna("Clear").astype(str).str.strip()

        if "Lighting_Condition" not in df.columns or df["Lighting_Condition"].isnull().all():
            df["Lighting_Condition"] = df["Is_Night"].map({1: "Darkness - lights lit", 0: "Daylight"}) if "Is_Night" in df.columns else "Daylight"
        else:
            df["Lighting_Condition"] = df["Lighting_Condition"].fillna("Daylight").astype(str).str.strip()

        if "Road_Surface" not in df.columns or df["Road_Surface"].isnull().all():
            df["Road_Surface"] = df["Is_Adverse_Weather"].map({1: "Wet", 0: "Dry"}) if "Is_Adverse_Weather" in df.columns else "Dry"
        else:
            df["Road_Surface"] = df["Road_Surface"].fillna("Dry").astype(str).str.strip()

        if "Vehicle_Type" not in df.columns or df["Vehicle_Type"].isnull().all():
            df["Vehicle_Type"] = "Motor Vehicle"
        else:
            df["Vehicle_Type"] = df["Vehicle_Type"].fillna("Motor Vehicle").astype(str).str.strip()

        return df
