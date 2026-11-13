"""
ROADGUARD AI - PCA (Principal Component Analysis) Engine
Reduces high-dimensional accident feature spaces to 2D and 3D coordinate systems for visual
cluster inspection, and extracts component loading matrices with academic documentation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA

class PCAEngine:
    @staticmethod
    def fit_transform(X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Fits a 3-component PCA on scaled accident features.
        Returns coordinates, explained variance, and feature loadings.
        """
        n_components = min(3, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)

        exp_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        # Loading matrix (components_ has shape [n_components, n_features])
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=feature_names,
        )

        # Top feature drivers per principal component
        top_drivers = {}
        for col in loadings_df.columns:
            abs_loadings = loadings_df[col].abs().sort_values(ascending=False)
            top_drivers[col] = [
                {"feature": feat, "loading": round(loadings_df.loc[feat, col], 4), "abs_loading": round(val, 4)}
                for feat, val in abs_loadings.head(4).items()
            ]

        academic_note = (
            "PCA ACADEMIC METHODOLOGY:\n"
            "Principal Component Analysis (PCA) performs an orthogonal linear transformation to project "
            "correlated multi-variable accident attributes onto linearly uncorrelated principal components "
            "that maximize variance. "
            "Crucially, PCA is strictly an exploratory dimensionality reduction and projection tool for "
            "geometric inspection in 2D/3D space. It does NOT represent a causal model or emergency prediction."
        )

        return {
            "model": pca,
            "coordinates": X_pca,
            "n_components": n_components,
            "explained_variance_ratio": [round(float(v), 4) for v in exp_var],
            "cumulative_variance": [round(float(v), 4) for v in cum_var],
            "loadings_df": loadings_df,
            "top_drivers": top_drivers,
            "academic_note": academic_note,
        }
