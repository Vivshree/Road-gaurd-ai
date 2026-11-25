# ROADGUARD AI — Road Accident Intelligence & Hotspot Segmentation System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research-0284C7.svg)]()

> **Tagline:** Road Safety Intelligence & Accident Hotspot Analytics  
> **Classification:** Unsupervised Machine Learning Exploratory Decision-Support System  
> **Academic Integrity Notice:** This system is an exploratory decision-support platform designed to assist civil engineers, municipal authorities, and traffic departments in prioritizing road safety interventions. It is **not** a real-time emergency dispatch or deterministic accident prediction system.

---

## 📌 1. Project Overview & Objectives

Road traffic collisions represent one of the leading global causes of preventable mortality and severe trauma. Traditional roadway safety analyses rely predominantly on reactive police records or simplistic frequency counts, often neglecting the multi-dimensional interactions between ambient illumination, adverse precipitation, roadway classification, posted speed limits, and casualty severity.

**ROADGUARD AI** formulates accident hotspot discovery as an **unsupervised machine learning problem**. Without requiring ground-truth causal labels, the system partitions multi-attribute accident records into homogeneous clusters (hotspot archetypes). This enables traffic planners to:
1. **Discover Latent Collision Archetypes:** Automatically isolate high-speed expressway pileups, monsoon precipitation skids, unlit nocturnal arterial crashes, and urban daytime shunts.
2. **Prioritize Targeted Safety Interventions:** Align engineering countermeasures (such as anti-skid resurfacing, streetlighting lux upgrades, or automated speed cameras) with the dominant empirical traits of each corridor.
3. **Ingest Real-World Datasets:** Provide a flexible **Real-World Dataset Upload Center** with automated schema mapping for arbitrary CSV and Excel accident files.
4. **Conduct Rigorous Academic Validation:** Evaluate cluster separation mathematically using Within-Cluster Sum of Squares (WCSS / Inertia), Silhouette Analysis, and the Davies-Bouldin Index.

---

## 🏗️ 2. Architectural Structure

```
d:\road accident project/
│
├── .streamlit/
│   └── config.toml               # Command-center dark navy aesthetic (#0B1220, #131E30, #22D3EE)
│
├── data/
│   ├── sample_accidents.csv      # Synthetic academic benchmark dataset (3,500 multi-corridor records)
│   └── generate_sample.py        # Statistical dataset generator with 5 latent collision archetypes
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # System constants, canonical schemas, column synonyms, and color palettes
│   ├── styles.py                 # Custom CSS stylesheet, KPI cards, badges, and layout components
│   ├── data_loader.py            # CSV/Excel reader with multi-encoding fallback, schema auto-mapper, and quality audit
│   ├── preprocessor.py           # Missing value imputation, standard/robust scaling, and temporal feature engineering
│   ├── clustering.py             # K-Means, Agglomerative clustering, Euclidean vs Manhattan comparisons, elbow curves
│   ├── pca_engine.py             # 2D & 3D PCA projection, explained variance ratio, and eigenvector feature loadings
│   ├── profiling.py              # Empirical cluster aggregation, semantic label synthesis, and Z-score deviation profiles
│   ├── visualizer.py             # Dark-themed Plotly charts (time-series, sunburst, radar, heatmaps, 3D scatter)
│   ├── map_engine.py             # Interactive Folium maps, CartoDB dark tiles, heatmaps, and forensic popups
│   └── report_engine.py          # Publication-grade PDF generator (ReportLab) and CSV export utilities
│
├── pages_app/
│   ├── __init__.py
│   ├── upload_center.py          # Real-World Dataset Upload Center, preview, schema mapping, and preprocessing controls
│   ├── executive_overview.py     # Command-center KPIs, temporal trends, atmospheric patterns, and prone corridors
│   ├── hotspot_map.py            # Spatial Folium map, heatmap layer, cluster filters, and forensic popups
│   ├── cluster_explorer.py       # K-Means granularity explorer (K=2-10), cluster cards, and multi-attribute radar
│   ├── ml_lab.py                 # Academic flagship: K-Means, Agglomerative, Euclidean vs Manhattan, Elbow, PCA 2D/3D
│   ├── pattern_analysis.py       # Hour x Weekday heatmap, roadway cross-tabs, and feature Z-score deviations
│   ├── location_intel.py         # Municipal node diagnostics, peer-cluster comparisons, and safety intervention hypotheses
│   └── report_generator.py       # Academic PDF compiler, CSV export center, and viva defense brief
│
├── app.py                        # Streamlit central orchestrator, navigation router, and session manager
├── requirements.txt              # Pinned production dependencies
└── README.md                     # Comprehensive academic documentation and defense viva questions
```

---

## 🔬 3. Machine Learning Methodology & Mathematical Formulations

### A. K-Means Clustering
K-Means partitions $N$ observations into $K$ mutually exclusive clusters by minimizing the **Within-Cluster Sum of Squares (WCSS)**, also known as **Inertia**:

$$\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$$

where $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ is the centroid (arithmetic mean) of cluster $C_k$.

### B. Euclidean ($L_2$) vs. Manhattan ($L_1$) Distance Foundations
- **Squared Euclidean ($L_2^2$):** $d_2(x, y)^2 = \sum_{j=1}^D (x_j - y_j)^2$. The arithmetic mean is analytically proven to be the unique minimizer of squared Euclidean deviations.
- **Manhattan ($L_1$):** $d_1(x, y) = \sum_{j=1}^D |x_j - y_j|$. Minimizing $L_1$ distances requires the **geometric median** (or coordinate-wise median, as implemented in K-Medians / PAM).
- **Academic Distinction:** Standard Lloyd's K-Means does **not** natively optimize Manhattan distance. Forcing Manhattan distance into a standard mean-based centroid update produces sub-optimal, mathematically inconsistent cluster centers. Therefore, ROADGUARD AI utilizes **Agglomerative Hierarchical Clustering** under Complete/Average linkages to evaluate Manhattan metric structures rigorously.

### C. Agglomerative Hierarchical Clustering
Agglomerative clustering initializes each record as a singleton cluster and iteratively merges the closest pair of clusters according to a chosen linkage criterion:
- **Ward Linkage:** Minimizes total within-cluster variance. *Mathematically restricted strictly to Euclidean distance ($L_2$).*
- **Complete Linkage:** $\max \{d(u, v) : u \in A, v \in B\}$
- **Average Linkage:** $\frac{1}{|A||B|} \sum_{u \in A} \sum_{v \in B} d(u, v)$
- **Single Linkage:** $\min \{d(u, v) : u \in A, v \in B\}$

### D. Unsupervised Validation Metrics
1. **Silhouette Coefficient:** Evaluates the trade-off between cluster cohesion $a(i)$ and cluster separation $b(i)$:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$
2. **Davies-Bouldin Index (DBI):** Evaluates the average similarity between each cluster and its most similar counterpart. **Lower values indicate superior partition compactness and separation.**
   $$DB = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)} \right)$$
3. **Calinski-Harabasz Index:** Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).

### E. Principal Component Analysis (PCA)
PCA computes the eigenvectors of the empirical feature covariance matrix $\mathbf{\Sigma} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$, projecting the standardized high-dimensional feature space onto orthogonal axes that maximize captured variance. **Crucial Note:** PCA is used exclusively for 2D/3D geometric visualization and dimensionality reduction, not causal explanation.

---

## 🛠️ 4. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- Windows / macOS / Linux

### Setup Instructions

1. **Clone or Navigate to Project Repository:**
   ```powershell
   cd "d:\road accident project"
   ```

2. **Create and Activate Virtual Environment:**
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Launch the ROADGUARD AI Command Center:**
   ```powershell
   streamlit run app.py
   ```

5. **Access Application:**
   Open your browser and navigate to `http://localhost:8501`.

---

## 📊 5. Dataset Schema & Intelligent Auto-Mapping

ROADGUARD AI accepts arbitrary real-world CSV or Excel datasets. The ingestion engine automatically matches raw headers to canonical schema definitions:

| Canonical Column | Data Type | Required | Description | Example Synonyms |
| :--- | :--- | :---: | :--- | :--- |
| `Accident_ID` | String | No | Unique incident key | `id`, `crash_id`, `collision_id`, `report_num` |
| `Date` | Datetime | **Yes** | Date of collision | `date`, `crash_date`, `incident_date`, `timestamp` |
| `Time` | String / Time | No | Time of occurrence | `time`, `crash_time`, `hour_min`, `time_of_day` |
| `State` | String | No | Administrative region | `state`, `province`, `region`, `territory` |
| `City` | String | No | Municipality / Node | `city`, `district`, `town`, `local_authority` |
| `Latitude` | Float | No | Geographic Latitude | `lat`, `latitude`, `y`, `gps_lat` |
| `Longitude` | Float | No | Geographic Longitude | `lon`, `lng`, `longitude`, `x`, `gps_lng` |
| `Road_Type` | Categorical | No | Infrastructure class | `road_type`, `roadway`, `route_type`, `highway` |
| `Weather` | Categorical | No | Atmospheric condition | `weather`, `weather_condition`, `climate` |
| `Lighting_Condition` | Categorical | No | Ambient illumination | `lighting`, `light_condition`, `day_night` |
| `Road_Surface` | Categorical | No | Pavement condition | `surface`, `road_surface`, `pavement_condition` |
| `Vehicle_Type` | Categorical | No | Vehicle classification | `vehicle_type`, `veh_type`, `transport_type` |
| `Number_of_Vehicles` | Integer | No | Vehicles involved | `num_vehicles`, `vehicles_involved`, `veh_count` |
| `Number_of_Casualties`| Integer | No | Total individuals | `casualties`, `num_casualties`, `victims` |
| `Number_of_Fatalities`| Integer | No | Fatal casualties | `fatalities`, `deaths`, `killed`, `persons_killed`|
| `Number_of_Injuries` | Integer | No | Non-fatal injuries | `injuries`, `hurt`, `injured`, `persons_injured` |
| `Accident_Severity` | Categorical | No | Severity level | `severity`, `accident_severity`, `crash_severity`|
| `Speed_Limit` | Float | No | Posted speed (km/h) | `speed_limit`, `posted_speed`, `max_speed` |

---

## 🎓 6. Academic Viva Voce / Defense Examination Guide (16 Questions & Answers)

### Theoretical & ML Foundations
#### Q1: Why did you adopt Unsupervised Machine Learning rather than Supervised Classification?
> **Answer:** In road traffic safety analysis, future accident occurrences do not possess ground-truth class labels. Supervised models are confined to predicting historical target variables (such as binary severity) and suffer from severe reporting class imbalance. Unsupervised clustering groups multi-dimensional data objectively by natural cohesion, enabling traffic authorities to uncover latent risk archetypes (e.g. weather-induced skids vs high-speed nocturnal pileups) across unlabelled geographic corridors.

#### Q2: What is the objective function of standard K-Means clustering?
> **Answer:** Standard Lloyd's K-Means minimizes the Within-Cluster Sum of Squares (WCSS / Inertia):
> $$\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$$
> It iteratively alternates between assigning observations to the nearest Euclidean centroid and updating each centroid to the arithmetic mean of its assigned points.

#### Q3: Why is standard K-Means restricted to squared Euclidean distance ($L_2$)? Can it use Manhattan distance ($L_1$)?
> **Answer:** Standard K-Means cannot natively optimize Manhattan distance ($L_1$). Mathematically, the arithmetic mean is the unique point that minimizes sum of squared Euclidean errors ($\frac{\partial}{\partial \mu} \sum (x_i - \mu)^2 = 0 \implies \mu = \bar{x}$). If one substitutes Manhattan distance ($\sum |x_i - \mu|$), the optimal centroid is the **coordinate-wise median**, not the mean. Forcing Manhattan distance into Lloyd's algorithm creates a mathematical mismatch where the centroid update no longer minimizes the objective function. To evaluate Manhattan distance, one must use K-Medians, K-Medoids (PAM), or Agglomerative Hierarchical Clustering.

#### Q4: Why must Ward's linkage strictly utilize Euclidean distance?
> **Answer:** Ward's minimum variance criterion merges the pair of clusters that minimizes the increase in total within-cluster variance ($\Delta \text{ESS}$). Because variance is fundamentally an $L_2$ concept defined by squared deviations from the centroid, Ward's objective is defined only in Euclidean space. Non-Euclidean metrics violate the underlying geometric variance equations.

#### Q5: How do Complete, Average, and Single linkage differ in Agglomerative Clustering?
> **Answer:**
> - **Complete Linkage:** Measures the maximum pairwise distance between points in two clusters, resulting in compact, spherical clusters.
> - **Average Linkage:** Measures the average pairwise distance between all point pairs, balancing variance and handling arbitrary distance metrics (like Manhattan).
> - **Single Linkage:** Measures the minimum distance between any point in cluster A and cluster B. It can discover arbitrary geometries but is vulnerable to the "chaining effect" (long straggly clusters connected by noise).

### Evaluation & Hyperparameter Selection
#### Q6: How does the Silhouette Coefficient evaluate cluster validity?
> **Answer:** For observation $i$:
> $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
> where $a(i)$ is the mean intra-cluster distance to all other points in the same cluster, and $b(i)$ is the mean nearest-cluster distance. A score near $+1$ indicates excellent cohesion and separation; $0$ denotes cluster boundary overlapping; and negative values indicate incorrect cluster assignments.

#### Q7: What are the mathematical properties of the Davies-Bouldin Index?
> **Answer:** The Davies-Bouldin Index computes the average similarity ratio of each cluster to its most similar competitor, where similarity is defined as within-cluster dispersion divided by centroid separation. Lower DBI values indicate tighter clusters that are farther apart. Unlike Silhouette score, DBI does not require calculating full $O(N^2)$ pairwise point distances, making it computationally faster.

#### Q8: How is the "Elbow Point" identified on an Inertia vs. K curve?
> **Answer:** As $K$ increases, WCSS monotonically decreases (reaching $0$ when $K=N$). The elbow represents the point of maximum curvature where additional clusters yield diminishing returns in variance reduction. Mathematically, it corresponds to the peak in the discrete second difference ($\Delta^2 \text{WCSS} = \text{WCSS}_{K-1} - 2\text{WCSS}_K + \text{WCSS}_{K+1}$).

#### Q9: What is the purpose of Principal Component Analysis (PCA) in ROADGUARD AI?
> **Answer:** Standard accident records contain high-dimensional features ($D \ge 10$). PCA computes the eigenvectors of the covariance matrix to linearly project the data onto 2 or 3 orthogonal components that capture the maximum variance. In this system, PCA is strictly an exploratory tool to visualize cluster separability in 2D and 3D space; it is **not** a predictive or causal model.

### Preprocessing & Engineering
#### Q10: Why must numerical features be standardized prior to clustering?
> **Answer:** Clustering algorithms rely on geometric distance metrics. If unscaled, features with large absolute ranges (e.g. `Speed_Limit` spanning 30–120 km/h) would dominate the distance calculation over features with small scales (e.g. `Severity_Numeric` spanning 1–3 or binary flags 0–1). Standardization transforms all variables to zero mean and unit variance ($\mu=0, \sigma=1$).

#### Q11: When would you use RobustScaler over StandardScaler?
> **Answer:** In traffic accident records, casualty counts frequently exhibit extreme positive skewness (e.g. rare pileups involving 8+ casualties). `StandardScaler` uses the empirical mean and standard deviation, which are sensitive to outliers. `RobustScaler` scales using the median and Interquartile Range ($\text{IQR} = Q_3 - Q_1$), preventing catastrophic collisions from skewing the scaling parameters.

#### Q12: Why are record identifiers (like `Accident_ID`) excluded from the clustering feature matrix?
> **Answer:** Identifiers are high-cardinality nominal keys that contain zero generalizable physical information about roadway risk. Including IDs would introduce spurious noise and induce arbitrary clustering based on sequential record keys rather than physical crash dynamics.

### Academic Integrity, Ethics & Domain Bias
#### Q13: What is "Reporting Bias" in road accident records, and how does it affect clustering?
> **Answer:** Minor collisions (slight vehicle scratches, minor shunts) are substantially underreported to police departments, whereas fatal and severe collisions receive mandatory legal and forensic documentation. This creates a survivor and reporting bias in the dataset. Unsupervised clusters may reflect police recording protocols or urban enforcement density rather than absolute danger.

#### Q14: Why must cluster labels be distinguished from official legal categories?
> **Answer:** Cluster labels generated by ROADGUARD AI (e.g. "Severe Night-Time Collision Corridor") are empirical statistical descriptions synthesized from cluster centroid means. They do not constitute statutory legal fault classifications or judicial determinations of liability.

#### Q15: Why is correlation not causation in this decision-support system?
> **Answer:** An empirical cluster demonstrating an association between wet roads and nighttime collisions does not prove that darkness or rain *caused* the collision (confounding factors like driver fatigue, intoxication, or tire tread wear may be unobserved). The system's recommendations are exploratory decision-support hypotheses to prioritize engineering surveys, not causal assertions.

#### Q16: How does ROADGUARD AI handle missing geographical coordinates?
> **Answer:** In strict accordance with scientific integrity, ROADGUARD AI **never fabricates or hallucinates geographical coordinates**. When an uploaded dataset lacks latitude/longitude attributes, the application displays an informative notification, disables spatial Folium maps, and provides an alternative tabular and administrative breakdown (e.g. by City or District).

---

## ☁️ 7. Cloud Deployment Guide

ROADGUARD AI is built with Streamlit and requires a persistent background process with WebSocket support.

### Option A: Streamlit Community Cloud (Recommended — 100% Free & 1-Click)
Streamlit Community Cloud is the official hosting platform for Streamlit applications:
1. Fork or push this repository to GitHub.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"** and enter:
   - **Repository:** `Vivshree/Road-gaurd-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** Your app will be live with a permanent public URL and automatic re-deployments on every `git push`.

### Option B: Render.com (Free Web Service)
1. Sign in to **[Render.com](https://dashboard.render.com)** and create a new **Web Service**.
2. Connect this GitHub repository.
3. Configuration:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **Create Web Service**.

### Option C: Docker Containerization
A production-ready [Dockerfile](Dockerfile) is included:
```bash
# Build Docker image
docker build -t roadguard-ai .

# Run container locally on port 8501
docker run -p 8501:8501 roadguard-ai
```

> **Note on Vercel:** Vercel is a serverless platform designed for stateless HTTP APIs (Next.js, Flask, FastAPI). Because Streamlit is an interactive stateful application requiring persistent WebSockets, Vercel's serverless runtime raises `[PYTHON_ENTRYPOINT_NOT_FOUND]`. Please use Streamlit Community Cloud or Render for cloud hosting.

---

## 📄 8. Academic License & Citation

This software is developed as an academic engineering research project for unsupervised machine learning decision-support.  
**Academic Citation:**  
*ROADGUARD AI: Road Accident Intelligence & Hotspot Segmentation System. Department of Computer Science & Engineering, 2024.*
