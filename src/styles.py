"""
ROADGUARD AI - Custom CSS Stylesheet
Implements a Command Center Dark Navy Theme with Cyan (#22D3EE) and Amber (#F59E0B) Accents.
"""

import streamlit as st

def inject_custom_styles():
    """Injects responsive, modern command-center CSS into the Streamlit app."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B1220;
        --bg-panel: #131E30;
        --bg-card: #18253B;
        --bg-card-hover: #1F304B;
        --accent-cyan: #22D3EE;
        --accent-cyan-glow: rgba(34, 211, 238, 0.15);
        --accent-orange: #F59E0B;
        --accent-red: #EF4444;
        --accent-green: #10B981;
        --border-color: #24354D;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* Global Base */
    html, body, [class*="st-"] {
        font-family: var(--font-main);
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(at 0% 0%, rgba(34, 211, 238, 0.04) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(245, 158, 11, 0.03) 0px, transparent 50%);
    }

    /* Top Header & Branding */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 24px;
    }

    .brand-logo {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #0284C7 0%, #22D3EE 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(34, 211, 238, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #22D3EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .brand-tagline {
        font-size: 13px;
        color: var(--accent-cyan);
        font-weight: 500;
        letter-spacing: 0.3px;
        margin-top: 3px;
    }

    /* Command Center KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-cyan);
        box-shadow: 0 8px 24px rgba(34, 211, 238, 0.12);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #FFFFFF;
        font-family: var(--font-mono);
        line-height: 1;
        margin-bottom: 6px;
    }

    .kpi-delta {
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .kpi-delta.cyan { color: var(--accent-cyan); }
    .kpi-delta.orange { color: var(--accent-orange); }
    .kpi-delta.red { color: var(--accent-red); }
    .kpi-delta.green { color: var(--accent-green); }

    /* Surface Cards & Panels */
    .dashboard-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .panel-title {
        font-size: 16px;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .panel-subtitle {
        font-size: 13px;
        color: var(--text-muted);
    }

    /* Badges */
    .rg-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .rg-badge-cyan {
        background: rgba(34, 211, 238, 0.12);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }

    .rg-badge-orange {
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .rg-badge-red {
        background: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .rg-badge-green {
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Drag & Drop Upload Container */
    .upload-box {
        border: 2px dashed rgba(34, 211, 238, 0.35);
        background: rgba(19, 30, 48, 0.7);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 24px;
    }

    .upload-box:hover {
        border-color: var(--accent-cyan);
        background: rgba(34, 211, 238, 0.04);
    }

    /* Academic Banner / Alert */
    .academic-callout {
        background: rgba(19, 30, 48, 0.85);
        border-left: 4px solid var(--accent-cyan);
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13.5px;
        color: #CBD5E1;
        line-height: 1.5;
    }

    .academic-callout strong {
        color: #FFFFFF;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #080D17 !important;
        border-right: 1px solid var(--border-color);
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #E2E8F0 !important;
        font-weight: 600;
        font-size: 13px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0E7490 0%, #0891B2 100%);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 14px;
        border: 1px solid rgba(34, 211, 238, 0.4);
        border-radius: 8px;
        padding: 8px 18px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0891B2 0%, #06B6D4 100%);
        border-color: #22D3EE;
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.4);
        transform: translateY(-1px);
        color: #FFFFFF;
    }

    /* Secondary / Reset Buttons */
    .stButton > button[kind="secondary"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
    }

    /* Footer */
    .rg-footer {
        margin-top: 48px;
        padding: 20px 0;
        border-top: 1px solid var(--border-color);
        text-align: center;
        font-size: 12px;
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }

    .rg-footer a {
        color: var(--accent-cyan);
        text-decoration: none;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        color: #FFFFFF !important;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Renders the top command center branding bar."""
    st.markdown("""
    <div class="brand-container">
        <div class="brand-logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M12 8v4"/>
                <path d="M12 16h.01"/>
            </svg>
        </div>
        <div>
            <h1 class="brand-title">ROADGUARD AI</h1>
            <div class="brand-tagline">Road Safety Intelligence &amp; Accident Hotspot Segmentation System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(label, value, delta=None, delta_color="cyan", tooltip=None):
    """Generates HTML for a single command-center KPI card."""
    delta_html = f'<div class="kpi-delta {delta_color}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card" title="{tooltip or label}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def render_footer():
    """Renders the official academic project footer."""
    st.markdown("""
    <div class="rg-footer">
        <strong>ROADGUARD AI</strong> &bull; Academic Decision-Support System &bull; 
        Unsupervised Machine Learning &amp; Spatial Hotspot Segmentation &bull; 
        <em>Data-driven exploratory platform &mdash; Not for real-time emergency dispatch</em>
    </div>
    """, unsafe_allow_html=True)
