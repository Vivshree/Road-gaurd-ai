"""
ROADGUARD AI - Custom CSS Stylesheet
Implements a Modern, Clean Light Intelligence Theme with Sky Blue (#0284C7) and Slate accents.
"""

import streamlit as st

def inject_custom_styles():
    """Injects responsive, modern clean light CSS into the Streamlit app."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #F8FAFC;
        --bg-panel: #FFFFFF;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F8FAFC;
        --accent-cyan: #0284C7;
        --accent-cyan-glow: rgba(2, 132, 199, 0.12);
        --accent-orange: #D97706;
        --accent-red: #DC2626;
        --accent-green: #059669;
        --border-color: #E2E8F0;
        --border-color-subtle: #F1F5F9;
        --text-primary: #0F172A;
        --text-secondary: #475569;
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
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(2, 132, 199, 0.04) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(217, 119, 6, 0.03) 0px, transparent 50%);
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
        background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 60%, #0284C7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .brand-tagline {
        font-size: 13px;
        color: var(--accent-cyan);
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-top: 3px;
    }

    /* Light Theme KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #0284C7;
        box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.1), 0 4px 6px -4px rgba(2, 132, 199, 0.05);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #0284C7, transparent);
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #0F172A !important;
        font-family: var(--font-mono);
        line-height: 1;
        margin-bottom: 6px;
    }

    .kpi-delta {
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .kpi-delta.cyan { color: #0284C7; }
    .kpi-delta.orange { color: #D97706; }
    .kpi-delta.red { color: #DC2626; }
    .kpi-delta.green { color: #059669; }

    /* Surface Cards & Panels */
    .dashboard-panel {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E2E8F0;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .panel-subtitle {
        font-size: 13px;
        color: #64748B;
    }

    /* Badges */
    .rg-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .rg-badge-cyan {
        background: #F0F9FF;
        color: #0284C7;
        border: 1px solid #BAE6FD;
    }

    .rg-badge-orange {
        background: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }

    .rg-badge-red {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }

    .rg-badge-green {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }

    /* Interactive Drag & Drop Upload Container */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #0284C7 !important;
        background-color: #F8FAFC !important;
        border-radius: 14px !important;
        padding: 28px 20px !important;
        text-align: center !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F0F9FF !important;
        border-color: #0369A1 !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.12) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 6px 20px !important;
        transition: background 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #0369A1 !important;
    }

    [data-testid="stFileUploaderDropzone"] span {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: #64748B !important;
    }

    .upload-box {
        border: 2px dashed #BAE6FD;
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .upload-box:hover {
        border-color: #0284C7;
        background: #F0F9FF;
    }

    /* Academic Banner / Alert */
    .academic-callout {
        background: #F0F9FF;
        border-left: 4px solid #0284C7;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13.5px;
        color: #1E293B;
        line-height: 1.5;
        border-top: 1px solid #E0F2FE;
        border-right: 1px solid #E0F2FE;
        border-bottom: 1px solid #E0F2FE;
    }

    .academic-callout strong {
        color: #0369A1;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #1E293B !important;
        font-weight: 600;
        font-size: 13px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 14px;
        border: 1px solid #0284C7;
        border-radius: 8px;
        padding: 8px 18px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
        border-color: #0369A1;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        transform: translateY(-1px);
        color: #FFFFFF !important;
    }

    /* Secondary / Reset Buttons */
    .stButton > button[kind="secondary"] {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0;
        color: #475569 !important;
    }

    /* Footer */
    .rg-footer {
        margin-top: 48px;
        padding: 20px 0;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        font-size: 12px;
        color: #64748B;
        letter-spacing: 0.5px;
    }

    .rg-footer a {
        color: #0284C7;
        text-decoration: none;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        color: #0F172A !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        background: #FFFFFF;
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
    """Generates HTML for a single modern light KPI card."""
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
