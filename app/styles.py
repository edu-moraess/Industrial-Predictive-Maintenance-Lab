"""
Industrial Control Center - Design Tokens & CSS System (Sober & Minimalist)
Palette: #0F1115 (Background), #171A21 (Surface), #D4A84F (Accent)
Typography: Arial, Helvetica, system-ui
"""

INDUSTRIAL_THEME_CSS = """
<style>
    :root {
        --background: #0F1115;
        --surface: #171A21;
        --surface-secondary: #1D2129;
        --border: #2A2F38;
        --text-primary: #F2F2F2;
        --text-secondary: #9A9FA8;
        --accent: #D4A84F;
        --success: #4CAF78;
        --warning: #D9A441;
        --critical: #D95C5C;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text-primary);
        font-family: Arial, Helvetica, system-ui, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: Arial, Helvetica, system-ui, sans-serif;
        color: var(--text-primary);
        font-weight: 600;
    }
    .mono-text, [data-testid="stMetricValue"] {
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace !important;
    }

    /* Industrial Cards */
    .ind-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    
    .ind-card-header {
        font-size: 0.70rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    .ind-card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: SFMono-Regular, Consolas, monospace;
    }

    .ind-card-unit {
        font-size: 0.8rem;
        color: var(--accent);
        margin-left: 4px;
        font-weight: 500;
    }

    .ind-card-desc {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: SFMono-Regular, Consolas, monospace;
    }
    .badge-success { background-color: rgba(76, 175, 120, 0.12); color: var(--success); border: 1px solid rgba(76, 175, 120, 0.25); }
    .badge-warning { background-color: rgba(217, 164, 65, 0.12); color: var(--warning); border: 1px solid rgba(217, 164, 65, 0.25); }
    .badge-critical { background-color: rgba(217, 92, 92, 0.12); color: var(--critical); border: 1px solid rgba(217, 92, 92, 0.25); }
    .badge-info { background-color: rgba(212, 168, 79, 0.12); color: var(--accent); border: 1px solid rgba(212, 168, 79, 0.25); }

    /* Primary button — industrial accent, restrained */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: var(--surface-secondary) !important;
        border: 1px solid var(--accent) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    .stButton > button {
        border-radius: 3px !important;
        border: 1px solid var(--border) !important;
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
    }

    /* Metrics / widgets */
    [data-testid="stMetric"] {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 8px 12px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--background); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    /* Mobile: stack KPI cards */
    @media (max-width: 768px) {
        .ind-card-value { font-size: 1.2rem; }
    }
</style>
"""
