"""
Industrial Control Center - Design Tokens & CSS System
Palette: Industrial Dark (#0B1220)
Typography: Inter (UI) & JetBrains Mono (Technical Data)
"""

INDUSTRIAL_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Settings */
    .stApp {
        background-color: #0B1220;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #253044;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }
    code, pre, .mono-text, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Industrial Cards */
    .ind-card {
        background-color: #111827;
        border: 1px solid #253044;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .ind-card-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 8px;
    }

    .ind-card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .ind-card-unit {
        font-size: 0.85rem;
        color: #38BDF8;
        margin-left: 4px;
        font-weight: 500;
    }

    .ind-card-desc {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .badge-success { background-color: rgba(34, 197, 94, 0.1); color: #22C55E; border-color: rgba(34, 197, 94, 0.2); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.1); color: #F59E0B; border-color: rgba(245, 158, 11, 0.2); }
    .badge-critical { background-color: rgba(239, 68, 68, 0.1); color: #EF4444; border-color: rgba(239, 68, 68, 0.2); }
    .badge-info { background-color: rgba(56, 189, 248, 0.1); color: #38BDF8; border-color: rgba(56, 189, 248, 0.2); }

    /* Custom scrollbars and minor polish */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0B1220; }
    ::-webkit-scrollbar-thumb { background: #253044; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
"""
