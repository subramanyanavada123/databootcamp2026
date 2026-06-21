import streamlit as st

st.set_page_config(
    page_title="Data Science Workshop 2026",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

  /* Main background */
  .main .block-container { padding-top: 2rem; max-width: 1400px; }

  /* Cards */
  .ds-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .ds-card-accent {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border: 1px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid #3b82f6;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
  }
  .hero-banner h1 { color: #f8fafc; font-size: 2.2rem; font-weight: 800; margin: 0; }
  .hero-banner p  { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; }

  /* Topic badges */
  .badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .badge-blue   { background: #1d4ed8; color: #bfdbfe; }
  .badge-green  { background: #166534; color: #bbf7d0; }
  .badge-orange { background: #92400e; color: #fde68a; }
  .badge-purple { background: #5b21b6; color: #ddd6fe; }
  .badge-red    { background: #991b1b; color: #fecaca; }

  /* Pipeline flow */
  .pipeline {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding: 1rem;
    background: #0f172a;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
  }
  .pipeline-step {
    background: #1e3a5f;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    color: #93c5fd;
    white-space: nowrap;
  }
  .pipeline-arrow { color: #475569; font-size: 1.2rem; }

  /* Quiz */
  .quiz-option {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .quiz-option:hover { border-color: #3b82f6; }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .metric-card {
    flex: 1;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
  .metric-card .label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

  /* Section headers */
  .section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
  }
  .section-header h3 { color: #f1f5f9; margin: 0; font-size: 1.1rem; font-weight: 600; }

  /* Stmetric overrides */
  [data-testid="stMetricValue"] { color: #38bdf8 !important; }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

  /* Code blocks */
  .stCodeBlock { border-radius: 8px !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; border-bottom: 1px solid #334155; }
  .stTabs [data-baseweb="tab"] { background: #1e293b; border-radius: 8px 8px 0 0; color: #94a3b8; border: 1px solid #334155; border-bottom: none; padding: 0.5rem 1.2rem; }
  .stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; border-color: #1d4ed8 !important; }

  /* Chat messages */
  .chat-user { background: #1e3a5f; border-radius: 12px 12px 4px 12px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #3b82f6; }
  .chat-ai   { background: #1e293b; border-radius: 12px 12px 12px 4px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #10b981; }

  /* Success/warning/error boxes */
  .box-success { background: #14532d; border: 1px solid #16a34a; border-radius: 8px; padding: 1rem; color: #bbf7d0; }
  .box-warning { background: #78350f; border: 1px solid #d97706; border-radius: 8px; padding: 1rem; color: #fde68a; }
  .box-info    { background: #1e3a5f; border: 1px solid #3b82f6; border-radius: 8px; padding: 1rem; color: #bfdbfe; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Button overrides */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    box-shadow: 0 4px 12px rgba(59,130,246,0.4);
  }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "quiz_score": 0,
    "quiz_total": 0,
    "challenge_active": False,
    "completed_topics": [],
    "current_topic": "🏠 Home",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
      <div style="font-size:2.5rem;">🔬</div>
      <div style="font-size:1rem; font-weight:700; color:#f8fafc;">DS Workshop 2026</div>
      <div style="font-size:0.7rem; color:#64748b; margin-top:0.2rem;">Data Science Foundation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    PAGES = {
        "🏠 Home":                    "home",
        "📊 1. Understanding Data":   "understanding_data",
        "🔌 2. Data Collection":      "data_collection",
        "🧹 3. Data Cleaning":        "data_cleaning",
        "⚙️ 4. Data Processing":      "data_processing",
        "🔍 5. Exploratory Analysis": "eda",
        "🛠️ 6. Feature Engineering":  "feature_engineering",
        "📈 7. Data Visualization":   "visualization",
        "🤖 8. ML Fundamentals":      "ml_fundamentals",
        "🧠 9. AI Applications":      "ai_applications",
        "🚀 10. MLOps & Production":  "mlops",
        "─────────────────":          None,
        "⚡ Challenge Mode":          "challenge",
        "🏆 Capstone Project":        "capstone",
        "💬 AI Mentor Chat":          "chat",
    }

    page = st.selectbox(
        "NAVIGATE",
        options=[k for k in PAGES if PAGES[k] is not None],
        index=0,
    )
    st.session_state.current_topic = page

    st.markdown("---")

    # Progress tracker
    completed = len(st.session_state.completed_topics)
    total_topics = 10
    pct = int((completed / total_topics) * 100)
    st.markdown(f"""
    <div style="margin-bottom:0.5rem;">
      <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em;">WORKSHOP PROGRESS</div>
      <div style="font-size:1.4rem; font-weight:700; color:#38bdf8;">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100)

    st.markdown(f"""
    <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">
      {completed}/{total_topics} topics completed<br/>
      Quiz score: {st.session_state.quiz_score}/{st.session_state.quiz_total}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#475569; text-align:center;">
      Data → Information → Insight<br/>→ Prediction → Decision → Automation
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ───────────────────────────────────────────────────────────────
page_id = PAGES[page]

if page_id == "home":
    from app.pages.home import render
elif page_id == "understanding_data":
    from app.pages.p01_understanding_data import render
elif page_id == "data_collection":
    from app.pages.p02_data_collection import render
elif page_id == "data_cleaning":
    from app.pages.p03_data_cleaning import render
elif page_id == "data_processing":
    from app.pages.p04_data_processing import render
elif page_id == "eda":
    from app.pages.p05_eda import render
elif page_id == "feature_engineering":
    from app.pages.p06_feature_engineering import render
elif page_id == "visualization":
    from app.pages.p07_visualization import render
elif page_id == "ml_fundamentals":
    from app.pages.p08_ml_fundamentals import render
elif page_id == "ai_applications":
    from app.pages.p09_ai_applications import render
elif page_id == "mlops":
    from app.pages.p10_mlops import render
elif page_id == "challenge":
    from app.pages.challenge import render
elif page_id == "capstone":
    from app.pages.capstone import render
elif page_id == "chat":
    from app.pages.chat import render

render()
