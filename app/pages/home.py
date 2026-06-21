import streamlit as st

def render():
    st.markdown("""
    <div class="hero-banner">
      <div style="font-size:3rem; margin-bottom:0.5rem;">🔬</div>
      <h1>Data Science Foundation Workshop</h1>
      <p>For Engineers — Interactive · Visual · Industry-Ready</p>
      <div style="margin-top:1rem; display:flex; justify-content:center; gap:0.75rem; flex-wrap:wrap;">
        <span class="badge badge-blue">10 Topics</span>
        <span class="badge badge-green">Live Code</span>
        <span class="badge badge-orange">Real Datasets</span>
        <span class="badge badge-purple">AI Mentor</span>
        <span class="badge badge-red">Challenges</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    st.markdown("""
    <div class="pipeline">
      <span class="pipeline-step">📦 Data</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">ℹ️ Information</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">💡 Insight</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">🎯 Prediction</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">✅ Decision</span>
      <span class="pipeline-arrow">→</span>
      <span class="pipeline-step">⚡ Automation</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Topic grid
    st.markdown("### Workshop Curriculum")

    topics = [
        ("📊", "1. Understanding Data",   "What data IS, types, quality dimensions",               "understanding_data",  "blue"),
        ("🔌", "2. Data Collection",      "APIs, sensors, databases, web scraping, streams",        "data_collection",     "green"),
        ("🧹", "3. Data Cleaning",        "Missing values, outliers, duplicates, validation",       "data_cleaning",       "orange"),
        ("⚙️", "4. Data Processing",     "Pipelines, transformation, encoding, scaling",            "data_processing",     "purple"),
        ("🔍", "5. Exploratory Analysis", "EDA, statistical summaries, pattern discovery",          "eda",                 "blue"),
        ("🛠️", "6. Feature Engineering", "Creating signal, selection, dimensionality reduction",    "feature_engineering", "green"),
        ("📈", "7. Data Visualization",   "Plotly charts, dashboards, storytelling with data",      "visualization",       "orange"),
        ("🤖", "8. ML Fundamentals",      "Supervised/unsupervised, training, evaluation",          "ml_fundamentals",     "purple"),
        ("🧠", "9. AI Applications",      "LLMs, computer vision, NLP, generative AI",             "ai_applications",     "red"),
        ("🚀", "10. MLOps & Production", "Deployment, monitoring, CI/CD for ML, data drift",       "mlops",               "blue"),
    ]

    cols = st.columns(2)
    for i, (icon, title, desc, page_id, color) in enumerate(topics):
        completed = page_id in st.session_state.get("completed_topics", [])
        check = "✅ " if completed else ""
        with cols[i % 2]:
            st.markdown(f"""
            <div class="ds-card" style="border-left: 3px solid {'#3b82f6' if color=='blue' else '#10b981' if color=='green' else '#f59e0b' if color=='orange' else '#8b5cf6' if color=='purple' else '#ef4444'};">
              <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
              <div style="font-weight:700; color:#f1f5f9; margin-bottom:0.2rem;">{check}{title}</div>
              <div style="font-size:0.8rem; color:#64748b;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Special modes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ds-card-accent">
          <div style="font-size:2rem;">⚡</div>
          <div style="font-weight:700; color:#f1f5f9; margin:0.3rem 0;">Challenge Mode</div>
          <div style="font-size:0.8rem; color:#94a3b8;">Hands-on mini-projects with real datasets, tasks, hints and expected outputs. Test your skills!</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ds-card-accent">
          <div style="font-size:2rem;">🏆</div>
          <div style="font-weight:700; color:#f1f5f9; margin:0.3rem 0;">Capstone Project</div>
          <div style="font-size:0.8rem; color:#94a3b8;">Full end-to-end project: business problem → data pipeline → ML model → deployment architecture.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="ds-card-accent">
          <div style="font-size:2rem;">💬</div>
          <div style="font-weight:700; color:#f1f5f9; margin:0.3rem 0;">AI Mentor Chat</div>
          <div style="font-size:0.8rem; color:#94a3b8;">Ask anything. Claude answers as your personal data science mentor — with code, diagrams, and examples.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="box-info">
      <strong>🚀 How to navigate:</strong> Use the sidebar to jump to any topic, challenge, or chat.
      Each topic page has <strong>Explain → Visualize → Code → Quiz</strong> sections.
      Complete topics to track your progress!
    </div>
    """, unsafe_allow_html=True)
