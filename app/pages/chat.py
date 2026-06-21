import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.ai_mentor import stream_response, ANTHROPIC_AVAILABLE

QUICK_PROMPTS = [
    "What is a DataFrame and how does it work?",
    "Explain the difference between AI, ML, Deep Learning, and Generative AI",
    "What is feature engineering and why does it matter?",
    "How does a Random Forest work?",
    "What is data drift and how do I detect it?",
    "How do I handle imbalanced datasets?",
    "Explain overfitting vs underfitting with examples",
    "What is cross-validation and when should I use it?",
    "How does transfer learning work?",
    "What is RAG (Retrieval Augmented Generation)?",
    "Explain the bias-variance tradeoff",
    "What metrics should I use for imbalanced classification?",
]

TOPIC_STARTERS = {
    "📊 Understanding Data":   "Teach me about data types and quality dimensions with engineering examples",
    "🔌 Data Collection":      "What are the main data collection methods in industrial IoT systems?",
    "🧹 Data Cleaning":        "Walk me through a complete data cleaning workflow for sensor data",
    "⚙️ Data Processing":     "Explain the difference between StandardScaler, MinMaxScaler, and RobustScaler",
    "🔍 EDA":                  "What are the most important EDA steps before training a model?",
    "🛠️ Feature Engineering": "Give me 10 feature engineering techniques for time series data",
    "📈 Visualization":        "What charts should I use for different types of data questions?",
    "🤖 ML Fundamentals":      "Compare Random Forest vs XGBoost vs Neural Networks for tabular data",
    "🧠 AI Applications":      "How do I build an anomaly detection system for manufacturing sensors?",
    "🚀 MLOps":                "What does a production ML pipeline look like end to end?",
}

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">💬 AI Mentor Chat</h1>
      <p>Ask anything about data science, ML, AI, and Python — get expert answers with code</p>
    </div>
    """, unsafe_allow_html=True)

    if not ANTHROPIC_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        st.markdown("""
        <div class="box-warning">
        ⚠️ <strong>API Key not set.</strong>
        To enable live AI responses, set your Anthropic API key:<br/><br/>
        <code>export ANTHROPIC_API_KEY=sk-ant-...</code><br/>
        or add it to a <code>.env</code> file in the project root.<br/><br/>
        <strong>Pre-built answers for common questions still work below!</strong>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Quick Prompts")
        st.markdown("Click to ask:")
        for prompt in QUICK_PROMPTS[:8]:
            if st.button(prompt[:45] + "..." if len(prompt) > 45 else prompt,
                         use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", "content": prompt
                })
                st.rerun()

        st.markdown("### By Topic")
        for topic, starter in TOPIC_STARTERS.items():
            if st.button(topic, use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user", "content": starter
                })
                st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="box-info">
                <strong>👋 Welcome to your AI Mentor!</strong><br/><br/>
                I'm here to answer any data science questions:<br/>
                • "Why?" — go deeper on any concept<br/>
                • "What if?" — explore edge cases<br/>
                • "Show me code for X" — code-first explanations<br/>
                • "Compare A vs B" — side-by-side analysis<br/>
                • "Give me a challenge on topic X" — practice problem<br/><br/>
                Use the quick prompts on the right, or type your question below.
                </div>
                """, unsafe_allow_html=True)
            else:
                for i, msg in enumerate(st.session_state.chat_history):
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-user">
                          <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.3rem;">YOU</div>
                          {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-ai">
                          <div style="font-size:0.75rem; color:#10b981; margin-bottom:0.3rem;">🔬 AI MENTOR</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(msg["content"])

        # Check if last message is from user (needs response)
        needs_response = (
            st.session_state.chat_history and
            st.session_state.chat_history[-1]["role"] == "user"
        )

        if needs_response:
            with st.spinner("Thinking..."):
                user_msg = st.session_state.chat_history[-1]["content"]
                history_for_api = st.session_state.chat_history[:-1]

                response = stream_response(user_msg, history_for_api)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                })
                st.rerun()

        # Input box
        st.markdown("---")
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([6, 1])
            user_input = c1.text_input(
                "Ask your question:",
                placeholder='e.g., "How do I detect data drift in production?" or "Show me code for K-Means"',
                label_visibility="collapsed"
            )
            submitted = c2.form_submit_button("Send ↗", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input.strip()
            })
            st.rerun()

        # Suggested follow-ups
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
            st.markdown("**Suggested follow-ups:**")
            followups = [
                "Show me Python code for this",
                "Give me a real industry example",
                "What are the edge cases?",
                "How does this connect to ML?",
                "Give me a practice challenge",
            ]
            cols = st.columns(len(followups))
            for i, fu in enumerate(followups):
                with cols[i]:
                    if st.button(fu, key=f"fu_{i}", use_container_width=True):
                        st.session_state.chat_history.append({
                            "role": "user", "content": fu
                        })
                        st.rerun()
