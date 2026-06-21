import os
import streamlit as st

try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False
    _CLIENT = None

SYSTEM_PROMPT = """You are an expert Data Science Mentor, Industry Data Scientist, AI Engineer, and Python Coach.

Your audience consists of technically strong engineers attending a Data Science Foundation workshop.

Teaching style:
- Interactive, conversational, and highly visual using ASCII diagrams
- Industry-focused with examples from aerospace, manufacturing, telecom, automotive, healthcare, finance, and IoT
- Practical rather than academic
- Engineering-oriented

For every answer:
1. Give a simple explanation first, then an engineering-level explanation
2. Use ASCII diagrams, flowcharts, and tables where relevant
3. Show Python code examples when helpful
4. Connect concepts to: Data → Information → Insight → Prediction → Decision → Automation
5. When relevant, explain how AI/ML uses the concept

Available topics:
1. Understanding Data  2. Data Collection  3. Data Cleaning  4. Data Processing
5. Exploratory Analysis  6. Feature Engineering  7. Data Visualization
8. ML Fundamentals  9. AI Applications  10. MLOps & Production

Keep responses concise but rich. Use markdown formatting. Always encourage curiosity."""


def stream_response(user_message: str, history: list) -> str:
    if not ANTHROPIC_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback_response(user_message)

    messages = []
    for h in history[-10:]:  # last 10 turns for context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    full = ""
    placeholder = st.empty()
    try:
        with _CLIENT.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full += text
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)
    except Exception as e:
        full = f"⚠️ API error: {e}\n\n" + _fallback_response(user_message)
        placeholder.markdown(full)
    return full


def _fallback_response(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["dataframe", "pandas", "df"]):
        return """## DataFrame — The Engineer's Spreadsheet

A **DataFrame** is a 2D labeled data structure — think programmable spreadsheet backed by NumPy arrays.

```
         Col 0       Col 1      Col 2
         engine_id   temp_C     alert
Index 0  ENG-L-001   642.7      False
Index 1  ENG-R-001   651.2      True
```

```python
import pandas as pd
df = pd.read_csv('data.csv')
df.head()          # first 5 rows
df.dtypes          # column types
df.describe()      # statistics
df[df['alert']]    # filter rows
```

**Key insight:** Each column is a NumPy array → vectorized operations → 100x faster than Python loops."""

    if any(w in q for w in ["machine learning", "ml", "model"]):
        return """## Machine Learning — Teaching Machines with Data

```
HUMAN LEARNING           ML LEARNING
──────────────           ───────────
See examples    →        Training data  (X, y)
Spot patterns   →        Algorithm finds patterns
Apply to new    →        model.predict(X_new)
```

**Three types:**
| Type | Example | Algorithm |
|---|---|---|
| Supervised | Predict churn | Random Forest |
| Unsupervised | Find segments | K-Means |
| Reinforcement | Game AI | Q-Learning |

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```"""

    if any(w in q for w in ["missing", "null", "nan", "clean"]):
        return """## Handling Missing Data

```
STRATEGIES
──────────
Remove  → df.dropna()           ← only if <5% missing
Fill    → df.fillna(mean)       ← numerical columns
Impute  → KNNImputer()          ← smarter filling
Flag    → df['was_null'] = ...  ← preserve signal
```

```python
# Check missing values
df.isnull().sum()

# Fill with median (robust to outliers)
df['column'].fillna(df['column'].median(), inplace=True)

# Or use scikit-learn imputer
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
df_clean = imp.fit_transform(df)
```

⚠️ Never blindly drop rows — missing data often carries signal."""

    return """## Great question!

To get AI-powered answers, add your **ANTHROPIC_API_KEY** to the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or create a `.env` file in the project root.

**In the meantime**, here are the quick-access topics:
- `What is a DataFrame?`
- `How does machine learning work?`
- `What is feature engineering?`
- `Explain data cleaning`

The AI Mentor works fully offline with the interactive lessons in each topic page!"""
