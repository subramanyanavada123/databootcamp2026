import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.utils.visualizations import plot_timeseries, fig_defaults
import plotly.graph_objects as go

def render():
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem;">
      <h1 style="font-size:1.8rem;">🔌 Data Collection</h1>
      <p>Where data comes from — APIs, sensors, databases, streams, and scraping</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Explain", "🎨 Visualize", "💻 Code", "✅ Quiz"])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### How Data Enters the Pipeline")
            st.markdown("""
Data collection is the **first transformation** in the lifecycle:
converting real-world events into storable, processable signals.

**Engineering view:** Every data source is essentially a **producer** in a
producer-consumer architecture. Your job is to reliably consume it.
            """)
            st.code("""
DATA SOURCES                    COLLECTION METHOD
────────────────────────────────────────────────────
 Manufacturing sensors    ──►  MQTT / OPC-UA / Modbus
 Web user behaviour       ──►  REST API / Webhooks
 Database records         ──►  SQL queries / CDC
 Social media / News      ──►  Web scraping / APIs
 Financial markets        ──►  WebSocket streams
 IoT devices              ──►  MQTT / CoAP / HTTP
 Lab instruments          ──►  RS-232 / USB / GPIB
 Satellite telemetry      ──►  SDR / Binary protocol
            """, language="text")

        with col2:
            st.markdown("### Collection Strategies")
            strategies = {
                "Strategy": ["Batch", "Micro-batch", "Streaming", "Change Data Capture"],
                "Frequency": ["Hours/Days", "Minutes", "Real-time", "On change"],
                "Tool": ["SQL + cron", "Spark Streaming", "Kafka/Kinesis", "Debezium"],
                "Use Case": ["Reports", "Dashboards", "Alerts", "Sync"],
            }
            st.dataframe(pd.DataFrame(strategies), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### The Data Collection Pipeline")
        st.code("""
SOURCE          INGESTION       TRANSPORT       STORAGE
──────────────────────────────────────────────────────────────────
Sensor          Collector       Message Queue   Data Lake (S3)
  │                │               │               │
  │  raw bytes     │  parsed msg   │  topic/stream │  Parquet files
  └───────────►────┘───────────►───┘───────────►───┘
                                                    │
                                              Data Warehouse
                                            (BigQuery/Redshift)
                                                    │
                                              Your DataFrame ✓
        """, language="text")

        st.markdown("---")
        st.markdown("""
        <div class="box-info">
        <strong>🧠 AI Perspective:</strong> The quality of your AI model is bounded by your data collection.
        A model can never learn patterns that weren't captured.
        Poor sampling rate → missed anomalies. Biased collection → biased predictions.
        <br><br>
        <strong>Amazon</strong> famously discovered their hiring AI was biased against women —
        because training data came from 10 years of historically male-dominated CVs.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Live Simulated Data Streams")

        stream_type = st.selectbox("Choose a stream to simulate:", [
            "🌡️ IoT Temperature Sensor",
            "📡 Network Packet Monitor",
            "⚙️ CNC Machine Vibration",
            "📈 Stock Price Feed",
        ])

        n_points = st.slider("Data points:", 50, 500, 200)
        np.random.seed(42)
        t = pd.date_range("2026-06-21 00:00", periods=n_points, freq="1min")

        if "IoT" in stream_type:
            df = pd.DataFrame({
                "timestamp": t,
                "temperature_C": 22 + 5*np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 0.5, n_points),
                "humidity_pct": 55 + 10*np.cos(np.linspace(0, 2*np.pi, n_points)) + np.random.normal(0, 1, n_points),
            })
            fig = plot_timeseries(df, "timestamp", ["temperature_C", "humidity_pct"])

        elif "Network" in stream_type:
            df = pd.DataFrame({
                "timestamp": t,
                "packets_per_sec": np.random.poisson(1000, n_points),
                "latency_ms": np.random.exponential(5, n_points),
            })
            fig = plot_timeseries(df, "timestamp", ["packets_per_sec"])

        elif "CNC" in stream_type:
            signal = np.sin(np.linspace(0, 20*np.pi, n_points)) * np.random.uniform(0.8, 1.2, n_points)
            signal[150:160] *= 5  # inject spike
            df = pd.DataFrame({"timestamp": t, "vibration_g": signal})
            fig = plot_timeseries(df, "timestamp", ["vibration_g"])

        else:
            price = 100 + np.cumsum(np.random.normal(0, 1, n_points))
            df = pd.DataFrame({"timestamp": t, "price_usd": price})
            fig = plot_timeseries(df, "timestamp", ["price_usd"])

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("#### Collection Latency Comparison")
        latency_df = pd.DataFrame({
            "Method": ["REST API (polling)", "WebSocket", "MQTT", "Kafka Consumer", "Database CDC"],
            "Latency": [500, 10, 5, 20, 100],
            "Throughput_msgs_sec": [100, 10000, 50000, 1000000, 5000],
        })
        import plotly.express as px
        fig2 = px.scatter(latency_df, x="Latency", y="Throughput_msgs_sec",
                          text="Method", size=[30]*5, color="Method",
                          log_x=True, log_y=True,
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_traces(textposition="top center")
        fig2 = fig_defaults(fig2, "Latency vs Throughput by Collection Method", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### Code Examples")

        method = st.selectbox("Choose method:", [
            "REST API", "MQTT Sensor", "SQL Database", "Web Scraping", "CSV/File Ingestion"
        ])

        if method == "REST API":
            st.code("""
import requests
import pandas as pd
from datetime import datetime

def collect_from_api(endpoint: str, api_key: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(endpoint, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["results"])
    df["collected_at"] = datetime.now()
    return df

# Example: Weather station API
df = collect_from_api(
    "https://api.openweathermap.org/data/2.5/stations",
    api_key="YOUR_KEY"
)

# Pagination pattern
def collect_all_pages(base_url, api_key, max_pages=100):
    all_records = []
    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}&limit=100"
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
        data = resp.json()
        if not data["results"]:
            break
        all_records.extend(data["results"])
    return pd.DataFrame(all_records)
            """, language="python")

        elif method == "MQTT Sensor":
            st.code("""
import paho.mqtt.client as mqtt
import pandas as pd
import json
from datetime import datetime

records = []

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    records.append({
        "timestamp": datetime.now(),
        "device_id": payload["device_id"],
        "temperature": payload["temperature"],
        "humidity": payload["humidity"],
        "topic": msg.topic,
    })

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt.factory.internal", port=1883)
client.subscribe("sensors/floor1/#")   # wildcard subscription
client.loop_start()

# Collect for 60 seconds
import time
time.sleep(60)
client.loop_stop()

df = pd.DataFrame(records)
df.to_parquet("sensor_batch.parquet", index=False)
print(f"Collected {len(df)} readings from {df['device_id'].nunique()} devices")
            """, language="python")

        elif method == "SQL Database":
            st.code("""
import pandas as pd
import sqlalchemy as sa
from datetime import datetime, timedelta

engine = sa.create_engine("postgresql://user:pass@host:5432/production_db")

# Incremental extraction (only new records)
def extract_incremental(last_run: datetime) -> pd.DataFrame:
    query = sa.text(\"\"\"
        SELECT
            event_id,
            machine_id,
            event_type,
            value,
            unit,
            recorded_at
        FROM sensor_events
        WHERE recorded_at > :last_run
          AND event_type IN ('temperature', 'pressure', 'vibration')
        ORDER BY recorded_at
        LIMIT 100000
    \"\"\")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"last_run": last_run},
                         parse_dates=["recorded_at"])
    return df

# Pivot wide format: one row per event_time
def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index=["machine_id", "recorded_at"],
        columns="event_type",
        values="value",
        aggfunc="first"
    ).reset_index()
            """, language="python")

        elif method == "Web Scraping":
            st.code("""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_product_table(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    table = soup.find("table", {"class": "data-table"})
    for tr in table.find_all("tr")[1:]:   # skip header
        cells = [td.text.strip() for td in tr.find_all("td")]
        rows.append(cells)

    df = pd.DataFrame(rows, columns=["product", "price", "stock", "rating"])
    df["price"] = df["price"].str.replace("$", "").astype(float)
    df["scraped_at"] = pd.Timestamp.now()
    return df

# Polite scraping: add delays
def scrape_multiple_pages(base_url, n_pages):
    all_dfs = []
    for i in range(1, n_pages + 1):
        df = scrape_product_table(f"{base_url}?page={i}")
        all_dfs.append(df)
        time.sleep(1)   # be polite — 1 req/sec
    return pd.concat(all_dfs, ignore_index=True)
            """, language="python")

        else:
            st.code("""
import pandas as pd
import glob
import os

# Single file
df = pd.read_csv("data/sensors_2026_06.csv", parse_dates=["timestamp"])

# Multiple files matching pattern
files = glob.glob("data/sensors_*.csv")
df = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in files],
               ignore_index=True)

# Parquet (10x faster, compressed)
df = pd.read_parquet("data/sensors.parquet")

# Excel with multiple sheets
dfs = pd.read_excel("quality_report.xlsx", sheet_name=None)   # all sheets
df_all = pd.concat(dfs.values(), ignore_index=True)

# Large files: chunked reading
chunk_iter = pd.read_csv("large_log.csv", chunksize=100_000)
df = pd.concat(
    [chunk[chunk["severity"] == "ERROR"] for chunk in chunk_iter],
    ignore_index=True
)
print(f"Loaded {len(df):,} error records")
            """, language="python")

    with tab4:
        st.markdown("### Quiz — Data Collection")
        questions = [
            {"q": "Which protocol is most common for IoT sensor data in manufacturing?",
             "options": ["HTTP REST", "MQTT", "FTP", "SMTP"],
             "answer": "MQTT", "explanation": "MQTT is designed for low-bandwidth, high-reliability IoT messaging."},
            {"q": "You need sub-10ms latency for stock prices. Which method fits?",
             "options": ["Batch SQL", "CSV files", "WebSocket stream", "Email digest"],
             "answer": "WebSocket stream", "explanation": "WebSockets maintain persistent connections with near-real-time push."},
            {"q": "What is CDC (Change Data Capture) used for?",
             "options": ["Compressing CSV", "Streaming only new DB changes", "Scraping websites", "Encrypting data"],
             "answer": "Streaming only new DB changes", "explanation": "CDC captures row-level changes in a database and streams them downstream."},
        ]
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['q']}")
            choice = st.radio("", q["options"], key=f"quiz_02_q{i}", index=None, horizontal=True)
            if choice:
                if choice == q["answer"]:
                    st.markdown(f'<div class="box-success">✅ Correct! {q["explanation"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="box-warning">❌ Answer: **{q["answer"]}**. {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")

        if "data_collection" not in st.session_state.completed_topics:
            if st.button("✅ Mark topic as complete"):
                st.session_state.completed_topics.append("data_collection")
                st.success("Topic marked complete!")
