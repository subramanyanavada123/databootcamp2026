import pandas as pd
import numpy as np

def get_engine_dataset(n=500, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "timestamp":       pd.date_range("2026-06-01", periods=n, freq="1min"),
        "engine_id":       np.random.choice(["ENG-L-001", "ENG-R-001"], n),
        "egt_celsius":     np.random.normal(640, 15, n).round(2),
        "n1_fan_speed":    np.random.normal(94, 2, n).round(2),
        "vibration":       np.random.exponential(0.4, n).round(4),
        "fuel_flow_kg_hr": np.random.normal(2850, 50, n).round(1),
        "altitude_ft":     np.random.choice([0, 10000, 20000, 35000], n),
        "alert_active":    np.random.choice([True, False], n, p=[0.05, 0.95]),
        "severity":        np.random.choice(["Normal", "Caution", "Warning"], n, p=[0.90, 0.08, 0.02]),
    })
    # inject some missing values and anomalies
    idx_miss = np.random.choice(n, size=20, replace=False)
    df.loc[idx_miss[:10], "egt_celsius"] = np.nan
    df.loc[idx_miss[10:], "vibration"] = np.nan
    idx_anom = np.random.choice(n, size=5, replace=False)
    df.loc[idx_anom, "egt_celsius"] = np.random.uniform(750, 900, 5).round(2)
    return df


def get_manufacturing_dataset(n=300, seed=7):
    np.random.seed(seed)
    df = pd.DataFrame({
        "part_id":        [f"PART-{i:04d}" for i in range(n)],
        "material":       np.random.choice(["Steel", "Aluminum", "Copper"], n),
        "thickness_mm":   np.random.normal(10.0, 0.3, n).round(3),
        "hardness_HRC":   np.random.normal(58, 3, n).round(1),
        "surface_finish": np.random.choice(["Rough", "Medium", "Fine"], n),
        "weight_kg":      np.random.normal(2.5, 0.1, n).round(3),
        "temperature_C":  np.random.normal(22, 5, n).round(1),
        "pressure_bar":   np.random.normal(100, 10, n).round(1),
        "passed_qc":      np.random.choice([True, False], n, p=[0.92, 0.08]),
        "defect_count":   np.random.poisson(0.15, n),
    })
    idx_miss = np.random.choice(n, size=15, replace=False)
    df.loc[idx_miss[:8], "hardness_HRC"] = np.nan
    df.loc[idx_miss[8:], "thickness_mm"] = np.nan
    return df


def get_telecom_dataset(n=400, seed=21):
    np.random.seed(seed)
    df = pd.DataFrame({
        "customer_id":       [f"CUST-{i:05d}" for i in range(n)],
        "tenure_months":     np.random.randint(1, 72, n),
        "monthly_charges":   np.random.normal(65, 20, n).round(2),
        "total_charges":     np.nan,  # to be computed
        "contract_type":     np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "internet_service":  np.random.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.45, 0.20]),
        "tech_support":      np.random.choice(["Yes", "No"], n),
        "num_complaints":    np.random.poisson(0.5, n),
        "avg_call_duration": np.random.exponential(8, n).round(2),
        "data_usage_gb":     np.random.exponential(15, n).round(2),
        "churned":           np.random.choice([True, False], n, p=[0.27, 0.73]),
    })
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.95, 1.05, n)).round(2)
    idx_miss = np.random.choice(n, size=10, replace=False)
    df.loc[idx_miss, "avg_call_duration"] = np.nan
    return df


def get_iot_dataset(n=1000, seed=99):
    np.random.seed(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamp":       t,
        "device_id":       np.random.choice([f"DEVICE-{i:03d}" for i in range(1, 6)], n),
        "temperature_C":   (20 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.5, n)).round(2),
        "humidity_pct":    np.clip(np.random.normal(55, 10, n), 0, 100).round(1),
        "pressure_hPa":    np.random.normal(1013, 5, n).round(1),
        "co2_ppm":         np.random.normal(400, 50, n).round(0),
        "motion_detected": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "battery_pct":     np.clip(100 - np.arange(n) * 0.08 + np.random.normal(0, 1, n), 0, 100).round(1),
    })
    return df


def get_healthcare_dataset(n=250, seed=55):
    np.random.seed(seed)
    df = pd.DataFrame({
        "patient_id":      [f"PAT-{i:04d}" for i in range(n)],
        "age":             np.random.randint(25, 80, n),
        "gender":          np.random.choice(["M", "F"], n),
        "bmi":             np.random.normal(27, 5, n).round(1),
        "blood_pressure":  np.random.normal(120, 20, n).round(0),
        "cholesterol":     np.random.normal(200, 40, n).round(0),
        "glucose":         np.random.normal(100, 25, n).round(0),
        "smoker":          np.random.choice([True, False], n, p=[0.25, 0.75]),
        "exercise_hrs_wk": np.random.exponential(3, n).round(1),
        "readmitted":      np.random.choice([True, False], n, p=[0.18, 0.82]),
    })
    idx_miss = np.random.choice(n, size=12, replace=False)
    df.loc[idx_miss[:6], "bmi"] = np.nan
    df.loc[idx_miss[6:], "cholesterol"] = np.nan
    return df


DATASETS = {
    "✈️ Aircraft Engine (Aerospace)":     get_engine_dataset,
    "🔩 Manufacturing QC (Industry)":     get_manufacturing_dataset,
    "📱 Telecom Churn (Telco)":           get_telecom_dataset,
    "🌡️ IoT Sensors (Smart Building)":    get_iot_dataset,
    "🏥 Patient Readmission (Healthcare)": get_healthcare_dataset,
}
