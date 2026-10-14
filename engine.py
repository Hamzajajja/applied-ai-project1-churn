from __future__ import annotations

import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    ATTRIBUTION_GROUPS, BILL_MAX_USD, BILL_MIN_USD, CITIES, CONTRACTS, INTERNET,
    PAYMENTS, RAW_COLS, REFERENCE, SERVICES, TIERS,
)

log = logging.getLogger("churn")

MODEL = None
META = None
COLS: list = []


@st.cache_resource(show_spinner="Loading model...")
def _load_files():
    if not (os.path.exists("churn_model.joblib") and os.path.exists("model_meta.json")):
        log.error("model files not found in %s", os.getcwd())
        return None, None
    try:
        model = joblib.load("churn_model.joblib")
        with open("model_meta.json") as f:
            meta = json.load(f)
        return model, meta
    except Exception:
        log.exception("could not load model files")
        return None, None


def load() -> bool:
    global MODEL, META, COLS
    MODEL, META = _load_files()
    if MODEL is None or META is None:
        return False
    COLS = META["feature_columns"]
    return True


# ---------------------------------------------------------------- inference

def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(raw[RAW_COLS])
    return x.reindex(columns=COLS, fill_value=0).astype(float)


def predict(raw: pd.DataFrame) -> np.ndarray:
    start = time.perf_counter()
    try:
        proba = MODEL.predict_proba(prepare(raw))[:, 1]
    except Exception as e:
        log.exception("prediction failed")
        st.error(f"Prediction failed: {e}")
        return np.zeros(len(raw))
    st.session_state["latency_ms"] = (time.perf_counter() - start) * 1000
    return proba


@st.cache_data(show_spinner="Scoring customers...")
def score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["churn_probability"] = predict(out)
    return out


def tier_of(p: float, t: float):
    if p >= t:
        return "HIGH RISK", "#ef4444", "Priority retention call with a tailored offer"
    if p >= t / 2:
        return "WATCHLIST", "#f59e0b", "Add to loyalty campaign and review in 30 days"
    return "STABLE", "#22c55e", "No action needed, keep normal service"


def tier_series(p: pd.Series, t: float) -> pd.Series:
    return pd.cut(p, bins=[-1, t / 2, t, 2], labels=TIERS)


def fix_dependencies(row: dict) -> dict:
    """Keeps internet / phone dependent fields consistent after we change something."""
    if row["InternetService"] == "No":
        for s in SERVICES:
            row[s] = "No internet service"
    else:
        for s in SERVICES:
            if row[s] == "No internet service":
                row[s] = "No"
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    elif row["MultipleLines"] == "No phone service":
        row["MultipleLines"] = "No"
    row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
    return row


# ---------------------------------------------------------------- single customer analysis

@st.cache_data(show_spinner=False)
def attributions(customer_json: str) -> pd.DataFrame:
    """Swap each group of features for the reference value and see how much the risk moves."""
    cust = json.loads(customer_json)
    rows = [cust]
    for cols in ATTRIBUTION_GROUPS.values():
        r = dict(cust)
        for c in cols:
            r[c] = REFERENCE[c]
        rows.append(fix_dependencies(r))
    p = predict(pd.DataFrame(rows))
    out = pd.DataFrame({"Factor": list(ATTRIBUTION_GROUPS), "Contribution": p[0] - p[1:]})
    return out.sort_values("Contribution", key=np.abs).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def reference_probability() -> float:
    return float(predict(pd.DataFrame([fix_dependencies(dict(REFERENCE))]))[0])


@st.cache_data(show_spinner=False)
def curve_tenure(customer_json: str) -> pd.DataFrame:
    cust = json.loads(customer_json)
    rows = []
    for contract in CONTRACTS:
        for months in range(0, 73, 2):
            r = fix_dependencies(dict(cust, Contract=contract, tenure=months))
            rows.append({**r, "_group": contract})
    df = pd.DataFrame(rows)
    df["p"] = predict(df)
    return df[["_group", "tenure", "p"]].rename(columns={"_group": "Contract"})


@st.cache_data(show_spinner=False)
def curve_price(customer_json: str) -> pd.DataFrame:
    cust = json.loads(customer_json)
    rows = []
    for net in INTERNET:
        for usd in np.arange(BILL_MIN_USD, BILL_MAX_USD + 1, 3.0):
            r = fix_dependencies(dict(cust, InternetService=net, MonthlyCharges=float(usd)))
            rows.append({**r, "_group": net})
    df = pd.DataFrame(rows)
    df["p"] = predict(df)
    return df[["_group", "MonthlyCharges", "p"]].rename(columns={"_group": "Internet", "MonthlyCharges": "usd"})


@st.cache_data(show_spinner=False)
def risk_surface(customer_json: str) -> pd.DataFrame:
    cust = json.loads(customer_json)
    rows = []
    for months in range(0, 73, 6):
        for usd in range(20, 121, 10):
            r = fix_dependencies(dict(cust, tenure=months, MonthlyCharges=float(usd)))
            rows.append({**r, "_t": months, "_m": usd})
    df = pd.DataFrame(rows)
    df["p"] = predict(df)
    return df.pivot(index="_m", columns="_t", values="p")


def contract_payment_grid(cust: dict) -> pd.DataFrame:
    rows = []
    for c in CONTRACTS:
        for p in PAYMENTS:
            r = fix_dependencies(dict(cust, Contract=c, PaymentMethod=p))
            rows.append({**r, "_c": c, "_p": p})
    df = pd.DataFrame(rows)
    df["p"] = predict(df)
    return df[["_c", "_p", "p"]]


def retention_levers(cust: dict) -> list:
    levers = []
    if cust["Contract"] == "Month-to-month":
        levers.append(("Move to 1-year contract", {"Contract": "One year"}))
        levers.append(("Move to 2-year contract", {"Contract": "Two year"}))
    elif cust["Contract"] == "One year":
        levers.append(("Move to 2-year contract", {"Contract": "Two year"}))
    if cust["PaymentMethod"] in ("Electronic check", "Mailed check"):
        levers.append(("Switch to card autopay", {"PaymentMethod": "Credit card (automatic)"}))
        levers.append(("Switch to bank autopay", {"PaymentMethod": "Bank transfer (automatic)"}))
    if cust["InternetService"] != "No":
        for col, label in [("TechSupport", "Add tech support"), ("OnlineSecurity", "Add online security"),
                           ("OnlineBackup", "Add online backup"), ("DeviceProtection", "Add device protection")]:
            if cust[col] == "No":
                levers.append((label, {col: "Yes"}))
    if cust["PaperlessBilling"] == "Yes":
        levers.append(("Back to paper billing", {"PaperlessBilling": "No"}))
    if cust["MonthlyCharges"] > 30:
        levers.append(("10% loyalty discount", {"MonthlyCharges": round(cust["MonthlyCharges"] * 0.9, 2)}))
    return levers


def combined_plan(cust: dict, levers: list) -> dict:
    """Everything that doesn't conflict: 2-year contract, card autopay, all add-ons, paper bill."""
    plan = dict(cust)
    for label, change in levers:
        if label in ("Move to 2-year contract", "Switch to card autopay", "Back to paper billing") \
                or label.startswith("Add"):
            plan.update(change)
    return plan


# ---------------------------------------------------------------- portfolio data

def synth_portfolio(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    net = rng.choice(INTERNET, n, p=[0.44, 0.34, 0.22])
    tenure = rng.integers(0, 73, n)
    bill = np.where(
        net == "Fiber optic", rng.uniform(70, 110, n),
        np.where(net == "DSL", rng.uniform(40, 72, n), rng.uniform(18, 30, n)),
    ).round(2)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    cities = list(CITIES)

    df = pd.DataFrame({
        "customerID": [f"PK-{i + 1:05d}" for i in range(n)],
        "City": rng.choice(cities, n, p=[CITIES[c][2] for c in cities]),
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": net,
        "Contract": rng.choice(CONTRACTS, n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "PaymentMethod": rng.choice(PAYMENTS, n, p=[0.34, 0.23, 0.22, 0.21]),
        "MonthlyCharges": bill,
        "TotalCharges": (tenure * bill).round(2),
    })
    for s in SERVICES:
        df[s] = np.where(net == "No", "No internet service", rng.choice(["Yes", "No"], n))
    return df


@st.cache_data(show_spinner=False)
def template_csv() -> str:
    return synth_portfolio(3)[["customerID", "City"] + RAW_COLS].to_csv(index=False)


def validate_and_clean(df: pd.DataFrame):
    """Returns (clean_df or None, errors, warnings)."""
    errors, notes = [], []
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {', '.join(missing)}"], notes

    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    blank_total = int(df["TotalCharges"].isna().sum())
    if blank_total:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
        notes.append(f"{blank_total} blank TotalCharges filled with tenure x MonthlyCharges.")

    bad = df[["tenure", "MonthlyCharges", "SeniorCitizen"]].isna().any(axis=1)
    if bad.any():
        notes.append(f"{int(bad.sum())} rows dropped because tenure / MonthlyCharges / SeniorCitizen were not numeric.")
        df = df[~bad]

    odd = (df["tenure"] < 0) | (df["tenure"] > 120) | (df["MonthlyCharges"] < 0)
    if odd.any():
        notes.append(f"{int(odd.sum())} rows have out-of-range numbers (kept, but worth checking).")

    for col, allowed in {"Contract": CONTRACTS, "PaymentMethod": PAYMENTS, "InternetService": INTERNET}.items():
        n_bad = int((~df[col].isin(allowed)).sum())
        if n_bad:
            notes.append(f"{n_bad} rows have a value in '{col}' the model has never seen.")

    if "customerID" in df.columns and df["customerID"].duplicated().any():
        notes.append(f"{int(df['customerID'].duplicated().sum())} duplicate customerID values found.")

    if df.empty:
        errors.append("No usable rows left after cleaning.")
        return None, errors, notes
    return df.reset_index(drop=True), errors, notes


def parse_labels(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )
    return mapped


# ---------------------------------------------------------------- model introspection

def feature_importance():
    est = MODEL.steps[-1][1] if hasattr(MODEL, "steps") else MODEL
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    else:
        return None
    if len(vals) != len(COLS):
        return None
    return pd.DataFrame({"Feature": COLS, "Importance": vals}).sort_values("Importance", ascending=False)


def estimator_name() -> str:
    est = MODEL.steps[-1][1] if hasattr(MODEL, "steps") else MODEL
    return type(est).__name__


def gain_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-p)
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    share = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame({"share": share, "gain": cum, "lift": cum / share})


def suggest_action(row) -> str:
    """Rule-based talking point for the call centre, not model output."""
    if row["Contract"] == "Month-to-month" and row["tenure"] < 12:
        return "Offer 1-year plan with first-month discount"
    if row["PaymentMethod"] == "Electronic check":
        return "Offer autopay cashback"
    if row["InternetService"] != "No" and row.get("TechSupport") == "No":
        return "Free 3-month tech-support trial"
    if row["Contract"] == "Month-to-month":
        return "Loyalty bundle with contract upgrade"
    return "Courtesy call and satisfaction check"
