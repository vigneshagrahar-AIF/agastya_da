# app.py — Agastya DA Dashboard (polished UI, cached, filter form, first-sheet-only)
# Author focus: clarity, speed, and craft. Safe for large sheets; avoids recompute on filter changes.

import re
import warnings
import traceback
import hashlib
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ========================= App Configuration =========================
st.set_page_config(
    page_title="Agastya – DA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:ops@agastya.org",
        "Report a bug": "mailto:ops@agastya.org",
        "About": "Agastya – Data Assessment Dashboard • Clean → Score → Aggregate → Visualize",
    },
)
APP_VER = "AIO v2.0 – polished UI (first-sheet only + cached)"

warnings.filterwarnings("ignore", category=FutureWarning)

# ========================= Theme / UX polish =========================
# Subtle, readable typography, refined spacing, and accent color system
st.markdown(
    """
    <style>
    :root {
        --accent:#4C6FFF;            /* calm indigo */
        --accent-2:#2BB673;          /* success green */
        --accent-3:#FF7A59;          /* warm orange */
        --muted:#6b7280;             /* gray-500 */
        --surface:#0b0d0f00;
        --radius:10px;
    }
    /* page */
    .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
    h1,h2,h3 { letter-spacing: 0.2px; }
    /* metric cards */
    .kpi { border:1px solid rgba(125,125,125,0.15); border-radius: var(--radius); padding: 12px 14px; }
    .kpi h3 { font-size: 0.9rem; color: var(--muted); margin: 0; }
    .kpi .v { font-size: 1.6rem; font-weight: 700; margin-top: 6px; }
    .kpi .s { font-size: 0.85rem; color: var(--muted); }
    /* section cards */
    .card { border:1px solid rgba(125,125,125,0.18); border-radius: var(--radius); padding: 14px; }
    .tight { margin-top: 0.4rem; margin-bottom: 0.8rem; }
    /* sidebar header spacing */
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px; border: 1px solid rgba(0,0,0,0.08);
        padding: .5rem .8rem; font-weight: 600;
    }
    .good { color: var(--accent-2); }
    .warn { color: var(--accent-3); }
    .pill { padding:.12rem .5rem; border-radius:999px; font-size:.78rem; border:1px solid rgba(0,0,0,.08) }
    .small { font-size: .9rem; color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================= Constants / Config =========================
MISSING_TOKENS = {
    None, "", " ", "-", "--", "NA", "N/A", "NULL", "null", "None", "nan", "NaN", "NAN", "na", "n/a", "0"
}
VALID_CHOICES = {"A", "B", "C", "D", "E"}

ALIASES = {
    "Region": ["Region"],
    "Instructor Name": ["Instructor Name", "Ignator Name", "Ignator", "Educator"],
    "Date": ["Date", "Date_Pre", "Pre Date"],
    "Date_Post": ["Date_Post", "Post Date", "Date Post"],
    "Subject": ["Subject"],
    "Topic Name": ["Topic Name", "Topic"],
    "Program Id": ["Program Id", "ProgramID"],
    "Program Type": ["Program Type"],
    "Donor": ["Donor"],
    "School Name": ["School Name"],
    "State": ["State"],
    "Student Name": ["Student Name", "Learner Name"],
    "Student Id": ["Student Id", "StudentID", "Learner Id"],
    "Gender": ["Gender"],
    "Class": ["Class"],
    "Roll No": ["Roll No", "RollNo"],
}

REQUIRED_DIMENSIONS = ["Region", "Instructor Name"]  # fill "Unknown" if missing

# ========================= Small Utils =========================
def as_bool_series(x) -> pd.Series:
    if not isinstance(x, pd.Series):
        return pd.Series([False], dtype=bool)
    if pd.api.types.is_bool_dtype(x) or pd.api.types.is_boolean_dtype(x):
        return x.fillna(False).astype(bool)
    return x.fillna(False).astype(bool)

def normalize_token(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s in MISSING_TOKENS else s

def normalize_choice(x):
    x = normalize_token(x)
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    return s if s in VALID_CHOICES else np.nan

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    colmap = {}
    for canon, variants in ALIASES.items():
        for v in variants:
            exact = [c for c in df.columns if c == v]
            if exact:
                colmap[exact[0]] = canon
                break
            approx = [c for c in df.columns if c.lower() == v.lower()]
            if approx:
                colmap[approx[0]] = canon
                break
    return df.rename(columns=colmap)

def detect_q_columns(df: pd.DataFrame):
    norm = {c: re.sub(r"\s+", "", c).lower() for c in df.columns}
    pre_q_pat   = re.compile(r"^q(\d+)$", re.IGNORECASE)
    pre_key_pat = re.compile(r"^q(\d+)answer$", re.IGNORECASE)
    post_q_pat  = re.compile(r"^q(\d+)_?post$", re.IGNORECASE)
    post_key_pat= re.compile(r"^q(\d+)_?answer_?post$", re.IGNORECASE)

    pre_q, pre_key, post_q, post_key = {}, {}, {}, {}
    for c, n in norm.items():
        n2 = n.replace("_", "")
        m = pre_q_pat.match(n2)
        if m: pre_q[int(m.group(1))] = c; continue
        m = pre_key_pat.match(n2)
        if m: pre_key[int(m.group(1))] = c; continue
        m = post_q_pat.match(n2)
        if m: post_q[int(m.group(1))] = c; continue
        m = post_key_pat.match(n2)
        if m: post_key[int(m.group(1))] = c; continue

    pre_q_cols    = [pre_q[k]   for k in sorted(pre_q)]
    pre_key_cols  = [pre_key[k] for k in sorted(pre_key)]
    post_q_cols   = [post_q[k]  for k in sorted(post_q)]
    post_key_cols = [post_key[k]for k in sorted(post_key)]
    return pre_q_cols, pre_key_cols, post_q_cols, post_key_cols

def any_non_null_row(row, cols):
    for c in cols:
        if c in row and not pd.isna(row[c]):
            return True
    return False

def score_row(row, qs, keys):
    score, total = 0, 0
    for q, k in zip(qs, keys):
        ans = row.get(q, np.nan); key = row.get(k, np.nan)
        if pd.isna(ans) and pd.isna(key):
            continue
        total += 1
        if not pd.isna(ans) and not pd.isna(key) and ans == key:
            score += 1
    return score, total

def add_completion_rate(df):
    if df is None or df.empty:
        return df
    if {"DA_Post","DA_Pre"}.issubset(df.columns) and "Completion_Rate_%" not in df.columns:
        a = pd.to_numeric(df["DA_Post"], errors="coerce")
        b = pd.to_numeric(df["DA_Pre"], errors="coerce")
        df["Completion_Rate_%"] = np.where(b.fillna(0) > 0, (a/b)*100, np.nan)
    return df

# ---------- file helpers (caching keys) ----------
def _file_bytes(u):
    pos = u.tell() if hasattr(u, "tell") else None
    raw = u.read()
    if hasattr(u, "seek"): u.seek(0)
    return raw

def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

# ========================= I/O (first sheet only) =========================
def read_raw_first_sheet(upload) -> tuple[pd.DataFrame, dict]:
    """
    Always reads the FIRST sheet if Excel; reads CSV as-is.
    Returns (df, meta) where meta includes 'source', 'sheet_name'
    """
    name = upload.name if hasattr(upload, "name") else "uploaded"
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    meta = {"source": name, "sheet_name": None}

    try:
        if suffix == "csv":
            df = pd.read_csv(upload, dtype=str, encoding="utf-8", low_memory=False)
        elif suffix in {"xlsx", "xlsm", "xls"}:
            raw = upload.read()
            bio = BytesIO(raw)
            xls = pd.ExcelFile(bio, engine="openpyxl" if suffix in {"xlsx","xlsm"} else None)
            first = xls.sheet_names[0]
            meta["sheet_name"] = first
            bio.seek(0)
            if suffix in {"xlsx","xlsm"}:
                df = pd.read_excel(bio, dtype=str, engine="openpyxl", sheet_name=first)
            else:  # legacy xls
                df = pd.read_excel(bio, dtype=str, engine="xlrd", sheet_name=first)
        else:
            raw = upload.read()
            bio = BytesIO(raw)
            try:
                xls = pd.ExcelFile(bio)
                first = xls.sheet_names[0]
                meta["sheet_name"] = first
                bio.seek(0)
                df = pd.read_excel(bio, dtype=str, sheet_name=first)
            except Exception:
                bio.seek(0)
                df = pd.read_csv(bio, dtype=str, encoding="utf-8", low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Could not read '{name}'. Error: {e}")

    if df.empty:
        raise RuntimeError("The file loaded but is empty.")
    return df, meta

# ========================= Core Processing (with progress) =========================
def process_all(df_raw: pd.DataFrame, fast_summary: bool = False):
    steps = [
        "Canonicalize & normalize",
        "Parse dates",
        "Detect Q/Key columns",
        "Normalize choices & flags",
        "Score rows",
        "Anomalies",
        "Type tweaks",
        "Aggregations",
        "Per-question stats" if not fast_summary else "(skipped) Per-question stats",
    ]
    progress = st.progress(0, text="Starting…")
    done = 0
    def bump():
        nonlocal done
        done += 1
        progress.progress(int(done/len(steps)*100), text=steps[done-1])

    # ---- Canonicalize & normalize ----
    df = canonicalize_columns(df_raw)
    for c in df.columns:
        df[c] = df[c].map(normalize_token)
    for c in ["Region","Instructor Name","School Name","Subject","Topic Name","State","Class"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    bump()

    # ---- Parse dates ----
    if "Date" in df.columns:
        df["Date"] = parse_date_series(df["Date"])
    if "Date_Post" in df.columns:
        df["Date_Post"] = parse_date_series(df["Date_Post"])
    bump()

    # ---- Detect Q/Key columns ----
    pre_q_cols, pre_key_cols, post_q_cols, post_key_cols = detect_q_columns(df)
    bump()

    # ---- Normalize choices & flags ----
    for c in pre_q_cols + pre_key_cols + post_q_cols + post_key_cols:
        if c in df.columns:
            df[c] = df[c].map(normalize_choice)

    pre_presence_cols  = [c for c in (pre_q_cols + ["Date"]) if c in df.columns]
    post_presence_cols = [c for c in (post_q_cols + ["Date_Post"]) if c in df.columns]

    if pre_presence_cols:
        df["DA_Pre_Done"] = df[pre_presence_cols].apply(lambda r: any_non_null_row(r, pre_presence_cols), axis=1)
    else:
        df["DA_Pre_Done"] = False
    if post_presence_cols:
        df["DA_Post_Done"] = df[post_presence_cols].apply(lambda r: any_non_null_row(r, post_presence_cols), axis=1)
    else:
        df["DA_Post_Done"] = False

    df["DA_Pre_Done"]  = as_bool_series(df["DA_Pre_Done"])
    df["DA_Post_Done"] = as_bool_series(df["DA_Post_Done"])
    bump()

    # ---- Scoring ----
    pre_scores, pre_totals, post_scores, post_totals = [], [], [], []
    for _, row in df.iterrows():
        ps, pt = score_row(row, pre_q_cols, pre_key_cols) if pre_q_cols and pre_key_cols else (np.nan, 0)
        qs, qt = score_row(row, post_q_cols, post_key_cols) if post_q_cols and post_key_cols else (np.nan, 0)
        pre_scores.append(ps); pre_totals.append(pt)
        post_scores.append(qs); post_totals.append(qt)
    df["Pre_Score"]  = pre_scores
    df["Pre_Max"]    = pre_totals
    df["Post_Score"] = post_scores
    df["Post_Max"]   = post_totals
    df["Gain"]       = df["Post_Score"] - df["Pre_Score"]

    same_max = (df["Pre_Max"].fillna(0) == df["Post_Max"].fillna(0)) & (df["Pre_Max"].fillna(0) > 0)
    denom = (df["Pre_Max"] - df["Pre_Score"]).where(same_max)
    df["Norm_Gain"] = ((df["Post_Score"] - df["Pre_Score"]).where(same_max)) / denom.replace(0, np.nan)
    bump()

    # ---- Anomalies ----
    has_date = "Date" in df.columns
    has_post_date = "Date_Post" in df.columns
    if has_date and has_post_date:
        df["Anom_PostBeforePre"] = (df["Date"].notna()) & (df["Date_Post"].notna()) & (df["Date_Post"] < df["Date"])
        df["Anom_SameTimestamp"] = (df["Date"].notna()) & (df["Date_Post"].notna()) & (df["Date_Post"] == df["Date"])
    else:
        df["Anom_PostBeforePre"] = False
        df["Anom_SameTimestamp"] = False
    df["Anom_PostBeforePre"] = as_bool_series(df["Anom_PostBeforePre"])
    df["Anom_SameTimestamp"] = as_bool_series(df["Anom_SameTimestamp"])
    bump()

    # ---- Type tweaks & safe columns ----
    if "Student Name" not in df.columns:
        df["Student Name"] = "Unknown Student"
    for col in REQUIRED_DIMENSIONS + ["School Name"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype("category")
    bump()

    # ---- Aggregations ----
    reg = (df.groupby("Region", dropna=False, observed=True)
             .agg(
                 Rows=("Student Name", "count"),
                 DA_Pre=("DA_Pre_Done", "sum"),
                 DA_Post=("DA_Post_Done", "sum"),
                 Unique_Ignators=("Instructor Name", pd.Series.nunique),
                 Unique_Schools=("School Name", pd.Series.nunique),
                 Mean_Pre=("Pre_Score", "mean"),
                 Median_Pre=("Pre_Score", "median"),
                 Mean_Post=("Post_Score", "mean"),
                 Median_Post=("Post_Score", "median"),
                 Mean_Gain=("Gain", "mean"),
                 Median_Gain=("Gain", "median"),
                 Mean_Norm_Gain=("Norm_Gain", "mean"),
             ).reset_index())

    ign = (df.groupby("Instructor Name", dropna=False, observed=True)
             .agg(
                 Rows=("Student Name", "count"),
                 DA_Pre=("DA_Pre_Done", "sum"),
                 DA_Post=("DA_Post_Done", "sum"),
                 Regions=("Region", pd.Series.nunique),
                 Schools=("School Name", pd.Series.nunique),
                 Mean_Pre=("Pre_Score", "mean"),
                 Mean_Post=("Post_Score", "mean"),
                 Mean_Gain=("Gain", "mean"),
                 Median_Gain=("Gain", "median"),
                 Mean_Norm_Gain=("Norm_Gain", "mean"),
             ).reset_index())

    ri = (df.groupby(["Region","Instructor Name"], dropna=False, observed=True)
            .agg(
                Rows=("Student Name", "count"),
                DA_Pre=("DA_Pre_Done", "sum"),
                DA_Post=("DA_Post_Done", "sum"),
                Schools=("School Name", pd.Series.nunique),
                Mean_Pre=("Pre_Score", "mean"),
                Mean_Post=("Post_Score", "mean"),
                Mean_Gain=("Gain", "mean"),
                Median_Gain=("Gain", "median"),
                Mean_Norm_Gain=("Norm_Gain", "mean"),
            ).reset_index())

    for dfx in (reg, ign, ri):
        add_completion_rate(dfx)

    # Region-wise ignator completion counts
    reg_ign_status = (
        df.groupby(["Region","Instructor Name"], observed=True)
          .agg(Pre_Done=("DA_Pre_Done","any"),
               Post_Done=("DA_Post_Done","any"))
          .reset_index()
    )
    pre_b2  = as_bool_series(reg_ign_status["Pre_Done"])
    post_b2 = as_bool_series(reg_ign_status["Post_Done"])
    reg_ign_status["Completion_Status"] = np.select(
        [pre_b2 & post_b2, pre_b2 & (~post_b2), (~pre_b2) & post_b2],
        ["Both", "Pre only", "Post only"], default="None"
    )
    ric = (
        reg_ign_status.groupby(["Region","Completion_Status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Both","Pre only","Post only","None"]:
        if col not in ric.columns:
            ric[col] = 0
    ric["Total_Ignators"] = ric["Both"] + ric["Pre only"] + ric["Post only"] + ric["None"]
    bump()

    # ---- Per-question stats (skip in fast mode) ----
    qp = pd.DataFrame()
    if not fast_summary:
        rows = []
        for i, (qpre, kpre) in enumerate(zip(pre_q_cols, pre_key_cols), start=1):
            if qpre not in df or kpre not in df:
                continue
            pre_total = df[qpre].notna().sum()
            pre_correct = (df[qpre] == df[kpre]).sum()
            row = {"Q": f"Q{i}", "Pre_Attempted": int(pre_total), "Pre_Correct_%": (100*pre_correct/max(pre_total,1))}
            if i <= len(post_q_cols) and post_q_cols[i-1] in df and post_key_cols[i-1] in df:
                qpost, kpost = post_q_cols[i-1], post_key_cols[i-1]
                post_total = df[qpost].notna().sum()
                post_correct = (df[qpost] == df[kpost]).sum()
                row.update({
                    "Post_Attempted": int(post_total),
                    "Post_Correct_%": (100*post_correct/max(post_total,1)) if post_total else np.nan,
                    "Delta_Correct_%": (100*post_correct/max(post_total,1) - row["Pre_Correct_%"]) if post_total else np.nan
                })
            rows.append(row)
        qp = pd.DataFrame(rows)
    bump()

    progress.progress(100, text="Done")

    meta = {
        "rows": int(len(df)),
        "pre_q_cols": pre_q_cols, "pre_key_cols": pre_key_cols,
        "post_q_cols": post_q_cols, "post_key_cols": post_key_cols,
    }
    return df, reg, ign, ri, ric, qp, meta

# ========================= Export =========================
def build_excel_report(reg, ign, ri, ric, dq, anomalies, qp, meta) -> bytes:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    info = pd.DataFrame(
        [
            ("Generated At", ts),
            ("Rows", meta.get("rows")),
            ("Detected Pre Q", ", ".join(meta.get("pre_q_cols", []))),
            ("Detected Pre Keys", ", ".join(meta.get("pre_key_cols", []))),
            ("Detected Post Q", ", ".join(meta.get("post_q_cols", []))),
            ("Detected Post Keys", ", ".join(meta.get("post_key_cols", []))),
        ], columns=["Metric","Value"]
    )
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        info.to_excel(w, sheet_name="Report_Info", index=False)
        add_completion_rate(reg.copy()).to_excel(w, sheet_name="Region_Summary", index=False)
        add_completion_rate(ign.copy()).to_excel(w, sheet_name="Ignator_Summary", index=False)
        add_completion_rate(ri.copy()).to_excel(w, sheet_name="Region_Ignator", index=False)
        ric.copy().to_excel(w, sheet_name="Region_Ignator_Completed", index=False)
        dq.copy().to_excel(w, sheet_name="Data_Quality", index=False)
        anomalies.copy().to_excel(w, sheet_name="Anomalies", index=False)
        if qp is not None and not qp.empty:
            qp.to_excel(w, sheet_name="Per_Question_Stats", index=False)
    bio.seek(0)
    return bio.getvalue()

# ========================= Caching wrappers =========================
@st.cache_data(show_spinner=False)
def cached_read_first_sheet(file_bytes: bytes, filename: str):
    from io import BytesIO

    class _U:
        """Minimal file-like object with a .name and a BytesIO backend."""
        def __init__(self, name: str, b: bytes):
            self._name = name
            self._bio = BytesIO(b)

        @property
        def name(self) -> str:
            return self._name

        # Pandas may call read(n) with a size; default -1 = read all
        def read(self, n: int = -1) -> bytes:
            return self._bio.read(None if n == -1 else n)

        # Some code paths may seek/tell; make it robust
        def seek(self, offset: int, whence: int = 0) -> int:
            return self._bio.seek(offset, whence)

        def tell(self) -> int:
            return self._bio.tell()

        # (Optional niceties)
        def readable(self) -> bool: return True
        def close(self): self._bio.close()

    u = _U(filename, file_bytes)
    return read_raw_first_sheet(u)

@st.cache_data(show_spinner=False)
def cached_process(df_raw: pd.DataFrame, fast_summary: bool):
    return process_all(df_raw, fast_summary=fast_summary)

@st.cache_data(show_spinner=False)
def cached_exclude_dupes(df_clean, reg, ign, ri):
    if {"Student Id","Topic Name","Date"}.issubset(df_clean.columns):
        df_nd = df_clean.sort_values(["Student Id","Topic Name","Date"])
        mask_dup = df_nd.duplicated(subset=["Student Id","Topic Name","Date"], keep="first")
        if mask_dup.any():
            reg2 = (df_nd.groupby("Region", dropna=False, observed=True)
                        .agg(DA_Pre=("DA_Pre_Done","sum"), DA_Post=("DA_Post_Done","sum"),
                             Rows=("Student Name","count"),
                             Unique_Ignators=("Instructor Name", pd.Series.nunique),
                             Unique_Schools=("School Name", pd.Series.nunique),
                             Mean_Pre=("Pre_Score","mean"), Mean_Post=("Post_Score","mean"),
                             Mean_Gain=("Gain","mean"), Median_Gain=("Gain","median"),
                             Mean_Norm_Gain=("Norm_Gain","mean"))
                        .reset_index()); add_completion_rate(reg2)
            ign2 = (df_nd.groupby("Instructor Name", dropna=False, observed=True)
                        .agg(DA_Pre=("DA_Pre_Done","sum"), DA_Post=("DA_Post_Done","sum"),
                             Rows=("Student Name","count"),
                             Regions=("Region", pd.Series.nunique),
                             Schools=("School Name", pd.Series.nunique),
                             Mean_Pre=("Pre_Score","mean"), Mean_Post=("Post_Score","mean"),
                             Mean_Gain=("Gain","mean"), Median_Gain=("Gain","median"),
                             Mean_Norm_Gain=("Norm_Gain","mean"))
                        .reset_index()); add_completion_rate(ign2)
            ri2 = (df_nd.groupby(["Region","Instructor Name"], dropna=False, observed=True)
                        .agg(DA_Pre=("DA_Pre_Done","sum"), DA_Post=("DA_Post_Done","sum"),
                             Rows=("Student Name","count"),
                             Schools=("School Name", pd.Series.nunique),
                             Mean_Pre=("Pre_Score","mean"), Mean_Post=("Post_Score","mean"),
                             Mean_Gain=("Gain","mean"), Median_Gain=("Gain","median"),
                             Mean_Norm_Gain=("Norm_Gain","mean"))
                        .reset_index()); add_completion_rate(ri2)
            return df_nd, reg2, ign2, ri2
    return df_clean, reg, ign, ri

# ========================= Header =========================
st.markdown(
    f"""
    <div class="card">
        <div style="display:flex; justify-content: space-between; align-items:center;">
            <div>
                <h2 style="margin-bottom:4px;">Agastya — Data Assessment</h2>
                <div class="small">Clean → Score → Aggregate → Visualize</div>
            </div>
            <div class="pill">Build {APP_VER}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================= Controls (top) =========================
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    uploaded = st.file_uploader("Upload raw data (.xls / .xlsx / .csv)", type=["xls","xlsx","csv"])
with c2:
    fast_summary = st.toggle("Fast summary", value=False, help="Skip per-question & heavy tables")
with c3:
    show_sample = st.toggle("Show sample (200)", value=False)
with c4:
    exclude_dupes = st.toggle("Exclude duplicates", value=True, help="Student Id + Topic + Date")

if not uploaded:
    st.info("Upload your file. For Excel, only the FIRST sheet is processed.")
    st.stop()

# ========================= Read + Process (cached) =========================
try:
    file_bytes = _file_bytes(uploaded)
    with st.status("Reading file… (first sheet only)", expanded=False) as s:
        df_raw, meta_in = cached_read_first_sheet(file_bytes, uploaded.name)
        s.update(label=f"File read ✓  (sheet: {meta_in.get('sheet_name') or '—'})", state="complete")
except Exception as e:
    st.error(f"Read error: {e}")
    st.stop()

if show_sample:
    st.markdown("**Sample (first 200 rows):**")
    st.dataframe(df_raw.head(200), use_container_width=True, hide_index=True)

try:
    with st.status("Processing… (cleaning, scoring, aggregating)", expanded=True) as s:
        df_clean, reg, ign, ri, ric, qp, meta = cached_process(df_raw, fast_summary)
        s.update(label="Processing complete ✓", state="complete")
except Exception as e:
    tb = traceback.format_exc()
    st.error("Processing failed.\n\n**Error:**\n```\n" + str(e) + "\n```\n**Traceback:**\n```\n" + tb + "\n```")
    st.stop()

# Data Quality (light)
diag = {}
for c in meta["pre_q_cols"] + meta["post_q_cols"]:
    if c in df_clean.columns:
        diag[f"{c}_Missing_%"] = [round(100 * df_clean[c].isna().mean(), 2)]
if "Date" in df_clean.columns:
    diag["Date_Missing_%"] = [round(100 * df_clean["Date"].isna().mean(), 2)]
if "Date_Post" in df_clean.columns:
    diag["Date_Post_Missing_%"] = [round(100 * df_clean["Date_Post"].isna().mean(), 2)]
if "Anom_PostBeforePre" in df_clean.columns:
    diag["Anom_PostBeforePre_%"] = [round(100 * df_clean["Anom_PostBeforePre"].mean(), 2)]
if "Anom_SameTimestamp" in df_clean.columns:
    diag["Anom_SameTimestamp_%"] = [round(100 * df_clean["Anom_SameTimestamp"].mean(), 2)]
dq = pd.DataFrame(diag) if diag else pd.DataFrame()

# Optional duplicate exclusion (cached)
if exclude_dupes:
    df_clean, reg, ign, ri = cached_exclude_dupes(df_clean, reg, ign, ri)
    reg_ign_status = (
        df_clean.groupby(["Region","Instructor Name"], observed=True)
             .agg(Pre_Done=("DA_Pre_Done","any"),
                  Post_Done=("DA_Post_Done","any"))
             .reset_index()
    )
    pre_b2 = as_bool_series(reg_ign_status["Pre_Done"]); post_b2 = as_bool_series(reg_ign_status["Post_Done"])
    reg_ign_status["Completion_Status"] = np.select(
        [pre_b2 & post_b2, pre_b2 & (~post_b2), (~pre_b2) & post_b2],
        ["Both", "Pre only", "Post only"], default="None"
    )
    ric = (reg_ign_status.groupby(["Region","Completion_Status"]).size()
              .unstack(fill_value=0).reset_index())
    for col in ["Both","Pre only","Post only","None"]:
        if col not in ric.columns: ric[col] = 0
    ric["Total_Ignators"] = ric["Both"] + ric["Pre only"] + ric["Post only"] + ric["None"]

# ========================= Sidebar Filters (FORM) =========================
st.sidebar.header("Filters")

with st.sidebar.form("filters_form", clear_on_submit=False):
    regions = sorted(reg["Region"].dropna().astype(str).unique()) if "Region" in reg.columns else []
    region_sel = st.multiselect("Region", regions, default=regions[: min(5, len(regions))] if regions else None)

    ri_for_ign = ri.copy()
    if region_sel and "Region" in ri_for_ign.columns:
        ri_for_ign = ri_for_ign[ri_for_ign["Region"].astype(str).isin(region_sel)]
    ignators = sorted(ri_for_ign["Instructor Name"].dropna().astype(str).unique()) if "Instructor Name" in ri_for_ign.columns else []
    ign_sel = st.multiselect("ignator", ignators)

    extra_dims = [d for d in ["Subject","Topic Name","Class","Program Type","Donor","School Name","State"] if d in ri.columns]
    extra_filters = {}
    for dim in extra_dims:
        opts = sorted(ri[dim].dropna().astype(str).unique())
        pick = st.multiselect(dim, opts, key=f"filter_{dim}")
        if pick:
            extra_filters[dim] = set(pick)

    submitted = st.form_submit_button("Apply filters")

if "filters" not in st.session_state or submitted:
    st.session_state["filters"] = {"region_sel": region_sel, "ign_sel": ign_sel, "extra_filters": extra_filters}

f = st.session_state["filters"]
region_sel = f["region_sel"]
ign_sel = f["ign_sel"]
extra_filters = f["extra_filters"]

def apply_filters(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    if "Region" in df.columns and region_sel:
        df = df[df["Region"].astype(str).isin(region_sel)]
    if "Instructor Name" in df.columns and ign_sel:
        df = df[df["Instructor Name"].astype(str).isin(ign_sel)]
    for dim, vals in extra_filters.items():
        if dim in df.columns:
            df = df[df[dim].astype(str).isin(vals)]
    return df

reg_v = apply_filters(reg.copy())
ign_v = apply_filters(ign.copy())
ri_v  = apply_filters(ri.copy())
ric_v = apply_filters(ric.copy())
anom_v = apply_filters(df_clean[(df_clean.get("Anom_PostBeforePre", False)) | (df_clean.get("Anom_SameTimestamp", False))].copy()) if not fast_summary else pd.DataFrame()

# ========================= KPIs =========================
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1:
    v = int(reg_v["DA_Pre"].sum()) if "DA_Pre" in reg_v else 0
    st.markdown(f'<div class="kpi"><h3>Total DA-Pre</h3><div class="v">{v:,}</div><div class="s">entries</div></div>', unsafe_allow_html=True)
with k2:
    v = int(reg_v["DA_Post"].sum()) if "DA_Post" in reg_v else 0
    st.markdown(f'<div class="kpi"><h3>Total DA-Post</h3><div class="v">{v:,}</div><div class="s">entries</div></div>', unsafe_allow_html=True)
with k3:
    if "DA_Post" in reg_v and "DA_Pre" in reg_v:
        denom = max(reg_v["DA_Pre"].sum(), 1)
        rate = (reg_v["DA_Post"].sum()/denom)*100
        cls = "good" if rate >= 70 else "warn" if rate < 40 else ""
        st.markdown(f'<div class="kpi"><h3>Completion Rate</h3><div class="v {cls}">{rate:,.2f}%</div><div class="s">Post / Pre</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi"><h3>Completion Rate</h3><div class="v">—</div></div>', unsafe_allow_html=True)
with k4:
    v = f"{reg_v['Mean_Pre'].mean():.2f}" if "Mean_Pre" in reg_v and not reg_v.empty else "—"
    st.markdown(f'<div class="kpi"><h3>Avg Pre Score</h3><div class="v">{v}</div></div>', unsafe_allow_html=True)
with k5:
    v = f"{reg_v['Mean_Post'].mean():.2f}" if "Mean_Post" in reg_v and not reg_v.empty else "—"
    st.markdown(f'<div class="kpi"><h3>Avg Post Score</h3><div class="v">{v}</div></div>', unsafe_allow_html=True)
with k6:
    v = f"{reg_v['Mean_Gain'].mean():.2f}" if "Mean_Gain" in reg_v and not reg_v.empty else "—"
    st.markdown(f'<div class="kpi"><h3>Avg Gain</h3><div class="v">{v}</div></div>', unsafe_allow_html=True)

st.divider()

# ========================= Tabs =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Region Deep-Dive", "Ignator Leaderboards", "Quality & Anomalies", "Per-Question"
])

# ---------- Overview ----------
with tab1:
    cA, cB = st.columns(2, gap="large")
    with cA:
        st.subheader("Regions by Completion Rate (%)", anchor=False)
        if not reg_v.empty and "Completion_Rate_%" in reg_v.columns:
            d = reg_v.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False])
            fig = px.bar(
                d, x="Region", y="Completion_Rate_%",
                hover_data=[c for c in ["DA_Pre","DA_Post","Mean_Gain","Mean_Norm_Gain","Unique_Ignators","Unique_Schools"] if c in d.columns],
            )
            fig.update_layout(xaxis_title="", yaxis_title="Completion Rate (%)", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No region rows after filters.")
    with cB:
        st.subheader("Top ignators by Completion (Region × ignator)", anchor=False)
        if not ri_v.empty and "Completion_Rate_%" in ri_v.columns:
            d = ri_v[ri_v["DA_Pre"].fillna(0) >= 10].sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]).head(25)
            if not d.empty:
                fig = px.bar(
                    d, x="Instructor Name", y="Completion_Rate_%", color="Region",
                    hover_data=[c for c in ["DA_Pre","DA_Post","Mean_Gain","Mean_Norm_Gain"] if c in d.columns],
                )
                fig.update_layout(xaxis_title="", yaxis_title="Completion Rate (%)", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ignators meet DA_Pre ≥ 10.")
        else:
            st.info("No Region × ignator rows.")
    st.subheader("Region leaderboard", anchor=False)
    if not reg_v.empty:
        tbl = reg_v.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]).reset_index(drop=True)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Region leaderboard (CSV)", tbl.to_csv(index=False).encode("utf-8"), "region_leaderboard.csv", "text/csv")
    else:
        st.info("No region summary available.")

# ---------- Region Deep-Dive ----------
with tab2:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Ignator completion status by Region", anchor=False)
        needed = {"Region","Both","Pre only","Post only","None"}
        if not ric_v.empty and needed.issubset(ric_v.columns):
            m = ric_v.melt(id_vars=["Region"], value_vars=["Both","Pre only","Post only","None"],
                           var_name="Completion_Status", value_name="Ignator_Count")
            m["Completion_Status"] = pd.Categorical(m["Completion_Status"],
                                                    categories=["Both","Pre only","Post only","None"], ordered=True)
            fig = px.bar(m, x="Region", y="Ignator_Count", color="Completion_Status", barmode="stack")
            fig.update_layout(xaxis_title="", yaxis_title="Ignators", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇ Download Region→ignator completion counts (CSV)",
                               ric_v.to_csv(index=False).encode("utf-8"),
                               "region_ignator_completed.csv", "text/csv")
        else:
            st.info("Region_Ignator_Completed missing required columns.")
    with col2:
        st.subheader("Mean Gain by Region", anchor=False)
        if not reg_v.empty and "Mean_Gain" in reg_v.columns:
            d = reg_v.sort_values("Mean_Gain", ascending=False)
            fig = px.bar(d, x="Region", y="Mean_Gain",
                         hover_data=[c for c in ["Mean_Pre","Mean_Post","Completion_Rate_%"] if c in d.columns])
            fig.update_layout(xaxis_title="", yaxis_title="Mean Gain", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Mean_Gain not available.")

    st.subheader("Region × ignator detail", anchor=False)
    if not ri_v.empty:
        d = ri_v.sort_values(["Region","Completion_Rate_%","DA_Pre"], ascending=[True, False, False]).reset_index(drop=True)
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Region × ignator detail (CSV)",
                           d.to_csv(index=False).encode("utf-8"),
                           "region_ignator_detail.csv", "text/csv")
    else:
        st.info("No Region × ignator records.")

    st.subheader("Weekly DA-Pre vs DA-Post trend", anchor=False)
    if {"Date","DA_Pre_Done","DA_Post_Done"}.issubset(df_clean.columns):
        temp = df_clean.copy()
        temp["Week"] = temp["Date"].dt.to_period("W").apply(lambda r: r.start_time) if "Date" in temp.columns else np.nan
        tt = temp.groupby("Week", observed=True).agg(DA_Pre=("DA_Pre_Done","sum"),
                                                     DA_Post=("DA_Post_Done","sum")).reset_index()
        tt = tt.dropna(subset=["Week"])
        if not tt.empty:
            fig = px.line(tt, x="Week", y=["DA_Pre","DA_Post"])
            fig.update_layout(xaxis_title="", yaxis_title="Count", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dated rows to plot time trend.")
    else:
        st.info("No dated rows to plot time trend.")

# ---------- Ignator Leaderboards ----------
with tab3:
    st.info("Adjust thresholds to reduce noise.")
    c1,c2,c3 = st.columns(3)
    with c1:
        min_pre_comp = st.number_input("Min DA_Pre for Completion ranking", min_value=0, value=10, step=1)
    with c2:
        min_pre_gain = st.number_input("Min DA_Pre for Mean Gain ranking", min_value=0, value=10, step=1)
    with c3:
        min_pre_norm = st.number_input("Min DA_Pre for Normalized Gain (ignator overall)", min_value=0, value=20, step=1)

    st.subheader("Top Completion (Region × ignator)", anchor=False)
    d = ri_v[ri_v["DA_Pre"].fillna(0) >= min_pre_comp].copy()
    if not d.empty:
        d = d.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]).head(25)
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download table (CSV)",
                           d.to_csv(index=False).encode("utf-8"),
                           "top_completion_region_ignator.csv", "text/csv")
    else:
        st.info("No ignators meet threshold.")

    st.subheader("Top Mean Gain (Region × ignator)", anchor=False)
    d2 = ri_v[ri_v["DA_Pre"].fillna(0) >= min_pre_gain].copy()
    if not d2.empty and "Mean_Gain" in d2.columns:
        d2 = d2.sort_values(["Mean_Gain","DA_Pre"], ascending=[False, False]).head(25)
        st.dataframe(d2, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download table (CSV)",
                           d2.to_csv(index=False).encode("utf-8"),
                           "top_mean_gain_region_ignator.csv", "text/csv")
    else:
        st.info("No ignators meet threshold or Mean_Gain missing.")

    st.subheader("Top Normalized Gain (ignator overall)", anchor=False)
    d3 = ign_v[ign_v["DA_Pre"].fillna(0) >= min_pre_norm].copy()
    if not d3.empty and "Mean_Norm_Gain" in d3.columns:
        d3 = d3.sort_values(["Mean_Norm_Gain","DA_Pre"], ascending=[False, False]).head(25)
        st.dataframe(d3, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download table (CSV)",
                           d3.to_csv(index=False).encode("utf-8"),
                           "top_norm_gain_ignator.csv", "text/csv")
    else:
        st.info("No ignators meet threshold or Mean_Norm_Gain missing.")

# ---------- Quality & Anomalies ----------
with tab4:
    colA, colB = st.columns([1,2], gap="large")
    with colA:
        st.subheader("Data Quality (light)", anchor=False)
        if not dq.empty:
            st.dataframe(dq, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download Data Quality (CSV)",
                               dq.to_csv(index=False).encode("utf-8"),
                               "data_quality.csv", "text/csv")
        else:
            st.info("No Data_Quality available.")
    with colB:
        st.subheader("Anomalies", anchor=False)
        if not anom_v.empty:
            st.dataframe(anom_v, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download Anomalies (CSV)",
                               anom_v.to_csv(index=False).encode("utf-8"),
                               "anomalies.csv", "text/csv")
        else:
            st.info("No anomalies detected or (Fast summary enabled).")

# ---------- Per-Question ----------
with tab5:
    if fast_summary:
        st.info("Per-question stats skipped in Fast summary.")
    elif qp is None or qp.empty:
        st.info("No per-question stats available (could not detect Q/Answer columns).")
    else:
        st.dataframe(qp, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Per-Question Stats (CSV)",
                           qp.to_csv(index=False).encode("utf-8"),
                           "per_question_stats.csv", "text/csv")

st.divider()

# ========================= Download generated Excel =========================
st.subheader("Download generated Excel report", anchor=False)
colx, coly = st.columns([1,3])
with colx:
    if st.button("Generate Excel now"):
        report_bytes = build_excel_report(reg_v, ign_v, ri_v, ric_v, dq, anom_v, qp if not fast_summary else pd.DataFrame(), meta)
        st.download_button(
            "⬇ Download Excel (multi-sheet)",
            data=report_bytes,
            file_name=f"Agastya_DA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
with coly:
    st.caption("Excel contains: Region Summary, Ignator Summary, Region×Ignator, Completion breakdown, Data Quality, Anomalies, and (if enabled) Per-Question Stats.")

st.caption(f"Built for Agastya • {APP_VER}")
