# app.py — Agastya DA (v1.6 insights) — first-sheet only, fast, robust, many lenses
# Raw .xls/.xlsx/.csv -> clean -> detect Pre/Post -> score -> aggregates -> visual dashboards -> Excel

import re
import warnings
import traceback
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ───────────────────────── App config ─────────────────────────
st.set_page_config(page_title="Agastya – DA (All-in-One)", page_icon="📊", layout="wide")
APP_VER = "AIO v1.6 (insights+)"
warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────── Constants ─────────────────────────
MISSING_TOKENS = {
    None, "", " ", "-", "--", "NA", "N/A", "NULL", "null", "None", "nan", "NaN", "NAN", "na", "n/a", "0"
}
VALID_CHOICES = {"A", "B", "C", "D", "E"}

ALIASES = {
    "Region": ["Region"],
    "Instructor Name": ["Instructor Name", "Ignator Name", "Ignator", "Educator"],
    "Date": ["Date", "Date_Pre", "Pre Date"],
    "Date_Post": ["Date_Post", "Post Date", "Date Post"],
    "Time": ["Time", "Time_Pre"],
    "Time Post": ["Time Post", "Time_Post"],
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

REQUIRED_DIMENSIONS = ["Region", "Instructor Name"]  # filled as "Unknown" if missing

# ───────────────────────── Helpers ─────────────────────────
def as_bool_series(x) -> pd.Series:
    if not isinstance(x, pd.Series):
        return pd.Series([False], dtype=bool)
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

def parse_time_series(s: pd.Series) -> pd.Series:
    # returns datetime.time or NaT
    try:
        t = pd.to_datetime(s, errors="coerce").dt.time
    except Exception:
        t = pd.to_datetime(s, errors="coerce", format="%H:%M:%S").dt.time
    return t

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
    return any((c in row and not pd.isna(row[c])) for c in cols)

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

# ───────────────────────── I/O (first sheet only) ─────────────────────────
def read_raw_first_sheet(upload) -> tuple[pd.DataFrame, dict]:
    name = upload.name if hasattr(upload, "name") else "uploaded"
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    meta = {"source": name, "sheet_name": None}

    try:
        if suffix == "csv":
            df = pd.read_csv(upload, dtype=str, encoding="utf-8", low_memory=False)
        elif suffix in {"xlsx", "xlsm", "xls"}:
            raw = upload.read()
            bio = BytesIO(raw)
            try:
                xls = pd.ExcelFile(bio, engine="openpyxl" if suffix in {"xlsx","xlsm"} else None)
            except Exception:
                # last resort: let pandas sniff engine
                bio.seek(0); xls = pd.ExcelFile(bio)
            first = xls.sheet_names[0]
            meta["sheet_name"] = first
            bio.seek(0)
            engine = "openpyxl" if suffix in {"xlsx","xlsm"} else "xlrd"
            df = pd.read_excel(bio, dtype=str, engine=engine, sheet_name=first)
        else:
            raw = upload.read(); bio = BytesIO(raw)
            try:
                xls = pd.ExcelFile(bio)
                first = xls.sheet_names[0]; meta["sheet_name"] = first
                bio.seek(0); df = pd.read_excel(bio, dtype=str, sheet_name=first)
            except Exception:
                bio.seek(0); df = pd.read_csv(bio, dtype=str, encoding="utf-8", low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Could not read '{name}'. Error: {e}")

    if df.empty:
        raise RuntimeError("The file loaded but is empty.")
    return df, meta

# ───────────────────────── Core Processing (with progress) ─────────────────────────
def process_all(df_raw: pd.DataFrame, fast_summary: bool = False):
    steps = [
        "Canonicalize & normalize",
        "Parse dates/times",
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

    # Canonicalize & normalize
    df = canonicalize_columns(df_raw)
    for c in df.columns:
        df[c] = df[c].map(normalize_token)
    for c in ["Region","Instructor Name","School Name","Subject","Topic Name","State","Class","Gender"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    bump()

    # Dates & times
    if "Date" in df.columns: df["Date"] = parse_date_series(df["Date"])
    if "Date_Post" in df.columns: df["Date_Post"] = parse_date_series(df["Date_Post"])
    if "Time" in df.columns: df["Time"] = parse_time_series(df["Time"])
    if "Time Post" in df.columns: df["Time Post"] = parse_time_series(df["Time Post"])
    bump()

    # Detect Q/Key
    pre_q_cols, pre_key_cols, post_q_cols, post_key_cols = detect_q_columns(df)
    bump()

    # Normalize choices & presence flags
    for c in pre_q_cols + pre_key_cols + post_q_cols + post_key_cols:
        if c in df.columns:
            df[c] = df[c].map(normalize_choice)

    pre_presence_cols  = [c for c in (pre_q_cols + ["Date"]) if c in df.columns]
    post_presence_cols = [c for c in (post_q_cols + ["Date_Post"]) if c in df.columns]
    df["DA_Pre_Done"]  = df[pre_presence_cols].apply(lambda r: any_non_null_row(r, pre_presence_cols), axis=1) if pre_presence_cols else False
    df["DA_Post_Done"] = df[post_presence_cols].apply(lambda r: any_non_null_row(r, post_presence_cols), axis=1) if post_presence_cols else False
    df["DA_Pre_Done"]  = as_bool_series(df["DA_Pre_Done"])
    df["DA_Post_Done"] = as_bool_series(df["DA_Post_Done"])
    bump()

    # Scoring
    pre_scores, pre_totals, post_scores, post_totals = [], [], [], []
    for _, row in df.iterrows():
        ps, pt = score_row(row, pre_q_cols, pre_key_cols) if pre_q_cols and pre_key_cols else (np.nan, 0)
        qs, qt = score_row(row, post_q_cols, post_key_cols) if post_q_cols and post_key_cols else (np.nan, 0)
        pre_scores.append(ps); pre_totals.append(pt)
        post_scores.append(qs); post_totals.append(qt)
    df["Pre_Score"]  = pre_scores; df["Pre_Max"]  = pre_totals
    df["Post_Score"] = post_scores; df["Post_Max"] = post_totals
    df["Gain"]       = df["Post_Score"] - df["Pre_Score"]
    same_max = (df["Pre_Max"].fillna(0) == df["Post_Max"].fillna(0)) & (df["Pre_Max"].fillna(0) > 0)
    denom = (df["Pre_Max"] - df["Pre_Score"]).where(same_max)
    df["Norm_Gain"] = ((df["Post_Score"] - df["Pre_Score"]).where(same_max)) / denom.replace(0, np.nan)
    bump()

    # Anomalies
    if {"Date","Date_Post"}.issubset(df.columns):
        df["Anom_PostBeforePre"] = (df["Date"].notna()) & (df["Date_Post"].notna()) & (df["Date_Post"] < df["Date"])
        df["Anom_SameTimestamp"] = (df["Date"].notna()) & (df["Date_Post"].notna()) & (df["Date_Post"] == df["Date"])
    else:
        df["Anom_PostBeforePre"] = False; df["Anom_SameTimestamp"] = False
    df["Anom_PostBeforePre"] = as_bool_series(df["Anom_PostBeforePre"])
    df["Anom_SameTimestamp"] = as_bool_series(df["Anom_SameTimestamp"])
    bump()

    # Types & safe cols
    if "Student Name" not in df.columns:
        df["Student Name"] = "Unknown Student"
    for col in REQUIRED_DIMENSIONS + ["School Name"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype("category")
    bump()

    # Aggregations
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
    for dfx in (reg, ign, ri): add_completion_rate(dfx)

    # Region × ignator completion
    reg_ign_status = (
        df.groupby(["Region","Instructor Name"], observed=True)
          .agg(Pre_Done=("DA_Pre_Done","any"), Post_Done=("DA_Post_Done","any"))
          .reset_index()
    )
    pre_b2, post_b2 = as_bool_series(reg_ign_status["Pre_Done"]), as_bool_series(reg_ign_status["Post_Done"])
    reg_ign_status["Completion_Status"] = np.select(
        [pre_b2 & post_b2, pre_b2 & (~post_b2), (~pre_b2) & post_b2],
        ["Both", "Pre only", "Post only"], default="None"
    )
    ric = (reg_ign_status.groupby(["Region","Completion_Status"]).size()
              .unstack(fill_value=0).reset_index())
    for col in ["Both","Pre only","Post only","None"]:
        if col not in ric.columns: ric[col] = 0
    ric["Total_Ignators"] = ric["Both"] + ric["Pre only"] + ric["Post only"] + ric["None"]
    bump()

    # Per-question stats (optionally heavy)
    qp = pd.DataFrame()
    if not fast_summary:
        rows = []
        for i, (qpre, kpre) in enumerate(zip(pre_q_cols, pre_key_cols), start=1):
            if qpre not in df or kpre not in df: continue
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

# ───────────────────────── Extra analytics helpers ─────────────────────────
def distractor_table(df, q_col, key_col):
    """Return counts (%) for A–E choices for a single question column."""
    if q_col not in df.columns or key_col not in df.columns:
        return pd.DataFrame()
    sub = df[[q_col, key_col]].copy()
    sub["Correct"] = (sub[q_col] == sub[key_col])
    counts = sub[q_col].value_counts(dropna=True).reindex(list("ABCDE")).fillna(0).astype(int)
    total = counts.sum() if counts.sum() else 1
    pct = (counts / total * 100).round(2)
    out = pd.DataFrame({"Choice": list("ABCDE"), "Count": counts.values, "Percent": pct.values})
    # top distractor = the wrong option with max count
    wrong = out[out["Choice"].isin(VALID_CHOICES)]
    # remove the correct option from wrong choices:
    return out

def build_subject_topic(df):
    """Subject/Topic summaries with correct% and gains."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    # Row-level correctness ratio (when keys exist)
    row = df.copy()
    subj_cols = [c for c in ["Subject","Topic Name"] if c in row.columns]
    pre, post = [], []
    # simple per-row % correct when totals available
    if {"Pre_Score","Pre_Max"}.issubset(row.columns):
        row["Pre_%"] = np.where(row["Pre_Max"].fillna(0) > 0, row["Pre_Score"]/row["Pre_Max"]*100, np.nan)
    if {"Post_Score","Post_Max"}.issubset(row.columns):
        row["Post_%"] = np.where(row["Post_Max"].fillna(0) > 0, row["Post_Score"]/row["Post_Max"]*100, np.nan)
    # Subject summary
    s_cols = ["Subject"] if "Subject" in row.columns else []
    t_cols = ["Subject","Topic Name"] if set(["Subject","Topic Name"]).issubset(row.columns) else []
    sub = pd.DataFrame(); topic = pd.DataFrame()
    if s_cols:
        sub = (row.groupby(s_cols, observed=True)
                 .agg(
                    Rows=("Student Name","count"),
                    DA_Pre=("DA_Pre_Done","sum"),
                    DA_Post=("DA_Post_Done","sum"),
                    Mean_PreScore=("Pre_Score","mean"),
                    Mean_PostScore=("Post_Score","mean"),
                    Mean_Gain=("Gain","mean"),
                    Mean_PrePct=("Pre_%","mean"),
                    Mean_PostPct=("Post_%","mean"),
                 ).reset_index())
        add_completion_rate(sub)
    if t_cols:
        topic = (row.groupby(t_cols, observed=True)
                   .agg(
                      Rows=("Student Name","count"),
                      DA_Pre=("DA_Pre_Done","sum"),
                      DA_Post=("DA_Post_Done","sum"),
                      Mean_PreScore=("Pre_Score","mean"),
                      Mean_PostScore=("Post_Score","mean"),
                      Mean_Gain=("Gain","mean"),
                      Mean_PrePct=("Pre_%","mean"),
                      Mean_PostPct=("Post_%","mean"),
                   ).reset_index())
        add_completion_rate(topic)
    return sub, topic

def funnel_table(df, by_cols):
    """Assigned -> Attempted(Pre) -> Completed(Post) funnel."""
    if df.empty: return pd.DataFrame()
    d = df.copy()
    d["Assigned"] = True  # every row is an assignment record
    agg = (d.groupby(by_cols, observed=True)
             .agg(
                 Assigned=("Assigned","sum"),
                 Pre_Attempted=("DA_Pre_Done","sum"),
                 Post_Completed=("DA_Post_Done","sum"),
             ).reset_index())
    agg["Pre_Attempt_%"] = np.where(agg["Assigned"]>0, agg["Pre_Attempted"]/agg["Assigned"]*100, np.nan)
    agg["Post_Complete_%"] = np.where(agg["Pre_Attempted"]>0, agg["Post_Completed"]/agg["Pre_Attempted"]*100, np.nan)
    return agg

def equity_gaps(df, gap_dim="Gender"):
    """Compute completion & gain gaps between groups (e.g., M vs F)."""
    if df.empty or gap_dim not in df.columns: return pd.DataFrame()
    # roll up per-student first to avoid row-level noise
    if "Student Id" not in df.columns:
        return pd.DataFrame()
    stu = (df.groupby(["Student Id", gap_dim], observed=True)
             .agg(Pre=("DA_Pre_Done","sum"),
                  Post=("DA_Post_Done","sum"),
                  Mean_Gain=("Gain","mean"))
             .reset_index())
    stu["Completed_Both"] = (stu["Pre"]>0) & (stu["Post"]>0)
    g = (stu.groupby(gap_dim, observed=True)
            .agg(Students=("Student Id","nunique"),
                 Completed=("Completed_Both","sum"),
                 Avg_Gain=("Mean_Gain","mean"))
            .reset_index())
    g["Completion_Rate_%"] = np.where(g["Students"]>0, g["Completed"]/g["Students"]*100, np.nan)
    return g

def outlier_scan(df):
    """Simple outlier detection based on z-score of Gain and Norm_Gain; and repeated answers."""
    if df.empty: return pd.DataFrame()
    d = df.copy()
    out = {}
    for col in ["Gain","Norm_Gain"]:
        if col in d.columns:
            z = (d[col] - d[col].mean())/ (d[col].std(ddof=0) if d[col].std(ddof=0) else 1)
            out[f"{col}_z"] = z
    # repeated identical answers across Qs as a heuristic (e.g., all 'A')
    qcols = [c for c in d.columns if re.match(r"^Q\d+$", c, re.I)]
    if qcols:
        d["_same_ans_count"] = d[qcols].apply(lambda r: r.value_counts(dropna=True).max() if r.notna().any() else 0, axis=1)
        d["_qs_attempted"] = d[qcols].count(axis=1)
        d["_same_ans_ratio"] = np.where(d["_qs_attempted"]>0, d["_same_ans_count"]/d["_qs_attempted"], np.nan)
        out["Same_Ans_Ratio"] = d["_same_ans_ratio"]
    if out:
        res = pd.DataFrame(out)
        res = pd.concat([d[["Region","Instructor Name","School Name","Student Id","Student Name","Date"]], res], axis=1)
        # mark potential outliers
        res["Outlier"] = (res.get("Gain_z",0).abs()>3) | (res.get("Norm_Gain_z",0).abs()>3) | (res.get("Same_Ans_Ratio",0)>0.9)
        return res.sort_values(["Outlier","Gain_z","Norm_Gain_z","Same_Ans_Ratio"], ascending=[False, False, False, False])
    return pd.DataFrame()

# ───────────────────────── Report Export ─────────────────────────
def build_excel_report(reg, ign, ri, ric, dq, anomalies, qp, meta,
                       subj=None, topic=None, funnel_r=None, funnel_i=None,
                       schools=None, states=None, gender_gap=None, outliers=None, students=None) -> bytes:
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
        if subj is not None and not subj.empty:
            subj.to_excel(w, sheet_name="Subject_Summary", index=False)
        if topic is not None and not topic.empty:
            topic.to_excel(w, sheet_name="Topic_Summary", index=False)
        if funnel_r is not None and not funnel_r.empty:
            funnel_r.to_excel(w, sheet_name="Funnel_Region", index=False)
        if funnel_i is not None and not funnel_i.empty:
            funnel_i.to_excel(w, sheet_name="Funnel_Ignator", index=False)
        if schools is not None and not schools.empty:
            schools.to_excel(w, sheet_name="School_Summary", index=False)
        if states is not None and not states.empty:
            states.to_excel(w, sheet_name="State_Summary", index=False)
        if gender_gap is not None and not gender_gap.empty:
            gender_gap.to_excel(w, sheet_name="Gender_Gaps", index=False)
        if outliers is not None and not outliers.empty:
            outliers.to_excel(w, sheet_name="Outliers", index=False)
        if students is not None and not students.empty:
            students.to_excel(w, sheet_name="Student_Summary", index=False)
    bio.seek(0)
    return bio.getvalue()

# ───────────────────────── UI ─────────────────────────
st.markdown(f"### Agastya – DA (All-in-One)  \n*{APP_VER}*  \n*(Excel files: only the **first sheet** is processed)*")

# Performance toggles
c1, c2, c3 = st.columns(3)
with c1:   fast_summary = st.toggle("Fast Summary Mode", value=False, help="Skip heavy tables for huge files.")
with c2:   show_sample = st.toggle("Show a 200-row sample preview", value=False)
with c3:   exclude_dupes = st.toggle("Exclude duplicates (Student Id + Topic + Date)", value=True)

uploaded = st.file_uploader("Upload raw data (.xls / .xlsx / .csv)", type=["xls","xlsx","csv"])
if not uploaded:
    st.info("Upload your file to begin.")
    st.stop()

# Read first sheet
try:
    with st.status("Reading file (first sheet only)…", expanded=False) as s:
        df_raw, meta_in = read_raw_first_sheet(uploaded)
        s.update(label=f"File read ✓  (sheet: {meta_in.get('sheet_name') or '—'})", state="complete")
except Exception as e:
    st.error(f"Read error: {e}"); st.stop()

if show_sample:
    st.write("**Sample (first 200 rows of first sheet):**")
    st.dataframe(df_raw.head(200), use_container_width=True, hide_index=True)

# Process
try:
    with st.status("Processing… (cleaning, scoring, aggregating)", expanded=True) as s:
        df_clean, reg, ign, ri, ric, qp, meta = process_all(df_raw, fast_summary=fast_summary)
        s.update(label="Processing complete ✓", state="complete")
except Exception as e:
    tb = traceback.format_exc()
    st.error("Processing failed.\n\n**Error:**\n```\n" + str(e) + "\n```\n**Traceback:**\n```\n" + tb + "\n```")
    st.stop()

# Data Quality (light)
diag = {}
for c in meta["pre_q_cols"] + meta["post_q_cols"]:
    if c in df_clean.columns: diag[f"{c}_Missing_%"] = [round(100 * df_clean[c].isna().mean(), 2)]
if "Date" in df_clean.columns: diag["Date_Missing_%"] = [round(100 * df_clean["Date"].isna().mean(), 2)]
if "Date_Post" in df_clean.columns: diag["Date_Post_Missing_%"] = [round(100 * df_clean["Date_Post"].isna().mean(), 2)]
if "Anom_PostBeforePre" in df_clean.columns: diag["Anom_PostBeforePre_%"] = [round(100 * df_clean["Anom_PostBeforePre"].mean(), 2)]
if "Anom_SameTimestamp" in df_clean.columns: diag["Anom_SameTimestamp_%"] = [round(100 * df_clean["Anom_SameTimestamp"].mean(), 2)]
dq = pd.DataFrame(diag) if diag else pd.DataFrame()

# Full anomalies (skipped in fast mode)
if fast_summary:
    anomalies = pd.DataFrame()
else:
    anomalies = df_clean[(df_clean.get("Anom_PostBeforePre", False)) | (df_clean.get("Anom_SameTimestamp", False))].copy()

# Optional duplicate exclusion
if exclude_dupes and {"Student Id","Topic Name","Date"}.issubset(df_clean.columns):
    df_clean = df_clean.sort_values(["Student Id","Topic Name","Date"])
    mask_dup = df_clean.duplicated(subset=["Student Id","Topic Name","Date"], keep="first")
    if mask_dup.any():
        df_clean = df_clean.loc[~mask_dup].copy()
        # re-aggregate quickly after dedupe
        reg = (df_clean.groupby("Region", dropna=False, observed=True)
                 .agg(DA_Pre=("DA_Pre_Done","sum"), DA_Post=("DA_Post_Done","sum"),
                      Rows=("Student Name","count"),
                      Unique_Ignators=("Instructor Name", pd.Series.nunique),
                      Unique_Schools=("School Name", pd.Series.nunique),
                      Mean_Pre=("Pre_Score","mean"), Mean_Post=("Post_Score","mean"),
                      Mean_Gain=("Gain","mean"), Median_Gain=("Gain","median"),
                      Mean_Norm_Gain=("Norm_Gain","mean")).reset_index()); add_completion_rate(reg)
        ign = (df_clean.groupby("Instructor Name", dropna=False, observed=True)
                 .agg(DA_Pre=("DA_Pre_Done","sum"), DA_Post=("DA_Post_Done","sum"),
                      Rows=("Student Name","count"),
                      Regions=("Region", pd.Series.nunique),
                      Schools=("School Name", pd.Series.nunique),
                      Mean_Pre=("Pre_Score","mean"), Mean_Post=("Post_Score","mean"),
                      Mean_Gain=("Gain","mean"), Median_Gain=("Gain","median"),
                      Mean_Norm_Gain=("Norm_Gain","mean")).reset_index()); add_completion_rate(ign)
        ri = (df_clean.groupby(["Region","Instructor Name"], dropna=False, observed=True)
                 .agg(DA_Pre=("DA_Pre_Done","sum"), DA_Post=("DA_Post_Done","sum"),
                      Rows=("Student Name","count"),
                      Schools=("School Name", pd.Series.nunique),
                      Mean_Pre=("Pre_Score","mean"), Mean_Post=("Post_Score","mean"),
                      Mean_Gain=("Gain","mean"), Median_Gain=("Gain","median"),
                      Mean_Norm_Gain=("Norm_Gain","mean")).reset_index()); add_completion_rate(ri)
        reg_ign_status = (
            df_clean.groupby(["Region","Instructor Name"], observed=True)
                .agg(Pre_Done=("DA_Pre_Done","any"), Post_Done=("DA_Post_Done","any"))
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

# ───────────────────────── Filters ─────────────────────────
st.sidebar.header("Filters")
regions = sorted(reg["Region"].dropna().astype(str).unique()) if "Region" in reg.columns else []
region_sel = st.sidebar.multiselect("Region", regions, default=regions[: min(5, len(regions))] if regions else None)

ri_for_ign = ri.copy()
if region_sel and "Region" in ri_for_ign.columns:
    ri_for_ign = ri_for_ign[ri_for_ign["Region"].astype(str).isin(region_sel)]
ignators = sorted(ri_for_ign["Instructor Name"].dropna().astype(str).unique()) if "Instructor Name" in ri_for_ign.columns else []
ign_sel = st.sidebar.multiselect("ignator", ignators)

# Extra slicers
extra_dims = [d for d in ["Subject","Topic Name","Class","Gender","Program Type","Donor","School Name","State"] if d in ri.columns or d in df_clean.columns]
extra_filters = {}
for dim in extra_dims:
    source_df = ri if dim in ri.columns else df_clean
    opts = sorted(source_df[dim].dropna().astype(str).unique())
    pick = st.sidebar.multiselect(dim, opts)
    if pick: extra_filters[dim] = set(pick)

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

reg_v = apply_filters(reg.copy()); ign_v = apply_filters(ign.copy())
ri_v  = apply_filters(ri.copy());  ric_v = apply_filters(ric.copy())
anom_v = apply_filters(anomalies.copy())
df_clean_v = apply_filters(df_clean.copy())

# ───────────────────────── KPIs ─────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: st.metric("Total DA-Pre", int(reg_v["DA_Pre"].sum()) if "DA_Pre" in reg_v else 0)
with k2: st.metric("Total DA-Post", int(reg_v["DA_Post"].sum()) if "DA_Post" in reg_v else 0)
with k3:
    denom = max(reg_v["DA_Pre"].sum(), 1) if "DA_Pre" in reg_v else 1
    st.metric("Completion Rate (%)", f"{(reg_v['DA_Post'].sum()/denom)*100:,.2f}" if "DA_Post" in reg_v else "0.00")
with k4: st.metric("Avg Pre Score", f"{reg_v['Mean_Pre'].mean():.2f}" if "Mean_Pre" in reg_v else "—")
with k5: st.metric("Avg Post Score", f"{reg_v['Mean_Post'].mean():.2f}" if "Mean_Post" in reg_v else "—")
with k6: st.metric("Avg Gain", f"{reg_v['Mean_Gain'].mean():.2f}" if "Mean_Gain" in reg_v else "—")

st.divider()

# ───────────────────────── Tabs ─────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Overview", "Region Deep-Dive", "Ignator Leaderboards",
    "Quality & Anomalies", "Per-Question", "Subject/Topic",
    "Funnel & Coverage", "Students & Equity"
])

# ── Overview ──
with tab1:
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Regions by Completion Rate (%)")
        if not reg_v.empty:
            d = reg_v.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False])
            fig = px.bar(d, x="Region", y="Completion_Rate_%",
                         hover_data=[c for c in ["DA_Pre","DA_Post","Mean_Gain","Mean_Norm_Gain","Unique_Ignators","Unique_Schools"] if c in d.columns])
            fig.update_layout(xaxis_title="", yaxis_title="Completion Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No region rows after filters.")
    with cB:
        st.subheader("Top ignators by Completion (Region × ignator)")
        if not ri_v.empty:
            d = ri_v[ri_v["DA_Pre"].fillna(0) >= 10].sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]).head(25)
            if not d.empty:
                fig = px.bar(d, x="Instructor Name", y="Completion_Rate_%", color="Region",
                             hover_data=[c for c in ["DA_Pre","DA_Post","Mean_Gain","Mean_Norm_Gain"] if c in d.columns])
                fig.update_layout(xaxis_title="", yaxis_title="Completion Rate (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ignators meet DA_Pre ≥ 10.")
        else:
            st.info("No Region × ignator rows.")
    st.subheader("Region leaderboard")
    if not reg_v.empty:
        tbl = reg_v.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]).reset_index(drop=True)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button("Download Region leaderboard (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           "region_leaderboard.csv", "text/csv")

# ── Region Deep-Dive ──
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ignator completion status by Region")
        needed = {"Region","Both","Pre only","Post only","None"}
        if not ric_v.empty and needed.issubset(ric_v.columns):
            m = ric_v.melt(id_vars=["Region"], value_vars=["Both","Pre only","Post only","None"],
                           var_name="Completion_Status", value_name="Ignator_Count")
            m["Completion_Status"] = pd.Categorical(m["Completion_Status"],
                                                    categories=["Both","Pre only","Post only","None"], ordered=True)
            fig = px.bar(m, x="Region", y="Ignator_Count", color="Completion_Status", barmode="stack")
            fig.update_layout(xaxis_title="", yaxis_title="Ignators")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region_Ignator_Completed missing required columns.")
    with col2:
        st.subheader("Mean Gain by Region")
        if not reg_v.empty and "Mean_Gain" in reg_v.columns:
            d = reg_v.sort_values("Mean_Gain", ascending=False)
            fig = px.bar(d, x="Region", y="Mean_Gain",
                         hover_data=[c for c in ["Mean_Pre","Mean_Post","Completion_Rate_%"] if c in d.columns])
            fig.update_layout(xaxis_title="", yaxis_title="Mean Gain")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Mean_Gain not available.")

    st.subheader("Region × ignator detail")
    if not ri_v.empty:
        d = ri_v.sort_values(["Region","Completion_Rate_%","DA_Pre"], ascending=[True, False, False]).reset_index(drop=True)
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.download_button("Download Region × ignator detail (CSV)",
                           d.to_csv(index=False).encode("utf-8"),
                           "region_ignator_detail.csv", "text/csv")
    else:
        st.info("No Region × ignator records.")

    st.subheader("Weekly DA-Pre vs DA-Post trend")
    if {"Date","DA_Pre_Done","DA_Post_Done"}.issubset(df_clean_v.columns):
        temp = df_clean_v.copy()
        temp["Week"] = temp["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        tt = temp.groupby("Week", observed=True).agg(DA_Pre=("DA_Pre_Done","sum"),
                                                     DA_Post=("DA_Post_Done","sum")).reset_index()
        if not tt.empty:
            fig = px.line(tt, x="Week", y=["DA_Pre","DA_Post"])
            fig.update_layout(xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dated rows to plot time trend.")

# ── Ignator Leaderboards ──
with tab3:
    st.info("Adjust thresholds to reduce noise.")
    c1,c2,c3 = st.columns(3)
    with c1: min_pre_comp = st.number_input("Min DA_Pre for Completion ranking", min_value=0, value=10, step=1)
    with c2: min_pre_gain = st.number_input("Min DA_Pre for Mean Gain ranking", min_value=0, value=10, step=1)
    with c3: min_pre_norm = st.number_input("Min DA_Pre for Normalized Gain (ignator overall)", min_value=0, value=20, step=1)

    st.subheader("Top Completion (Region × ignator)")
    d = ri_v[ri_v["DA_Pre"].fillna(0) >= min_pre_comp].copy()
    if not d.empty:
        d = d.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]).head(25)
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.download_button("Download table (CSV)", d.to_csv(index=False).encode("utf-8"),
                           "top_completion_region_ignator.csv", "text/csv")
    else:
        st.info("No ignators meet threshold.")

    st.subheader("Top Mean Gain (Region × ignator)")
    d2 = ri_v[ri_v["DA_Pre"].fillna(0) >= min_pre_gain].copy()
    if not d2.empty and "Mean_Gain" in d2.columns:
        d2 = d2.sort_values(["Mean_Gain","DA_Pre"], ascending=[False, False]).head(25)
        st.dataframe(d2, use_container_width=True, hide_index=True)
        st.download_button("Download table (CSV)", d2.to_csv(index=False).encode("utf-8"),
                           "top_mean_gain_region_ignator.csv", "text/csv")
    else:
        st.info("No ignators meet threshold or Mean_Gain missing.")

    st.subheader("Top Normalized Gain (ignator overall)")
    d3 = ign_v[ign_v["DA_Pre"].fillna(0) >= min_pre_norm].copy()
    if not d3.empty and "Mean_Norm_Gain" in d3.columns:
        d3 = d3.sort_values(["Mean_Norm_Gain","DA_Pre"], ascending=[False, False]).head(25)
        st.dataframe(d3, use_container_width=True, hide_index=True)
        st.download_button("Download table (CSV)", d3.to_csv(index=False).encode("utf-8"),
                           "top_norm_gain_ignator.csv", "text/csv")
    else:
        st.info("No ignators meet threshold or Mean_Norm_Gain missing.")

# ── Quality & Anomalies ──
with tab4:
    st.subheader("Data Quality")
    if not dq.empty:
        st.dataframe(dq, use_container_width=True, hide_index=True)
        st.download_button("Download Data Quality (CSV)", dq.to_csv(index=False).encode("utf-8"),
                           "data_quality.csv", "text/csv")
    else:
        st.info("No Data_Quality available.")
    st.subheader("Anomalies")
    if not anom_v.empty:
        st.dataframe(anom_v, use_container_width=True, hide_index=True)
        st.download_button("Download Anomalies (CSV)", anom_v.to_csv(index=False).encode("utf-8"),
                           "anomalies.csv", "text/csv")
    else:
        st.info("No anomalies detected or (Fast Summary Mode enabled).")

    st.subheader("Outlier scan")
    out_tbl = outlier_scan(df_clean_v)
    if not out_tbl.empty:
        st.dataframe(out_tbl.head(500), use_container_width=True, hide_index=True)
        st.download_button("Download Outliers (CSV)", out_tbl.to_csv(index=False).encode("utf-8"),
                           "outliers.csv", "text/csv")
    else:
        st.info("No strong outliers detected with simple rules.")

# ── Per-Question ──
with tab5:
    if fast_summary:
        st.info("Per-question stats skipped in Fast Summary Mode.")
    elif qp is None or qp.empty:
        st.info("No per-question stats available (Q/Answer columns not detected).")
    else:
        st.dataframe(qp, use_container_width=True, hide_index=True)
        st.download_button("Download Per-Question Stats (CSV)", qp.to_csv(index=False).encode("utf-8"),
                           "per_question_stats.csv", "text/csv")

# ── Subject / Topic ──
with tab6:
    subj, topic = build_subject_topic(df_clean_v)
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Subject summary")
        if not subj.empty:
            d = subj.sort_values("Completion_Rate_%", ascending=False)
            fig = px.bar(d, x="Subject", y="Completion_Rate_%",
                         hover_data=["Rows","DA_Pre","DA_Post","Mean_Gain","Mean_PrePct","Mean_PostPct"])
            fig.update_layout(xaxis_title="", yaxis_title="Completion Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(d, use_container_width=True, hide_index=True)
            st.download_button("Download Subject Summary (CSV)", d.to_csv(index=False).encode("utf-8"),
                               "subject_summary.csv", "text/csv")
        else:
            st.info("No Subject column found.")
    with cB:
        st.subheader("Topic summary")
        if not topic.empty:
            d = topic.sort_values(["Subject","Completion_Rate_%"], ascending=[True, False])
            st.dataframe(d, use_container_width=True, hide_index=True)
            st.download_button("Download Topic Summary (CSV)", d.to_csv(index=False).encode("utf-8"),
                               "topic_summary.csv", "text/csv")
        else:
            st.info("No Topic Name column found.")

# ── Funnel & Coverage ──
with tab7:
    st.subheader("Funnel: Assigned → Pre Attempted → Post Completed")
    fr = funnel_table(df_clean_v, ["Region"]) if "Region" in df_clean_v.columns else pd.DataFrame()
    fi = funnel_table(df_clean_v, ["Instructor Name"]) if "Instructor Name" in df_clean_v.columns else pd.DataFrame()
    fs = funnel_table(df_clean_v, ["Subject"]) if "Subject" in df_clean_v.columns else pd.DataFrame()

    if not fr.empty:
        fig = px.bar(fr.sort_values("Post_Complete_%", ascending=False), x="Region", y="Post_Complete_%",
                     hover_data=["Assigned","Pre_Attempted","Post_Completed","Pre_Attempt_%"])
        fig.update_layout(xaxis_title="", yaxis_title="Post Complete (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Region Funnel (CSV)", fr.to_csv(index=False).encode("utf-8"),
                           "funnel_region.csv", "text/csv")
    else:
        st.info("Region funnel not available.")

    st.markdown("### School & State coverage")
    schools = states = pd.DataFrame()
    if "School Name" in df_clean_v.columns:
        schools = (df_clean_v.groupby("School Name", observed=True)
                    .agg(Rows=("Student Name","count"),
                         DA_Pre=("DA_Pre_Done","sum"),
                         DA_Post=("DA_Post_Done","sum"),
                         Mean_Gain=("Gain","mean")).reset_index())
        add_completion_rate(schools)
        st.dataframe(schools.sort_values(["Completion_Rate_%","DA_Pre"], ascending=[False, False]), use_container_width=True, hide_index=True)
    if "State" in df_clean_v.columns:
        states = (df_clean_v.groupby("State", observed=True)
                   .agg(Rows=("Student Name","count"),
                        DA_Pre=("DA_Pre_Done","sum"),
                        DA_Post=("DA_Post_Done","sum"),
                        Mean_Gain=("Gain","mean")).reset_index())
        add_completion_rate(states)
        st.download_button("Download School Summary (CSV)", schools.to_csv(index=False).encode("utf-8") if not schools.empty else b"",
                           "school_summary.csv", "text/csv", disabled=schools.empty)
        st.download_button("Download State Summary (CSV)", states.to_csv(index=False).encode("utf-8") if not states.empty else b"",
                           "state_summary.csv", "text/csv", disabled=states.empty)

# ── Students & Equity ──
with tab8:
    st.subheader("Student-wise insights")
    needed_cols = ["Student Id","Student Name","Gender","Class","Region","Instructor Name",
                   "DA_Pre_Done","DA_Post_Done","Pre_Score","Post_Score","Gain","Norm_Gain","Topic Name"]
    for c in needed_cols:
        if c not in df_clean_v.columns: df_clean_v[c] = np.nan
    stu = (df_clean_v.groupby(["Student Id","Student Name","Gender","Class","Region","Instructor Name"], dropna=False, observed=True)
              .agg(Rows=("Topic Name","count"),
                   DA_Pre=("DA_Pre_Done","sum"),
                   DA_Post=("DA_Post_Done","sum"),
                   Mean_Pre=("Pre_Score","mean"),
                   Mean_Post=("Post_Score","mean"),
                   Mean_Gain=("Gain","mean"),
                   Median_Gain=("Gain","median"),
                   Mean_Norm_Gain=("Norm_Gain","mean")).reset_index())
    stu["Completed_Both"] = (stu["DA_Pre"].fillna(0)>0) & (stu["DA_Post"].fillna(0)>0)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Students seen", len(stu))
    with c2: st.metric("Students completed (both)", int(stu["Completed_Both"].sum()))
    with c3: st.metric("Avg Gain (students)", f"{stu['Mean_Gain'].mean():.2f}" if not stu.empty else "—")
    with c4: st.metric("Avg Norm Gain (students)", f"{stu['Mean_Norm_Gain'].mean():.2f}" if not stu.empty else "—")

    st.markdown("### Class-wise completion and gains")
    if "Class" in stu.columns and not stu.empty:
        cls = (stu.groupby("Class", observed=True)
                 .agg(Students=("Student Id","nunique"),
                      Completed=("Completed_Both","sum"),
                      Avg_Gain=("Mean_Gain","mean"),
                      Avg_Pre=("Mean_Pre","mean"),
                      Avg_Post=("Mean_Post","mean")).reset_index())
        cls["Completion_Rate_%"] = np.where(cls["Students"]>0, cls["Completed"]/cls["Students"]*100, np.nan)
        cA,cB = st.columns(2)
        with cA:
            fig = px.bar(cls.sort_values("Completion_Rate_%", ascending=False), x="Class", y="Completion_Rate_%",
                         hover_data=["Students","Completed","Avg_Gain","Avg_Pre","Avg_Post"])
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            fig2 = px.bar(cls.sort_values("Avg_Gain", ascending=False), x="Class", y="Avg_Gain",
                          hover_data=["Students","Completion_Rate_%","Avg_Pre","Avg_Post"])
            st.plotly_chart(fig2, use_container_width=True)
        st.download_button("Download Class-wise table (CSV)", cls.to_csv(index=False).encode("utf-8"),
                           "class_wise_students.csv", "text/csv")

    st.markdown("### Gender-wise completion and gains")
    gen_gaps = equity_gaps(df_clean_v, "Gender")
    if not gen_gaps.empty:
        cC,cD = st.columns(2)
        with cC:
            fig3 = px.bar(gen_gaps.sort_values("Completion_Rate_%", ascending=False), x="Gender", y="Completion_Rate_%",
                          hover_data=["Students","Completed","Avg_Gain"])
            st.plotly_chart(fig3, use_container_width=True)
        with cD:
            if "Gender" in stu.columns:
                fig4 = px.box(stu.dropna(subset=["Gender","Mean_Gain"]), x="Gender", y="Mean_Gain", points="suspectedoutliers")
                st.plotly_chart(fig4, use_container_width=True)
        st.download_button("Download Gender-wise (CSV)", gen_gaps.to_csv(index=False).encode("utf-8"),
                           "gender_wise_gaps.csv", "text/csv")
    else:
        st.info("No Gender data for equity view.")

    st.markdown("### Top student improvements")
    topn = st.number_input("Top N students by Mean Gain", min_value=5, max_value=1000, value=50, step=5)
    top_students = (stu.sort_values(["Mean_Gain","DA_Pre"], ascending=[False, False]).head(int(topn)))
    st.dataframe(top_students, use_container_width=True, hide_index=True)
    st.download_button("Download Top students (CSV)", top_students.to_csv(index=False).encode("utf-8"),
                       "top_students_by_gain.csv", "text/csv")

# ───────────────────────── Download Excel ─────────────────────────
st.divider()
st.subheader("Download generated Excel report")
report_bytes = build_excel_report(
    reg_v, ign_v, ri_v, ric_v, dq, anom_v, qp if not fast_summary else pd.DataFrame(), meta,
    subj=build_subject_topic(df_clean_v)[0], topic=build_subject_topic(df_clean_v)[1],
    funnel_r=funnel_table(df_clean_v, ["Region"]) if "Region" in df_clean_v.columns else pd.DataFrame(),
    funnel_i=funnel_table(df_clean_v, ["Instructor Name"]) if "Instructor Name" in df_clean_v.columns else pd.DataFrame(),
    schools=(df_clean_v.groupby("School Name", observed=True)
             .agg(Rows=("Student Name","count"),DA_Pre=("DA_Pre_Done","sum"),DA_Post=("DA_Post_Done","sum"),Mean_Gain=("Gain","mean")).reset_index()
             if "School Name" in df_clean_v.columns else pd.DataFrame()),
    states=(df_clean_v.groupby("State", observed=True)
            .agg(Rows=("Student Name","count"),DA_Pre=("DA_Pre_Done","sum"),DA_Post=("DA_Post_Done","sum"),Mean_Gain=("Gain","mean")).reset_index()
            if "State" in df_clean_v.columns else pd.DataFrame()),
    gender_gap=equity_gaps(df_clean_v, "Gender"),
    outliers=outlier_scan(df_clean_v),
    students=(df_clean_v.groupby(["Student Id","Student Name","Gender","Class","Region","Instructor Name"], observed=True)
              .agg(Rows=("Topic Name","count"),DA_Pre=("DA_Pre_Done","sum"),DA_Post=("DA_Post_Done","sum"),
                   Mean_Pre=("Pre_Score","mean"),Mean_Post=("Post_Score","mean"),Mean_Gain=("Gain","mean"),
                   Median_Gain=("Gain","median"),Mean_Norm_Gain=("Norm_Gain","mean")).reset_index()
             if "Student Id" in df_clean_v.columns else pd.DataFrame())
)
st.download_button(
    "Download Excel (multi-sheet)",
    data=report_bytes,
    file_name=f"Agastya_DA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(f"Built for Agastya • {APP_VER}")
