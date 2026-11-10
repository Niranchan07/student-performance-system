import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend so plots work inside a web server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, Response, jsonify, request, send_file
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from waitress import serve

NUMERIC_FEATURES = ["study_hours_weekly", "attendance_rate", "past_failures", "absences"]
CATEGORICAL_FEATURES = ["internet_access", "parental_education", "extracurricular"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.png")


# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================
def generate_student_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    study_hours_weekly = rng.normal(loc=15, scale=6, size=n_samples).clip(1, 40)
    attendance_rate = rng.beta(a=7, b=2, size=n_samples) * 100
    past_failures = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.70, 0.18, 0.08, 0.04])
    internet_access = rng.choice(["Yes", "No"], size=n_samples, p=[0.85, 0.15])
    parental_education = rng.choice(
        ["None", "HighSchool", "Bachelors", "Masters"],
        size=n_samples,
        p=[0.1, 0.45, 0.35, 0.1],
    )
    extracurricular = rng.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])
    absences = rng.poisson(lam=4, size=n_samples).clip(0, 30)

    latent_score = (
        0.35 * (study_hours_weekly / 40)
        + 0.45 * (attendance_rate / 100)
        - 0.30 * (past_failures / 3)
        - 0.20 * (absences / 30)
        + 0.10 * (internet_access == "Yes")
        + 0.08 * (np.isin(parental_education, ["Bachelors", "Masters"]))
        + rng.normal(0, 0.12, size=n_samples)
    )

    target_pass = (latent_score >= np.median(latent_score)).astype(int)

    return pd.DataFrame({
        "study_hours_weekly": study_hours_weekly,
        "attendance_rate": attendance_rate,
        "past_failures": past_failures,
        "absences": absences,
        "internet_access": internet_access,
        "parental_education": parental_education,
        "extracurricular": extracurricular,
        "passed": target_pass,
    })


# ==========================================
# 2. MODEL PIPELINE BUILDER
# ==========================================
def get_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# ==========================================
# 3. TRAINING & EVALUATION BENCHMARK
# ==========================================
def train_and_benchmark(df: pd.DataFrame, target_col: str = "passed"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    candidate_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, random_state=42),
    }

    model_param_grids = {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [4, 6, 8],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 4],
        },
    }

    best_score = -1.0
    best_model_name = None
    best_pipeline = None
    benchmark = []

    print("--- BENCHMARK RESULTS ---")
    for name, model in candidate_models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        print(f"[{name}] F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
        benchmark.append({"model": name, "f1": round(float(f1), 4), "roc_auc": round(float(roc), 4)})

        if roc > best_score:
            best_score = roc
            best_model_name = name
            best_pipeline = pipe

    print(f"\nOptimal Model Selected: {best_model_name} (ROC-AUC: {best_score:.4f})")

    grid_search = GridSearchCV(
        best_pipeline,
        param_grid=model_param_grids[best_model_name],
        cv=4,
        scoring="roc_auc",
        n_jobs=1,
    )
    grid_search.fit(X_train, y_train)

    final_model = grid_search.best_estimator_
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    print("\n--- FINAL TEST EVALUATION ---")
    print(classification_report(y_test, y_test_pred, target_names=["At-Risk", "Passing"]))

    return {
        "model": final_model,
        "best_model_name": best_model_name,
        "benchmark": benchmark,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
    }


# ==========================================
# 4. EXPLAINABILITY & VISUALIZATION DASHBOARD
# ==========================================
def extract_feature_importance(pipeline, numeric_features, categorical_features) -> pd.DataFrame:
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    all_features = numeric_features + encoded_cat_features

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.zeros(len(all_features))

    return pd.DataFrame({"Feature": all_features, "Importance": importances}).sort_values(
        by="Importance", ascending=True
    )


def plot_analytics_dashboard(df, importance_df, y_test, y_test_pred, y_test_proba, save_path):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Student Performance Predictive Analytics Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)

    scatter_df = df.copy()
    scatter_df["Status"] = scatter_df["passed"].map({1: "Passing", 0: "At-Risk"})
    sns.scatterplot(
        ax=axes[0, 0],
        data=scatter_df.sample(min(500, len(df)), random_state=42),
        x="study_hours_weekly", y="attendance_rate", hue="Status",
        palette={"Passing": "#2ecc71", "At-Risk": "#e74c3c"},
        alpha=0.7, edgecolor="w", s=60,
    )
    axes[0, 0].set_title("1. Attendance vs. Study Hours Correlation", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Weekly Study Hours")
    axes[0, 0].set_ylabel("Attendance Rate (%)")

    axes[0, 1].barh(importance_df["Feature"], importance_df["Importance"], color="#3498db")
    axes[0, 1].set_title("2. Key Predictors of Academic Success", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Importance / Absolute Coefficient Weight")

    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0], cbar=False,
                xticklabels=["At-Risk", "Passing"], yticklabels=["At-Risk", "Passing"],
                annot_kws={"size": 14, "fontweight": "bold"})
    axes[1, 0].set_title("3. Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Predicted Label")
    axes[1, 0].set_ylabel("Actual Label")

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_score = roc_auc_score(y_test, y_test_proba)
    axes[1, 1].plot(fpr, tpr, color="#e67e22", lw=2.5, label=f"ROC Curve (AUC = {roc_score:.3f})")
    axes[1, 1].plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--")
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_title("4. Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("False Positive Rate")
    axes[1, 1].set_ylabel("True Positive Rate")
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Dashboard saved as '{save_path}'.")


# ==========================================
# 5. INFERENCE SYSTEM
# ==========================================
class StudentInferenceEngine:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict_risk(self, student_records: pd.DataFrame) -> pd.DataFrame:
        predictions = self.model.predict(student_records)
        probabilities = self.model.predict_proba(student_records)[:, 1]

        results = student_records.copy()
        results["Predicted_Pass"] = predictions
        results["Pass_Probability"] = np.round(probabilities, 4)
        results["Intervention_Required"] = np.where(results["Pass_Probability"] < 0.50, "Yes", "No")
        return results


# ==========================================
# 6. WEBSITE (HTML + CSS + JS)
# ==========================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Performance Predictor</title>
<style>
  :root {
    --bg:#f4f6fb; --card:#fff; --text:#1f2937; --muted:#6b7280;
    --primary:#4f46e5; --good:#16a34a; --bad:#dc2626; --border:#e5e7eb;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { background:linear-gradient(135deg,#4f46e5,#7c3aed); color:#fff; padding:36px 20px; text-align:center; }
  header h1 { margin:0 0 6px; font-size:28px; }
  header p { margin:0; opacity:.9; }
  main { max-width:1100px; margin:-24px auto 40px; padding:0 16px; display:grid; gap:20px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:800px){ .grid2{ grid-template-columns:1fr; } }
  .card { background:var(--card); border-radius:14px; padding:22px; box-shadow:0 4px 14px rgba(0,0,0,.06); }
  .card h2 { margin:0 0 16px; font-size:18px; }
  label { display:block; font-size:13px; font-weight:600; margin:12px 0 4px; }
  input, select { width:100%; padding:10px; border:1px solid var(--border); border-radius:8px; font-size:14px; }
  input[type=range] { padding:0; }
  .val { color:var(--primary); font-weight:700; }
  button { margin-top:18px; width:100%; padding:12px; background:var(--primary); color:#fff; border:0; border-radius:8px; font-size:15px; font-weight:600; cursor:pointer; }
  button:hover { background:#4338ca; }
  #result { text-align:center; display:none; }
  .gauge { width:150px; height:150px; border-radius:50%; margin:6px auto 12px; display:grid; place-items:center; }
  .gauge span { background:#fff; width:112px; height:112px; border-radius:50%; display:grid; place-items:center; font-size:28px; font-weight:800; }
  .badge { display:inline-block; padding:6px 14px; border-radius:20px; font-weight:700; color:#fff; }
  .badge.good { background:var(--good); } .badge.bad { background:var(--bad); }
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:14px; }
  .stat { background:var(--bg); border-radius:10px; padding:12px; text-align:center; }
  .stat b { display:block; font-size:20px; color:var(--primary); }
  .stat small { color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { padding:8px; text-align:left; border-bottom:1px solid var(--border); }
  .bar-row { display:flex; align-items:center; gap:8px; margin:6px 0; font-size:13px; }
  .bar-row .name { width:170px; flex-shrink:0; }
  .bar { height:12px; background:var(--primary); border-radius:6px; }
  img.dash { width:100%; border-radius:10px; }
  .err { color:var(--bad); margin-top:10px; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>🎓 Student Performance Predictor</h1>
  <p>Machine-learning powered early-warning system for at-risk students</p>
</header>

<main>
  <div class="grid2">
    <section class="card">
      <h2>Enter Student Details</h2>
      <form id="form">
        <label>Weekly Study Hours: <span class="val" id="v_study">15</span></label>
        <input type="range" id="study_hours_weekly" min="1" max="40" step="0.5" value="15">

        <label>Attendance Rate (%): <span class="val" id="v_att">80</span></label>
        <input type="range" id="attendance_rate" min="0" max="100" step="0.5" value="80">

        <label>Past Failures</label>
        <select id="past_failures"><option>0</option><option>1</option><option>2</option><option>3</option></select>

        <label>Absences (0–30)</label>
        <input type="number" id="absences" min="0" max="30" value="4">

        <label>Internet Access</label>
        <select id="internet_access"><option>Yes</option><option>No</option></select>

        <label>Parental Education</label>
        <select id="parental_education">
          <option>None</option><option selected>HighSchool</option><option>Bachelors</option><option>Masters</option>
        </select>

        <label>Extracurricular Activities</label>
        <select id="extracurricular"><option>Yes</option><option>No</option></select>

        <button type="submit">Predict Outcome</button>
        <div class="err" id="err"></div>
      </form>
    </section>

    <section class="card" id="resultCard">
      <h2>Prediction</h2>
      <div id="placeholder" style="color:var(--muted)">Fill in the form and click <b>Predict Outcome</b> to see the result.</div>
      <div id="result">
        <div class="gauge" id="gauge"><span id="prob">0%</span></div>
        <p style="margin:0 0 10px;color:var(--muted)">Probability of passing</p>
        <span class="badge" id="badge"></span>
        <p id="advice" style="margin-top:14px"></p>
      </div>
    </section>
  </div>

  <section class="card">
    <h2>Model Performance</h2>
    <div class="stats">
      <div class="stat"><b id="m_best">–</b><small>Selected model</small></div>
      <div class="stat"><b id="m_auc">–</b><small>Test ROC-AUC</small></div>
      <div class="stat"><b id="m_acc">–</b><small>Test accuracy</small></div>
    </div>
    <div class="grid2">
      <div>
        <h3 style="margin-top:0">Benchmark</h3>
        <table id="bench"><thead><tr><th>Model</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody></tbody></table>
      </div>
      <div>
        <h3 style="margin-top:0">Top Predictive Factors</h3>
        <div id="imp"></div>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Analytics Dashboard</h2>
    <img class="dash" src="/dashboard.png" alt="Analytics dashboard">
  </section>
</main>

<script>
const $ = id => document.getElementById(id);

$("study_hours_weekly").oninput = e => $("v_study").textContent = e.target.value;
$("attendance_rate").oninput = e => $("v_att").textContent = e.target.value;

fetch("/api/metrics").then(r => r.json()).then(m => {
  $("m_best").textContent = m.best_model;
  $("m_auc").textContent = m.roc_auc.toFixed(3);
  $("m_acc").textContent = (m.accuracy * 100).toFixed(1) + "%";
  $("bench").querySelector("tbody").innerHTML = m.benchmark
    .map(b => `<tr><td>${b.model}</td><td>${b.f1.toFixed(3)}</td><td>${b.roc_auc.toFixed(3)}</td></tr>`).join("");
  const max = Math.max(...m.importance.map(i => i.importance)) || 1;
  $("imp").innerHTML = m.importance.slice(0, 8).map(i =>
    `<div class="bar-row"><span class="name">${i.feature}</span>
     <div class="bar" style="width:${(i.importance / max) * 100}%"></div>
     <span>${i.importance.toFixed(3)}</span></div>`).join("");
});

$("form").addEventListener("submit", async e => {
  e.preventDefault();
  $("err").textContent = "";
  const payload = {};
  ["study_hours_weekly","attendance_rate","past_failures","absences",
   "internet_access","parental_education","extracurricular"].forEach(k => payload[k] = $(k).value);

  try {
    const res = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");

    const pct = Math.round(data.pass_probability * 100);
    const risk = data.intervention_required === "Yes";
    const color = risk ? "var(--bad)" : "var(--good)";
    $("placeholder").style.display = "none";
    $("result").style.display = "block";
    $("prob").textContent = pct + "%";
    $("gauge").style.background = `conic-gradient(${color} ${pct * 3.6}deg, #e5e7eb 0deg)`;
    $("badge").textContent = risk ? "At-Risk – Intervention Required" : "On Track";
    $("badge").className = "badge " + (risk ? "bad" : "good");
    $("advice").textContent = risk
      ? "Recommend academic counselling, attendance monitoring and extra study support."
      : "Student is predicted to pass. Keep up the current habits.";
  } catch (err) {
    $("err").textContent = err.message;
  }
});
</script>
</body>
</html>
"""


# ==========================================
# 7. FLASK APP
# ==========================================
app = Flask(__name__)


def build_artifacts():
    """Train once; cache the model, metrics and dashboard image."""
    data = generate_student_data(n_samples=2000, random_state=42)
    res = train_and_benchmark(data)

    importance_df = extract_feature_importance(res["model"], NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    joblib.dump(res["model"], MODEL_PATH)
    plot_analytics_dashboard(
        data, importance_df, res["y_test"], res["y_test_pred"], res["y_test_proba"],
        save_path=DASHBOARD_PATH,
    )

    report = classification_report(
        res["y_test"], res["y_test_pred"], target_names=["At-Risk", "Passing"], output_dict=True
    )
    metrics = {
        "best_model": res["best_model_name"],
        "roc_auc": round(float(roc_auc_score(res["y_test"], res["y_test_proba"])), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "benchmark": res["benchmark"],
        "importance": [
            {"feature": r.Feature, "importance": round(float(r.Importance), 4)}
            for r in importance_df.sort_values("Importance", ascending=False).itertuples()
        ],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    return metrics


def load_or_build():
    if all(os.path.exists(p) for p in (MODEL_PATH, METRICS_PATH, DASHBOARD_PATH)):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return build_artifacts()


METRICS = load_or_build()
ENGINE = StudentInferenceEngine(MODEL_PATH)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/dashboard.png")
def dashboard():
    return send_file(DASHBOARD_PATH, mimetype="image/png")


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        p = request.get_json(force=True)
        record = pd.DataFrame([{
            "study_hours_weekly": float(p["study_hours_weekly"]),
            "attendance_rate": float(p["attendance_rate"]),
            "past_failures": int(p["past_failures"]),
            "absences": int(p["absences"]),
            "internet_access": p["internet_access"],
            "parental_education": p["parental_education"],
            "extracurricular": p["extracurricular"],
        }])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    out = ENGINE.predict_risk(record).iloc[0]
    return jsonify({
        "predicted_pass": int(out["Predicted_Pass"]),
        "pass_probability": float(out["Pass_Probability"]),
        "intervention_required": out["Intervention_Required"],
    })


if __name__ == "__main__":
    print("Serving with Waitress on http://127.0.0.1:5000 ...")
    serve(app, host="127.0.0.1", port=5000)