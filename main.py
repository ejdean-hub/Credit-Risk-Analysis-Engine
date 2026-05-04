import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import gaussian_kde


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')


#-------------------
# Load data
#-------------------
df = pd.read_csv("credit_risk_dataset.csv")

num_cols = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
]
cat_cols = [
    "person_home_ownership", "loan_intent",
    "loan_grade", "cb_person_default_on_file",
]
target = "loan_status"

X = df[num_cols + cat_cols]
y = df[target]

#--------------------
# Train / test split
#--------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#-------------------
# Preprocessing
#-------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), # Fills missing numeric values with the median of each column
    ("scaler", StandardScaler())                   # Mean 0 and standard deviation 1
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])



# -------------------------------
# Logistic regression PD model
# -------------------------------
model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"))
])
model.fit(X_train, y_train)

pd_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pd_test)
print("ROC-AUC:", auc)

# ---------------------------
# PD -> Score mapping
# ---------------------------
points_double_odds = 50
odds0 = 20
score0 = 600

B = points_double_odds / np.log(2)
A = score0 - B * np.log(odds0)

def pd_to_score(pd_vec):
    pd_vec = np.clip(pd_vec, 1e-6, 1 - 1e-6)
    odds = (1 - pd_vec) / pd_vec
    return A + B * np.log(odds) # score

scores_test = pd_to_score(pd_test)

# -----------------------
# Score band analysis
# -----------------------
test = X_test.copy()
test["loan_status"] = y_test.values
test["pd"] = pd_test
test["score"] = scores_test

bins   = [300, 500, 600, 700, 800, 900]
labels = ["300-500", "500-600", "600-700", "700-800", "800-900"]
test["band"] = pd.cut(test["score"], bins=bins, labels=labels, right=False)

summary = (
    test.groupby("band")
        .agg(n=("loan_status","size"),
             bad_rate=("loan_status","mean"),
             avg_score=("score","mean"))
)
print(summary)

# -------------------------
# Scorecard coefficients
# -------------------------
log_reg = model.named_steps["clf"]
prep    = model.named_steps["prep"]

num_pipe   = prep.named_transformers_["num"]
scaler     = num_pipe.named_steps["scaler"]
num_means  = scaler.mean_
num_scales = scaler.scale_

cat_pipe    = prep.named_transformers_["cat"]
ohe         = cat_pipe.named_steps["ohe"]
cat_features = ohe.get_feature_names_out(cat_cols)

beta   = log_reg.coef_[0]
beta0  = log_reg.intercept_[0]

n_num            = len(num_cols)
beta_num_scaled  = beta[:n_num]
beta_cat         = beta[n_num:]

beta_num_orig  = beta_num_scaled / num_scales
beta0_orig     = beta0 - np.sum(beta_num_scaled * num_means / num_scales)

points_num_per_unit = -B * beta_num_orig
points_cat_if_true  = -B * beta_cat
intercept_points    = -B * beta0_orig

# Parse categorical rows
cat_rows = []
for f, b, p in zip(cat_features, beta_cat, points_cat_if_true):
    for col in cat_cols:
        prefix = col + "_"
        if f.startswith(prefix):
            cat_val = f[len(prefix):]
            cat_rows.append({"feature": col, "type": "categorical",
                             "category": cat_val, "coef_orig": b,
                             "points_if_true": p})
            break

scorecard = pd.DataFrame([
    *[{"feature": f, "type": "numeric", "category": None,
       "coef_orig": b, "points_per_unit": p}
      for f, b, p in zip(num_cols, beta_num_orig, points_num_per_unit)],
    *cat_rows
])

print("\n=== SCORECARD (first 40 rows) ===")
print(scorecard.head(40))
print("\nIntercept points:", intercept_points)

# --------------------------
# Manual scoring function
# --------------------------
def compute_score_from_raw(row: pd.Series) -> float:
    score = A + intercept_points
    # numeric: skip NaN values (imputer handles them in the pipeline)
    for f, p in zip(num_cols, points_num_per_unit):
        val = row[f]
        if pd.notna(val):
            score += p * val
    # categorical: unseen or NaN values contribute 0 pts (matches OHE handle_unknown='ignore')
    for f, p in zip(cat_features, points_cat_if_true):
        for col in cat_cols:
            prefix = col + "_"
            if f.startswith(prefix):
                cat_val = f[len(prefix):]
                row_val = row[col]
                if pd.notna(row_val) and str(row_val) == cat_val:
                    score += p
                break
    return score

print("\n--- SCORE CHECK (model vs manual) ---")
for idx in test.index[:5]:
    row          = X_test.loc[idx]
    score_model  = test.loc[idx, "score"]
    score_manual = compute_score_from_raw(row)
    print(f"idx={idx}  model={score_model:.4f}  manual={score_manual:.4f}")







# VISUALISATIONS
#------------------------
# Colour palette
#------------------------
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
MUTED     = "#8b949e"
WHITE     = "#e6edf3"

def _apply_dark_style(fig, axes):
    """
    Sets up the background for the dashboard.
    """
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")


# ----------------------------------------------------------
# Figure 1 — ROC + Calibration
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model diagnostics", fontsize=14, fontweight="bold", color=WHITE, y=0.99)
_apply_dark_style(fig, axes)

# ROC
fpr, tpr, _ = roc_curve(y_test, pd_test)
ax = axes[0]
ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"Model (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--", label="Random")
ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve", color=WHITE)
ax.legend(loc="lower right", fontsize=9, facecolor=PANEL_BG, edgecolor=MUTED, labelcolor=MUTED)




# Calibration
fraction_pos, mean_pred = calibration_curve(y_test, pd_test, n_bins=10)
ax = axes[1]
ax.plot(mean_pred, fraction_pos, marker="o", color=ACCENT, lw=2, label="Model")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--", label="Perfect")
ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.12, color=YELLOW)
ax.set_xlabel("Mean predicted PD")
ax.set_ylabel("Observed bad rate")
ax.set_title("Calibration curve", color=WHITE)
ax.legend(loc="upper left", fontsize=9, facecolor=PANEL_BG, edgecolor=MUTED, labelcolor=MUTED)



plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# Figure 2 — Score distribution
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
_apply_dark_style(fig, [ax])

for status, label, color, ls in [
    (0, "Good (repaid)", ACCENT, "-"),
    (1, "Bad (default)", RED,    "--")
]:
    subset = test[test["loan_status"] == status]["score"].values
    kde = gaussian_kde(subset)
    xs = np.linspace(subset.min(), subset.max(), 300)
    ax.plot(xs, kde(xs), color=color, lw=2, linestyle=ls, label=label)
    ax.fill_between(xs, kde(xs), alpha=0.20, color=color)
    ax.axvline(np.median(subset), color=color, linestyle=":", linewidth=1.5,
               label=f"{label} median ({np.median(subset):.0f})")

ax.set_xlabel("Credit score")
ax.set_ylabel("Density")
ax.set_title("Score distribution by loan outcome", fontweight="bold", color=WHITE)
ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=MUTED, labelcolor=MUTED)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# Figure 3 — Score band summary
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Score band analysis", fontsize=14, fontweight="bold", color=WHITE)
_apply_dark_style(fig, axes)

band_colors = plt.cm.RdYlGn(np.linspace(0, 1, len(summary)))[::-1]

# Bad rate
ax = axes[0]
bars = ax.bar(summary.index.astype(str), summary["bad_rate"] * 100,
              color=band_colors, edgecolor=WHITE, linewidth=0.8)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color=WHITE)
ax.set_xlabel("Score band")
ax.set_ylabel("Bad rate (%)")
ax.set_title("Bad rate by score band", color=WHITE)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

# Population
ax = axes[1]
bars2 = ax.bar(summary.index.astype(str), summary["n"],
               color=band_colors, edgecolor=WHITE, linewidth=0.8)
ax.bar_label(bars2, fmt="%d", padding=3, fontsize=9, color=WHITE)
ax.set_xlabel("Score band")
ax.set_ylabel("Number of loans")
ax.set_title("Population by score band", color=WHITE)

plt.tight_layout()
plt.show()



# ----------------------------------------------------------
# Figure 4 — PD -> Score calibration
# ----------------------------------------------------------

score_range = np.linspace(300, 900, 500)
pd_range = 1 / (1 + np.exp((score_range - A) / B))

fig, ax = plt.subplots(figsize=(9, 5))
_apply_dark_style(fig, [ax])

ax.plot(score_range, pd_range * 100, color=ACCENT, lw=2.5)
ax.fill_between(score_range, pd_range * 100, alpha=0.10, color=ACCENT)

for s in [500, 600, 700, 800]:
    lo  = (s - A) / B
    pd_ = 1 / (1 + np.exp(lo)) * 100
    ax.annotate(f"{s}: {pd_:.1f}% PD",
                xy=(s, pd_), xytext=(s + 15, pd_ + 4),
                fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

ax.set_xlabel("Credit score")
ax.set_ylabel("Probability of default (%)")
ax.set_title("PD → Score calibration (log-odds mapping)", fontweight="bold", color=WHITE)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xlim(300, 900)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.show()


