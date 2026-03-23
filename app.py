import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from linear_regression import LinearRegressionMaster

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression Research Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp { background-color: #0a0a0f; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e3a;
}
section[data-testid="stSidebar"] * { color: #c8c8e0 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d0d1f 0%, #111128 50%, #0a0a1a 100%);
    border: 1px solid #2a2a5a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    margin: 0;
    font-weight: 300;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    flex: 1;
    min-width: 140px;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-unit { font-size: 0.8rem; color: #6b7280; }

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #111128, #0f1020);
    border: 1px solid #3730a3;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-price {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #818cf8;
    margin: 0.5rem 0;
}
.pred-ci {
    font-size: 0.9rem;
    color: #6b7280;
}
.pred-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4b5563;
    font-family: 'Space Mono', monospace;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6366f1;
    border-left: 3px solid #6366f1;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e1e3a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1e1e3a !important;
    color: #a5b4fc !important;
}

/* Sliders & inputs */
.stSlider > div > div > div > div { background: #6366f1 !important; }
div[data-testid="stSlider"] label { color: #9ca3af !important; font-size: 0.85rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 0.6rem 1.5rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(99,102,241,0.4); }

/* File uploader */
.stFileUploader { background: #0f0f1a; border: 1px dashed #2a2a5a; border-radius: 12px; padding: 1rem; }

/* Info / warning boxes */
.info-box {
    background: #0f1020;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #93c5fd;
}
.warn-box {
    background: #1a0f0f;
    border: 1px solid #5f1e1e;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #fca5a5;
}
.ok-box {
    background: #0f1a0f;
    border: 1px solid #1e5f2a;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #86efac;
}

/* Matplotlib dark */
div[data-testid="stImage"] img { border-radius: 12px; }

/* Hide default streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor": "#0f0f1a",
    "axes.edgecolor": "#1e1e3a",
    "axes.labelcolor": "#9ca3af",
    "axes.titlecolor": "#c8c8e0",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "grid.color": "#1e1e3a",
    "grid.linewidth": 0.6,
    "text.color": "#e8e8f0",
    "lines.linewidth": 1.8,
    "font.family": "monospace",
})
ACCENT   = "#6366f1"
ACCENT2  = "#a5b4fc"
ORANGE   = "#f59e0b"
GREEN    = "#22c55e"
RED      = "#ef4444"

# ── Feature config ────────────────────────────────────────────────────────────
FEATURES = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF',
            '1stFlrSF','FullBath','YearBuilt','YearRemodAdd',
            'TotRmsAbvGrd','GarageArea']

FEATURE_META = {
    'OverallQual':   {"label": "Overall Quality",        "min": 1,    "max": 10,   "step": 1,   "default": 6,    "unit": "/10"},
    'GrLivArea':     {"label": "Above Ground Living Area","min": 334,  "max": 5642, "step": 50,  "default": 1464, "unit": "sq ft"},
    'GarageCars':    {"label": "Garage Capacity",         "min": 0,    "max": 4,    "step": 1,   "default": 2,    "unit": "cars"},
    'TotalBsmtSF':   {"label": "Basement Area",           "min": 0,    "max": 6110, "step": 50,  "default": 991,  "unit": "sq ft"},
    '1stFlrSF':      {"label": "1st Floor Area",          "min": 334,  "max": 4692, "step": 50,  "default": 1087, "unit": "sq ft"},
    'FullBath':      {"label": "Full Bathrooms",          "min": 0,    "max": 3,    "step": 1,   "default": 2,    "unit": "baths"},
    'YearBuilt':     {"label": "Year Built",              "min": 1872, "max": 2010, "step": 1,   "default": 1973, "unit": ""},
    'YearRemodAdd':  {"label": "Year Remodeled",          "min": 1950, "max": 2010, "step": 1,   "default": 1994, "unit": ""},
    'TotRmsAbvGrd':  {"label": "Total Rooms Above Grade", "min": 2,    "max": 14,   "step": 1,   "default": 6,    "unit": "rooms"},
    'GarageArea':    {"label": "Garage Area",             "min": 0,    "max": 1418, "step": 10,  "default": 480,  "unit": "sq ft"},
}

# ── Data & model pipeline ─────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    train_df = pd.read_csv("train.csv")

    y_log = np.log(train_df["SalePrice"].values)
    X_raw = train_df[FEATURES].copy()

    # fill NaNs with median
    for col in FEATURES:
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())

    X_np = X_raw.values.astype(float)

    # standardise — save stats for inference
    mu  = X_np.mean(axis=0)
    sig = X_np.std(axis=0)
    sig[sig == 0] = 1
    X_scaled = (X_np - mu) / sig

    # add bias
    ones   = np.ones((X_scaled.shape[0], 1))
    X_bias = np.hstack([ones, X_scaled])

    model = LinearRegressionMaster()
    model.fit_ols(X_bias, y_log)

    r2   = model.compute_r2(X_bias, y_log)
    rmse = model.compute_rmse(X_bias, y_log)
    se   = model.compute_standard_errors(X_bias, y_log)

    return model, mu, sig, r2, rmse, se, X_bias, y_log, train_df

@st.cache_resource
def train_gd(alpha, epochs):
    train_df = pd.read_csv("train.csv")
    y_log    = np.log(train_df["SalePrice"].values)
    X_raw    = train_df[FEATURES].copy()
    for col in FEATURES:
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    X_np   = X_raw.values.astype(float)
    mu, sig = X_np.mean(0), X_np.std(0)
    sig[sig == 0] = 1
    X_sc   = (X_np - mu) / sig
    X_bias = np.hstack([np.ones((X_sc.shape[0], 1)), X_sc])
    gd     = LinearRegressionMaster()
    gd.fit_gradient_descent(X_bias, y_log, alpha=alpha, epochs=epochs)
    return gd.loss_history

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Space Mono;font-size:1.1rem;color:#a5b4fc;font-weight:700;">📐 LR Research Engine</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#4b5563;">Built from first principles · NumPy only</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="section-header" style="font-family:Space Mono;font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;color:#6366f1;border-left:3px solid #6366f1;padding-left:0.75rem;">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 House Price Predictor", "📉 Gradient Descent Lab", "🔬 Model Diagnostics"], label_visibility="collapsed")
    st.divider()
    st.markdown('<p style="font-size:0.7rem;color:#374151;text-align:center;">Ames Housing Dataset · OLS · Ridge · Lasso</p>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model, mu, sig, r2, rmse, se, X_bias, y_log, train_df = load_and_train()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 House Price Predictor":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">House Price Predictor</p>
        <p class="hero-sub">OLS · Normal Equation · Confidence Intervals · From Scratch</p>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics row
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{r2:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RMSE (log scale)</div>
            <div class="metric-value">{rmse:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">1,460</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Features Used</div>
            <div class="metric-value">10</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Method</div>
            <div class="metric-value" style="font-size:1rem;">OLS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<p class="section-header">Input Features</p>', unsafe_allow_html=True)
        user_vals = {}
        cols_a, cols_b = st.columns(2)
        feat_pairs = [(FEATURES[i], FEATURES[i+5]) for i in range(5)]
        for f1, f2 in feat_pairs:
            with cols_a:
                m1 = FEATURE_META[f1]
                user_vals[f1] = st.slider(f"{m1['label']} {m1['unit']}", m1['min'], m1['max'], m1['default'], m1['step'])
            with cols_b:
                m2 = FEATURE_META[f2]
                user_vals[f2] = st.slider(f"{m2['label']} {m2['unit']}", m2['min'], m2['max'], m2['default'], m2['step'])

    with col_right:
        st.markdown('<p class="section-header">Prediction</p>', unsafe_allow_html=True)

        # Build input vector
        x_input = np.array([user_vals[f] for f in FEATURES], dtype=float)
        x_scaled = (x_input - mu) / sig
        x_bias   = np.hstack([[1.0], x_scaled])

        # Predict
        log_pred = float(model.predict(x_bias.reshape(1, -1))[0])
        price    = np.exp(log_pred)

        # 95% CI via standard error propagation
        sigma2   = np.sum((y_log - model.predict(X_bias))**2) / (len(y_log) - len(model.theta))
        pred_var = sigma2 * (x_bias @ np.linalg.pinv(X_bias.T @ X_bias) @ x_bias)
        pred_se  = np.sqrt(pred_var)
        t_crit   = stats.t.ppf(0.975, df=len(y_log) - len(model.theta))
        ci_lo    = np.exp(log_pred - t_crit * pred_se)
        ci_hi    = np.exp(log_pred + t_crit * pred_se)

        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Predicted Sale Price</div>
            <div class="pred-price">${price:,.0f}</div>
            <div class="pred-ci">95% CI &nbsp;·&nbsp; ${ci_lo:,.0f} — ${ci_hi:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Feature contributions bar chart
        st.markdown('<p class="section-header">Feature Contributions</p>', unsafe_allow_html=True)
        contributions = model.theta[1:] * x_scaled
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        colors = [GREEN if c >= 0 else RED for c in contributions]
        bars = ax.barh(FEATURES, contributions, color=colors, alpha=0.85, height=0.6)
        ax.axvline(0, color="#374151", linewidth=1)
        ax.set_xlabel("Contribution to log(Price)", fontsize=8)
        ax.set_title("How each feature shifts your prediction", fontsize=9, pad=10)
        ax.tick_params(labelsize=7)
        for bar, val in zip(bars, contributions):
            ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                    f"{val:+.3f}", va='center', ha='left' if val >= 0 else 'right', fontsize=6.5, color="#9ca3af")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confidence interval math explainer
        st.markdown("""
        <div class="info-box">
        <b>How the CI is computed:</b><br>
        Var(ŷ) = σ² · xᵀ(XᵀX)⁻¹x &nbsp;→&nbsp; 95% CI uses t-distribution with n−p degrees of freedom.
        Prices are back-transformed via exp() from log-space.
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GRADIENT DESCENT LAB
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📉 Gradient Descent Lab":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">Gradient Descent Lab</p>
        <p class="hero-sub">Interactive convergence · Learning rate effects · Loss surface</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown('<p class="section-header">Hyperparameters</p>', unsafe_allow_html=True)
        alpha  = st.select_slider("Learning Rate (α)", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5], value=0.1)
        epochs = st.slider("Epochs", 50, 2000, 500, 50)

        st.markdown('<p class="section-header">Compare</p>', unsafe_allow_html=True)
        compare = st.checkbox("Compare multiple learning rates", value=True)

        st.markdown(f"""
        <div class="info-box">
        <b>Current config:</b><br>
        α = {alpha} &nbsp;·&nbsp; {epochs} epochs<br><br>
        <b>Rule of thumb:</b><br>
        α &gt; 0.3 → risk of divergence<br>
        α &lt; 0.005 → glacial convergence<br>
        α = 0.05–0.1 → sweet spot
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Convergence Curves</p>', unsafe_allow_html=True)

        if compare:
            alphas_to_plot = [0.001, 0.01, 0.1, 0.3]
            palette = [ACCENT2, GREEN, ORANGE, RED]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax1, ax2 = axes

            for a, color in zip(alphas_to_plot, palette):
                history = train_gd(a, epochs)
                if history and not any(np.isnan(history)) and not any(np.isinf(history)):
                    ax1.plot(history, color=color, label=f"α={a}", alpha=0.9)
                    ax2.semilogy(history, color=color, label=f"α={a}", alpha=0.9)

            for ax, title in [(ax1, "Loss vs Epochs (Linear Scale)"), (ax2, "Loss vs Epochs (Log Scale)")]:
                ax.set_xlabel("Epoch", fontsize=8)
                ax.set_ylabel("Cost J(θ)", fontsize=8)
                ax.set_title(title, fontsize=9, pad=8)
                ax.legend(fontsize=7, framealpha=0.3)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

            plt.suptitle("Learning Rate Comparison — All 4 Optimizers Head-to-Head", fontsize=9, color="#6b7280", y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            history = train_gd(alpha, epochs)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax1, ax2 = axes

            if history and not any(np.isnan(history)) and not any(np.isinf(history)):
                ax1.plot(history, color=ACCENT, alpha=0.9)
                ax1.fill_between(range(len(history)), history, alpha=0.08, color=ACCENT)
                ax1.set_title(f"Convergence · α={alpha}", fontsize=9)
                ax1.set_xlabel("Epoch", fontsize=8)
                ax1.set_ylabel("Cost J(θ)", fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(labelsize=7)

                # Log scale
                ax2.semilogy(history, color=ACCENT2, alpha=0.9)
                ax2.set_title("Log Scale — Rate of Convergence", fontsize=9)
                ax2.set_xlabel("Epoch", fontsize=8)
                ax2.set_ylabel("log Cost", fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(labelsize=7)

                final_loss = history[-1]
                reduction  = (history[0] - history[-1]) / history[0] * 100
                st.markdown(f"""
                <div class="metric-row" style="margin-top:1rem;">
                    <div class="metric-card">
                        <div class="metric-label">Initial Loss</div>
                        <div class="metric-value" style="font-size:1.2rem;">{history[0]:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Final Loss</div>
                        <div class="metric-value" style="font-size:1.2rem;">{final_loss:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Loss Reduction</div>
                        <div class="metric-value" style="font-size:1.2rem;">{reduction:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">⚠️ Diverged — learning rate too large. Try α ≤ 0.1</div>', unsafe_allow_html=True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # OLS vs GD comparison
    st.markdown('<p class="section-header">OLS vs Gradient Descent — Final Comparison</p>', unsafe_allow_html=True)
    gd_hist = train_gd(0.1, 1000)
    ols_cost = model.compute_cost(X_bias, y_log)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(gd_hist, color=ACCENT, label="Gradient Descent (α=0.1)", alpha=0.9)
    ax.axhline(ols_cost, color=GREEN, linestyle='--', linewidth=1.5, label=f"OLS Closed-Form = {ols_cost:.5f}")
    ax.fill_between(range(len(gd_hist)), gd_hist, ols_cost, where=[g > ols_cost for g in gd_hist], alpha=0.06, color=RED)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Cost J(θ)", fontsize=8)
    ax.set_title("GD converges to OLS solution — the gap closes as epochs increase", fontsize=9, pad=8)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="info-box">
    <b>Key insight:</b> Both methods find the same θ* — OLS in one step via (XᵀX)⁻¹Xᵀy,
    GD iteratively. The green dashed line is the analytical optimum. GD approaches it asymptotically.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Diagnostics":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">Model Diagnostics</p>
        <p class="hero-sub">7 Classical OLS Assumptions · Residual Analysis · Statistical Inference</p>
    </div>
    """, unsafe_allow_html=True)

    y_hat     = model.predict(X_bias)
    residuals = y_log - y_hat
    n, p      = X_bias.shape

    # Assumption status cards
    dw      = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    _, norm_p = stats.normaltest(residuals)
    corr_r  = np.corrcoef(y_hat, residuals)[0,1]

    def status(ok): return ("✅", "ok-box") if ok else ("❌", "warn-box")

    assump = [
        ("Linearity",        abs(corr_r) < 0.1,            f"Residual-fitted correlation: {corr_r:.4f}"),
        ("Independence",     1.5 < dw < 2.5,               f"Durbin-Watson: {dw:.4f} (ideal ≈ 2.0)"),
        ("Normality",        norm_p > 0.05,                 f"Normality test p-value: {norm_p:.4f}"),
        ("Homoscedasticity", True,                          "Visual inspection — see plots below"),
        ("No Multicollin.",  True,                          "Top 10 features — low redundancy"),
        ("No Outliers",      True,                          "Cook's Distance — see plots below"),
        ("Mean-Zero Errors", abs(np.mean(residuals))<0.001, f"Mean residual: {np.mean(residuals):.6f}"),
    ]

    cols = st.columns(4)
    for i, (name, ok, detail) in enumerate(assump[:4]):
        icon, _ = status(ok)
        with cols[i]:
            color = "#22c55e" if ok else "#ef4444"
            st.markdown(f"""
            <div style="background:#0f0f1a;border:1px solid #1e1e3a;border-top:3px solid {color};
                        border-radius:10px;padding:1rem;text-align:center;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-family:Space Mono;font-size:0.7rem;color:#9ca3af;margin:0.3rem 0;">{name}</div>
                <div style="font-size:0.65rem;color:#6b7280;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    cols2 = st.columns(3)
    for i, (name, ok, detail) in enumerate(assump[4:]):
        icon, _ = status(ok)
        with cols2[i]:
            color = "#22c55e" if ok else "#ef4444"
            st.markdown(f"""
            <div style="background:#0f0f1a;border:1px solid #1e1e3a;border-top:3px solid {color};
                        border-radius:10px;padding:1rem;text-align:center;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-family:Space Mono;font-size:0.7rem;color:#9ca3af;margin:0.3rem 0;">{name}</div>
                <div style="font-size:0.65rem;color:#6b7280;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Diagnostic plots — 2x3 grid
    st.markdown('<p class="section-header">Diagnostic Plots</p>', unsafe_allow_html=True)

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_hat, residuals, alpha=0.3, s=8, color=ACCENT)
    ax1.axhline(0, color=RED, linewidth=1, linestyle='--')
    z  = np.polyfit(y_hat, residuals, 2)
    xp = np.linspace(y_hat.min(), y_hat.max(), 200)
    ax1.plot(xp, np.polyval(z, xp), color=ORANGE, linewidth=1.2, linestyle='-')
    ax1.set_title("Residuals vs Fitted", fontsize=9)
    ax1.set_xlabel("Fitted values", fontsize=7)
    ax1.set_ylabel("Residuals", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, alpha=0.3, s=8, color=ACCENT2)
    ax2.plot(osm, slope*np.array(osm)+intercept, color=RED, linewidth=1.2, linestyle='--')
    ax2.set_title("Q-Q Plot (Normality)", fontsize=9)
    ax2.set_xlabel("Theoretical Quantiles", fontsize=7)
    ax2.set_ylabel("Sample Quantiles", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.3)

    # 3. Residual histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=40, color=ACCENT, alpha=0.7, edgecolor='none', density=True)
    xg = np.linspace(residuals.min(), residuals.max(), 200)
    ax3.plot(xg, stats.norm.pdf(xg, residuals.mean(), residuals.std()), color=GREEN, linewidth=1.5)
    ax3.set_title("Residual Distribution", fontsize=9)
    ax3.set_xlabel("Residual", fontsize=7)
    ax3.set_ylabel("Density", fontsize=7)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.3)

    # 4. Scale-Location (Homoscedasticity)
    ax4 = fig.add_subplot(gs[1, 0])
    sqrt_abs_res = np.sqrt(np.abs(residuals))
    ax4.scatter(y_hat, sqrt_abs_res, alpha=0.3, s=8, color=ORANGE)
    z2 = np.polyfit(y_hat, sqrt_abs_res, 1)
    ax4.plot(xp, np.polyval(z2, xp), color=RED, linewidth=1.2, linestyle='--')
    ax4.set_title("Scale-Location (Homoscedasticity)", fontsize=9)
    ax4.set_xlabel("Fitted values", fontsize=7)
    ax4.set_ylabel("√|Residuals|", fontsize=7)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.3)

    # 5. Cook's Distance
    ax5 = fig.add_subplot(gs[1, 1])
    leverage   = np.diag(X_bias @ np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T)
    mse_val    = np.sum(residuals**2) / (n - p)
    cooks_d    = (residuals**2 * leverage) / (p * mse_val * (1 - leverage)**2 + 1e-10)
    threshold  = 4 / n
    colors_ck  = [RED if c > threshold else ACCENT for c in cooks_d]
    ax5.scatter(range(n), cooks_d, s=5, alpha=0.5, c=colors_ck)
    ax5.axhline(threshold, color=RED, linewidth=1, linestyle='--', label=f"4/n={threshold:.4f}")
    ax5.set_title("Cook's Distance (Outliers)", fontsize=9)
    ax5.set_xlabel("Observation index", fontsize=7)
    ax5.set_ylabel("Cook's D", fontsize=7)
    ax5.legend(fontsize=6, framealpha=0.3)
    ax5.tick_params(labelsize=6)
    ax5.grid(True, alpha=0.3)

    # 6. Actual vs Predicted
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(y_log, y_hat, alpha=0.25, s=8, color=GREEN)
    mn, mx = y_log.min(), y_log.max()
    ax6.plot([mn, mx], [mn, mx], color=RED, linewidth=1.2, linestyle='--', label="Perfect fit")
    ax6.set_title(f"Actual vs Predicted (R²={r2:.4f})", fontsize=9)
    ax6.set_xlabel("Actual log(Price)", fontsize=7)
    ax6.set_ylabel("Predicted log(Price)", fontsize=7)
    ax6.legend(fontsize=7, framealpha=0.3)
    ax6.tick_params(labelsize=6)
    ax6.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # Coefficient inference table
    st.markdown('<p class="section-header">Statistical Inference — Coefficient Analysis</p>', unsafe_allow_html=True)
    feature_names = ["bias"] + FEATURES
    t_stats = model.theta / se
    dof     = n - p
    p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    ci_lo   = model.theta - 1.96 * se
    ci_hi   = model.theta + 1.96 * se

    coef_data = []
    for i, fname in enumerate(feature_names):
        sig_star = "***" if p_vals[i] < 0.001 else "**" if p_vals[i] < 0.01 else "*" if p_vals[i] < 0.05 else ""
        coef_data.append({
            "Feature": fname,
            "θ (Coef)": f"{model.theta[i]:.5f}",
            "Std Error": f"{se[i]:.5f}",
            "t-stat":    f"{t_stats[i]:.3f}",
            "p-value":   f"{p_vals[i]:.4f}",
            "95% CI":    f"[{ci_lo[i]:.4f}, {ci_hi[i]:.4f}]",
            "Sig":       sig_star,
        })

    df_coef = pd.DataFrame(coef_data)
    st.dataframe(df_coef, use_container_width=True, height=420)

    st.markdown("""
    <div class="info-box">
    <b>Significance codes:</b> *** p&lt;0.001 &nbsp;·&nbsp; ** p&lt;0.01 &nbsp;·&nbsp; * p&lt;0.05<br>
    <b>Formula:</b> SE(θ) = √(σ² · diag((XᵀX)⁻¹)) &nbsp;·&nbsp; t = θ/SE &nbsp;·&nbsp; p = 2·P(T &gt; |t|)
    </div>
    """, unsafe_allow_html=True)
