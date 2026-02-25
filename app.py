import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# =============================
# Core Simulation Engine
# =============================

def simulate_gbm(S0, mu, sigma, T, steps, n_paths):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    W = np.random.randn(n_paths, steps + 1)
    W[:, 0] = 0
    W = np.cumsum(W * np.sqrt(dt), axis=1)
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return S

def simulate_merton_jump(S0, mu, sigma, T, steps, n_paths, jump_lambda, jump_mu, jump_sigma):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    W = np.random.randn(n_paths, steps + 1)
    W[:, 0] = 0
    W = np.cumsum(W * np.sqrt(dt), axis=1)
    N = np.random.poisson(jump_lambda * dt, (n_paths, steps + 1))
    J = np.random.normal(jump_mu, jump_sigma, (n_paths, steps + 1))
    jump_sum = np.cumsum(N * J, axis=1)
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W + jump_sum)
    return S

def simulate_heston(S0, mu, sigma, T, steps, n_paths, kappa, theta, xi, rho):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    S = np.zeros((n_paths, steps + 1))
    v = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    v[:, 0] = sigma ** 2
    for i in range(steps):
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2) * np.sqrt(dt)
        v[:, i + 1] = np.abs(v[:, i] + kappa * (theta - v[:, i]) * dt + xi * np.sqrt(v[:, i]) * W2)
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * v[:, i]) * dt + np.sqrt(v[:, i]) * W1)
    return S

# =============================
# Visualization Functions
# =============================

def plot_3d_price_cloud(S, T, steps):
    n_paths = S.shape[0]
    t = np.linspace(0, T, steps + 1)
    fig = go.Figure()
    for i in range(n_paths):
        fig.add_trace(go.Scatter3d(
            x=t,
            y=np.full_like(t, i),
            z=S[i],
            mode='lines',
            line=dict(color='cyan', width=2),
            opacity=0.7
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Path Index',
            zaxis_title='Price',
            bgcolor='black',
            xaxis=dict(showgrid=True, zeroline=False, color='white', gridcolor='gray'),
            yaxis=dict(showgrid=True, zeroline=False, color='white', gridcolor='gray'),
            zaxis=dict(showgrid=True, zeroline=False, color='white', gridcolor='gray'),
            aspectmode='cube',
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
    )
    fig.update_traces(line=dict(color='cyan'))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    return fig

def plot_3d_density_surface(S, T, steps, bins=100):
    n_paths = S.shape[0]
    t = np.linspace(0, T, steps + 1)
    price_bins = np.linspace(np.min(S), np.max(S), bins)
    density = np.zeros((bins, steps + 1))
    for i in range(steps + 1):
        hist, _ = np.histogram(S[:, i], bins=price_bins, density=True)
        density[:, i] = np.append(hist, 0)[:bins]

    # Animation frames for surface morphing
    frames = []
    for i in range(steps + 1):
        frames.append(go.Frame(
            data=[go.Surface(
                x=price_bins,
                y=t,
                z=density.T,
                surfacecolor=density.T,
                colorscale='Inferno',
                opacity=0.98,
                showscale=False,
            )],
            name=f"frame{i}"
        ))

    fig = go.Figure(
        data=[go.Surface(
            x=price_bins,
            y=t,
            z=density.T,
            surfacecolor=density.T,
            colorscale='Inferno',
            opacity=0.98,
            showscale=False,
        )],
        frames=frames
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='Price',
            yaxis_title='Time',
            zaxis_title='Density',
            bgcolor='black',
            xaxis=dict(showgrid=True, zeroline=False, color='white', gridcolor='gray'),
            yaxis=dict(showgrid=True, zeroline=False, color='white', gridcolor='gray'),
            zaxis=dict(showgrid=True, zeroline=False, color='white', gridcolor='gray'),
            aspectmode='cube',
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="bottom"
            )
        ]
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    return fig

# =============================
# Streamlit UI
# =============================

st.set_page_config(
    page_title="Monte Carlo Options Universe — Quant Lab Edition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧪"
)
st.markdown("""
<style>
body { background-color: #111 !important; }
[data-testid="stAppViewContainer"] { background-color: #111 !important; }
[data-testid="stSidebar"] { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Monte Carlo Options Universe — Quant Lab Edition")
st.markdown("""
#### Institutional Quant Simulation Lab
""")

with st.sidebar:
    st.header("Simulation Parameters")
    model = st.selectbox("Model", ["Geometric Brownian Motion", "Merton Jump Diffusion", "Heston Stochastic Volatility"])
    S0 = st.slider("Initial Price", 1, 1000, 100)
    mu = st.slider("Drift (μ)", -0.1, 0.2, 0.05, step=0.01)
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
    T = st.slider("Time to Maturity (years)", 0.1, 5.0, 1.0, step=0.1)
    steps = st.slider("Steps", 50, 1000, 250, step=10)
    n_paths = st.slider("Number of Paths", 100, 5000, 3000, step=100)
    show_jump = st.checkbox("Enable Jump Diffusion Parameters")
    jump_lambda = st.slider("Jump Intensity (λ)", 0.0, 2.0, 0.1, step=0.01) if show_jump else 0.0
    jump_mu = st.slider("Jump Mean (μJ)", -0.2, 0.2, 0.0, step=0.01) if show_jump else 0.0
    jump_sigma = st.slider("Jump Std (σJ)", 0.01, 0.5, 0.1, step=0.01) if show_jump else 0.0
    show_heston = st.checkbox("Enable Heston Parameters")
    kappa = st.slider("Heston Kappa (κ)", 0.01, 5.0, 1.0, step=0.01) if show_heston else 1.0
    theta = st.slider("Heston Theta (θ)", 0.01, 1.0, 0.2, step=0.01) if show_heston else 0.2
    xi = st.slider("Heston Xi (ξ)", 0.01, 1.0, 0.3, step=0.01) if show_heston else 0.3
    rho = st.slider("Heston Rho (ρ)", -1.0, 1.0, -0.5, step=0.01) if show_heston else -0.5
    vis_mode = st.radio("Visualization Mode", ["3D Price Path Cloud", "3D Probability Density Surface"])

@st.cache_data(show_spinner=False, max_entries=10)
def run_simulation(model, S0, mu, sigma, T, steps, n_paths, jump_lambda, jump_mu, jump_sigma, kappa, theta, xi, rho):
    if model == "Geometric Brownian Motion":
        return simulate_gbm(S0, mu, sigma, T, steps, n_paths)
    elif model == "Merton Jump Diffusion":
        return simulate_merton_jump(S0, mu, sigma, T, steps, n_paths, jump_lambda, jump_mu, jump_sigma)
    elif model == "Heston Stochastic Volatility":
        return simulate_heston(S0, mu, sigma, T, steps, n_paths, kappa, theta, xi, rho)
    else:
        return simulate_gbm(S0, mu, sigma, T, steps, n_paths)

with st.spinner("Running Monte Carlo simulation..."):
    S = run_simulation(model, S0, mu, sigma, T, steps, n_paths, jump_lambda, jump_mu, jump_sigma, kappa, theta, xi, rho)

st.subheader(f"{model} — {vis_mode}")

if vis_mode == "3D Price Path Cloud":
    fig = plot_3d_price_cloud(S, T, steps)
    st.plotly_chart(fig, width='stretch')
elif vis_mode == "3D Probability Density Surface":
    fig = plot_3d_density_surface(S, T, steps, bins=120)
    st.plotly_chart(fig, width='stretch')

st.markdown("""
<small>Built for institutional quant research. Powered by NumPy, Plotly, Streamlit.</small>
""")
