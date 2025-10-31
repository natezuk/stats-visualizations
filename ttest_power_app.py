"""
Interactive T-Test Power Analysis Application

To run this application:
1. Install dependencies: pip install -r requirements.txt
2. Run the app: streamlit run ttest_power_app.py

The app will open in your browser where you can interactively adjust:
- Effect size (Cohen's d)
- Alpha level (Type I error rate)
- Power (1 - Type II error rate)
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="T-Test Power Analysis", layout="wide")

st.title("Interactive T-Test Power Analysis")
st.markdown("""
This application allows you to explore how different parameters affect power analysis for t-tests.
Adjust the effect size (Cohen's d), alpha level, and desired power to see how they impact the required sample size.
""")

# Sidebar for input controls
st.sidebar.header("Parameters")

# User inputs
effect_size = st.sidebar.slider(
    "Effect Size (Cohen's d)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Standardized difference between groups"
)

alpha = st.sidebar.slider(
    "Alpha (Type I Error Rate)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Significance level / probability of false positive"
)

power = st.sidebar.slider(
    "Power (1 - Type II Error Rate)",
    min_value=0.5,
    max_value=0.99,
    value=0.8,
    step=0.01,
    help="Probability of detecting a true effect"
)

# Sample size range
max_n = st.sidebar.slider(
    "Maximum Sample Size",
    min_value=50,
    max_value=500,
    value=100,
    step=10,
    help="Maximum sample size to consider"
)

# Calculations
n_samples = np.array(range(2, max_n + 1))

# Calculate the threshold t-statistic
t_thres = stats.t.isf(alpha / 2, n_samples - 1)

# Calculate resulting t-statistic with the desired effect size and power
t_effect = stats.nct.isf(power, n_samples - 1, nc=effect_size * np.sqrt(n_samples))

# Find minimum sample size where effect crosses threshold
crossing_points = np.where(t_effect > t_thres)[0]
if len(crossing_points) > 0:
    min_idx = crossing_points[0]
    min_n = n_samples[min_idx]
    min_t = t_effect[min_idx]
    
    # Calculate p-value and power for minimum n
    pval = 1 - stats.t.cdf(min_t, min_n - 1)
    pwr_val = 1 - stats.nct.cdf(min_t, min_n - 1, nc=effect_size * np.sqrt(min_n))
else:
    min_n = None
    min_t = None
    pval = None
    pwr_val = None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Power Analysis Plot")
    
    # Plot both curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_samples, t_thres, 'r', linewidth=2, label=f'Threshold (α={alpha})')
    ax.plot(n_samples, t_effect, 'b', linewidth=2, label=f'Effect (d={effect_size})')
    
    if min_n is not None:
        ax.axvline(min_n, color='green', linestyle='--', alpha=0.5, label=f'Min n={min_n}')
        ax.axhline(min_t, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('t-statistic', fontsize=12)
    ax.set_title('T-Statistic vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Distribution plot
    if min_n is not None:
        st.subheader("Null vs Alternative Distribution")
        n = min_n
        
        # Null distribution: t with n-1 df, noncentrality=0
        x = np.linspace(-5, 8, 500)
        null_dist = stats.t(df=n - 1)
        alt_dist = stats.nct(df=n - 1, nc=effect_size * np.sqrt(n))
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x, null_dist.pdf(x), 'b-', linewidth=2, label='Null (d=0)')
        ax2.plot(x, alt_dist.pdf(x), 'r-', linewidth=2, label=f'Alternative (d={effect_size})')
        ax2.axvline(min_t, color='green', linestyle='--', linewidth=2, 
                   label=f'Threshold t={min_t:.3f}')
        
        # Shade rejection region
        x_reject_left = x[x < -abs(min_t)]
        x_reject_right = x[x > abs(min_t)]
        if len(x_reject_left) > 0:
            ax2.fill_between(x_reject_left, 0, null_dist.pdf(x_reject_left), 
                           alpha=0.3, color='red', label='Rejection Region')
        if len(x_reject_right) > 0:
            ax2.fill_between(x_reject_right, 0, null_dist.pdf(x_reject_right), 
                           alpha=0.3, color='red')
        
        ax2.set_xlabel('t-statistic', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title(f'Null vs. Alternative Distribution (n={n})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)

with col2:
    st.subheader("Results")
    
    if min_n is not None:
        st.metric("Minimum Sample Size", min_n)
        st.metric("t-statistic", f"{min_t:.3f}")
        st.metric("p-value", f"{pval:.4f}")
        st.metric("Power", f"{pwr_val:.4f}")
        
        st.markdown("---")
        st.markdown("### Interpretation")
        st.info(f"""
        With an effect size of **{effect_size}**, alpha of **{alpha}**, 
        and desired power of **{power}**, you need at least **{min_n}** 
        samples per group to detect the effect.
        """)
    else:
        st.warning("⚠️ No crossing point found within the specified range. Try increasing the maximum sample size or adjusting parameters.")
    
    # Power analysis explanation
    st.markdown("---")
    with st.expander("About Power Analysis"):
        st.markdown("""
        **Power analysis** helps determine the sample size needed to detect an effect.
        
        - **Effect Size (d)**: Standardized difference between groups (Cohen's d)
        - **Alpha (α)**: Probability of false positive (Type I error)
        - **Power (1-β)**: Probability of detecting a true effect
        
        The minimum sample size is where the effect t-statistic exceeds the threshold.
        """)

