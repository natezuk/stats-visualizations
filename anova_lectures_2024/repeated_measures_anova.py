import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app
def main():
    st.title("Repeated Measures ANOVA Simulation")

    # User inputs
    num_participants = st.slider("Number of participants", 10, 100, 30)
    num_conditions = st.slider("Number of conditions", 2, 5, 3)
    effect_size = st.slider("Effect size", 0.0, 1.0, 0.3, 0.1)

    # Generate simulated data
    data = generate_data(num_participants, num_conditions, effect_size)

    # Perform repeated measures ANOVA
    f_value, p_value = repeated_measures_anova(data)

    # Display results
    st.subheader("ANOVA Results")
    st.write(f"F-value: {f_value:.4f}")
    st.write(f"p-value: {p_value:.4f}")

    # Plot the data
    fig = plot_data(data)
    st.pyplot(fig)

def generate_data(num_participants, num_conditions, effect_size):
    base_mean = 50
    base_std = 10
    
    data = np.random.normal(base_mean, base_std, (num_participants, num_conditions))
    
    # Add effect
    for i in range(num_conditions):
        data[:, i] += i * effect_size * base_std
    
    return data

def repeated_measures_anova(data):
    num_participants, num_conditions = data.shape
    grand_mean = np.mean(data)
    
    # Calculate SSt (Total Sum of Squares)
    ss_total = np.sum((data - grand_mean) ** 2)
    
    # Calculate SSb (Between-subjects Sum of Squares)
    subject_means = np.mean(data, axis=1)
    ss_between = num_conditions * np.sum((subject_means - grand_mean) ** 2)
    
    # Calculate SSw (Within-subjects Sum of Squares)
    ss_within = ss_total - ss_between
    
    # Calculate SSc (Conditions Sum of Squares)
    condition_means = np.mean(data, axis=0)
    ss_conditions = num_participants * np.sum((condition_means - grand_mean) ** 2)
    
    # Calculate SSr (Residual Sum of Squares)
    ss_residual = ss_within - ss_conditions
    
    # Degrees of freedom
    df_conditions = num_conditions - 1
    df_residual = (num_participants - 1) * (num_conditions - 1)
    
    # Mean Squares
    ms_conditions = ss_conditions / df_conditions
    ms_residual = ss_residual / df_residual
    
    # F-value
    f_value = ms_conditions / ms_residual
    
    # p-value
    p_value = 1 - stats.f.cdf(f_value, df_conditions, df_residual)
    
    return f_value, p_value

def plot_data(data):
    df = pd.DataFrame(data, columns=[f"Condition {i+1}" for i in range(data.shape[1])])
    df_melted = df.melt(var_name="Condition", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Condition", y="Score", data=df_melted, ax=ax)
    sns.lineplot(x="Condition", y="Score", data=df_melted, color="0.25", ax=ax, markers=True)

    ax.set_title("Repeated Measures ANOVA - Simulated Data")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Score")

    return fig

if __name__ == "__main__":
    main()