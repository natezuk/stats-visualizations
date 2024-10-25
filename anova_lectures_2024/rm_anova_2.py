import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate example data
n_subjects = 30
n_conditions = 3
subject_ids = np.repeat(np.arange(1, n_subjects + 1), n_conditions)
conditions = np.tile(['A', 'B', 'C'], n_subjects)

# Generate random data with a slight effect
data = np.random.normal(loc=100, scale=15, size=n_subjects * n_conditions)
data[conditions == 'B'] += 5  # Add a small effect to condition B
data[conditions == 'C'] += 10  # Add a larger effect to condition C

# Create a DataFrame
df = pd.DataFrame({
    'Subject': subject_ids,
    'Condition': conditions,
    'Score': data
})

# Run repeated measures ANOVA
aov = pg.rm_anova(data=df, dv='Score', within='Condition', subject='Subject')

# Display the ANOVA results
print("Repeated Measures ANOVA Results:")
print(aov)

# Run post-hoc pairwise t-tests
post_hoc = pg.pairwise_tests(data=df, dv='Score', within='Condition', subject='Subject', padjust='bonf')

# Display post-hoc results
print("\nPost-hoc Pairwise T-tests:")
print(post_hoc)

# Calculate effect size (eta-squared)
eta_squared = aov['ng2'].values[0]
print(f"\nEffect size (eta-squared): {eta_squared:.3f}")

# Calculate observed power
observed_power = pg.power_anova(eta=np.sqrt(eta_squared), power=None, n=n_subjects, k=n_conditions)
print(f"Observed power: {observed_power:.3f}")

# Create line plots
plt.figure(figsize=(12, 6))

# Plot individual subject data
sns.lineplot(data=df, x='Condition', y='Score', hue='Subject', legend=False, alpha=0.3)

# Plot mean scores
sns.pointplot(data=df, x='Condition', y='Score', color='black', markers='D', linestyles='-', ci=68)

plt.title('Repeated Measures: Individual Subjects and Mean Scores Across Conditions')
plt.xlabel('Condition')
plt.ylabel('Score')

# Add text annotations for mean scores
for i, condition in enumerate(['A', 'B', 'C']):
    mean_score = df[df['Condition'] == condition]['Score'].mean()
    plt.text(i, mean_score, f'{mean_score:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Condition', y='Score')
sns.swarmplot(data=df, x='Condition', y='Score', color=".25", size=3, alpha=0.5)

plt.title('Boxplot of Scores Across Conditions')
plt.xlabel('Condition')
plt.ylabel('Score')

plt.tight_layout()
plt.show()