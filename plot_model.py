import pandas as pd
import matplotlib.pyplot as plt
from analysis import load_and_prepare_data, plot_single_model_points

# Load data
df = load_and_prepare_data()
valid_df = df[df['error'] == ""]

# Plot for specific model (replace with your model name)
model_name = "gpt-4o-mini"
fig = plot_single_model_points(valid_df, model_name)
plt.savefig(f'plots/points_{model_name}.png', dpi=300, bbox_inches='tight')
plt.close()

# Print basic statistics
model_df = valid_df[valid_df['model_used'] == model_name]
print(f"\nStatistics for {model_name}:")
print(f"Mean points: {model_df['points_earned'].mean():.1f}")
print(f"Median points: {model_df['points_earned'].median():.1f}")
print(f"Std dev: {model_df['points_earned'].std():.1f}")
print(f"Min points: {model_df['points_earned'].min():.1f}")
print(f"Max points: {model_df['points_earned'].max():.1f}") 