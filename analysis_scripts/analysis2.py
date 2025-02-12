import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from scipy import stats

def load_and_prepare_data(filepath: str = "game_results.csv") -> pd.DataFrame:
    """Load and prepare the data for analysis."""
    df = pd.read_csv(filepath)
    
    # Convert string representations of JSON objects to Python objects.
    for col in ['action_log', 'prize_values', 'basket_counts', 'revealed_cells']:
        df[col] = df[col].apply(json.loads)
    
    # Ensure the 'error' column exists and is a string.
    if 'error' in df.columns:
        df['error'] = df['error'].fillna("").astype(str)
    else:
        df['error'] = ""
    
    # Extract additional columns.
    df['num_reveals'] = df['revealed_cells'].apply(len)
    df['accepted_default'] = df['action_log'].apply(
        lambda x: any('accepted default' in str(action).lower() for action in x)
    )
    df['final_decision'] = df.apply(
        lambda row: 'Accepted Default' if row['accepted_default']
        else 'Direct Choice' if row['num_reveals'] == 0
        else 'After Reveals', axis=1
    )
    
    return df

def plot_decision_distribution_by_model_nudge(df: pd.DataFrame) -> plt.Figure:
    """
    Create a faceted count plot showing the distribution of final decision types
    split by nudge condition for each model.
    """
    g = sns.catplot(
        data=df,
        kind="count",
        x="final_decision",
        hue="nudge_present",
        col="model_used",
        palette="deep",
        height=4,
        aspect=1,
        dodge=True,
        col_wrap=3
    )
    g.set_axis_labels("Final Decision", "Count")
    g.fig.suptitle("Decision Distribution by Model and Nudge Condition", y=1.02)
    return g.fig

def plot_nudge_acceptance_rate_by_model(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the percentage of nudged games where the default basket was accepted,
    grouped by model.
    """
    nudged_df = df[df['nudge_present'] == True]
    rates = nudged_df.groupby('model_used')['final_decision'].apply(
        lambda x: x.str.contains("accepted default", case=False).mean() * 100
    ).reset_index(name='acceptance_rate')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=rates, x='model_used', y='acceptance_rate', ax=ax, palette="deep")
    ax.set_title("Nudge Acceptance Rate by Model")
    ax.set_xlabel("Model Used")
    ax.set_ylabel("Acceptance Rate (%)")
    for i, rate in enumerate(rates['acceptance_rate']):
        ax.text(i, rate, f"{rate:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    return fig

def plot_points_by_decision_by_model(df: pd.DataFrame) -> plt.Figure:
    """
    Create a box plot comparing points earned by decision type, split by nudge condition,
    and faceted by model.
    """
    g = sns.catplot(
        data=df,
        kind="box",
        x="final_decision",
        y="points_earned",
        hue="nudge_present",
        col="model_used",
        palette="deep",
        height=4,
        aspect=1,
        dodge=True,
        col_wrap=3
    )
    g.set_axis_labels("Final Decision", "Points Earned")
    g.fig.suptitle("Points Earned by Decision Type (by Model)", y=1.02)
    return g.fig

def plot_error_rate_by_model(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the error rate (percentage of games with errors) for each model.
    """
    total_by_model = df.groupby('model_used').size()
    error_by_model = df[df['error'] != ""].groupby('model_used').size()
    error_rate = (error_by_model / total_by_model * 100).reset_index(name='error_rate')
    error_rate['error_rate'] = error_rate['error_rate'].fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=error_rate, x='model_used', y='error_rate', ax=ax, palette="deep")
    ax.set_title("Error Rate by Model")
    ax.set_xlabel("Model Used")
    ax.set_ylabel("Error Rate (%)")
    for i, rate in enumerate(error_rate['error_rate']):
        ax.text(i, rate, f"{rate:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    return fig

def plot_overall_performance_by_model(df: pd.DataFrame) -> plt.Figure:
    """
    Create a box plot comparing the overall points earned by each model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='model_used', y='points_earned', ax=ax, palette="deep")
    ax.set_title("Overall Performance by Model (Points Earned)")
    ax.set_xlabel("Model Used")
    ax.set_ylabel("Points Earned")
    plt.tight_layout()
    return fig

def scatter_plot_reveals_vs_points(df: pd.DataFrame) -> plt.Figure:
    """
    Create a scatter plot of number of reveals vs. points earned.
    Color the points by nudge condition and facet by model.
    """
    g = sns.FacetGrid(df, col="model_used", hue="nudge_present", height=4, aspect=1, col_wrap=3, palette="deep")
    g.map(sns.scatterplot, "num_reveals", "points_earned")
    g.set_axis_labels("Number of Reveals", "Points Earned")
    g.fig.suptitle("Reveals vs. Points Earned by Model and Nudge Condition", y=1.02)
    return g.fig

def generate_analysis_report():
    """Generate complete analysis report with plots for nudging susceptibility."""
    df = load_and_prepare_data()
    
    # We will only consider valid games (i.e., error field is empty) for performance analyses.
    valid_df = df[df['error'] == ""]
    # For error analysis, consider the error games separately.
    error_df = df[df['error'] != ""]
    
    os.makedirs('plots', exist_ok=True)
    
    plot_funcs = {
        'decision_distribution_by_model_nudge': plot_decision_distribution_by_model_nudge,
        'nudge_acceptance_rate_by_model': plot_nudge_acceptance_rate_by_model,
        'points_by_decision_by_model': plot_points_by_decision_by_model,
        'error_rate_by_model': plot_error_rate_by_model,
        'overall_performance_by_model': plot_overall_performance_by_model,
        'scatter_reveals_vs_points': scatter_plot_reveals_vs_points
    }
    
    for name, plot_func in plot_funcs.items():
        fig = plot_func(valid_df)
        plt.savefig(f'plots/{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("=== Analysis Report ===")
    print("\nValid Games: ")
    print(valid_df.describe(include="all"))
    print("\nError Games Statistics:")
    print("Total error games:", len(error_df))
    if not error_df.empty:
        print(error_df['error'].value_counts())
    
    # Additionally, generate a model comparison plot if needed.
    fig = plot_overall_performance_by_model(valid_df)
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_analysis_report()
