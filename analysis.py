import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import List, Dict
import numpy as np

def load_and_prepare_data(filepath: str = "game_results.csv") -> pd.DataFrame:
    """Load and prepare the data for analysis."""
    df = pd.read_csv(filepath)
    
    # Convert string representations to Python objects
    for col in ['action_log', 'prize_values', 'basket_counts', 'revealed_cells']:
        df[col] = df[col].apply(json.loads)
    
    # Extract decision patterns
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

def plot_decision_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot decision distribution by nudge condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    decision_counts = pd.crosstab(
        df['nudge_present'], 
        df['final_decision'],
        normalize='index'
    ) * 100
    
    decision_counts.plot(kind='bar', ax=ax)
    ax.set_title('Decision Types by Nudge Condition')
    ax.set_xlabel('Nudge Present')
    ax.set_ylabel('Percentage')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    return fig

def plot_reveals_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot distribution of number of reveals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(
        data=df,
        x='num_reveals',
        hue='nudge_present',
        multiple="dodge",
        ax=ax,
        discrete=True
    )
    ax.set_title('Number of Reveals Distribution')
    ax.set_xlabel('Number of Reveals Made')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    return fig

def plot_points_by_decision(df: pd.DataFrame) -> plt.Figure:
    """Plot points earned by decision type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=df,
        x='final_decision',
        y='points_earned',
        hue='nudge_present',
        ax=ax
    )
    ax.set_title('Points Earned by Decision Type')
    ax.set_xlabel('Decision Type')
    ax.set_ylabel('Points Earned')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_learning_curve(df: pd.DataFrame) -> plt.Figure:
    """Plot learning curve of reveals over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df['trial_number'] = range(len(df))
    
    sns.regplot(
        data=df[df['nudge_present']],
        x='trial_number',
        y='num_reveals',
        scatter=True,
        label='Nudge Present',
        ax=ax
    )
    sns.regplot(
        data=df[~df['nudge_present']],
        x='trial_number',
        y='num_reveals',
        scatter=True,
        label='Control',
        ax=ax
    )
    ax.set_title('Number of Reveals Over Time')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Number of Reveals')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_nudge_optimality(df: pd.DataFrame) -> plt.Figure:
    """Plot analysis of nudge basket's value compared to other baskets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter for cases where nudge was present
    nudge_df = df[df['nudge_present']]
    
    # Calculate if the nudged basket was actually the best
    def is_nudge_optimal(row):
        basket_values = {}
        for basket, counts in row['basket_counts'].items():
            value = sum(count * row['prize_values'][prize] 
                       for prize, count in counts.items())
            basket_values[int(basket)] = value
        max_value = max(basket_values.values())
        nudge_value = basket_values[row['default_basket']]
        return {
            'is_optimal': nudge_value == max_value,
            'nudge_value': nudge_value,
            'max_value': max_value,
            'value_difference': max_value - nudge_value
        }
    
    # Apply the analysis to each row
    nudge_analysis = nudge_df.apply(is_nudge_optimal, axis=1)
    optimal_stats = pd.DataFrame(nudge_analysis.tolist())
    
    # Plot 1: Optimality Rate
    optimal_rate = optimal_stats['is_optimal'].value_counts(normalize=True) * 100
    optimal_rate.plot(kind='bar', ax=ax1)
    ax1.set_title('Was Nudge the Optimal Choice?')
    ax1.set_xlabel('Nudge Was Optimal')
    ax1.set_ylabel('Percentage of Games')
    for i, v in enumerate(optimal_rate):
        ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Value Difference Distribution
    sns.histplot(
        data=optimal_stats,
        x='value_difference',
        ax=ax2,
        bins=20
    )
    ax2.set_title('Value Difference: Best Basket vs Nudged Basket')
    ax2.set_xlabel('Points Difference')
    ax2.set_ylabel('Count')
    
    # Add mean difference as text
    mean_diff = optimal_stats['value_difference'].mean()
    ax2.text(0.95, 0.95, f'Mean Difference: {mean_diff:.1f} points',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def generate_statistics(df: pd.DataFrame) -> None:
    """Generate and print detailed statistics about AI decisions."""
    print("\n=== Decision Pattern Analysis ===")
    
    # Overall statistics
    print("\nOverall Decision Distribution:")
    print(df['final_decision'].value_counts(normalize=True).mul(100).round(1))
    
    # Decision patterns by nudge condition
    print("\nDecision Distribution by Nudge Condition (%):")
    decision_dist = pd.crosstab(
        df['nudge_present'], 
        df['final_decision'],
        normalize='index'
    ) * 100
    print(decision_dist.round(1))
    
    # Reveal patterns
    print("\nReveal Statistics:")
    reveal_stats = df.groupby('nudge_present')['num_reveals'].agg(['mean', 'std', 'max'])
    print(reveal_stats.round(2))
    
    # Performance analysis
    print("\nPoints Earned by Decision Type:")
    performance = df.groupby(['final_decision', 'nudge_present'])['points_earned'].agg(['mean', 'std'])
    print(performance.round(2))
    
    # Statistical tests
    from scipy import stats
    
    nudge = df[df['nudge_present']]['points_earned']
    control = df[~df['nudge_present']]['points_earned']
    t_stat, p_val = stats.ttest_ind(nudge, control)
    
    print("\nStatistical Tests:")
    print(f"Points Earned t-test (nudge vs control):")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.3f}")

def generate_nudge_optimality_stats(df: pd.DataFrame) -> None:
    """Generate statistics about nudge optimality."""
    nudge_df = df[df['nudge_present']]
    
    def analyze_basket_values(row):
        basket_values = {}
        for basket, counts in row['basket_counts'].items():
            value = sum(count * row['prize_values'][prize] 
                       for prize, count in counts.items())
            basket_values[int(basket)] = value
        max_value = max(basket_values.values())
        nudge_value = basket_values[row['default_basket']]
        return pd.Series({
            'is_optimal': nudge_value == max_value,
            'value_difference': max_value - nudge_value,
            'relative_rank': sorted(basket_values.values(), reverse=True).index(nudge_value) + 1
        })
    
    analysis = nudge_df.apply(analyze_basket_values, axis=1)
    
    print("\n=== Nudge Optimality Analysis ===")
    print("\nOptimality Rate:")
    print(analysis['is_optimal'].value_counts(normalize=True).mul(100).round(1))
    
    print("\nValue Difference Statistics (when not optimal):")
    non_optimal_diff = analysis[~analysis['is_optimal']]['value_difference']
    print(non_optimal_diff.agg(['count', 'mean', 'std', 'min', 'max']).round(2))
    
    print("\nNudge Basket Rank Distribution:")
    rank_dist = analysis['relative_rank'].value_counts().sort_index()
    print(rank_dist.div(len(analysis)).mul(100).round(1))

def generate_analysis_report():
    """Generate complete analysis report with plots and statistics."""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Set up plotting style
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Generate and save individual plots
    plots = {
        'decision_distribution': plot_decision_distribution,
        'reveals_distribution': plot_reveals_distribution,
        'points_by_decision': plot_points_by_decision,
        'learning_curve': plot_learning_curve,
        'nudge_optimality': plot_nudge_optimality
    }
    
    for name, plot_func in plots.items():
        fig = plot_func(df)
        plt.savefig(f'plots/{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate statistics
    generate_statistics(df)
    generate_nudge_optimality_stats(df)

if __name__ == "__main__":
    generate_analysis_report() 