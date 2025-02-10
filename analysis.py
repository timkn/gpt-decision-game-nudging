# analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import List, Dict
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
    
    nudge_df = df[df['nudge_present']]
    
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
    
    nudge_analysis = nudge_df.apply(is_nudge_optimal, axis=1)
    optimal_stats = pd.DataFrame(nudge_analysis.tolist())
    
    # Plot optimality rate.
    optimal_rate = optimal_stats['is_optimal'].value_counts(normalize=True) * 100
    optimal_rate.plot(kind='bar', ax=ax1)
    ax1.set_title('Nudge Optimality Rate')
    ax1.set_xlabel('Nudge Was Optimal')
    ax1.set_ylabel('Percentage')
    for i, v in enumerate(optimal_rate):
        ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # Plot value difference distribution.
    sns.histplot(
        data=optimal_stats,
        x='value_difference',
        ax=ax2,
        bins=20
    )
    ax2.set_title('Value Difference: Best Basket vs Nudged Basket')
    ax2.set_xlabel('Points Difference')
    ax2.set_ylabel('Count')
    mean_diff = optimal_stats['value_difference'].mean()
    ax2.text(0.95, 0.95, f'Mean Diff: {mean_diff:.1f} pts',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_basket_values_debug(df: pd.DataFrame) -> plt.Figure:
    """Plot the values of all baskets for each game to debug nudge optimality."""
    def get_basket_values(row):
        values = {}
        for basket, counts in row['basket_counts'].items():
            value = sum(count * row['prize_values'][prize] 
                       for prize, count in counts.items())
            values[int(basket)] = value
        return values
    
    df['basket_values'] = df.apply(get_basket_values, axis=1)
    game_data = []
    for idx, row in df.iterrows():
        for basket, value in row['basket_values'].items():
            game_data.append({
                'Game': idx + 1,
                'Basket': basket,
                'Value': value,
                'Is Default': basket == row.get('default_basket', None),
                'Nudge Present': row['nudge_present']
            })
    
    plot_df = pd.DataFrame(game_data)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    sns.scatterplot(
        data=plot_df,
        x='Game',
        y='Value',
        hue='Basket',
        style='Is Default',
        size='Is Default',
        sizes={False: 50, True: 200},
        alpha=0.6,
        ax=ax
    )
    
    default_baskets = plot_df[plot_df['Is Default']]
    sns.scatterplot(
        data=default_baskets,
        x='Game',
        y='Value',
        color='red',
        marker='*',
        s=200,
        label='Default Basket',
        ax=ax
    )
    
    for game in plot_df['Game'].unique():
        ax.axvline(x=game, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_title('Basket Values for Each Game (Default marked with *)')
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Basket Value')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(title='Baskets', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def generate_statistics(df: pd.DataFrame) -> None:
    """Generate basic statistics about AI decisions on valid games."""
    print("\n=== Basic Decision Statistics (Valid Games) ===")
    
    print("\nOverall Decision Distribution:")
    print(df['final_decision'].value_counts(normalize=True).mul(100).round(1))
    
    print("\nDecision Distribution by Nudge Condition (%):")
    decision_dist = pd.crosstab(
        df['nudge_present'], 
        df['final_decision'],
        normalize='index'
    ) * 100
    print(decision_dist.round(1))
    
    print("\nReveal Statistics:")
    reveal_stats = df.groupby('nudge_present')['num_reveals'].agg(['mean', 'std', 'max'])
    print(reveal_stats.round(2))
    
    print("\nPoints Earned by Decision Type:")
    performance = df.groupby(['final_decision', 'nudge_present'])['points_earned'].agg(['mean', 'std'])
    print(performance.round(2))
    
    t_stat, p_val = stats.ttest_ind(df[df['nudge_present']]['points_earned'], df[~df['nudge_present']]['points_earned'])
    print("\nPoints Earned t-test (nudge vs control):")
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

def generate_additional_statistics(df: pd.DataFrame) -> None:
    """Generate additional statistics such as cost efficiency and correlations."""
    print("\n=== Additional Statistics ===")
    df['cost_efficiency'] = df['total_reveal_cost'] / (df['total_reveal_cost'] + df['points_earned'])
    print("Average cost efficiency ratio (reveal cost / (reveal cost + payout)) by final decision:")
    print(df.groupby('final_decision')['cost_efficiency'].mean().round(2))
    
    corr = df['wrong_moves'].corr(df['points_earned'])
    print(f"\nCorrelation between wrong moves and points earned: {corr:.2f}")

def generate_model_comparison_stats(df: pd.DataFrame) -> None:
    """Generate statistics comparing performance across different models, if available."""
    print("\n=== Model Comparison Statistics ===")
    if 'model_used' in df.columns:
        model_groups = df.groupby('model_used')
        for model, group in model_groups:
            print(f"\nModel: {model}")
            print("Average points earned:", group['points_earned'].mean())
            print("Average number of reveals:", group['num_reveals'].mean())
            print("Average wrong moves:", group['wrong_moves'].mean())
    else:
        print("No model comparison available, 'model_used' column missing.")

def generate_error_statistics(df: pd.DataFrame) -> None:
    """Generate statistics for games that ended with an error."""
    error_df = df[df['error'] != ""]
    print("\n=== Error Games Statistics ===")
    print("Total games with errors:", len(error_df))
    if len(error_df) > 0:
        print("Error Distribution:")
        print(error_df['error'].value_counts())
    else:
        print("No error games recorded.")

def generate_analysis_report():
    """Generate complete analysis report with plots and statistics."""
    df = load_and_prepare_data()
    
    # Separate valid games from error games.
    valid_df = df[df['error'] == ""]
    error_df = df[df['error'] != ""]
    
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    os.makedirs('plots', exist_ok=True)
    
    plots = {
        'decision_distribution': plot_decision_distribution,
        'reveals_distribution': plot_reveals_distribution,
        'points_by_decision': plot_points_by_decision,
        'learning_curve': plot_learning_curve,
        'nudge_optimality': plot_nudge_optimality,
        'basket_values_debug': plot_basket_values_debug
    }
    
    # Generate plots only for valid games.
    for name, plot_func in plots.items():
        fig = plot_func(valid_df)
        plt.savefig(f'plots/{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n===== Basic Statistics (Valid Games) =====")
    generate_statistics(valid_df)
    
    print("\n===== Nudge Optimality Statistics (Valid Games) =====")
    generate_nudge_optimality_stats(valid_df)
    
    print("\n===== Additional Statistics (Valid Games) =====")
    generate_additional_statistics(valid_df)
    
    print("\n===== Model Comparison Statistics (Valid Games) =====")
    generate_model_comparison_stats(valid_df)
    
    # Report error statistics separately.
    print("\n===== Error Games Statistics =====")
    generate_error_statistics(error_df)

if __name__ == "__main__":
    generate_analysis_report()
