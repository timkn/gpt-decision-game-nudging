import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load data with error handling
try:
    df = pd.read_csv("game_results.csv")
    
    # Print data info for debugging
    print("DataFrame Info:")
    print(df.info())
    print("\nAvailable columns:", df.columns.tolist())
    
    # Check if required columns exist
    if 'model_used' not in df.columns or 'points_earned' not in df.columns:
        raise ValueError("Required columns 'model_used' or 'points_earned' not found in CSV")
    
    # Convert points_earned to numeric, handling any non-numeric values
    df['points_earned'] = pd.to_numeric(df['points_earned'], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna(subset=['points_earned', 'model_used'])
    
    # Calculate total points for each model
    total_points = df.groupby('model_used')['points_earned'].sum().reset_index()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=total_points, x='model_used', y='points_earned')
    
    # Add value labels on top of each bar
    for i, v in enumerate(total_points['points_earned']):
        plt.text(i, v, f'{int(v):,}', ha='center', va='bottom')
    
    plt.title('Total Points Earned by Model')
    plt.xlabel('Model')
    plt.ylabel('Total Points')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('plots/total_points_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nTotal Points by Model:")
    print(total_points.to_string(index=False))
    
    # Print detailed statistics
    print("\nDetailed Statistics by Model:")
    stats = df.groupby('model_used')['points_earned'].agg(['count', 'sum', 'mean', 'std']).round(2)
    print(stats)

except FileNotFoundError:
    print("Error: game_results.csv file not found!")
except Exception as e:
    print(f"Error: {str(e)}") 