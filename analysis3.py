import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Ensure numerical conversions are correct
df_success["nudge_present"] = df_success["nudge_present"].astype(bool)

# 1️⃣ Recalculate nudge decision-making (corrected plot)
def plot_nudge_decision_making():
    """Corrected: Compare the probability of choosing the best basket with and without a nudge."""
    df_success["chose_best_basket"] = df_success["final_choice"] == df_success["best_basket"]
    nudge_effect_data = df_success.groupby(["model_used", "nudge_present"])["chose_best_basket"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x="model_used", y="chose_best_basket", hue="nudge_present", data=nudge_effect_data)
    plt.xlabel("Modell")
    plt.ylabel("Wahrscheinlichkeit, den besten Korb zu wählen")
    plt.title("Einfluss von Nudging auf die Wahl des optimalen Korbs")
    plt.legend(title="Nudge vorhanden", labels=["Nein", "Ja"])
    plt.xticks(rotation=45)
    plt.show()

# 2️⃣ Points earned per model in different conditions
def plot_points_earned_by_model():
    """Compare the net points earned across models under nudging and no nudging."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="model_used", y="points_earned", hue="nudge_present", data=df_success)
    plt.xlabel("Modell")
    plt.ylabel("Erzielte Punkte")
    plt.title("Vergleich der erzielten Punkte mit und ohne Nudging")
    plt.xticks(rotation=45)
    plt.legend(title="Nudge vorhanden", labels=["Nein", "Ja"])
    plt.show()

# 3️⃣ More detailed analysis of reveals across models
def plot_reveals_by_model():
    """Compare number of reveals before decision under nudge vs. no nudge per model."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="model_used", y="num_reveals", hue="nudge_present", data=df_success)
    plt.xlabel("Modell")
    plt.ylabel("Anzahl der aufgedeckten Zellen")
    plt.title("Vergleich der Explorationsstrategie mit und ohne Nudging")
    plt.xticks(rotation=45)
    plt.legend(title="Nudge vorhanden", labels=["Nein", "Ja"])
    plt.show()

# 4️⃣ Statistical tests between models
# T-test for number of reveals between GPT-4o and o1-mini
t_stat_reveals, p_value_reveals = stats.ttest_ind(
    df_4o["num_reveals"].dropna(), df_o1_mini["num_reveals"].dropna(), equal_var=False
)

# T-test for points earned between GPT-4o and o1-mini
t_stat_points, p_value_points = stats.ttest_ind(
    df_4o["points_earned"].dropna(), df_o1_mini["points_earned"].dropna(), equal_var=False
)

# Logistic regression: Does nudging affect choosing the best basket differently for o1-mini?
import statsmodels.api as sm

df_o1_mini["nudge_present"] = df_o1_mini["nudge_present"].astype(int)  # Convert boolean to int for regression
X = sm.add_constant(df_o1_mini["nudge_present"])  # Add constant term
y = (df_o1_mini["final_choice"] == df_o1_mini["best_basket"]).astype(int)  # Whether final choice was the best

logit_model = sm.Logit(y, X)
logit_result_o1_mini = logit_model.fit()

# Execute plots
plot_nudge_decision_making()
plot_points_earned_by_model()
plot_reveals_by_model()

# Display statistical results
stats_summary = {
    "T-test (Reveals: GPT-4o vs. o1-mini)": {"t-statistic": t_stat_reveals, "p-value": p_value_reveals},
    "T-test (Points Earned: GPT-4o vs. o1-mini)": {"t-statistic": t_stat_points, "p-value": p_value_points},
    "Logistic Regression (Nudge -> Choosing Best Basket for o1-mini)": logit_result_o1_mini.summary()
}

stats_summary
