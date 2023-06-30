import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import stats

np.random.seed(42)

num_students = 200

teaching_methods = np.random.choice(["A", "B"], size=num_students)
study_hours = np.random.normal(loc=25, scale=5, size=num_students)
student_motivation = np.random.normal(loc=7, scale=2, size=num_students)
prior_knowledge = np.random.normal(loc=75, scale=10, size=num_students)


def generate_test_scores(
    teaching_methods, study_hours, student_motivation, prior_knowledge
):
    test_scores = []
    for i in range(num_students):
        if teaching_methods[i] == "A":
            score = (
                0.4 * prior_knowledge[i]
                + 0.2 * study_hours[i]
                + 0.3 * student_motivation[i]
                + np.random.normal(loc=10, scale=5)
            )
        else:
            score = (
                0.3 * prior_knowledge[i]
                + 0.1 * study_hours[i]
                + 0.2 * student_motivation[i]
                + np.random.normal(loc=8, scale=5)
            )
        test_scores.append(score)
    return test_scores


test_scores = generate_test_scores(
    teaching_methods, study_hours, student_motivation, prior_knowledge
)

data = {
    "Teaching_Method": teaching_methods,
    "Test_Scores": test_scores,
    "Study_Hours": study_hours,
    "Student_Motivation": student_motivation,
    "Prior_Knowledge": prior_knowledge,
}

df = pd.DataFrame(data)

# Hypothesis Testing (t-test)
group_A_scores = df.loc[df["Teaching_Method"] == "A", "Test_Scores"]
group_B_scores = df.loc[df["Teaching_Method"] == "B", "Test_Scores"]

t_statistic, p_value = stats.ttest_ind(group_A_scores, group_B_scores)

print("T-Statistic:", t_statistic)
print("p-value:", p_value)

confidence_interval_A = stats.t.interval(
    0.95,
    len(group_A_scores) - 1,
    loc=np.mean(group_A_scores),
    scale=stats.sem(group_A_scores),
)
confidence_interval_B = stats.t.interval(
    0.95,
    len(group_B_scores) - 1,
    loc=np.mean(group_B_scores),
    scale=stats.sem(group_B_scores),
)

print("Confidence Interval for Group A:", confidence_interval_A)
print("Confidence Interval for Group B:", confidence_interval_B)

# Analysis of Variance (ANOVA)
f_statistic, p_value_anova = stats.f_oneway(group_A_scores, group_B_scores)

print("F-Statistic (ANOVA):", f_statistic)
print("p-value (ANOVA):", p_value_anova)

# Logistic Regression
# pass_fail_labels = np.where(df["Test_Scores"] >= 60, 1, 0)
# X = df[["Teaching_Method", "Study_Hours", "Student_Motivation", "Prior_Knowledge"]]
# X_encoded = pd.get_dummies(X, drop_first=True)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_encoded, pass_fail_labels, test_size=0.2, random_state=42
# )

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

# predicted_probs = logreg.predict_proba(X_test)
# print("Predicted Probabilities:", predicted_probs)
# threshold = 0.5
# predicted_classifiers = np.where(predicted_probs[:, 1] >= threshold, 1, 0)
# accuracy = np.mean(predicted_classifiers == y_test)
# print("Accuracy:", accuracy)

# Non-parametric Test (Mann-Whitney U test)
u_statistic, p_value_mannwhitney = stats.mannwhitneyu(group_A_scores, group_B_scores)

print("U-Statistic (Mann-Whitney):", u_statistic)
print("p-value (Mann-Whitney):", p_value_mannwhitney)

# Structural Equation Modeling (SEM)
model = sm.OLS.from_formula(
    "Test_Scores ~ Teaching_Method + Study_Hours + Student_Motivation + Prior_Knowledge",
    data=df,
)
results = model.fit()

print(results.summary())
