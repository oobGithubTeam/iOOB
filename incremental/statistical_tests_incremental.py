from scipy.stats import ttest_ind
from tabulate import tabulate
import numpy as np

scores_for_metrics = [
    "incremental_recall.csv",
    "incremental_specificity.csv",
    "incremental_bac.csv",
]

stat_better_table_for_metrics = [
    'stat_better_table_recall',
    'stat_better_table_specificity',
    'stat_better_table_bac',
]

metric_names = [
    "recall",
    "specificity",
    "balanced accuracy"
]

clfs = [
    "SEA",
    "OB",
    "OOB",
    "UOB",
]

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

iterator = 0

for iterator in range(3):
    scores = np.genfromtxt(scores_for_metrics[iterator], delimiter=",")

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

    headers = ["SEA", "OB", "OOB", "UOB"]
    names_column = np.array([["SEA"], ["OB"], ["OOB"], ["UOB"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table, "\n")

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("Advantage:\n", advantage_table, "\n")

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print("Statistical significance (alpha = 0.05):\n", significance_table, "\n")

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("metrics name:", metric_names[iterator], "\nStatistically significantly better:\n", stat_better_table, "\n")

    np.save(stat_better_table_for_metrics[iterator], stat_better_table)
    
# create results.csv
result_recall = np.genfromtxt("incremental_recall.csv", delimiter=",")
result_specificity = np.genfromtxt("incremental_specificity.csv", delimiter=",")
result_bac = np.genfromtxt("incremental_bac.csv", delimiter=",")
recall_mean = np.mean(result_recall, axis=1)
specificity_mean = np.mean(result_specificity, axis=1)
bac_mean = np.mean(result_bac, axis=1)

results = [
    recall_mean.tolist(),
    specificity_mean.tolist(),
    bac_mean.tolist(),
]
np.savetxt("results.csv", results, delimiter=",")
    
    
