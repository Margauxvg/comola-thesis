import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Data for pop size 10 and 2 generations
original_code_small = [40, 37, 35, 35, 36, 35, 35, 35, 35, 35]
post_processing_small = [36, 35, 36, 36, 35, 35, 35, 35, 36, 35]
without_postprocessing_small = [35, 35, 36, 35, 35, 36, 36, 36, 35, 36]

# Data for pop size 50 and 10 generations
original_code_large = [575, 571, 567, 567, 573, 567, 567, 569, 572, 568]
post_processing_large = [575, 575, 578, 581, 574, 590, 572, 576, 583, 583]
without_postprocessing_large = [565, 563, 565, 565, 565, 565, 565, 565, 564, 562]

# Calculate means for each set of experiments
original_means_small = np.mean(original_code_small)
post_means_small = np.mean(post_processing_small)
without_means_small = np.mean(without_postprocessing_small)

original_means_large = np.mean(original_code_large)
post_means_large = np.mean(post_processing_large)
without_means_large = np.mean(without_postprocessing_large)

# Combine data and labels for Tukey's test
data_small = np.concatenate([original_code_small, post_processing_small, without_postprocessing_small])
labels_small = (['Original Code'] * 10) + (['Revised with Preprocessing Layer'] * 10) + (['Revised without Preprocessing Layer' ] * 10)

# Combine data and labels for Tukey's test
data_large = np.concatenate([original_code_large, post_processing_large, without_postprocessing_large])
labels_large = (['Original Code'] * 10) + (['Revised with Preprocessing Layer'] * 10) + (['Revised without Preprocessing Layer'] * 10)



versions = ['Original Code', 'Revised with Preprocessing Layer', 'Revised without Preprocessing Layer']
colors = ['skyblue', 'salmon', 'lightgreen']

# Create bar plot for small experiments
means_small = [original_means_small, post_means_small, without_means_small]

plt.figure(figsize=(8, 6))

for i, (mean, color) in enumerate(zip(means_small, colors)):
    plt.bar(i, mean, color=color, width=0.4, label=versions[i])
    plt.text(i, mean + 1, '{:.2f}'.format(mean), ha='center', va='bottom', fontsize=10)

plt.ylabel('Mean Execution Time (seconds)')
plt.title("Small Experiment", fontsize=10, style='italic')
plt.suptitle("Mean Execution Time of Algorithm Versions", fontsize=14, fontweight='bold')
plt.xticks(range(len(versions)), versions)
plt.tight_layout()

plt.show()

# Bar plot for mean execution time of each version (large experiments)
means_large = [original_means_large, post_means_large, without_means_large]

plt.figure(figsize=(8, 6))

for i, (mean, color) in enumerate(zip(means_large, colors)):
    plt.bar(i, mean, color=color, width=0.4, label=versions[i])
    plt.text(i, mean + 1, '{:.2f}'.format(mean), ha='center', va='bottom', fontsize=10)

plt.ylabel('Mean Execution Time (seconds)')
plt.title("Large Experiment", fontsize=10, style='italic')
plt.suptitle("Mean Execution Time of Algorithm Versions", fontsize=14, fontweight='bold')
plt.xticks(range(len(versions)), versions)
plt.tight_layout()

plt.show()


# # Statistical comparison (ANOVA) for small experiments
# f_statistic_small, p_value_small = f_oneway(original_code_small, post_processing_small, without_postprocessing_small)

# if p_value_small < 0.05:
#     print("ANOVA (Small Experiments): There is a significant difference between at least one pair of means.")
#     # Perform Tukey's HSD post-hoc test
#     tukey_result = pairwise_tukeyhsd(data_small, labels_small, alpha=0.05)
#     print(tukey_result)
# else:
#     print("ANOVA (Small Experiments): No significant difference between the groups.")

# # Statistical comparison (ANOVA) for large experiments
# f_statistic_large, p_value_large = f_oneway(original_code_large, post_processing_large, without_postprocessing_large)

# if p_value_large < 0.05:
#     print("ANOVA (Large Experiments): There is a significant difference between at least one pair of means.")
#     # Perform Tukey's HSD post-hoc test
#     tukey_result = pairwise_tukeyhsd(data_large, labels_large, alpha=0.05)
#     print(tukey_result)
# else:
#     print("ANOVA (Large Experiments): No significant difference between the groups.")