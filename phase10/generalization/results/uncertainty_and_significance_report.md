# Step 4: Statistical Uncertainty Quantification & Hypothesis Testing

Defensible statistical evidence evaluating whether depth refinement gains are statistically significant ($p < 0.05$) across paired images.

## 1. Summary of Statistical Significance

| Domain | Test Images | Base Mean | Ours Mean | Mean Delta AbsRel | 95% Bootstrap CI | Paired t-test (p) | Wilcoxon (p) | Win Rate | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Camo** | 76 | 20.4080 | 19.1613 | **+1.2467** | [1.0377, 1.4725] | p = 2.54e-17 | p = 6.04e-14 | **97.4%** | Yes (p < 0.001) |
| **Overall** | 80 | 20.0089 | 18.8425 | **+1.1664** | [0.9469, 1.3969] | p = 8.63e-16 | p = 1.94e-13 | **95.0%** | Yes (p < 0.001) |


## 2. Statistical Interpretation

1. **Zero Excluded from 95% Confidence Intervals:** Across domains, the 95% bootstrap confidence interval for mean error reduction is strictly positive, verifying that error reduction is not an artifact of random sampling.
2. **Hypothesis Rejection:** The null hypothesis (H0: adapter does not reduce depth error) is rejected with p < 0.01 on both parametric (paired Student's t-test) and non-parametric (Wilcoxon signed-rank test) metrics.
3. **Consistency Across Domains:** The per-image win rate confirms consistency across heterogeneous edge cases.
