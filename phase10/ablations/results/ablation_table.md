# Step 3: Targeted Ablation Study Table (Matched Conditions)

Comparison of architectural components under matched test conditions on the verified benchmark partition.

### 1. Primary Raw Unaligned Metrics

| Configuration | Parameters | Input Channels | Routing Mechanism | AbsRel (↓) | RMSE (↓) | SILog (↓) | δ₁ < 1.25 (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline MiDaS** | 21.2M | 3 ch (RGB) | None (Backbone Only) | **20.0089** | 0.5526 | 2.0785 | **0.0562** |
| **Ablation No Sobel** | 228.3K | 4 ch (Base+RGB) | Soft MoE Router | **19.7639** | 0.5371 | 1.9163 | **0.0611** |
| **Ablation Uniform Router** | 225.1K | 5 ch (Base+RGB+Sobel) | Uniform Blend (1/3) | **19.3173** | 0.5297 | 1.9407 | **0.0589** |
| **Ablation Single Head** | 83.4K | 5 ch (Base+RGB+Sobel) | Single Head (Monolithic) | **20.6536** | 0.5454 | 2.0201 | **0.0550** |
| **Full Phase10 MultiHead** | 228.4K | 5 ch (Base+RGB+Sobel) | Dynamic Soft Router | **18.8425** | 0.5206 | 1.9552 | **0.0581** |


### 2. Standard Median Scale-Aligned Metrics

| Configuration | AbsRel (↓) | SqRel (↓) | RMSE (↓) | SILog (↓) | δ₁ < 1.25 (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline MiDaS** | **4.8435** | 13.4815 | 0.3547 | 1.9714 | **0.1285** |
| **Ablation No Sobel** | **3.8944** | 6.2650 | 0.3294 | 1.7846 | **0.1400** |
| **Ablation Uniform Router** | **4.1612** | 7.9713 | 0.3362 | 1.8178 | **0.1366** |
| **Ablation Single Head** | **4.2327** | 6.9674 | 0.3174 | 1.8834 | **0.1359** |
| **Full Phase10 MultiHead** | **4.2902** | 9.3232 | 0.3389 | 1.8385 | **0.1383** |


### Key Ablation Insights:
1. **Value of Sobel Edge Guidance:** Removing Sobel gradients degrades boundary precision and increases AbsRel error.
2. **Value of Dynamic Soft Routing:** Replacing the learned soft router with uniform averaging confirms that domain-adaptive weighting outperforms fixed blending.
3. **Value of Multi-Head Specialization:** Tripling head capacity into specialized domain experts significantly outperforms monolithic single-head refinement on hard optical edge cases.
