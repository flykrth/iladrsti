# Ilādṛṣṭi Final Evaluation Suite: Primary Benchmark Comparison
*Track 6: AI for Science & Society — Deep Learning Hackathon, Amrita Vishwa Vidyapeetham*
*Evaluation Split: Strictly held-out test data ($N=40$ tiles) | Seed Locked: 42*

### Comparative Benchmark Matrix

| Model Name | Spectral Bands | Objective Loss | Test Dice ($F_1$) | Test Mean IoU | Test Loss | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Model 1: Baseline BCE (RGB)** | 3 (RGB) | BCEWithLogitsLoss | **`0.7071`** | **`0.6297`** | `0.4203` | Verified |
| **Model 2: Focal-Tversky (RGB)** | 3 (RGB) | Focal-Tversky Loss | **`0.5027`** | **`0.4245`** | `0.5154` | Verified |
| **Model 3: Focal-Tversky (RGB + NIR)** | 4 (RGB+NIR) | Focal-Tversky Loss | **`0.5796`** | **`0.5013`** | `0.5102` | Verified |
| *Spectral Gain: NIR vs RGB (M3 - M2)* | *+1 Band (NIR)* | *Focal-Tversky* | *+0.0769* | *+0.0767* | *-0.0052* | *Ablation Delta* |
| *Loss Gain: Focal-Tversky vs BCE (M2 - M1)* | *3 (RGB)* | *FTL vs BCE* | *-0.2044* | *-0.2052* | — | *Loss Delta* |

### Metric Definitions
- **Test Dice Coefficient ($F_1$)**: Harmonic mean of precision and recall: $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$. Primary measure of burned scar spatial overlap.
- **Test Mean IoU (Jaccard Index)**: Intersection over Union: $\frac{TP}{TP + FP + FN}$. Strict measure of scar contour alignment.
- **Evaluation Integrity**: Zero leakage verified; test data contains no spatial or temporal overlap with training or validation splits.
