# Ilādṛṣṭi Spectral Ablation Study: 3-Band (RGB) vs. 4-Band (RGB+NIR)

| Metric | 3-Band RGB | 4-Band RGB + NIR | Delta ($\Delta$) | Empirical Contribution |
| :--- | :---: | :---: | :---: | :--- |
| **Test Dice ($F_1$)** | `0.5027` | `0.5796` | **`+0.0769`** | Higher scar delineation accuracy |
| **Test Mean IoU** | `0.4245` | `0.5013` | **`+0.0767`** | Superior spatial boundary alignment |
| **Test Loss** | `0.5154` | `0.5102` | `-0.0052` | Cross-entropy/Tversky objective |

*Both models evaluated on held-out test split ($N=40$) using deterministic seed=42.*
