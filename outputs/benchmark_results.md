# Ilādṛṣṭi: Benchmark Results & Ablation Study
Track 6 (AI for Science & Society) — Deep Learning Hackathon, Amrita Vishwa Vidyapeetham

| Experiment | Architecture | Backbone | Bands | Loss Function | Val Dice | Val IoU | Test Loss | Test Dice | Test IoU | Seed | Notes |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mandatory Baseline** | Standard U-Net | ResNet-50 (Pretrained) | 4 (RGB+NIR) | BCEWithLogitsLoss | 0.6924 | 0.6100 | 0.3781 | **0.7834** | **0.7044** | 42 | Mandatory comparison floor (Seed locked 42) |
