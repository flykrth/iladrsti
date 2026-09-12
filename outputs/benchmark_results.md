# Ilādṛṣṭi: Benchmark Results & Ablation Study
Track 6 (AI for Science & Society) — Deep Learning Hackathon, Amrita Vishwa Vidyapeetham

| Experiment | Architecture | Backbone | Bands | Loss Function | Val Dice | Val IoU | Test Loss | Test Dice | Test IoU | Seed | Notes |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mandatory Baseline** | Standard U-Net | ResNet-50 (Pretrained) | 4 (RGB+NIR) | BCE | 0.6924 | 0.6100 | 0.3781 | **0.7834** | **0.7044** | 42 | Evaluation on held-out test split (Seed locked 42) |
| **Ablation RGB (3-band)** | Standard U-Net | ResNet-50 (Pretrained) | 3 (RGB) | FOCAL_TVERSKY | 0.4414 | 0.3653 | 0.5154 | **0.5027** | **0.4245** | 42 | Evaluation on held-out test split (Seed locked 42) |
| **Ablation RGB+NIR (4-band)** | Standard U-Net | ResNet-50 (Pretrained) | 4 (RGB+NIR) | FOCAL_TVERSKY | 0.4840 | 0.4034 | 0.5102 | **0.5796** | **0.5013** | 42 | Evaluation on held-out test split (Seed locked 42) |
