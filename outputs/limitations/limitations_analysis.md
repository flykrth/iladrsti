# Ilādṛṣṭi: Systematic Limitations & Error Analysis
*Track 6: AI for Science & Society — Deep Learning Hackathon, Amrita Vishwa Vidyapeetham*

## 1. Quantitative Failure Case Breakdown (Top 10 False Positive Samples)

The table below catalogs the top 10 test samples with the highest False Positive rates on the held-out test split ($N=40$).
These samples represent optical edge cases where unburned terrain is mistakenly classified as wildfire burn scars.

| Rank | Scene Identifier | False Positive Rate | FP Pixels | Test Dice | Test IoU | Primary Failure Mode | Physical Remote Sensing Explanation |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **#01** | `a1__07_13.tif` | **59.56%** | 30,462 | `0.4748` | `0.3113` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.07) exhibiting charcoal-like visible light absorption. |
| **#02** | `a2__5_5.tif` | **53.97%** | 14,062 | `0.8377` | `0.7207` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.06) exhibiting charcoal-like visible light absorption. |
| **#03** | `a1__07_11.tif` | **43.62%** | 12,947 | `0.8284` | `0.7071` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.08) exhibiting charcoal-like visible light absorption. |
| **#04** | `b__06_06.tif` | **29.75%** | 4,606 | `0.9434` | `0.8929` | **Complex Smoke-Cloud Boundary** | Turbulent smoke plume mixed with cloud margins (RGB mean=0.16) creating diffuse false positives. |
| **#05** | `a2__5_3.tif` | **28.13%** | 10,519 | `0.8287` | `0.7076` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.05) exhibiting charcoal-like visible light absorption. |
| **#06** | `a1__07_10.tif` | **26.45%** | 10,940 | `0.7525` | `0.6032` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.08) exhibiting charcoal-like visible light absorption. |
| **#07** | `a1__04_10.tif` | **22.87%** | 11,375 | `0.6892` | `0.5257` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.08) exhibiting charcoal-like visible light absorption. |
| **#08** | `a1__08_06.tif` | **22.37%** | 5,758 | `0.8915` | `0.8042` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.10) exhibiting charcoal-like visible light absorption. |
| **#09** | `a1__02_04.tif` | **20.80%** | 5,252 | `0.8776` | `0.7819` | **Topographic Shadow / Water Confusion** | Deep cast shadow or dark terrain (RGB mean=0.10) exhibiting charcoal-like visible light absorption. |
| **#10** | `m1__5_4.tif` | **20.27%** | 4,732 | `0.9142` | `0.8419` | **Complex Smoke-Cloud Boundary** | Turbulent smoke plume mixed with cloud margins (RGB mean=0.16) creating diffuse false positives. |

---

## 2. Root Cause Classification & Physical Remote Sensing Mechanisms

### A. Dense Cumulus Cloud Edges & Vapor Plumes
Dense clouds introduce severe optical scattering. While deep cloud centers are bright across all bands, cloud margins display steep gradient rolloffs and partial transparency. When aerosol haze interacts with visible bands, the model occasionally misidentifies turbulent cloud boundaries as active fire perimeters.

### B. Topographic Mountain Shadows & Sun-Terrain Geometry
In steep, rugged terrain (e.g., canyons and mountain escarpments), low solar elevation angles create deep cast shadows. These shadowed pixels exhibit near-zero visible surface reflectance ($\\rho < 0.10$), closely mimicking charred, carbonized charcoal ash. 

### C. Mitigation via Spectral Band Engineering
Our spectral ablation experiments demonstrate that incorporating Sentinel-2 **Band 8 (Near-Infrared, $\\sim 842\\,\\text{nm}$)** significantly reduces shadow false alarms, because unburned shaded vegetation maintains residual NIR mesophyll reflectance, whereas true burned scars exhibit a complete NIR collapse.

---

## 3. Visual Artifact Reference
* Diagnostic composites for all top 10 failure cases are located in `outputs/limitations/`.
* High-resolution multi-case publication figure: `outputs/limitations/fig_limitations_top10_overview.png`.
