/**
 * Ilādṛṣṭi Web Dashboard - Data Registry
 * Contains benchmark metrics, training progression history,
 * test scene telemetry, and architecture specifications.
 */

const ILADRISTI_DATA = {
  project: {
    name: "Ilādṛṣṭi",
    sanskritName: "इलादृष्टि",
    etymology: "इला (Earth) + दृष्टि (Vision)",
    subtitle: "Multispectral Semantic Segmentation for Wildfire Progression",
    track: "Track 6: AI for Science & Society",
    hackathon: "Deep Learning Hackathon @ Amrita Vishwa Vidyapeetham",
    seed: 42,
    device: "NVIDIA RTX 4050 Edge GPU",
    throughput: "36.4 FPS",
    fullTileLatency: "< 50 seconds (10980x10980 Sentinel-2)",
    bands: "4 Bands (B04 Red, B03 Green, B02 Blue, B08 NIR)",
    resolution: "10m GSD (Ground Sample Distance)",
    weights: "ImageNet Pretrained ResNet-50 + 4-Band Conv1 Adaptation"
  },

  benchmarks: [
    {
      id: "baseline",
      name: "Mandatory Baseline",
      bands: "4-Band (RGB + NIR)",
      loss: "BCEWithLogitsLoss",
      valDice: 0.6924,
      valIoU: 0.6100,
      testDice: 0.7834,
      testIoU: 0.7044,
      testLoss: 0.3781,
      peakEpoch: 5,
      isBaseline: true,
      badge: "Mandatory Baseline Winner"
    },
    {
      id: "ablation_rgb",
      name: "Ablation Run 1 (3-Band RGB)",
      bands: "3-Band (RGB-Only)",
      loss: "Focal-Tversky Loss (α=0.7, β=0.3, γ=1.333)",
      valDice: 0.4414,
      valIoU: 0.3653,
      testDice: 0.5027,
      testIoU: 0.4245,
      testLoss: 0.5154,
      peakEpoch: 12,
      isBaseline: false,
      badge: "3-Band RGB Control"
    },
    {
      id: "ablation_rgbnir",
      name: "Ablation Run 2 (4-Band RGB+NIR)",
      bands: "4-Band (RGB + NIR)",
      loss: "Focal-Tversky Loss (α=0.7, β=0.3, γ=1.333)",
      valDice: 0.4840,
      valIoU: 0.4034,
      testDice: 0.5796,
      testIoU: 0.5013,
      testLoss: 0.5102,
      peakEpoch: 15,
      isBaseline: false,
      badge: "4-Band Spectral Boost"
    }
  ],

  spectralGain: {
    diceDelta: "+0.0769",
    dicePct: "+15.3%",
    iouDelta: "+0.0768",
    iouPct: "+18.1%",
    lossDelta: "-0.0052",
    contribution: "NIR (Band 8, 842nm) penetrates particulate pyrocumulonimbus haze and detects sudden mesophyll chlorophyll collapse where visible RGB scatters."
  },

  trainingHistory: [
    { epoch: 1, trainLoss: 0.5488, trainDice: 0.4698, trainIoU: 0.3860, valLoss: 0.6584, valDice: 0.2157, valIoU: 0.1421, isBest: false },
    { epoch: 2, trainLoss: 0.4466, trainDice: 0.6321, trainIoU: 0.5482, valLoss: 0.4880, valDice: 0.3783, valIoU: 0.2987, isBest: false },
    { epoch: 3, trainLoss: 0.4281, trainDice: 0.5896, trainIoU: 0.5062, valLoss: 0.4282, valDice: 0.5177, valIoU: 0.4340, isBest: false },
    { epoch: 4, trainLoss: 0.3944, trainDice: 0.6375, trainIoU: 0.5591, valLoss: 0.3941, valDice: 0.6749, valIoU: 0.5943, isBest: false },
    { epoch: 5, trainLoss: 0.3895, trainDice: 0.6366, trainIoU: 0.5575, valLoss: 0.3598, valDice: 0.6924, valIoU: 0.6100, isBest: true },
    { epoch: 6, trainLoss: 0.3707, trainDice: 0.6848, trainIoU: 0.6089, valLoss: 0.3604, valDice: 0.6894, valIoU: 0.6091, isBest: false },
    { epoch: 7, trainLoss: 0.3567, trainDice: 0.6617, trainIoU: 0.5899, valLoss: 0.4002, valDice: 0.6004, valIoU: 0.5201, isBest: false },
    { epoch: 8, trainLoss: 0.3437, trainDice: 0.7393, trainIoU: 0.6722, valLoss: 0.3468, valDice: 0.6878, valIoU: 0.6113, isBest: false },
    { epoch: 9, trainLoss: 0.3434, trainDice: 0.7189, trainIoU: 0.6508, valLoss: 0.3440, valDice: 0.6613, valIoU: 0.5837, isBest: false },
    { epoch: 10, trainLoss: 0.3294, trainDice: 0.7120, trainIoU: 0.6433, valLoss: 0.3383, valDice: 0.6829, valIoU: 0.6061, isBest: false },
    { epoch: 11, trainLoss: 0.3275, trainDice: 0.7227, trainIoU: 0.6559, valLoss: 0.3297, valDice: 0.6593, valIoU: 0.5859, isBest: false },
    { epoch: 12, trainLoss: 0.3231, trainDice: 0.7254, trainIoU: 0.6587, valLoss: 0.3310, valDice: 0.6516, valIoU: 0.5772, isBest: false },
    { epoch: 13, trainLoss: 0.3155, trainDice: 0.7346, trainIoU: 0.6702, valLoss: 0.3308, valDice: 0.6624, valIoU: 0.5891, isBest: false },
    { epoch: 14, trainLoss: 0.3112, trainDice: 0.7410, trainIoU: 0.6781, valLoss: 0.3312, valDice: 0.6610, valIoU: 0.5878, isBest: false },
    { epoch: 15, trainLoss: 0.3203, trainDice: 0.7282, trainIoU: 0.6620, valLoss: 0.3315, valDice: 0.6595, valIoU: 0.5862, isBest: false }
  ],

  scenes: [
    {
      id: "scene1",
      name: "Mega-Fire Scar Complex",
      tile: "b__06_06.tif",
      category: "benchmark",
      location: "Northern Mediterranean Pine Forest",
      description: "Contiguous mega-fire burn scar (>46,000 burned pixels). Displays near-perfect boundary adherence and high confidence segmentation.",
      metrics: {
        dice: 0.9317,
        iou: 0.8747,
        precision: 0.9528,
        recall: 0.9234,
        scarAreaPct: 74.01,
        scarPixels: 48507,
        gtPixels: 50053
      },
      assets: {
        rgb: "assets/samples/scene1_rgb.png",
        nir: "assets/samples/scene1_nir.png",
        pred: "assets/samples/scene1_pred.png",
        gt: "assets/samples/scene1_gt.png",
        overlay: "assets/samples/scene1_overlay.png",
        confusion: "assets/samples/scene1_confusion.png"
      }
    },
    {
      id: "scene2",
      name: "Smoke-Penetrated Active Front",
      tile: "a1__02_04.tif",
      category: "penetration",
      location: "Sierra Nevada Foothills, Mixed Conifer",
      description: "Active wildfire front obscured by dense aerosol plumes. Band 8 (NIR) penetrates particulate scattering to reveal underlying active perimeter.",
      metrics: {
        dice: 0.8283,
        iou: 0.7248,
        precision: 0.9611,
        recall: 0.7858,
        scarAreaPct: 50.26,
        scarPixels: 32941,
        gtPixels: 40286
      },
      assets: {
        rgb: "assets/samples/scene2_rgb.png",
        nir: "assets/samples/scene2_nir.png",
        pred: "assets/samples/scene2_pred.png",
        gt: "assets/samples/scene2_gt.png",
        overlay: "assets/samples/scene2_overlay.png",
        confusion: "assets/samples/scene2_confusion.png"
      }
    },
    {
      id: "scene3",
      name: "Canopy Crown Fire & Ridge",
      tile: "m1__2_5.tif",
      category: "topography",
      location: "Mountain Ridge Basin, Dense Woodland",
      description: "Severe canopy fire with complex ridge contours and preserved unburnt green enclaves. Clean avoidance of unburned forest stands.",
      metrics: {
        dice: 0.9230,
        iou: 0.8602,
        precision: 0.9607,
        recall: 0.8954,
        scarAreaPct: 57.59,
        scarPixels: 37743,
        gtPixels: 40496
      },
      assets: {
        rgb: "assets/samples/scene3_rgb.png",
        nir: "assets/samples/scene3_nir.png",
        pred: "assets/samples/scene3_pred.png",
        gt: "assets/samples/scene3_gt.png",
        overlay: "assets/samples/scene3_overlay.png",
        confusion: "assets/samples/scene3_confusion.png"
      }
    },
    {
      id: "scene4",
      name: "Fragmented Spot Fires & Mosaics",
      tile: "a2__5_3.tif",
      category: "mosaic",
      location: "Semi-Arid Scrub & Agricultural Interface",
      description: "Discontinuous wildfire mosaic across mixed vegetation types showing high sensitivity along perimeter boundaries and firebreaks.",
      metrics: {
        dice: 0.8080,
        iou: 0.7053,
        precision: 0.8069,
        recall: 0.8715,
        scarAreaPct: 46.37,
        scarPixels: 30390,
        gtPixels: 28138
      },
      assets: {
        rgb: "assets/samples/scene4_rgb.png",
        nir: "assets/samples/scene4_nir.png",
        pred: "assets/samples/scene4_pred.png",
        gt: "assets/samples/scene4_gt.png",
        overlay: "assets/samples/scene4_overlay.png",
        confusion: "assets/samples/scene4_confusion.png"
      }
    },
    {
      id: "failure_cloud",
      name: "Cloud Cover & Edge Diffraction (False Alarm)",
      tile: "a1__07_13.tif",
      category: "failure",
      location: "Coastal Range with Low Stratocumulus Clouds",
      description: "Saturated cloud albedo creates sharp contrast against dark ground, causing border diffraction and 13,269 false positive pixels.",
      metrics: {
        dice: 0.6341,
        iou: 0.4947,
        precision: 0.4959,
        recall: 0.9067,
        scarAreaPct: 40.16,
        scarPixels: 26321,
        gtPixels: 14395
      },
      assets: {
        rgb: "assets/samples/failure_cloud_rgb.png",
        nir: "assets/samples/failure_cloud_nir.png",
        pred: "assets/samples/failure_cloud_pred.png",
        gt: "assets/samples/failure_cloud_gt.png",
        overlay: "assets/samples/failure_cloud_overlay.png",
        confusion: "assets/samples/failure_cloud_confusion.png"
      }
    },
    {
      id: "failure_shadow",
      name: "Deep Mountain Ravine Shadow (Ash Mimicry)",
      tile: "k__4_6.tif",
      category: "failure",
      location: "Steep Alpine Gorge & North-Facing Ravines",
      description: "Near-zero solar illumination in deep gorges drops reflectance below 0.05, mimicking charcoal ash absorption and causing 7,211 FP pixels.",
      metrics: {
        dice: 0.3249,
        iou: 0.2266,
        precision: 0.3543,
        recall: 0.4520,
        scarAreaPct: 17.04,
        scarPixels: 11168,
        gtPixels: 8754
      },
      assets: {
        rgb: "assets/samples/failure_shadow_rgb.png",
        nir: "assets/samples/failure_shadow_nir.png",
        pred: "assets/samples/failure_shadow_pred.png",
        gt: "assets/samples/failure_shadow_gt.png",
        overlay: "assets/samples/failure_shadow_overlay.png",
        confusion: "assets/samples/failure_shadow_confusion.png"
      }
    }
  ],

  failures: [
    {
      title: "Dense Cloud Cover & Fringe Scattering",
      severity: "Moderate",
      opticalMechanism: "Thick cumulus and cirrus cloud tops reflect >85% of incoming visible and NIR radiation. At cloud perimeters, rapid light attenuation creates a steep spatial gradient that the convolutional kernels occasionally misinterpret as fresh charcoal boundaries.",
      falsePositiveImpact: "Commission errors along cloud margins (+13,269 pixels in extreme scene a1__07_13).",
      mitigation: "Ingesting ESA Copernicus Scene Classification Layer (SCL) cloud masks and pre-filtering saturated pixels (Reflectance > 0.85)."
    },
    {
      title: "Deep Topographic Shadows & North-Facing Ravines",
      severity: "High in Alpine Terrain",
      opticalMechanism: "Steep gorges and relief slopes with high solar zenith angles receive negligible direct beam irradiance. Low diffuse skylight yields raw Digital Numbers (< 500 DN, < 0.05 reflectance), creating a flat dark spectral curve indistinguishable from charred biomass in optical bands.",
      falsePositiveImpact: "Localized false alarms clustered in steep valley bottoms.",
      mitigation: "DEM-derived hillshade terrain normalization and integrating Short-Wave Infrared (SWIR Band 12, 2190nm), which exhibits distinct soil-rock ratios compared to vegetative ash."
    },
    {
      title: "Low-Biomass Fast-Moving Grassland Fires",
      severity: "Low to Moderate",
      opticalMechanism: "Rapid fire spread in dry savanna leaves fragile, thin surface soot without destroying woody crown structure. Wind rapidly disperses light ash, causing rapid spectral reversion to unburned soil signatures.",
      falsePositiveImpact: "Omission errors (False Negatives) where fire scars fade rapidly within 48-72 hours.",
      mitigation: "Bi-temporal Sentinel-2 pre-fire vs post-fire differencing (dNBR) to detect transient vegetative biomass reduction."
    }
  ]
};

// Export to window
window.ILADRISTI_DATA = ILADRISTI_DATA;
