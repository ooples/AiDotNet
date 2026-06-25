---
title: "Metrics"
description: "All 27 public types in the AiDotNet.metrics namespace, organized by kind."
section: "API Reference"
---

**27** public types in this namespace, organized by kind.

## Models & Types (25)

| Type | Summary |
|:-----|:--------|
| [`AestheticScore<T>`](/docs/reference/wiki/metrics/aestheticscore/) | Aesthetic Score metric using CLIP for evaluating image aesthetics. |
| [`CLIPScore<T>`](/docs/reference/wiki/metrics/clipscore/) | CLIPScore metric for evaluating text-image alignment and image quality. |
| [`ChamferDistance<T>`](/docs/reference/wiki/metrics/chamferdistance/) | Chamfer Distance metric for 3D point cloud comparison. |
| [`CharacterErrorRate`](/docs/reference/wiki/metrics/charactererrorrate/) | Character Error Rate (CER) metric for speech recognition and OCR evaluation. |
| [`EarthMoversDistance<T>`](/docs/reference/wiki/metrics/earthmoversdistance/) | Earth Mover's Distance (EMD) / Wasserstein Distance for point cloud comparison. |
| [`FScore<T>`](/docs/reference/wiki/metrics/fscore/) | F-Score metric for 3D reconstruction evaluation. |
| [`FrechetInceptionDistance<T>`](/docs/reference/wiki/metrics/frechetinceptiondistance/) | Fréchet Inception Distance (FID) - A metric for evaluating the quality of generated images. |
| [`FrechetVideoDistance<T>`](/docs/reference/wiki/metrics/frechetvideodistance/) | Fréchet Video Distance (FVD) - A metric for evaluating the quality of generated videos. |
| [`InceptionScore<T>`](/docs/reference/wiki/metrics/inceptionscore/) |  |
| [`IoU3D<T>`](/docs/reference/wiki/metrics/iou3d/) | 3D Intersection over Union (3D IoU) for voxel and bounding box evaluation. |
| [`KernelInceptionDistance<T>`](/docs/reference/wiki/metrics/kernelinceptiondistance/) | Kernel Inception Distance (KID) - A metric for evaluating the quality of generated images. |
| [`MeanIntersectionOverUnion<T>`](/docs/reference/wiki/metrics/meanintersectionoverunion/) | Mean Intersection over Union (mIoU) metric for segmentation tasks. |
| [`OverallAccuracy<T>`](/docs/reference/wiki/metrics/overallaccuracy/) | Overall Accuracy metric for classification and segmentation. |
| [`PeakSignalToNoiseRatio<T>`](/docs/reference/wiki/metrics/peaksignaltonoiseratio/) | Peak Signal-to-Noise Ratio (PSNR) metric for image quality assessment. |
| [`PerceptualSpeechQuality<T>`](/docs/reference/wiki/metrics/perceptualspeechquality/) | Perceptual Evaluation of Speech Quality (PESQ) approximation metric. |
| [`ScaleInvariantSignalToDistortionRatio<T>`](/docs/reference/wiki/metrics/scaleinvariantsignaltodistortionratio/) | Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric for source separation evaluation. |
| [`ShortTimeObjectiveIntelligibility<T>`](/docs/reference/wiki/metrics/shorttimeobjectiveintelligibility/) | Short-Time Objective Intelligibility (STOI) metric for speech intelligibility assessment. |
| [`SignalToNoiseRatio<T>`](/docs/reference/wiki/metrics/signaltonoiseratio/) | Signal-to-Noise Ratio (SNR) metric for audio quality assessment. |
| [`StructuralSimilarity<T>`](/docs/reference/wiki/metrics/structuralsimilarity/) | Structural Similarity Index Measure (SSIM) for image quality assessment. |
| [`TemporalConsistency<T>`](/docs/reference/wiki/metrics/temporalconsistency/) | Temporal Consistency metric for evaluating video smoothness and coherence. |
| [`VideoPSNR<T>`](/docs/reference/wiki/metrics/videopsnr/) | Video Peak Signal-to-Noise Ratio (VPSNR) - Frame-averaged PSNR for video quality. |
| [`VideoQualityIndex<T>`](/docs/reference/wiki/metrics/videoqualityindex/) | Video Quality Index (VQI) - A composite metric combining multiple video quality aspects. |
| [`VideoQualityResult<T>`](/docs/reference/wiki/metrics/videoqualityresult/) | Results from comprehensive video quality evaluation. |
| [`VideoSSIM<T>`](/docs/reference/wiki/metrics/videossim/) | Video Structural Similarity (VSSIM) - Frame-averaged SSIM for video quality. |
| [`WordErrorRate`](/docs/reference/wiki/metrics/worderrorrate/) | Word Error Rate (WER) metric for speech recognition evaluation. |

## Enums (1)

| Type | Summary |
|:-----|:--------|
| [`FrameSamplingStrategy`](/docs/reference/wiki/metrics/framesamplingstrategy/) | Frame sampling strategies for video feature extraction. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`RankingMetrics<T>`](/docs/reference/wiki/metrics/rankingmetrics/) | Static helpers for evaluating the quality of a ranking, most notably Normalized Discounted Cumulative Gain (NDCG@k). |

