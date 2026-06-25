---
title: "Anomaly Detection"
description: "All 68 public types in the AiDotNet.anomalydetection namespace, organized by kind."
section: "API Reference"
---

**68** public types in this namespace, organized by kind.

## Models & Types (65)

| Type | Summary |
|:-----|:--------|
| [`ABODDetector<T>`](/docs/reference/wiki/anomalydetection/aboddetector/) | Detects anomalies using Angle-Based Outlier Detection (ABOD). |
| [`ARIMADetector<T>`](/docs/reference/wiki/anomalydetection/arimadetector/) | Detects anomalies in time series using ARIMA model residuals. |
| [`AnoGANDetector<T>`](/docs/reference/wiki/anomalydetection/anogandetector/) | Implements AnoGAN (Anomaly Detection with Generative Adversarial Networks). |
| [`AnomalyTransformerDetector<T>`](/docs/reference/wiki/anomalydetection/anomalytransformerdetector/) | Implements Anomaly Transformer for time series anomaly detection. |
| [`AutoencoderDetector<T>`](/docs/reference/wiki/anomalydetection/autoencoderdetector/) | Implements an Autoencoder-based method for anomaly detection using reconstruction error. |
| [`AveragingDetector<T>`](/docs/reference/wiki/anomalydetection/averagingdetector/) | Combines multiple anomaly detectors using score averaging. |
| [`BayesianDetector<T>`](/docs/reference/wiki/anomalydetection/bayesiandetector/) | Detects anomalies using Bayesian probability estimation. |
| [`CBLOFDetector<T>`](/docs/reference/wiki/anomalydetection/cblofdetector/) | Detects anomalies using Cluster-Based Local Outlier Factor (CBLOF). |
| [`COFDetector<T>`](/docs/reference/wiki/anomalydetection/cofdetector/) | Detects anomalies using Connectivity-Based Outlier Factor (COF). |
| [`COPODDetector<T>`](/docs/reference/wiki/anomalydetection/copoddetector/) | Detects anomalies using Copula-Based Outlier Detection (COPOD). |
| [`ChiSquareDetector<T>`](/docs/reference/wiki/anomalydetection/chisquaredetector/) | Detects anomalies using the Chi-Square test for multivariate data. |
| [`DAGMMDetector<T>`](/docs/reference/wiki/anomalydetection/dagmmdetector/) | Implements DAGMM (Deep Autoencoding Gaussian Mixture Model) for anomaly detection. |
| [`DBSCANDetector<T>`](/docs/reference/wiki/anomalydetection/dbscandetector/) | Detects anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise). |
| [`DeepSVDDDetector<T>`](/docs/reference/wiki/anomalydetection/deepsvdddetector/) | Detects anomalies using Deep SVDD (Support Vector Data Description). |
| [`DevNetDetector<T>`](/docs/reference/wiki/anomalydetection/devnetdetector/) | Implements DevNet (Deep Anomaly Detection Network) for end-to-end anomaly scoring. |
| [`DixonQTestDetector<T>`](/docs/reference/wiki/anomalydetection/dixonqtestdetector/) | Detects anomalies using Dixon's Q Test for small datasets. |
| [`ECODDetector<T>`](/docs/reference/wiki/anomalydetection/ecoddetector/) | Detects anomalies using ECOD (Empirical Cumulative Distribution Functions for Outlier Detection). |
| [`ESDDetector<T>`](/docs/reference/wiki/anomalydetection/esddetector/) | Detects anomalies using ESD-based (Extreme Studentized Deviate) scoring. |
| [`EllipticEnvelopeDetector<T>`](/docs/reference/wiki/anomalydetection/ellipticenvelopedetector/) | Detects anomalies using Elliptic Envelope (robust covariance estimation). |
| [`ExtendedIsolationForest<T>`](/docs/reference/wiki/anomalydetection/extendedisolationforest/) | Detects anomalies using Extended Isolation Forest (EIF). |
| [`FairCutForest<T>`](/docs/reference/wiki/anomalydetection/faircutforest/) | Implements Fair-Cut Forest (FCF) for anomaly detection with balanced tree construction. |
| [`FastABODDetector<T>`](/docs/reference/wiki/anomalydetection/fastaboddetector/) | Detects anomalies using Fast Angle-Based Outlier Detection (FastABOD). |
| [`FeatureBaggingDetector<T>`](/docs/reference/wiki/anomalydetection/featurebaggingdetector/) | Detects anomalies using Feature Bagging ensemble method. |
| [`GANomalyDetector<T>`](/docs/reference/wiki/anomalydetection/ganomalydetector/) | Implements GANomaly for anomaly detection using GAN-based reconstruction. |
| [`GESDDetector<T>`](/docs/reference/wiki/anomalydetection/gesddetector/) | Detects anomalies using GESD-based (Generalized Extreme Studentized Deviate) scoring. |
| [`GMMDetector<T>`](/docs/reference/wiki/anomalydetection/gmmdetector/) | Detects anomalies using Gaussian Mixture Models (GMM). |
| [`GrubbsTestDetector<T>`](/docs/reference/wiki/anomalydetection/grubbstestdetector/) | Detects anomalies using Grubbs' Test for a single outlier. |
| [`HDBSCANDetector<T>`](/docs/reference/wiki/anomalydetection/hdbscandetector/) | Detects anomalies using HDBSCAN (Hierarchical DBSCAN). |
| [`HampelDetector<T>`](/docs/reference/wiki/anomalydetection/hampeldetector/) | Detects anomalies using Hampel identifier (median-based outlier detection). |
| [`INFLODetector<T>`](/docs/reference/wiki/anomalydetection/inflodetector/) | Detects anomalies using Influenced Outlierness (INFLO). |
| [`IQRDetector<T>`](/docs/reference/wiki/anomalydetection/iqrdetector/) | Detects anomalies using the Interquartile Range (IQR) method. |
| [`IsolationForest<T>`](/docs/reference/wiki/anomalydetection/isolationforest/) | Implements the Isolation Forest algorithm for anomaly detection. |
| [`KMeansDetector<T>`](/docs/reference/wiki/anomalydetection/kmeansdetector/) | Detects anomalies using K-Means clustering distance. |
| [`KNNDetector<T>`](/docs/reference/wiki/anomalydetection/knndetector/) | Detects anomalies using K-Nearest Neighbors distance. |
| [`KernelPCADetector<T>`](/docs/reference/wiki/anomalydetection/kernelpcadetector/) | Detects anomalies using Kernel PCA reconstruction error. |
| [`LDCOFDetector<T>`](/docs/reference/wiki/anomalydetection/ldcofdetector/) | Detects anomalies using LDCOF (Local Density Cluster-Based Outlier Factor). |
| [`LOCIDetector<T>`](/docs/reference/wiki/anomalydetection/locidetector/) | Detects anomalies using Local Correlation Integral (LOCI). |
| [`LSCPDetector<T>`](/docs/reference/wiki/anomalydetection/lscpdetector/) | Detects anomalies using LSCP (Locally Selective Combination in Parallel Outlier Ensembles). |
| [`LSTMDetector<T>`](/docs/reference/wiki/anomalydetection/lstmdetector/) | Implements LSTM-based anomaly detection using prediction error. |
| [`LoOPDetector<T>`](/docs/reference/wiki/anomalydetection/loopdetector/) | Detects anomalies using Local Outlier Probability (LoOP). |
| [`LocalOutlierFactor<T>`](/docs/reference/wiki/anomalydetection/localoutlierfactor/) | Implements the Local Outlier Factor (LOF) algorithm for density-based anomaly detection. |
| [`MADDetector<T>`](/docs/reference/wiki/anomalydetection/maddetector/) | Detects anomalies using Median Absolute Deviation (MAD). |
| [`MCDDetector<T>`](/docs/reference/wiki/anomalydetection/mcddetector/) | Detects anomalies using Minimum Covariance Determinant (MCD). |
| [`MatrixProfileDetector<T>`](/docs/reference/wiki/anomalydetection/matrixprofiledetector/) | Detects anomalies using Matrix Profile for time series discord detection. |
| [`MaximumDetector<T>`](/docs/reference/wiki/anomalydetection/maximumdetector/) | Combines multiple anomaly detectors using maximum score strategy. |
| [`ModifiedZScoreDetector<T>`](/docs/reference/wiki/anomalydetection/modifiedzscoredetector/) | Detects anomalies using the Modified Z-Score method based on Median Absolute Deviation (MAD). |
| [`MovingAverageDetector<T>`](/docs/reference/wiki/anomalydetection/movingaveragedetector/) | Detects anomalies using moving average deviation in time series. |
| [`NBEATSDetector<T>`](/docs/reference/wiki/anomalydetection/nbeatsdetector/) | Implements N-BEATS (Neural Basis Expansion Analysis for Time Series) for anomaly detection. |
| [`NoOutlierRemoval<T, TInput, TOutput>`](/docs/reference/wiki/anomalydetection/nooutlierremoval/) | A no-operation outlier removal implementation that preserves all data without modification. |
| [`OCSVMDetector<T>`](/docs/reference/wiki/anomalydetection/ocsvmdetector/) | Detects anomalies using One-Class SVM with simplified SGD training. |
| [`OneClassSVM<T>`](/docs/reference/wiki/anomalydetection/oneclasssvm/) | Implements One-Class SVM for novelty/outlier detection using the RBF kernel. |
| [`OutlierRemovalAdapter<T, TInput, TOutput>`](/docs/reference/wiki/anomalydetection/outlierremovaladapter/) | Adapts any `IAnomalyDetector` to the legacy `IOutlierRemoval` interface. |
| [`PCADetector<T>`](/docs/reference/wiki/anomalydetection/pcadetector/) | Detects anomalies using Principal Component Analysis (PCA) reconstruction error. |
| [`PercentileDetector<T>`](/docs/reference/wiki/anomalydetection/percentiledetector/) | Detects anomalies using percentile-based thresholds. |
| [`RandomSubspaceDetector<T>`](/docs/reference/wiki/anomalydetection/randomsubspacedetector/) | Detects anomalies using Random Subspace ensemble method. |
| [`RobustPCADetector<T>`](/docs/reference/wiki/anomalydetection/robustpcadetector/) | Detects anomalies using Robust PCA (Principal Component Pursuit). |
| [`SCiForest<T>`](/docs/reference/wiki/anomalydetection/sciforest/) | Detects anomalies using SCiForest (Sparse Clustering-Integrated Isolation Forest). |
| [`SOSDetector<T>`](/docs/reference/wiki/anomalydetection/sosdetector/) | Detects anomalies using SOS (Stochastic Outlier Selection). |
| [`STLDetector<T>`](/docs/reference/wiki/anomalydetection/stldetector/) | Detects anomalies using STL (Seasonal and Trend decomposition using Loess). |
| [`SUODDetector<T>`](/docs/reference/wiki/anomalydetection/suoddetector/) | Detects anomalies using SUOD (Scalable Unsupervised Outlier Detection). |
| [`SeasonalHybridESDDetector<T>`](/docs/reference/wiki/anomalydetection/seasonalhybridesddetector/) | Detects anomalies in time series data using Seasonal Hybrid ESD (S-H-ESD). |
| [`SpectralResidualDetector<T>`](/docs/reference/wiki/anomalydetection/spectralresidualdetector/) | Detects anomalies in time series using Spectral Residual method. |
| [`VAEDetector<T>`](/docs/reference/wiki/anomalydetection/vaedetector/) | Detects anomalies using Variational Autoencoder (VAE). |
| [`XGBODDetector<T>`](/docs/reference/wiki/anomalydetection/xgboddetector/) | Detects anomalies using XGBOD (Extreme Gradient Boosting Outlier Detection). |
| [`ZScoreDetector<T>`](/docs/reference/wiki/anomalydetection/zscoredetector/) | Detects anomalies using the Z-Score method (standard score). |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`AnomalyDetectorBase<T>`](/docs/reference/wiki/anomalydetection/anomalydetectorbase/) | Base class for algorithmic anomaly detectors providing common functionality. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`CombinationMethod<T>`](/docs/reference/wiki/anomalydetection/combinationmethod/) | Method for combining detector scores. |
| [`KernelType<T>`](/docs/reference/wiki/anomalydetection/kerneltype/) | Type of kernel function. |

