---
title: "AnomalyDetectorType"
description: "Specifies the type of anomaly detection algorithm to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of anomaly detection algorithm to use.

## For Beginners

This enum lists all available anomaly detection algorithms.
Each category targets different types of anomalies:

## Fields

| Field | Summary |
|:-----|:--------|
| `ABOD` | Angle-Based Outlier Detection (ABOD). |
| `ARIMA` | ARIMA-based anomaly detection. |
| `AnoGAN` | AnoGAN (Anomaly GAN). |
| `AnomalyTransformer` | Anomaly Transformer. |
| `Autoencoder` | Autoencoder-based detection using reconstruction error. |
| `AverageEnsemble` | Averaging ensemble method. |
| `Bayesian` | Bayesian Network based detection. |
| `CBLOF` | Cluster-based Local Outlier Factor (CBLOF). |
| `COF` | Connectivity-based Outlier Factor (COF). |
| `COPOD` | Copula-based Outlier Detection (COPOD). |
| `ChiSquare` | Chi-Square based detection for multivariate data. |
| `DAGMM` | DAGMM (Deep Autoencoding Gaussian Mixture Model). |
| `DBSCAN` | DBSCAN-based anomaly detection. |
| `DeepSVDD` | Deep Support Vector Data Description (DeepSVDD). |
| `DevNet` | DevNet (Deep Anomaly Detection Network). |
| `DixonQTest` | Dixon's Q test for small datasets (n < 25). |
| `ECOD` | Empirical CDF Outlier Detection (ECOD). |
| `ESD` | Extreme Studentized Deviate (ESD) test. |
| `EllipticEnvelope` | Elliptic Envelope using Minimum Covariance Determinant. |
| `ExtendedIsolationForest` | Extended Isolation Forest. |
| `FairCutForest` | Fair-Cut Forest. |
| `FastABOD` | Fast approximation of ABOD. |
| `FeatureBagging` | Feature Bagging ensemble. |
| `GANomaly` | GANomaly for anomaly detection. |
| `GESD` | Generalized ESD test for detecting up to k outliers. |
| `GMM` | Gaussian Mixture Model (GMM) based detection. |
| `GrubbsTest` | Grubbs' test for detecting a single outlier. |
| `HDBSCAN` | HDBSCAN-based anomaly detection. |
| `Hampel` | Hampel identifier for outlier detection. |
| `INFLO` | Influenced Outlierness (INFLO). |
| `IQR` | Interquartile Range (IQR) method. |
| `IsolationForest` | Isolation Forest. |
| `KMeans` | K-Means based anomaly detection. |
| `KNN` | K-Nearest Neighbors based detection. |
| `KernelPCA` | Kernel PCA for non-linear anomaly detection. |
| `LDCOF` | Large Database Cluster Outlier Factor (LDCOF). |
| `LOCI` | Local Correlation Integral (LOCI). |
| `LSCP` | Locally Selective Combination of Parallel Outlier Ensembles (LSCP). |
| `LSTM` | LSTM-based anomaly detection. |
| `LoOP` | Local Outlier Probability (LoOP). |
| `LocalOutlierFactor` | Local Outlier Factor (LOF). |
| `MAD` | Median Absolute Deviation (MAD) based detection. |
| `MCD` | Minimum Covariance Determinant (MCD) based detection. |
| `MatrixProfile` | Matrix Profile for time series. |
| `MaximumEnsemble` | Maximum ensemble method. |
| `ModifiedZScore` | Modified Z-Score using Median Absolute Deviation (MAD). |
| `MovingAverage` | Moving Average based anomaly detection. |
| `NBEATS` | N-BEATS for anomaly detection. |
| `OneClassSVM` | One-Class SVM. |
| `PCA` | PCA-based detection using reconstruction error. |
| `Percentile` | Percentile-based outlier detection. |
| `RandomSubspace` | Random Subspace ensemble. |
| `RobustPCA` | Robust PCA using outlier-resistant decomposition. |
| `SCiForest` | SCiForest (Split-Selection Criterion Isolation Forest). |
| `SOS` | Stochastic Outlier Selection (SOS). |
| `STL` | STL decomposition-based detection. |
| `SUOD` | Scalable Unsupervised Outlier Detection (SUOD). |
| `SeasonalHybridESD` | Seasonal-Hybrid ESD (S-H-ESD) for seasonal time series. |
| `SpectralResidual` | Spectral Residual method. |
| `VAE` | Variational Autoencoder (VAE) for anomaly detection. |
| `XGBOD` | XGBoost-based Outlier Detection (XGBOD). |
| `ZScore` | Z-Score based detection. |

