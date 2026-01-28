namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of anomaly detection algorithm to use.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum lists all available anomaly detection algorithms.
/// Each category targets different types of anomalies:
/// </para>
/// <list type="bullet">
/// <item><b>Statistical</b>: Simple, interpretable methods for univariate or low-dimensional data</item>
/// <item><b>Distance/Density</b>: Detect anomalies based on local neighborhood density</item>
/// <item><b>Tree-based</b>: Fast, scalable methods for high-dimensional data</item>
/// <item><b>Linear</b>: Methods assuming anomalies lie outside a learned boundary</item>
/// <item><b>Neural Network</b>: Deep learning methods for complex patterns</item>
/// <item><b>Time Series</b>: Specialized methods for temporal data</item>
/// <item><b>Ensemble</b>: Combine multiple detectors for robust detection</item>
/// <item><b>Probabilistic</b>: Model-based approaches using probability distributions</item>
/// <item><b>Angle-based</b>: Methods using angular variance for outlier detection</item>
/// </list>
/// </remarks>
public enum AnomalyDetectorType
{
    // ==========================================
    // Statistical Methods (Simple, Interpretable)
    // ==========================================

    /// <summary>
    /// Z-Score based detection. Assumes normal distribution.
    /// Industry standard threshold: 3.0 (flags ~0.3% of normal data).
    /// </summary>
    ZScore,

    /// <summary>
    /// Modified Z-Score using Median Absolute Deviation (MAD).
    /// More robust than Z-Score for non-normal or heavily contaminated data.
    /// Industry standard threshold: 3.5 (Iglewicz and Hoaglin).
    /// </summary>
    ModifiedZScore,

    /// <summary>
    /// Interquartile Range (IQR) method.
    /// Robust, non-parametric. Same method used in box plots.
    /// Industry standard multiplier: 1.5 (mild outliers) or 3.0 (extreme).
    /// </summary>
    IQR,

    /// <summary>
    /// Grubbs' test for detecting a single outlier.
    /// Assumes normal distribution. Tests one point at a time.
    /// </summary>
    GrubbsTest,

    /// <summary>
    /// Dixon's Q test for small datasets (n &lt; 25).
    /// Simple ratio test for the most extreme value.
    /// </summary>
    DixonQTest,

    /// <summary>
    /// Extreme Studentized Deviate (ESD) test.
    /// Iteratively identifies outliers. Good for multiple outliers.
    /// </summary>
    ESD,

    /// <summary>
    /// Generalized ESD test for detecting up to k outliers.
    /// Extension of ESD for a known maximum number of outliers.
    /// </summary>
    GESD,

    /// <summary>
    /// Chi-Square based detection for multivariate data.
    /// Assumes multivariate normal distribution.
    /// </summary>
    ChiSquare,

    // ==========================================
    // Distance/Density-based Methods
    // ==========================================

    /// <summary>
    /// K-Nearest Neighbors based detection.
    /// Anomalies have larger distances to their k nearest neighbors.
    /// </summary>
    KNN,

    /// <summary>
    /// Local Outlier Factor (LOF).
    /// Compares local density to neighbors' densities.
    /// Industry standard k: 20 neighbors.
    /// </summary>
    LocalOutlierFactor,

    /// <summary>
    /// Local Correlation Integral (LOCI).
    /// Automatically determines the optimal radius.
    /// </summary>
    LOCI,

    /// <summary>
    /// Connectivity-based Outlier Factor (COF).
    /// Uses shortest path distances instead of Euclidean.
    /// </summary>
    COF,

    /// <summary>
    /// Influenced Outlierness (INFLO).
    /// Considers both k-nearest neighbors and reverse k-nearest neighbors.
    /// </summary>
    INFLO,

    /// <summary>
    /// Local Outlier Probability (LoOP).
    /// Returns probability of being an outlier (0 to 1).
    /// </summary>
    LoOP,

    /// <summary>
    /// DBSCAN-based anomaly detection.
    /// Points not in any cluster are anomalies.
    /// </summary>
    DBSCAN,

    /// <summary>
    /// HDBSCAN-based anomaly detection.
    /// Hierarchical DBSCAN with automatic parameter selection.
    /// </summary>
    HDBSCAN,

    /// <summary>
    /// K-Means based anomaly detection.
    /// Anomalies are far from cluster centroids.
    /// </summary>
    KMeans,

    /// <summary>
    /// Cluster-based Local Outlier Factor (CBLOF).
    /// Combines clustering with local outlier factor.
    /// </summary>
    CBLOF,

    /// <summary>
    /// Large Database Cluster Outlier Factor (LDCOF).
    /// Optimized CBLOF for large datasets.
    /// </summary>
    LDCOF,

    /// <summary>
    /// Stochastic Outlier Selection (SOS).
    /// Uses affinity-based probabilities for outlier detection.
    /// </summary>
    SOS,

    // ==========================================
    // Tree-based Methods (Fast, Scalable)
    // ==========================================

    /// <summary>
    /// Isolation Forest.
    /// Industry standard for high-dimensional anomaly detection.
    /// Defaults: 100 trees, 256 samples per tree.
    /// Reference: Liu et al., ICDM 2008.
    /// </summary>
    IsolationForest,

    /// <summary>
    /// Extended Isolation Forest.
    /// Uses hyperplanes instead of axis-aligned splits.
    /// Better for multi-dimensional anomalies.
    /// </summary>
    ExtendedIsolationForest,

    /// <summary>
    /// SCiForest (Split-Selection Criterion Isolation Forest).
    /// Improved split selection for better anomaly isolation.
    /// </summary>
    SCiForest,

    /// <summary>
    /// Fair-Cut Forest.
    /// Balanced tree construction for improved detection.
    /// </summary>
    FairCutForest,

    // ==========================================
    // Linear/Projection Methods
    // ==========================================

    /// <summary>
    /// One-Class SVM.
    /// Learns a boundary around normal data.
    /// Good for novelty detection.
    /// </summary>
    OneClassSVM,

    /// <summary>
    /// PCA-based detection using reconstruction error.
    /// Anomalies have high reconstruction error.
    /// </summary>
    PCA,

    /// <summary>
    /// Robust PCA using outlier-resistant decomposition.
    /// Separates low-rank structure from sparse anomalies.
    /// </summary>
    RobustPCA,

    /// <summary>
    /// Elliptic Envelope using Minimum Covariance Determinant.
    /// Fits a robust ellipse around normal data.
    /// </summary>
    EllipticEnvelope,

    /// <summary>
    /// Kernel PCA for non-linear anomaly detection.
    /// Projects data to higher-dimensional space.
    /// </summary>
    KernelPCA,

    // ==========================================
    // Neural Network-based Methods
    // ==========================================

    /// <summary>
    /// Autoencoder-based detection using reconstruction error.
    /// Anomalies have high reconstruction error.
    /// </summary>
    Autoencoder,

    /// <summary>
    /// Variational Autoencoder (VAE) for anomaly detection.
    /// Uses reconstruction probability instead of error.
    /// </summary>
    VAE,

    /// <summary>
    /// Deep Support Vector Data Description (DeepSVDD).
    /// Deep learning version of SVDD.
    /// Reference: Ruff et al., ICML 2018.
    /// </summary>
    DeepSVDD,

    /// <summary>
    /// GANomaly for anomaly detection.
    /// Uses GAN-based reconstruction for detection.
    /// </summary>
    GANomaly,

    /// <summary>
    /// AnoGAN (Anomaly GAN).
    /// Original GAN-based anomaly detection method.
    /// </summary>
    AnoGAN,

    /// <summary>
    /// DevNet (Deep Anomaly Detection Network).
    /// End-to-end deep learning for anomaly scoring.
    /// </summary>
    DevNet,

    /// <summary>
    /// DAGMM (Deep Autoencoding Gaussian Mixture Model).
    /// Combines autoencoder with GMM in end-to-end training.
    /// </summary>
    DAGMM,

    // ==========================================
    // Time Series Methods
    // ==========================================

    /// <summary>
    /// Seasonal-Hybrid ESD (S-H-ESD) for seasonal time series.
    /// Decomposes time series and applies ESD to residuals.
    /// Used by Twitter for anomaly detection.
    /// </summary>
    SeasonalHybridESD,

    /// <summary>
    /// ARIMA-based anomaly detection.
    /// Uses prediction errors from ARIMA model.
    /// </summary>
    ARIMA,

    /// <summary>
    /// STL decomposition-based detection.
    /// Anomalies in the remainder component after decomposition.
    /// </summary>
    STL,

    /// <summary>
    /// Matrix Profile for time series.
    /// Finds anomalous subsequences using STAMP/STOMP algorithm.
    /// </summary>
    MatrixProfile,

    /// <summary>
    /// LSTM-based anomaly detection.
    /// Uses prediction error from LSTM network.
    /// </summary>
    LSTM,

    /// <summary>
    /// Spectral Residual method.
    /// Detects anomalies in the frequency domain.
    /// Used by Microsoft for Azure anomaly detection.
    /// </summary>
    SpectralResidual,

    /// <summary>
    /// Anomaly Transformer.
    /// Transformer-based architecture for time series anomaly detection.
    /// </summary>
    AnomalyTransformer,

    /// <summary>
    /// N-BEATS for anomaly detection.
    /// Uses N-BEATS forecasting model for anomaly scoring.
    /// </summary>
    NBEATS,

    // ==========================================
    // Ensemble Methods
    // ==========================================

    /// <summary>
    /// Feature Bagging ensemble.
    /// Trains multiple detectors on random feature subsets.
    /// </summary>
    FeatureBagging,

    /// <summary>
    /// Locally Selective Combination of Parallel Outlier Ensembles (LSCP).
    /// Selects best detector for each region of the data.
    /// </summary>
    LSCP,

    /// <summary>
    /// XGBoost-based Outlier Detection (XGBOD).
    /// Uses anomaly scores from multiple detectors as features for XGBoost.
    /// </summary>
    XGBOD,

    /// <summary>
    /// Scalable Unsupervised Outlier Detection (SUOD).
    /// Fast ensemble method optimized for large datasets.
    /// </summary>
    SUOD,

    /// <summary>
    /// Random Subspace ensemble.
    /// Projects to random subspaces and combines scores.
    /// </summary>
    RandomSubspace,

    // ==========================================
    // Probabilistic Methods
    // ==========================================

    /// <summary>
    /// Gaussian Mixture Model (GMM) based detection.
    /// Anomalies have low likelihood under the GMM.
    /// </summary>
    GMM,

    /// <summary>
    /// Copula-based Outlier Detection (COPOD).
    /// Uses copula to model dependencies between features.
    /// Fast, parameter-free method.
    /// </summary>
    COPOD,

    /// <summary>
    /// Empirical CDF Outlier Detection (ECOD).
    /// Uses empirical cumulative distribution functions.
    /// Fast, parameter-free method.
    /// </summary>
    ECOD,

    /// <summary>
    /// Bayesian Network based detection.
    /// Models dependencies using a Bayesian network.
    /// </summary>
    Bayesian,

    // ==========================================
    // Angle-based Methods
    // ==========================================

    /// <summary>
    /// Angle-Based Outlier Detection (ABOD).
    /// Uses variance of angles to neighbors for detection.
    /// Good for high-dimensional data.
    /// </summary>
    ABOD,

    /// <summary>
    /// Fast approximation of ABOD.
    /// Uses only k nearest neighbors for speed.
    /// </summary>
    FastABOD
}
