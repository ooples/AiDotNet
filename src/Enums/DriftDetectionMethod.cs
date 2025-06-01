namespace AiDotNet.Enums;

/// <summary>
/// Specifies the method used for detecting concept drift in data streams.
/// </summary>
public enum DriftDetectionMethod
{
    /// <summary>
    /// No drift detection.
    /// </summary>
    None,
    
    /// <summary>
    /// Drift Detection Method (DDM) - Monitors error rate.
    /// </summary>
    DDM,
    
    /// <summary>
    /// Early Drift Detection Method (EDDM) - Monitors distance between errors.
    /// </summary>
    EDDM,
    
    /// <summary>
    /// ADaptive WINdowing (ADWIN) - Adaptive sliding window approach.
    /// </summary>
    ADWIN,
    
    /// <summary>
    /// Page-Hinkley Test - Sequential analysis technique.
    /// </summary>
    PageHinkley,
    
    /// <summary>
    /// Hoeffding's Bound with Drift Detection (HDDM).
    /// </summary>
    HDDM,
    
    /// <summary>
    /// Statistical Test of Equal Proportions (STEPD).
    /// </summary>
    STEPD,
    
    /// <summary>
    /// Paired Learners - Uses two models to detect drift.
    /// </summary>
    PairedLearners,
    
    /// <summary>
    /// CUSUM - Cumulative Sum control charts.
    /// </summary>
    CUSUM,
    
    /// <summary>
    /// Geometric Moving Average test.
    /// </summary>
    GeometricMovingAverage,
    
    /// <summary>
    /// Kolmogorov-Smirnov test for drift detection.
    /// </summary>
    KolmogorovSmirnov,
    
    /// <summary>
    /// Wasserstein Distance based drift detection.
    /// </summary>
    WassersteinDistance,
    
    /// <summary>
    /// Maximum Mean Discrepancy for drift detection.
    /// </summary>
    MaximumMeanDiscrepancy,
    
    /// <summary>
    /// Reactive Drift Detection Method.
    /// </summary>
    ReactiveDrift,
    
    /// <summary>
    /// Fisher's Exact Test for drift detection.
    /// </summary>
    FisherExactTest,
    
    /// <summary>
    /// Ensemble-based drift detection using multiple methods.
    /// </summary>
    EnsembleDrift
}