namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the drift detection method used in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These methods detect when client data distributions change over time.
/// Each has different strengths — some catch sudden changes quickly (PageHinkley, DDM), while
/// others are better at gradual drift (ADWIN). Model-based methods (GradientDivergence, WeightDivergence)
/// detect drift through changes in training behavior rather than raw statistics.</para>
/// </remarks>
public enum FederatedDriftMethod
{
    /// <summary>
    /// Page-Hinkley test: sequential analysis for detecting change in the mean of a process.
    /// Good at detecting sudden and gradual drift with configurable sensitivity.
    /// </summary>
    PageHinkley,

    /// <summary>
    /// ADWIN (Adaptive Windowing): maintains a variable-length window and detects drift
    /// when two sub-windows have significantly different means. Self-adapting to drift speed.
    /// </summary>
    ADWIN,

    /// <summary>
    /// DDM (Drift Detection Method): monitors error rate and standard deviation.
    /// Triggers warning when error increases by 2 sigma, drift at 3 sigma.
    /// </summary>
    DDM,

    /// <summary>
    /// Gradient divergence: detects drift by comparing gradient directions between rounds.
    /// Large angular changes in gradient direction suggest distribution shift.
    /// </summary>
    GradientDivergence,

    /// <summary>
    /// Weight divergence: detects drift by measuring how much model weights change
    /// relative to historical patterns. Abnormal weight changes signal drift.
    /// </summary>
    WeightDivergence
}
