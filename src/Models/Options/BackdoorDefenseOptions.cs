namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the backdoor detection strategy for federated learning.
/// </summary>
public enum BackdoorDetectionStrategy
{
    /// <summary>No backdoor defense — standard aggregation without detection.</summary>
    None,
    /// <summary>Neural Cleanse — reverse-engineers potential triggers via L1-norm outlier detection.</summary>
    NeuralCleanse,
    /// <summary>Direction Alignment Inspector — detects anomalous gradient directions per subspace.</summary>
    DirectionAlignmentInspector
}

/// <summary>
/// Configuration options for backdoor attack detection and mitigation in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Backdoor attacks are stealthy — the model works correctly on clean data
/// but misbehaves when a specific trigger pattern is present. Unlike Byzantine attacks (which degrade
/// overall performance), backdoors are targeted and hard to detect statistically. These detectors
/// analyze client updates for tell-tale signs of backdoor injection.</para>
/// </remarks>
public class BackdoorDefenseOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the detection strategy. Default: None.
    /// </summary>
    public BackdoorDetectionStrategy Strategy { get; set; } = BackdoorDetectionStrategy.None;

    /// <summary>
    /// Gets or sets the suspicion threshold above which a client is filtered out. Default: 0.5.
    /// </summary>
    /// <remarks>
    /// Clients with suspicion scores above this threshold have their updates excluded from aggregation.
    /// Lower thresholds are more aggressive (filter more clients).
    /// </remarks>
    public double SuspicionThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of output classes for Neural Cleanse trigger analysis. Default: 10.
    /// </summary>
    public int NumClasses { get; set; } = 10;

    /// <summary>
    /// Gets or sets the MAD-based anomaly threshold for Neural Cleanse. Default: 2.0.
    /// </summary>
    /// <remarks>
    /// A group norm that deviates more than this many MADs from the median is flagged as anomalous.
    /// </remarks>
    public double AnomalyThreshold { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the number of parameter subspaces for direction alignment analysis. Default: 10.
    /// </summary>
    public int NumSubspaces { get; set; } = 10;

    /// <summary>
    /// Gets or sets the cosine similarity threshold below which a subspace is suspicious. Default: 0.3.
    /// </summary>
    /// <remarks>
    /// If a client's gradient direction in any subspace has cosine similarity below this threshold
    /// compared to the consensus direction, that subspace is considered suspicious.
    /// </remarks>
    public double AlignmentThreshold { get; set; } = 0.3;
}
