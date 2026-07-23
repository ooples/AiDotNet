namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for federated concept drift detection and adaptation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In production, client data changes over time (e.g., fraud patterns
/// evolve, user behavior shifts seasonally). These options configure how the FL system detects
/// when client data has drifted from the training distribution and how it adapts.</para>
/// </remarks>
public class FederatedDriftOptions
{
    /// <summary>
    /// Gets or sets whether drift detection is enabled. Default is false.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the drift detection method. Default is ADWIN.
    /// </summary>
    public FederatedDriftMethod Method { get; set; } = FederatedDriftMethod.ADWIN;

    /// <summary>
    /// Gets or sets the sensitivity threshold for drift detection.
    /// Lower values = more sensitive (catch drift earlier but more false positives).
    /// Default is 0.01.
    /// </summary>
    public double SensitivityThreshold { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the number of recent rounds to consider for drift analysis.
    /// Default is 20.
    /// </summary>
    public int LookbackWindowRounds { get; set; } = 20;

    /// <summary>
    /// Gets or sets whether to automatically adapt aggregation weights based on drift scores.
    /// Stable clients get higher weight; drifting clients get lower weight.
    /// Default is true.
    /// </summary>
    public bool AdaptAggregationWeights { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum weight multiplier for heavily drifting clients.
    /// Prevents drifting clients from being completely excluded. Default is 0.1 (10% of normal).
    /// </summary>
    public double MinDriftWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets how often to run drift detection (in rounds).
    /// Default is 1 (every round).
    /// </summary>
    public int DetectionFrequency { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to trigger selective retraining when global drift is detected.
    /// Default is false.
    /// </summary>
    public bool TriggerSelectiveRetraining { get; set; } = false;

    /// <summary>
    /// Gets or sets the global drift threshold above which all clients are notified.
    /// Default is 0.3 (30% of clients drifting = global drift).
    /// </summary>
    public double GlobalDriftThreshold { get; set; } = 0.3;
}
