using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.DriftDetection;

/// <summary>
/// Drift classification for a client's data distribution.
/// </summary>
public enum DriftType
{
    /// <summary>No drift detected.</summary>
    None,

    /// <summary>Warning: drift may be imminent.</summary>
    Warning,

    /// <summary>Sudden drift: abrupt change in distribution.</summary>
    Sudden,

    /// <summary>Gradual drift: slow transition between concepts.</summary>
    Gradual,

    /// <summary>Recurring drift: previously seen distribution returning.</summary>
    Recurring
}

/// <summary>
/// Recommended action for a drifting client.
/// </summary>
public enum DriftAction
{
    /// <summary>No action needed.</summary>
    None,

    /// <summary>Monitor more closely (increase detection frequency).</summary>
    Monitor,

    /// <summary>Reduce aggregation weight for this client.</summary>
    ReduceWeight,

    /// <summary>Request selective retraining from this client.</summary>
    SelectiveRetrain,

    /// <summary>Exclude this client temporarily until drift stabilizes.</summary>
    TemporaryExclude
}

/// <summary>
/// Per-client drift analysis result.
/// </summary>
public class ClientDriftResult
{
    /// <summary>
    /// Gets or sets the client ID.
    /// </summary>
    public int ClientId { get; set; }

    /// <summary>
    /// Gets or sets the drift score for this client. Range [0, 1]: 0 = no drift, 1 = severe drift.
    /// </summary>
    public double DriftScore { get; set; }

    /// <summary>
    /// Gets or sets the classified drift type.
    /// </summary>
    public DriftType DriftType { get; set; } = DriftType.None;

    /// <summary>
    /// Gets or sets the recommended action.
    /// </summary>
    public DriftAction RecommendedAction { get; set; } = DriftAction.None;

    /// <summary>
    /// Gets or sets the round at which drift was first detected for this client.
    /// -1 if no drift detected.
    /// </summary>
    public int DriftStartRound { get; set; } = -1;

    /// <summary>
    /// Gets or sets the suggested aggregation weight multiplier for this client.
    /// 1.0 = normal weight, less than 1.0 = reduced due to drift.
    /// </summary>
    public double SuggestedWeightMultiplier { get; set; } = 1.0;
}

/// <summary>
/// Comprehensive report of drift analysis across all federated clients.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After running drift detection, this report tells you which clients
/// are experiencing data distribution shifts, how severe they are, and what actions to take.
/// A global drift flag indicates when enough clients are drifting that the entire federation
/// should adapt (e.g., increase learning rate, trigger retraining).</para>
/// </remarks>
public class DriftReport
{
    /// <summary>
    /// Gets or sets the FL round at which this report was generated.
    /// </summary>
    public int Round { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp of the report.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets per-client drift results.
    /// </summary>
    public List<ClientDriftResult> ClientResults { get; set; } = new();

    /// <summary>
    /// Gets or sets whether global drift is detected (enough clients are drifting).
    /// </summary>
    public bool GlobalDriftDetected { get; set; }

    /// <summary>
    /// Gets or sets the fraction of clients currently experiencing drift.
    /// </summary>
    public double DriftingClientFraction { get; set; }

    /// <summary>
    /// Gets or sets the average drift score across all clients.
    /// </summary>
    public double AverageDriftScore { get; set; }

    /// <summary>
    /// Gets or sets the detection method used.
    /// </summary>
    public FederatedDriftMethod Method { get; set; }

    /// <summary>
    /// Gets or sets a human-readable summary of the drift analysis.
    /// </summary>
    public string Summary { get; set; } = string.Empty;
}
