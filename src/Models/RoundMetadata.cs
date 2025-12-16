namespace AiDotNet.Models;

/// <summary>
/// Contains detailed metrics for a single federated learning round.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Information about what happened in one specific training round.
/// </remarks>
public class RoundMetadata
{
    /// <summary>
    /// Gets or sets the round number (0-indexed).
    /// </summary>
    public int RoundNumber { get; set; }

    /// <summary>
    /// Gets or sets the global model loss after this round.
    /// </summary>
    public double GlobalLoss { get; set; }

    /// <summary>
    /// Gets or sets the global model accuracy after this round.
    /// </summary>
    public double GlobalAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the IDs of clients selected for this round.
    /// </summary>
    public List<int> SelectedClientIds { get; set; } = new List<int>();

    /// <summary>
    /// Gets or sets the time taken for this round in seconds.
    /// </summary>
    public double RoundTimeSeconds { get; set; }

    /// <summary>
    /// Gets or sets the communication cost for this round in megabytes.
    /// </summary>
    public double CommunicationMB { get; set; }

    /// <summary>
    /// Gets or sets the effective upload compression ratio for this round (1.0 means no compression).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the fraction of upload bandwidth used compared to sending a full update.
    /// For example, 0.1 means uploads were about 10% of the uncompressed size.
    /// </remarks>
    public double UploadCompressionRatio { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the average local loss across selected clients.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The average loss that clients achieved on their local data
    /// before sending updates. Comparing this to global loss can reveal overfitting.
    /// </remarks>
    public double AverageLocalLoss { get; set; }

    /// <summary>
    /// Gets or sets the privacy budget consumed in this round.
    /// </summary>
    public double PrivacyBudgetConsumed { get; set; }
}
