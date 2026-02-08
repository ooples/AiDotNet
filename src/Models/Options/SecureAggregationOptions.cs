namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for secure aggregation in federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Secure aggregation hides each client's update from the server so the server
/// can only see the final combined result. These options let you pick how secure aggregation should
/// behave, including whether the protocol can handle clients dropping out mid-round.
/// </remarks>
public class SecureAggregationOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets whether secure aggregation is enabled.
    /// </summary>
    /// <remarks>
    /// This is an alternative to <see cref="FederatedLearningOptions.UseSecureAggregation"/>. If either is enabled,
    /// secure aggregation is used.
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets which secure aggregation mode is used.
    /// </summary>
    public SecureAggregationMode Mode { get; set; } = SecureAggregationMode.FullParticipation;

    /// <summary>
    /// Gets or sets the minimum number of clients that must upload masked updates for the round to succeed.
    /// </summary>
    /// <remarks>
    /// If this is 0 or less, an industry-standard default is computed based on the number of selected clients.
    /// </remarks>
    public int MinimumUploaderCount { get; set; } = 0;

    /// <summary>
    /// Gets or sets the reconstruction threshold used by dropout-resilient secure aggregation.
    /// </summary>
    /// <remarks>
    /// If this is 0 or less, a default is chosen based on <see cref="MinimumUploaderCount"/>.
    /// The threshold is the minimum number of clients that must participate in the unmasking step
    /// to reconstruct missing masks when other clients drop out after uploading.
    /// </remarks>
    public int ReconstructionThreshold { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum fraction of selected clients that may drop out while still completing the round.
    /// </summary>
    /// <remarks>
    /// Used only when <see cref="MinimumUploaderCount"/> is not explicitly set. A value of 0.2 means
    /// "tolerate up to 20% dropouts" for the default minimum uploader calculation.
    /// </remarks>
    public double MaxDropoutFraction { get; set; } = 0.2;
}

