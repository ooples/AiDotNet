namespace AiDotNet.Models.Options;

/// <summary>
/// Determines which secure aggregation protocol variant is used.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Secure aggregation comes in different "flavors" depending on how
/// much client drop-out the protocol can tolerate.
/// </remarks>
public enum SecureAggregationMode
{
    /// <summary>
    /// Synchronous secure aggregation that requires full participation from the selected clients.
    /// </summary>
    /// <remarks>
    /// If any selected client drops out after masks are created, the round must be restarted.
    /// </remarks>
    FullParticipation = 0,

    /// <summary>
    /// Dropout-resilient secure aggregation with a reconstruction threshold.
    /// </summary>
    /// <remarks>
    /// This mode is based on practical secure aggregation techniques (e.g., Bonawitz et al.),
    /// allowing the server to recover the aggregate even if some clients fail to complete the
    /// round, provided a sufficient number of clients remain.
    /// </remarks>
    ThresholdDropoutResilient = 1
}

