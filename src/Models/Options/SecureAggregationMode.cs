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
    /// Runs the Bonawitz et al. (2017) protocol with threshold <c>t = n</c>: every selected client must complete
    /// every round, and any dropout aborts the round with a <c>SecureAggregationAbortedException</c>.
    /// </remarks>
    FullParticipation = 0,

    /// <summary>
    /// Dropout-resilient secure aggregation with a reconstruction threshold.
    /// </summary>
    /// <remarks>
    /// Runs the Bonawitz et al. (2017) protocol with a Shamir threshold <c>t &lt; n</c>
    /// (<see cref="SecureAggregationOptions.ReconstructionThreshold"/>; default <c>floor(2n/3) + 1</c>). The server
    /// recovers the exact aggregate of the survivors as long as at least <c>t</c> clients complete every round.
    /// </remarks>
    ThresholdDropoutResilient = 1
}

