namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Thrown when a secure aggregation round must abort, for example because fewer than the threshold <c>t</c> clients
/// remain, or a share fails authentication.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz et al. (2017), figure 4: at every round the server (and every client) aborts if fewer than <c>t</c>
/// participants are left. Aborting is the protocol's safety property: continuing below the threshold could let the
/// server unmask an individual client's input.
/// </para>
/// <para><b>For Beginners:</b> This is the clean failure you get instead of a wrong aggregate. The message says which
/// round failed and why; no client's input is revealed.</para>
/// </remarks>
public sealed class SecureAggregationAbortedException : InvalidOperationException
{
    /// <summary>Creates the exception for <paramref name="round"/>.</summary>
    public SecureAggregationAbortedException(SecureAggregationRound round, string message)
        : base(message)
    {
        Round = round;
    }

    /// <summary>Gets the round in which the protocol aborted.</summary>
    public SecureAggregationRound Round { get; }
}
