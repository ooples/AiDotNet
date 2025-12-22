namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Response after submitting a federated update.
/// </summary>
public class SubmitFederatedUpdateResponse
{
    /// <summary>
    /// Gets or sets whether the update was accepted.
    /// </summary>
    public bool Accepted { get; set; }

    /// <summary>
    /// Gets or sets the number of updates currently received for the round.
    /// </summary>
    public int ReceivedUpdatesForRound { get; set; }

    /// <summary>
    /// Gets or sets the minimum updates required before aggregation.
    /// </summary>
    public int MinUpdatesRequired { get; set; }
}

