namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Response after a client joins a federated run.
/// </summary>
public class JoinFederatedRunResponse
{
    /// <summary>
    /// Gets or sets the assigned client identifier.
    /// </summary>
    public int ClientId { get; set; }

    /// <summary>
    /// Gets or sets the current round number the client should train against.
    /// </summary>
    public int CurrentRound { get; set; }
}

