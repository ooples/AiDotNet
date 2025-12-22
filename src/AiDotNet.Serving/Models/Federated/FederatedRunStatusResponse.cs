namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Response describing federated run status.
/// </summary>
public class FederatedRunStatusResponse
{
    /// <summary>
    /// Gets or sets the run id.
    /// </summary>
    public string RunId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the current round number.
    /// </summary>
    public int CurrentRound { get; set; }

    /// <summary>
    /// Gets or sets the expected parameter count.
    /// </summary>
    public int ParameterCount { get; set; }

    /// <summary>
    /// Gets or sets the number of joined clients.
    /// </summary>
    public int JoinedClients { get; set; }

    /// <summary>
    /// Gets or sets the number of updates received for the current round.
    /// </summary>
    public int UpdatesReceivedForCurrentRound { get; set; }

    /// <summary>
    /// Gets or sets the minimum number of updates required for aggregation.
    /// </summary>
    public int MinUpdatesRequired { get; set; }
}

