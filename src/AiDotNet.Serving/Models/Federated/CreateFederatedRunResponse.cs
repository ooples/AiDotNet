namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Response after creating a federated run.
/// </summary>
public class CreateFederatedRunResponse
{
    /// <summary>
    /// Gets or sets the run identifier.
    /// </summary>
    public string RunId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the current round number.
    /// </summary>
    public int CurrentRound { get; set; }

    /// <summary>
    /// Gets or sets the expected parameter count for submitted updates.
    /// </summary>
    public int ParameterCount { get; set; }
}

