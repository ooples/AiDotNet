using AiDotNet.Models.Options;

namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Request to create a federated training run coordinated by AiDotNet.Serving.
/// </summary>
public class CreateFederatedRunRequest
{
    /// <summary>
    /// Gets or sets the name of a model already loaded in the serving repository.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets federated learning options used for this run.
    /// </summary>
    public FederatedLearningOptions Options { get; set; } = new FederatedLearningOptions();

    /// <summary>
    /// Gets or sets the minimum number of client updates required before aggregation is allowed for a round.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This prevents the server from aggregating when too few clients have contributed updates.
    /// </remarks>
    public int MinClientUpdatesPerRound { get; set; } = 1;
}

