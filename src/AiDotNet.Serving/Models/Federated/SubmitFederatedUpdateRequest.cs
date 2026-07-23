namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Request for a client to submit an updated parameter vector for a round.
/// </summary>
public class SubmitFederatedUpdateRequest
{
    /// <summary>
    /// Gets or sets the client identifier.
    /// </summary>
    public int ClientId { get; set; }

    /// <summary>
    /// Gets or sets the round number this update corresponds to.
    /// </summary>
    public int RoundNumber { get; set; }

    /// <summary>
    /// Gets or sets the client weight (typically proportional to sample count).
    /// </summary>
    public double ClientWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the full parameter vector after local training.
    /// </summary>
    public double[] Parameters { get; set; } = Array.Empty<double>();
}

