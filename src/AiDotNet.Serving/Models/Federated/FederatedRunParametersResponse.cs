namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Response payload containing the current global parameters for a federated run.
/// </summary>
public sealed class FederatedRunParametersResponse
{
    /// <summary>
    /// Gets or sets the federated run identifier.
    /// </summary>
    public string RunId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the current round number for which these parameters are valid.
    /// </summary>
    public int RoundNumber { get; set; }

    /// <summary>
    /// Gets or sets the number of parameters in the vector.
    /// </summary>
    public int ParameterCount { get; set; }

    /// <summary>
    /// Gets or sets the global parameter vector (double precision transport format).
    /// </summary>
    public double[] Parameters { get; set; } = Array.Empty<double>();
}

