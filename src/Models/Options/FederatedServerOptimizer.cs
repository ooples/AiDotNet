namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies which server-side federated optimizer (FedOpt family) to use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> FedOpt optimizers run on the server after aggregation and can improve convergence.
/// </remarks>
public enum FederatedServerOptimizer
{
    /// <summary>
    /// No server optimizer (use aggregated parameters directly).
    /// </summary>
    None = 0,

    /// <summary>
    /// FedAvg with server momentum.
    /// </summary>
    FedAvgM = 1,

    /// <summary>
    /// FedAdagrad server optimizer.
    /// </summary>
    FedAdagrad = 2,

    /// <summary>
    /// FedAdam server optimizer.
    /// </summary>
    FedAdam = 3,

    /// <summary>
    /// FedYogi server optimizer.
    /// </summary>
    FedYogi = 4
}

