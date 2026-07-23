namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how to down-weight stale updates in asynchronous federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> If an update was computed on an old global model, it can be less helpful.
/// Staleness weighting reduces the influence of older updates.
/// </remarks>
public enum FederatedStalenessWeighting
{
    /// <summary>
    /// No additional staleness weighting (constant weight).
    /// </summary>
    Constant = 0,

    /// <summary>
    /// Weight = 1 / (1 + staleness).
    /// </summary>
    Inverse = 1,

    /// <summary>
    /// Weight = exp(-rate * staleness).
    /// </summary>
    Exponential = 2,

    /// <summary>
    /// Weight = 1 / (1 + staleness)^rate.
    /// </summary>
    Polynomial = 3
}

