namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies which federated aggregation strategy to use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In federated learning, each client trains locally and sends an update.
/// The server then combines those updates using an aggregation strategy.
/// </remarks>
public enum FederatedAggregationStrategy
{
    /// <summary>
    /// Federated Averaging (FedAvg).
    /// </summary>
    FedAvg = 0,

    /// <summary>
    /// Federated Proximal (FedProx) for heterogeneity.
    /// </summary>
    FedProx = 1,

    /// <summary>
    /// Federated Batch Normalization (FedBN).
    /// </summary>
    FedBN = 2,

    /// <summary>
    /// Coordinate-wise median aggregation.
    /// </summary>
    Median = 3,

    /// <summary>
    /// Coordinate-wise trimmed mean aggregation.
    /// </summary>
    TrimmedMean = 4,

    /// <summary>
    /// Coordinate-wise winsorized mean aggregation.
    /// </summary>
    WinsorizedMean = 5,

    /// <summary>
    /// Robust Federated Aggregation (geometric median / RFA).
    /// </summary>
    Rfa = 6,

    /// <summary>
    /// Krum (Byzantine-robust selection).
    /// </summary>
    Krum = 7,

    /// <summary>
    /// Multi-Krum (select m central updates, then average).
    /// </summary>
    MultiKrum = 8,

    /// <summary>
    /// Bulyan (Multi-Krum selection + trimming).
    /// </summary>
    Bulyan = 9
}

