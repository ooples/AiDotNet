namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the asynchronous federated learning mode.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In async federated learning, clients can send updates at different times.
/// The server mixes updates as they arrive instead of waiting for a strict round barrier.
/// </remarks>
public enum FederatedAsyncMode
{
    /// <summary>
    /// Disable asynchronous modes (standard synchronous rounds).
    /// </summary>
    None = 0,

    /// <summary>
    /// FedAsync-style staleness-aware mixing.
    /// </summary>
    FedAsync = 1,

    /// <summary>
    /// FedBuff-style buffered aggregation.
    /// </summary>
    FedBuff = 2,

    /// <summary>
    /// AsyncFedED: entropy-driven client scheduling prioritizing most informative clients. (2024)
    /// </summary>
    AsyncFedED = 3,

    /// <summary>
    /// Semi-Async: hybrid sync/async with periodic barriers every K rounds. (Wu et al., IEEE TPDS 2023)
    /// </summary>
    SemiAsync = 4
}

