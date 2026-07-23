namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies which heterogeneity correction algorithm to use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Client data can be non-IID and clients may perform different amounts of local work.
/// Heterogeneity correction methods reduce client drift and improve convergence.
/// </remarks>
public enum FederatedHeterogeneityCorrection
{
    /// <summary>
    /// No heterogeneity correction.
    /// </summary>
    None = 0,

    /// <summary>
    /// SCAFFOLD control variates.
    /// </summary>
    Scaffold = 1,

    /// <summary>
    /// FedNova normalization.
    /// </summary>
    FedNova = 2,

    /// <summary>
    /// FedDyn dynamic regularization.
    /// </summary>
    FedDyn = 3
}

