namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for federated heterogeneity correction algorithms.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In federated learning, clients often have different data and different compute speeds.
/// Heterogeneity correction methods help reduce "client drift" so the global model converges more reliably on non-IID data.
/// </remarks>
public class FederatedHeterogeneityCorrectionOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the heterogeneity correction algorithm.
    /// </summary>
    public FederatedHeterogeneityCorrection Algorithm { get; set; } = FederatedHeterogeneityCorrection.None;

    /// <summary>
    /// Gets or sets the client learning rate used by methods that need it (e.g., SCAFFOLD control variates).
    /// </summary>
    public double ClientLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the FedDyn regularization strength (alpha).
    /// </summary>
    public double FedDynAlpha { get; set; } = 0.01;
}

