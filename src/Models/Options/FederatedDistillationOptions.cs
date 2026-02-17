namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated knowledge distillation strategy.
/// </summary>
public enum FederatedDistillationStrategy
{
    /// <summary>No distillation — standard parameter aggregation.</summary>
    None,
    /// <summary>FedMD — mutual distillation on a public dataset.</summary>
    FedMD,
    /// <summary>FedDF — ensemble distillation using unlabeled public data.</summary>
    FedDF,
    /// <summary>FedGEN — data-free distillation using server-side generative model.</summary>
    FedGEN
}

/// <summary>
/// Configuration options for federated knowledge distillation, enabling model-heterogeneous FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard federated learning, all clients must use the exact same model
/// architecture. Knowledge distillation removes this constraint — a phone can use a tiny model while a
/// server uses a large model, and they still learn from each other.</para>
/// </remarks>
public class FederatedDistillationOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the distillation strategy. Default: None.
    /// </summary>
    public FederatedDistillationStrategy Strategy { get; set; } = FederatedDistillationStrategy.None;

    /// <summary>
    /// Gets or sets the softmax temperature for soft label generation. Default: 3.0.
    /// </summary>
    /// <remarks>
    /// Higher temperatures produce softer probability distributions that reveal more about
    /// class relationships. Typical range: 1.0–10.0.
    /// </remarks>
    public double Temperature { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the weight for distillation loss vs task-specific loss. Default: 0.5.
    /// </summary>
    /// <remarks>
    /// 0.0 = only task loss (no distillation), 1.0 = only distillation loss.
    /// </remarks>
    public double DistillationAlpha { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of local distillation epochs per round. Default: 5.
    /// </summary>
    public int DistillationEpochs { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of output classes for FedGEN. Default: 10.
    /// </summary>
    public int NumClasses { get; set; } = 10;

    /// <summary>
    /// Gets or sets the feature dimensionality for FedGEN synthetic data. Default: 64.
    /// </summary>
    public int FeatureDimension { get; set; } = 64;
}
