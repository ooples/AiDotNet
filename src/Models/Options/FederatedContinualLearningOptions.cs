namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated continual learning strategy for preventing catastrophic forgetting across rounds.
/// </summary>
public enum FederatedContinualLearningStrategy
{
    /// <summary>No federated continual learning — standard aggregation without forgetting prevention.</summary>
    None,
    /// <summary>Federated EWC — Fisher-information-based importance weighting aggregated across clients.</summary>
    FederatedEWC,
    /// <summary>Federated Orthogonal Projection — projects gradients orthogonal to important directions.</summary>
    OrthogonalProjection
}

/// <summary>
/// Configuration options for federated continual learning (preventing catastrophic forgetting in FL).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When federated models learn new tasks over time, they can forget what they
/// learned before (catastrophic forgetting). These strategies identify which model parameters are important
/// for previous tasks and protect them during future training rounds. Each client computes importance locally,
/// and the server aggregates these importance estimates across all clients.</para>
/// </remarks>
public class FederatedContinualLearningOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the strategy. Default: None.
    /// </summary>
    public FederatedContinualLearningStrategy Strategy { get; set; } = FederatedContinualLearningStrategy.None;

    /// <summary>
    /// Gets or sets the regularization strength for EWC penalty. Default: 400.0.
    /// </summary>
    /// <remarks>
    /// Higher values provide stronger protection against forgetting but may slow learning of new tasks.
    /// Typical range: 100–5000.
    /// </remarks>
    public double RegularizationStrength { get; set; } = 400.0;

    /// <summary>
    /// Gets or sets the number of data samples for Fisher information estimation. Default: 200.
    /// </summary>
    public int FisherSamples { get; set; } = 200;

    /// <summary>
    /// Gets or sets the projection threshold for orthogonal projection. Default: 0.01.
    /// </summary>
    /// <remarks>
    /// Directions with importance above this threshold are protected. Lower values protect more directions.
    /// </remarks>
    public double ProjectionThreshold { get; set; } = 0.01;
}
