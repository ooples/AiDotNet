namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for server-side federated optimizers (FedOpt family).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> These optimizers run on the server after aggregation to update the global model.
/// If <see cref="Optimizer"/> is <see cref="FederatedServerOptimizer.None"/>, the server uses the aggregated parameters directly (FedAvg-style).
/// </remarks>
public class FederatedServerOptimizerOptions
{
    /// <summary>
    /// Gets or sets the server optimizer.
    /// </summary>
    public FederatedServerOptimizer Optimizer { get; set; } = FederatedServerOptimizer.None;

    /// <summary>
    /// Gets or sets the server learning rate (step size).
    /// </summary>
    public double LearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the server momentum coefficient for FedAvgM.
    /// </summary>
    public double Momentum { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the beta1 coefficient (Adam/Yogi).
    /// </summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the beta2 coefficient (Adam/Yogi).
    /// </summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the epsilon value used for numerical stability.
    /// </summary>
    public double Epsilon { get; set; } = 1e-8;
}

