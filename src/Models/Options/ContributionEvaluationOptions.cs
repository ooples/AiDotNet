namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for client contribution evaluation in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure how the system measures each client's
/// contribution to the global model. This is important for detecting free-riders, rewarding
/// valuable participants, and identifying low-quality or poisoned updates.</para>
/// </remarks>
public class ContributionEvaluationOptions
{
    /// <summary>
    /// Gets or sets the contribution evaluation method. Default is DataShapley.
    /// </summary>
    public ContributionMethod Method { get; set; } = ContributionMethod.DataShapley;

    /// <summary>
    /// Gets or sets the number of Monte Carlo sampling rounds for Data Shapley.
    /// More rounds give more accurate estimates but cost more compute.
    /// Default is 100.
    /// </summary>
    public int SamplingRounds { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate contributions (in FL rounds).
    /// Set to 1 to evaluate every round, or higher to save compute.
    /// Default is 5 (every 5 rounds).
    /// </summary>
    public int EvaluationFrequency { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use a performance cache to avoid redundant model evaluations
    /// during Shapley value computation. Default is true.
    /// </summary>
    public bool UsePerformanceCache { get; set; } = true;

    /// <summary>
    /// Gets or sets the convergence tolerance for Monte Carlo Shapley estimation.
    /// Sampling stops early if values change by less than this between rounds.
    /// Default is 0.01.
    /// </summary>
    public double ConvergenceTolerance { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the minimum contribution score below which a client is flagged as a free-rider.
    /// Default is 0.01 (clients contributing less than 1% of average are flagged).
    /// </summary>
    public double FreeRiderThreshold { get; set; } = 0.01;
}
