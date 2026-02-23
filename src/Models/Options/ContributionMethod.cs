namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the method used to evaluate client contributions in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Not all clients contribute equally to the global model. Some provide
/// high-quality data that greatly improves accuracy, while others may free-ride. These methods
/// measure each client's contribution so you can fairly compensate them or detect problems.</para>
/// </remarks>
public enum ContributionMethod
{
    /// <summary>
    /// Exact Shapley value: measures marginal contribution by testing all possible coalitions.
    /// Most accurate but exponential cost O(2^N) where N = number of clients.
    /// Only practical for small federations (fewer than ~15 clients).
    /// </summary>
    ShapleyValue,

    /// <summary>
    /// Data Shapley: Monte Carlo approximation of Shapley values.
    /// Samples random permutations of clients to estimate marginal contributions.
    /// Configurable accuracy vs. cost tradeoff via sampling rounds.
    /// </summary>
    DataShapley,

    /// <summary>
    /// Prototypical contribution: evaluates contribution using prototype representations.
    /// Each client's prototypes (class centroids) are compared against a validation set.
    /// Constant cost per round, suitable for large federations.
    /// </summary>
    Prototypical
}
