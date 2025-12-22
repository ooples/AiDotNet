namespace AiDotNet.Interfaces;

/// <summary>
/// Defines strategies for aggregating model updates from multiple clients in federated learning.
/// </summary>
/// <remarks>
/// This interface represents different methods for combining model updates from distributed clients
/// into a single improved global model.
///
/// <b>For Beginners:</b> An aggregation strategy is like a voting system or consensus mechanism
/// that decides how to combine different opinions into a single decision.
///
/// Think of aggregation strategies as different ways to combine contributions:
/// - Simple average: Everyone's input counts equally
/// - Weighted average: Some contributors' inputs count more based on criteria (data size, accuracy)
/// - Robust methods: Ignore outliers or malicious contributions
///
/// For example, in a federated learning scenario with hospitals:
/// - Hospital A has 10,000 patients: gets weight of 10,000
/// - Hospital B has 5,000 patients: gets weight of 5,000
/// - The aggregation strategy might weight Hospital A's updates more heavily
///
/// Different strategies handle different challenges:
/// - FedAvg: Standard weighted averaging
/// - FedProx: Handles clients with different update frequencies
/// - Krum: Robust to Byzantine (malicious) clients
/// - Median aggregation: Resistant to outliers
/// </remarks>
/// <typeparam name="TModel">The type of model being aggregated.</typeparam>
public interface IAggregationStrategy<TModel>
{
    /// <summary>
    /// Aggregates model updates from multiple clients into a single global model update.
    /// </summary>
    /// <remarks>
    /// This method combines model updates from clients using the strategy's specific algorithm.
    ///
    /// <b>For Beginners:</b> Aggregation is like combining multiple rough drafts of a document
    /// into one polished version that incorporates the best parts of each.
    ///
    /// The aggregation process typically:
    /// 1. Takes model updates (weight changes) from each client
    /// 2. Considers the weight or importance of each client (based on data size, accuracy, etc.)
    /// 3. Combines these updates using the strategy's algorithm
    /// 4. Returns a single aggregated model that represents the collective improvement
    ///
    /// For example with weighted averaging (FedAvg):
    /// - Client 1 (1000 samples): model update A
    /// - Client 2 (500 samples): model update B
    /// - Client 3 (1500 samples): model update C
    /// - Aggregated update = (1000*A + 500*B + 1500*C) / 3000
    /// </remarks>
    /// <param name="clientModels">Dictionary mapping client IDs to their trained models.</param>
    /// <param name="clientWeights">Dictionary mapping client IDs to their aggregation weights (typically based on data size).</param>
    /// <returns>The aggregated global model.</returns>
    TModel Aggregate(Dictionary<int, TModel> clientModels, Dictionary<int, double> clientWeights);

    /// <summary>
    /// Gets the name of the aggregation strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This helps identify which aggregation method is being used,
    /// useful for logging, debugging, and comparing different strategies.
    /// </remarks>
    /// <returns>A string describing the aggregation strategy (e.g., "FedAvg", "FedProx", "Krum").</returns>
    string GetStrategyName();
}
