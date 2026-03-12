namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements the Federated Averaging (FedAvg) aggregation strategy.
/// </summary>
/// <remarks>
/// FedAvg is the foundational aggregation algorithm for federated learning, proposed by
/// McMahan et al. in 2017. It performs a weighted average of client model updates based
/// on the number of training samples each client has.
///
/// <b>For Beginners:</b> FedAvg is like calculating a weighted class average where students
/// who solved more practice problems have more influence on the final answer.
///
/// How FedAvg works:
/// 1. Each client trains on their local data and computes model updates
/// 2. Clients send their updated model weights to the server
/// 3. Server computes weighted average: weight = (client_samples / total_samples)
/// 4. New global model = Σ(weight_i × client_model_i)
///
/// For example, with 3 hospitals:
/// - Hospital A: 1000 patients, model accuracy 90%
/// - Hospital B: 500 patients, model accuracy 88%
/// - Hospital C: 1500 patients, model accuracy 92%
///
/// Total patients: 3000
/// Hospital A weight: 1000/3000 = 0.333
/// Hospital B weight: 500/3000 = 0.167
/// Hospital C weight: 1500/3000 = 0.500
///
/// For each model parameter:
/// global_param = 0.333 × A_param + 0.167 × B_param + 0.500 × C_param
///
/// Benefits:
/// - Simple and efficient
/// - Well-studied theoretically
/// - Works well when clients have similar data distributions (IID data)
///
/// Limitations:
/// - Assumes clients are equally reliable
/// - Can struggle with non-IID data (different distributions across clients)
/// - No built-in handling for stragglers (slow clients)
///
/// Reference: McMahan, H. B., et al. (2017). "Communication-Efficient Learning of Deep Networks
/// from Decentralized Data." AISTATS 2017.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public class FedAvgAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    /// <summary>
    /// Aggregates client models using weighted averaging based on the number of samples.
    /// </summary>
    /// <remarks>
    /// This method implements the core FedAvg algorithm:
    ///
    /// Mathematical formulation:
    /// w_global = Σ(n_k / n_total) × w_k
    ///
    /// where:
    /// - w_global: global model weights
    /// - w_k: client k's model weights
    /// - n_k: number of samples at client k
    /// - n_total: total samples across all clients
    ///
    /// <b>For Beginners:</b> This combines all client models into one by taking a weighted
    /// average, where clients with more data have more influence.
    ///
    /// Step-by-step process:
    /// 1. Calculate total samples across all clients
    /// 2. For each client, compute weight = client_samples / total_samples
    /// 3. For each model parameter, compute weighted sum
    /// 4. Return the aggregated model
    ///
    /// For example, if we have 2 clients with a simple model (one parameter):
    /// - Client 1: 300 samples, parameter value = 0.8
    /// - Client 2: 700 samples, parameter value = 0.6
    ///
    /// Total samples: 1000
    /// Client 1 weight: 300/1000 = 0.3
    /// Client 2 weight: 700/1000 = 0.7
    /// Aggregated parameter: 0.3 × 0.8 + 0.7 × 0.6 = 0.24 + 0.42 = 0.66
    /// </remarks>
    /// <param name="clientModels">Dictionary mapping client IDs to their model parameters.</param>
    /// <param name="clientWeights">Dictionary mapping client IDs to their sample counts (weights).</param>
    /// <returns>The aggregated global model parameters.</returns>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Gets the name of the aggregation strategy.
    /// </summary>
    /// <returns>The string "FedAvg".</returns>
    public override string GetStrategyName()
    {
        return "FedAvg";
    }
}
