namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements the Federated Batch Normalization (FedBN) aggregation strategy.
/// </summary>
/// <remarks>
/// FedBN is a specialized aggregation strategy that handles batch normalization layers
/// differently from other layers in neural networks. Proposed by Li et al. in 2021,
/// it addresses the challenge of non-IID data by keeping batch normalization parameters local.
///
/// <b>For Beginners:</b> FedBN recognizes that some parts of a neural network should remain
/// personalized to each client rather than being averaged globally.
///
/// The key insight:
/// - Batch Normalization (BN) layers learn statistics specific to each client's data
/// - Averaging BN parameters across clients with different data distributions hurts performance
/// - Solution: Keep BN layers local, only aggregate other layers (Conv, FC, etc.)
///
/// How FedBN works:
/// 1. During aggregation, identify batch normalization layers
/// 2. Aggregate only non-BN layers using weighted averaging
/// 3. Keep each client's BN layers unchanged (personalized)
/// 4. Send back global model with client-specific BN layers
///
/// For example, in a CNN with layers:
/// - Conv1 (filters) → BN1 (normalization) → ReLU → Conv2 → BN2 → FC (classification)
///
/// FedBN aggregates:
/// - ✓ Conv1 filters: Averaged across clients
/// - ✗ BN1 params: Kept local to each client
/// - ✓ Conv2 filters: Averaged across clients
/// - ✗ BN2 params: Kept local to each client
/// - ✓ FC weights: Averaged across clients
///
/// Why this matters:
/// - Different clients may have different data ranges, distributions
/// - Hospital A images: brightness range [0, 100]
/// - Hospital B images: brightness range [50, 200]
/// - Each needs different normalization parameters
/// - Shared feature extractors (Conv layers) + personalized normalization works better
///
/// When to use FedBN:
/// - Training deep neural networks (especially CNNs)
/// - Non-IID data with distribution shift
/// - Batch normalization or layer normalization in architecture
/// - Want to improve accuracy without changing training much
///
/// Benefits:
/// - Significantly improves accuracy on non-IID data
/// - Simple modification to FedAvg
/// - No additional communication cost
/// - Each client keeps personalized normalization
///
/// Limitations:
/// - Only helps when using batch normalization
/// - Doesn't address other heterogeneity challenges
/// - Requires identifying BN layers in model structure
///
/// Reference: Li, X., et al. (2021). "Federated Learning on Non-IID Data Silos: An Experimental Study."
/// ICDE 2021.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public class FedBNAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly HashSet<string> _batchNormLayerPatterns;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedBNAggregationStrategy{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a FedBN aggregator that knows how to identify
    /// batch normalization layers in your model.
    ///
    /// Common BN layer naming patterns:
    /// - "bn", "batchnorm", "batch_norm": Explicit BN layers
    /// - "gamma", "beta": BN trainable parameters
    /// - "running_mean", "running_var": BN statistics
    ///
    /// The strategy will exclude these from aggregation, keeping them client-specific.
    /// </remarks>
    /// <param name="batchNormLayerPatterns">
    /// Patterns to identify batch normalization layers. If null, uses default patterns.
    /// </param>
    public FedBNAggregationStrategy(HashSet<string>? batchNormLayerPatterns = null)
    {
        // Default patterns for identifying batch normalization layers
        _batchNormLayerPatterns = batchNormLayerPatterns ?? new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "bn",
            "batchnorm",
            "batch_norm",
            "batch_normalization",
            "gamma",
            "beta",
            "running_mean",
            "running_var",
            "moving_mean",
            "moving_variance"
        };
    }

    /// <summary>
    /// Aggregates client models while keeping batch normalization layers local.
    /// </summary>
    /// <remarks>
    /// This method implements selective aggregation:
    ///
    /// <b>For Beginners:</b> Think of this as a smart averaging that knows some parameters
    /// should stay personal (like BN layers) while others should be shared (like Conv layers).
    ///
    /// Step-by-step process:
    /// 1. For each layer in the model:
    ///    - Check if it's a batch normalization layer (by name matching)
    ///    - If BN: Keep first client's values (could be any client's, they stay local)
    ///    - If not BN: Compute weighted average across all clients
    /// 2. Return aggregated model
    ///
    /// Mathematical formulation:
    /// For non-BN layers:
    ///   w_global[layer] = Σ(n_k / n_total) × w_k[layer]
    ///
    /// For BN layers:
    ///   w_global[layer] = w_client[layer]  (keeps local values)
    ///
    /// For example, with 3 clients and a model with:
    /// - "conv1": [0.5, 0.6, 0.7] at clients → Average these
    /// - "bn1_gamma": [1.0, 1.2, 0.9] at clients → Keep local (don't average)
    /// - "conv2": [0.3, 0.4, 0.5] at clients → Average these
    /// - "bn2_beta": [0.1, 0.2, 0.15] at clients → Keep local (don't average)
    ///
    /// Note: In practice, each client would maintain their own BN parameters.
    /// The "global" model returned includes BN params that each client will replace
    /// with their local version upon receiving the update.
    /// </remarks>
    /// <param name="clientModels">Dictionary mapping client IDs to their model parameters.</param>
    /// <param name="clientWeights">Dictionary mapping client IDs to their sample counts (weights).</param>
    /// <returns>The aggregated global model parameters with BN layers excluded from aggregation.</returns>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        double totalWeight = GetTotalWeightOrThrow(clientWeights, clientModels.Keys, nameof(clientWeights));

        var firstClientModel = clientModels.First().Value;
        var aggregatedModel = new Dictionary<string, T[]>();

        // Process each layer
        foreach (var layerName in firstClientModel.Keys)
        {
            // Check if this is a batch normalization layer
            bool isBatchNormLayer = IsBatchNormalizationLayer(layerName);

            if (isBatchNormLayer)
            {
                // For BN layers, keep the first client's parameters (they stay local)
                // In practice, each client will maintain their own BN params
                aggregatedModel[layerName] = (T[])firstClientModel[layerName].Clone();
            }
            else
            {
                // For non-BN layers, perform weighted aggregation (like FedAvg)
                var layer = CreateZeroInitializedLayer(firstClientModel[layerName].Length);

                aggregatedModel[layerName] = layer;
                AggregateLayerWeightedAverageInto(layerName, clientModels, clientWeights, totalWeight, layer);
            }
        }

        return aggregatedModel;
    }

    /// <summary>
    /// Determines whether a layer is a batch normalization layer based on its name.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This checks if a layer name contains any of the known
    /// batch normalization patterns.
    ///
    /// For example:
    /// - "conv1_weights" → false (not BN)
    /// - "bn1_gamma" → true (contains "bn")
    /// - "batch_norm_2_beta" → true (contains "batch_norm")
    /// - "fc_bias" → false (not BN)
    /// </remarks>
    /// <param name="layerName">The name of the layer to check.</param>
    /// <returns>True if the layer is a batch normalization layer, false otherwise.</returns>
    private bool IsBatchNormalizationLayer(string layerName)
    {
        return _batchNormLayerPatterns.Any(pattern => layerName.Contains(pattern, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Gets the name of the aggregation strategy.
    /// </summary>
    /// <returns>The string "FedBN".</returns>
    public override string GetStrategyName()
    {
        return "FedBN";
    }

    /// <summary>
    /// Gets the batch normalization layer patterns used for identification.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Returns the list of patterns used to recognize which
    /// layers are batch normalization layers.
    /// </remarks>
    /// <returns>A set of BN layer patterns.</returns>
    public IReadOnlyCollection<string> GetBatchNormPatterns()
    {
        return _batchNormLayerPatterns;
    }
}
