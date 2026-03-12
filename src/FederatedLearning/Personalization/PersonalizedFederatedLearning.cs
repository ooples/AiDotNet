namespace AiDotNet.FederatedLearning.Personalization;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Implements personalized federated learning where each client maintains some client-specific parameters.
/// </summary>
/// <remarks>
/// Personalized Federated Learning (PFL) addresses the challenge of heterogeneous data distributions
/// across clients by allowing each client to maintain personalized model components while still
/// benefiting from collaborative learning.
///
/// <b>For Beginners:</b> Personalized FL is like having a shared textbook but personal notes.
/// Everyone learns from the same core material (global model) but adapts it to their specific
/// needs (personalized layers).
///
/// Key concept:
/// - Global layers: Shared across all clients, learn common patterns
/// - Personalized layers: Client-specific, adapt to local data distribution
/// - Clients train both but only global layers are aggregated
///
/// How it works:
/// 1. Model is split into global and personalized parts
/// 2. During local training, both parts are updated
/// 3. Only global parts are sent to server for aggregation
/// 4. Personalized parts stay on the client
/// 5. Client receives updated global parts and keeps personalized parts
///
/// For example, in healthcare:
/// - Hospital A: Urban population, young average age
/// - Hospital B: Rural population, old average age
/// - Hospital C: Suburban population, mixed age
///
/// Model structure:
/// - Global layers (shared): General disease detection features
/// - Personalized layers: Adapt to local demographics
///
/// Benefits:
/// - Better performance on non-IID data
/// - Each client gets a model optimized for their data
/// - Preserves privacy (personalized parts never leave client)
/// - Relatively simple to implement
///
/// Common approaches:
/// 1. Layer-wise personalization: Last few layers personalized
/// 2. Feature-wise personalization: Some features personalized
/// 3. Meta-learning: Learn how to adapt quickly to local data
/// 4. Multi-task learning: Treat each client as a separate task
///
/// When to use PFL:
/// - Clients have significantly different data distributions
/// - Standard FedAvg performance is poor
/// - Can afford client-side storage for personalized parameters
/// - Want better local performance even at cost of global performance
///
/// Limitations:
/// - Requires more storage on client (for personalized params)
/// - May sacrifice some global model quality
/// - Need to choose which layers to personalize
/// - Risk of overfitting to local data
///
/// Reference:
/// - Wang, K., et al. (2019). "Federated Evaluation of On-device Personalization." arXiv preprint.
/// - Fallah, A., et al. (2020). "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach." NeurIPS 2020.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public class PersonalizedFederatedLearning<T>
{
    private readonly double _personalizationFraction;
    private readonly HashSet<string> _personalizedLayers;

    /// <summary>
    /// Initializes a new instance of the <see cref="PersonalizedFederatedLearning{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up personalized federated learning with a specified
    /// fraction of the model kept personalized.
    ///
    /// The personalization fraction determines the split:
    /// - 0.0: No personalization (standard federated learning)
    /// - 0.2: Last 20% of layers personalized (common choice)
    /// - 0.5: Half personalized, half global
    /// - 1.0: Fully personalized (no collaboration)
    ///
    /// Typical strategy:
    /// - Keep early layers (feature extractors) global
    /// - Keep late layers (task-specific) personalized
    ///
    /// For example, in a CNN:
    /// - Conv layers 1-3: Global (learn general visual features)
    /// - Conv layers 4-5: Personalized (adapt to local image characteristics)
    /// - FC layers: Personalized (task-specific classification)
    /// </remarks>
    /// <param name="personalizationFraction">
    /// The fraction of model layers to keep personalized (0.0 to 1.0).
    /// Typically 0.2 for last 20% of layers.
    /// </param>
    public PersonalizedFederatedLearning(double personalizationFraction = 0.2)
    {
        if (personalizationFraction < 0.0 || personalizationFraction > 1.0)
        {
            throw new ArgumentException("Personalization fraction must be between 0 and 1.", nameof(personalizationFraction));
        }

        _personalizationFraction = personalizationFraction;
        _personalizedLayers = new HashSet<string>();
    }

    /// <summary>
    /// Identifies which layers should be personalized based on the model structure.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This decides which parts of the model will be personalized
    /// vs. shared globally.
    ///
    /// Common strategies:
    /// 1. Last-N layers: Personalize the final layers (default)
    /// 2. By name: Personalize layers matching specific patterns
    /// 3. By type: Personalize certain layer types (e.g., batch norm)
    ///
    /// For example, with 10 layers and 20% personalization:
    /// - Layers 0-7: Global (shared)
    /// - Layers 8-9: Personalized (last 2 layers = 20%)
    ///
    /// The intuition:
    /// - Early layers learn low-level features (edges, textures) → should be shared
    /// - Late layers learn high-level, task-specific features → can be personalized
    /// </remarks>
    /// <param name="modelStructure">The model structure with layer names.</param>
    /// <param name="strategy">The strategy for selecting personalized layers.</param>
    /// <param name="customPatterns">Optional patterns for ByPattern strategy.</param>
    public void IdentifyPersonalizedLayers(
        Dictionary<string, T[]> modelStructure,
        PersonalizedLayerSelectionStrategy strategy = PersonalizedLayerSelectionStrategy.LastN,
        HashSet<string>? customPatterns = null)
    {
        if (modelStructure == null || modelStructure.Count == 0)
        {
            throw new ArgumentException("Model structure cannot be null or empty.", nameof(modelStructure));
        }

        _personalizedLayers.Clear();

        switch (strategy)
        {
            case PersonalizedLayerSelectionStrategy.LastN:
            {
                // Personalize the last N% of layers
                int totalLayers = modelStructure.Count;
                int personalizedCount = (int)Math.Ceiling(totalLayers * _personalizationFraction);

                // Sort layer names deterministically. Dictionary enumeration order is not guaranteed
                // across .NET runtimes (especially net471 vs modern .NET). Using ordinal sort ensures
                // reproducible results. Callers should use layer naming that sorts naturally
                // (e.g., "layer_00", "layer_01") if ordering matters.
                var layerNames = modelStructure.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();

                // Take the last personalizedCount layers
                for (int i = totalLayers - personalizedCount; i < totalLayers; i++)
                {
                    _personalizedLayers.Add(layerNames[i]);
                }

                break;
            }

            case PersonalizedLayerSelectionStrategy.ByPattern:
            {
                if (customPatterns == null || customPatterns.Count == 0)
                {
                    throw new ArgumentException(
                        "Custom patterns are required for ByPattern strategy.", nameof(customPatterns));
                }

                // Personalize layers matching specific patterns
                foreach (var layerName in modelStructure.Keys
                    .Where(name => customPatterns.Any(pattern =>
                        name.Contains(pattern, StringComparison.OrdinalIgnoreCase))))
                {
                    _personalizedLayers.Add(layerName);
                }

                break;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown personalization strategy.");
        }
    }

    /// <summary>
    /// Separates a model into global and personalized components.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Splits the model into two parts:
    /// - Global part: Will be sent to server and aggregated
    /// - Personalized part: Stays on client
    ///
    /// For example:
    /// Original model: {layer1: [...], layer2: [...], layer3: [...], layer4: [...]}
    /// If layers 3-4 are personalized:
    /// - Global: {layer1: [...], layer2: [...]}
    /// - Personalized: {layer3: [...], layer4: [...]}
    ///
    /// This separation enables:
    /// - Efficient communication (only send global parts)
    /// - Privacy (personalized parts never leave client)
    /// - Flexibility (different personalization per client)
    /// </remarks>
    /// <param name="fullModel">The complete model with all layers.</param>
    /// <param name="globalPart">Output: The global layers to be aggregated.</param>
    /// <param name="personalizedPart">Output: The personalized layers to keep local.</param>
    public void SeparateModel(
        Dictionary<string, T[]> fullModel,
        out Dictionary<string, T[]> globalPart,
        out Dictionary<string, T[]> personalizedPart)
    {
        if (fullModel == null || fullModel.Count == 0)
        {
            throw new ArgumentException("Full model cannot be null or empty.", nameof(fullModel));
        }

        globalPart = new Dictionary<string, T[]>();
        personalizedPart = new Dictionary<string, T[]>();

        foreach (var layer in fullModel)
        {
            if (_personalizedLayers.Contains(layer.Key))
            {
                // This layer is personalized - keep local
                personalizedPart[layer.Key] = (T[])layer.Value.Clone();
            }
            else
            {
                // This layer is global - will be aggregated
                globalPart[layer.Key] = (T[])layer.Value.Clone();
            }
        }
    }

    /// <summary>
    /// Combines global model update with client's personalized layers.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After the server aggregates global layers, each client
    /// combines the updated global layers with their own personalized layers to form
    /// the complete model for the next round.
    ///
    /// Process:
    /// 1. Receive updated global layers from server
    /// 2. Retrieve client's personalized layers from local storage
    /// 3. Merge them into one complete model
    /// 4. Ready for next round of local training
    ///
    /// For example:
    /// Server sends global update: {layer1: [...], layer2: [...]}
    /// Client has personalized: {layer3: [...], layer4: [...]}
    /// Combined model: {layer1: [...], layer2: [...], layer3: [...], layer4: [...]}
    ///
    /// This ensures:
    /// - Global knowledge is incorporated (layers 1-2 updated)
    /// - Local adaptation is preserved (layers 3-4 unchanged)
    /// - Model structure remains consistent
    /// </remarks>
    /// <param name="globalUpdate">The updated global layers from server.</param>
    /// <param name="personalizedLayers">The client's personalized layers.</param>
    /// <returns>The complete model combining both parts.</returns>
    public Dictionary<string, T[]> CombineModels(
        Dictionary<string, T[]> globalUpdate,
        Dictionary<string, T[]> personalizedLayers)
    {
        if (globalUpdate == null)
        {
            throw new ArgumentNullException(nameof(globalUpdate));
        }

        if (personalizedLayers == null)
        {
            throw new ArgumentNullException(nameof(personalizedLayers));
        }

        var combinedModel = new Dictionary<string, T[]>();

        // Add all global layers
        foreach (var layer in globalUpdate)
        {
            combinedModel[layer.Key] = (T[])layer.Value.Clone();
        }

        // Add all personalized layers
        foreach (var layer in personalizedLayers)
        {
            combinedModel[layer.Key] = (T[])layer.Value.Clone();
        }

        return combinedModel;
    }

    /// <summary>
    /// Checks if a specific layer is personalized.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Returns whether a given layer should be kept local
    /// (personalized) or sent to the server (global).
    /// </remarks>
    /// <param name="layerName">The name of the layer to check.</param>
    /// <returns>True if the layer is personalized, false if global.</returns>
    public bool IsLayerPersonalized(string layerName)
    {
        return _personalizedLayers.Contains(layerName);
    }

    /// <summary>
    /// Gets the set of all personalized layer names.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Returns the list of which layers are personalized.
    /// Useful for logging, debugging, and understanding the model split.
    /// </remarks>
    /// <returns>A read-only set of personalized layer names.</returns>
    public IReadOnlyCollection<string> GetPersonalizedLayers()
    {
        return _personalizedLayers;
    }

    /// <summary>
    /// Gets the personalization fraction.
    /// </summary>
    /// <returns>The fraction of layers that are personalized.</returns>
    public double GetPersonalizationFraction()
    {
        return _personalizationFraction;
    }

    /// <summary>
    /// Calculates statistics about the model split.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Provides useful information about how the model is divided:
    /// - How many parameters are global vs. personalized
    /// - What percentage of the model is personalized
    /// - Communication savings from personalization
    ///
    /// This helps understand the trade-offs:
    /// - More personalized → Less communication, more storage per client
    /// - More global → More communication, less storage per client
    /// </remarks>
    /// <param name="fullModel">The complete model.</param>
    /// <returns>A dictionary with statistics.</returns>
    public Dictionary<string, double> GetModelSplitStatistics(Dictionary<string, T[]> fullModel)
    {
        if (fullModel == null || fullModel.Count == 0)
        {
            throw new ArgumentException("Full model cannot be null or empty.", nameof(fullModel));
        }

        int totalParams = fullModel.Values.Sum(layer => layer.Length);
        int personalizedParams = fullModel
            .Where(layer => _personalizedLayers.Contains(layer.Key))
            .Sum(layer => layer.Value.Length);
        int globalParams = totalParams - personalizedParams;

        int totalLayers = fullModel.Count;
        int personalizedLayerCount = fullModel.Keys.Count(layerName => _personalizedLayers.Contains(layerName));
        int globalLayerCount = totalLayers - personalizedLayerCount;

        return new Dictionary<string, double>
        {
            ["total_parameters"] = totalParams,
            ["global_parameters"] = globalParams,
            ["personalized_parameters"] = personalizedParams,
            ["global_parameter_fraction"] = totalParams > 0 ? (double)globalParams / totalParams : 0,
            ["personalized_parameter_fraction"] = totalParams > 0 ? (double)personalizedParams / totalParams : 0,
            ["total_layers"] = totalLayers,
            ["global_layers"] = globalLayerCount,
            ["personalized_layers"] = personalizedLayerCount,
            ["communication_reduction"] = totalParams > 0 ? (double)personalizedParams / totalParams : 0
        };
    }
}

/// <summary>
/// Strategy for selecting which layers to personalize in federated learning.
/// </summary>
public enum PersonalizedLayerSelectionStrategy
{
    /// <summary>
    /// Personalize the last N% of layers (sorted by ordinal name). Default and most common approach.
    /// Early layers learn general features and are shared; late layers are task-specific and personalized.
    /// Use zero-padded names (e.g., "layer_00", "layer_01") for natural ordering.
    /// </summary>
    LastN = 0,

    /// <summary>
    /// Personalize layers matching user-provided name patterns (e.g., "batch_norm", "classifier").
    /// Requires customPatterns to be provided.
    /// </summary>
    ByPattern = 1
}
