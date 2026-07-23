using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using ContrastiveMode = AiDotNet.KnowledgeDistillation.Strategies.ContrastiveMode;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Factory of named knowledge-distillation strategies.
/// </summary>
/// <remarks>
/// <para>Each named strategy is a public factory method returning an
/// <see cref="IDistillationStrategy{T}"/>. The strategy IS the parameter: callers pass a
/// strategy instance directly (e.g. via <c>KnowledgeDistillationOptions.Strategy</c> or
/// <c>ConfigureDistillationStrategy</c>). When no strategy is supplied, response-based
/// distillation (Hinton et al., 2015) is the industry-standard default — see
/// <see cref="ResolveStrategy"/>.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public static class DistillationStrategyFactory<T>
{
    /// <summary>
    /// Returns <paramref name="strategy"/> when non-null, otherwise the response-based default
    /// (standard Hinton distillation) configured with the supplied temperature and alpha.
    /// </summary>
    /// <param name="strategy">The caller-supplied strategy, or null for the default.</param>
    /// <param name="temperature">Softmax temperature used for the default (default 3.0).</param>
    /// <param name="alpha">Hard/soft loss weight used for the default (default 0.3).</param>
    public static IDistillationStrategy<T> ResolveStrategy(
        IDistillationStrategy<T>? strategy,
        double temperature = 3.0,
        double alpha = 0.3)
    {
        return strategy ?? CreateResponseBasedStrategy(temperature, alpha);
    }

    /// <summary>
    /// Response-based distillation (Hinton et al., 2015) — the industry-standard default.
    /// Matches the teacher's temperature-scaled softmax outputs.
    /// </summary>
    public static IDistillationStrategy<T> CreateResponseBasedStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        return new DistillationLoss<T>(temperature, alpha);
    }

    /// <summary>
    /// Feature-based / FitNets distillation (Romero et al., 2014). Not available through the
    /// factory — construct <c>FeatureDistillationStrategy&lt;T&gt;</c> directly with layer pairs.
    /// </summary>
    public static IDistillationStrategy<T> CreateFeatureBasedStrategy(
        Vector<string>? layerPairs = null,
        double featureWeight = 0.5)
    {
        // FeatureDistillationStrategy doesn't implement IDistillationStrategy
        // It's a separate utility class that computes feature loss
        throw new NotSupportedException(
            "FeatureBased distillation strategy requires layer-specific feature extraction and cannot be used " +
            "through the factory. Create it directly: new FeatureDistillationStrategy<T>(layerPairs, featureWeight)");
    }

    /// <summary>
    /// Attention-transfer distillation (Zagoruyko &amp; Komodakis, 2017), for transformer models.
    /// </summary>
    public static IDistillationStrategy<T> CreateAttentionBasedStrategy(
        Vector<string>? attentionLayers = null,
        double temperature = 3.0,
        double alpha = 0.3,
        double attentionWeight = 0.3)
    {
        // Provide default attention layers if none specified
        var defaultLayers = new[] { "layer.0.attention", "layer.1.attention", "layer.2.attention" };
        var layersArray = attentionLayers != null ?
            Enumerable.Range(0, attentionLayers.Length).Select(i => attentionLayers[i]).ToArray() :
            defaultLayers;

        return new AttentionDistillationStrategy<T>(
            attentionLayers: layersArray,
            attentionWeight: attentionWeight,
            temperature: temperature,
            alpha: alpha,
            matchingMode: AttentionMatchingMode.MSE);
    }

    /// <summary>
    /// Relational Knowledge Distillation / RKD (Park et al., 2019). Preserves distances and angles.
    /// </summary>
    public static IDistillationStrategy<T> CreateRelationBasedStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        // Use constructor defaults for distanceWeight and angleWeight
        return new RelationalDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Contrastive Representation Distillation / CRD (Tian et al., 2020).
    /// </summary>
    public static IDistillationStrategy<T> CreateContrastiveStrategy(
        double temperature = 3.0,
        double alpha = 0.3,
        ContrastiveMode mode = ContrastiveMode.InfoNCE)
    {
        // Use constructor defaults for contrastiveWeight and negativesSampleSize
        return new ContrastiveDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha,
            mode: mode);
    }

    /// <summary>
    /// Similarity-Preserving Distillation / SP (Tung &amp; Mori, 2019).
    /// </summary>
    public static IDistillationStrategy<T> CreateSimilarityPreservingStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        // Use constructor defaults for similarityWeight
        return new SimilarityPreservingStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Probabilistic Knowledge Transfer / PKT (Passalis &amp; Tefas, 2018).
    /// </summary>
    public static IDistillationStrategy<T> CreateProbabilisticStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        // Use constructor defaults for distributionWeight, mode, and mmdBandwidth
        return new ProbabilisticDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Variational Information Distillation / VID (Ahn et al., 2019).
    /// </summary>
    public static IDistillationStrategy<T> CreateVariationalStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        // Use constructor defaults for variationalWeight, mode, and betaIB
        return new VariationalDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Factor Transfer (Kim et al., 2018).
    /// </summary>
    public static IDistillationStrategy<T> CreateFactorTransferStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        // Use constructor defaults for factorWeight, mode, numFactors, and normalizeFactors
        return new FactorTransferDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Neuron Selectivity Transfer / NST (Huang &amp; Wang, 2017).
    /// </summary>
    public static IDistillationStrategy<T> CreateNeuronSelectivityStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        // Use constructor defaults for selectivityWeight and metric
        return new NeuronSelectivityDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Flow of Solution Procedure / FSP (Yim et al., 2017).
    /// </summary>
    public static IDistillationStrategy<T> CreateFlowBasedStrategy(
        double temperature = 3.0,
        double alpha = 0.3)
    {
        return new FlowBasedDistillationStrategy<T>(
            temperature: temperature,
            alpha: alpha);
    }

    /// <summary>
    /// Combined/hybrid distillation. When no strategies are supplied, combines response-based and
    /// relational with equal weight.
    /// </summary>
    public static IDistillationStrategy<T> CreateHybridStrategy(
        double temperature = 3.0,
        double alpha = 0.3,
        Vector<IDistillationStrategy<T>>? strategies = null,
        Vector<double>? strategyWeights = null)
    {
        // Create tuple array from strategies and weights
        (IDistillationStrategy<T> Strategy, double Weight)[] strategyTuples;

        if (strategies == null || strategies.Length == 0)
        {
            // Default: combine response-based and relational
            strategyTuples = new[]
            {
                (CreateResponseBasedStrategy(temperature, alpha), 0.5),
                (CreateRelationBasedStrategy(temperature, alpha), 0.5)
            };
        }
        else
        {
            // Create tuple array from provided strategies and weights
            if (strategyWeights == null)
            {
                // Equal weights if not specified
                double equalWeight = 1.0 / strategies.Length;
                strategyTuples = Enumerable.Range(0, strategies.Length)
                    .Select(i => (strategies[i], equalWeight))
                    .ToArray();
            }
            else
            {
                if (strategies.Length != strategyWeights.Length)
                {
                    throw new ArgumentException(
                        $"Number of strategies ({strategies.Length}) must match number of weights ({strategyWeights.Length})");
                }
                // Zip strategies with weights from Vector
                strategyTuples = Enumerable.Range(0, strategies.Length)
                    .Select(i => (strategies[i], Convert.ToDouble(strategyWeights[i])))
                    .ToArray();
            }
        }

        return new HybridDistillationStrategy<T>(strategyTuples, temperature, alpha);
    }
}
