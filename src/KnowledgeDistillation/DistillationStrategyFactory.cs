using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using ContrastiveMode = AiDotNet.KnowledgeDistillation.Strategies.ContrastiveMode;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Factory for creating distillation strategies from enums and configurations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public static class DistillationStrategyFactory<T>
{
    /// <summary>
    /// Creates a distillation strategy from the specified type and parameters.
    /// </summary>
    /// <param name="strategyType">The type of strategy to create.</param>
    /// <param name="temperature">Softmax temperature (default 3.0).</param>
    /// <param name="alpha">Weight for hard loss vs soft loss (default 0.3).</param>
    /// <param name="featureWeight">Weight for feature matching loss (for feature-based strategies).</param>
    /// <param name="attentionWeight">Weight for attention matching loss (for attention-based strategies).</param>
    /// <param name="contrastiveMode">Mode for contrastive loss (for contrastive strategies).</param>
    /// <param name="featureLayerPairs">Layer pairs for feature matching (for feature-based).</param>
    /// <param name="attentionLayers">Attention layers to match (for attention-based).</param>
    /// <param name="strategies">Vector of strategies to combine (for hybrid).</param>
    /// <param name="strategyWeights">Weights for combined strategies (for hybrid).</param>
    /// <returns>A configured distillation strategy.</returns>
    public static IDistillationStrategy<Vector<T>, T> CreateStrategy(
        DistillationStrategyType strategyType,
        double temperature = 3.0,
        double alpha = 0.3,
        double? featureWeight = null,
        double? attentionWeight = null,
        ContrastiveMode? contrastiveMode = null,
        Vector<string>? featureLayerPairs = null,
        Vector<string>? attentionLayers = null,
        Vector<IDistillationStrategy<Vector<T>, T>>? strategies = null,
        Vector<double>? strategyWeights = null)
    {
        return strategyType switch
        {
            DistillationStrategyType.ResponseBased => CreateResponseBasedStrategy(temperature, alpha),
            DistillationStrategyType.FeatureBased => CreateFeatureBasedStrategy(featureLayerPairs, featureWeight ?? 0.5),
            DistillationStrategyType.AttentionBased => CreateAttentionBasedStrategy(attentionLayers, temperature, alpha, attentionWeight ?? 0.3),
            DistillationStrategyType.RelationBased => CreateRelationBasedStrategy(temperature, alpha),
            DistillationStrategyType.ContrastiveBased => CreateContrastiveStrategy(temperature, alpha, contrastiveMode ?? ContrastiveMode.InfoNCE),
            DistillationStrategyType.SimilarityPreserving => CreateSimilarityPreservingStrategy(temperature, alpha),
            DistillationStrategyType.FlowBased => CreateProbabilisticStrategy(temperature, alpha), // Flow-based uses probabilistic
            DistillationStrategyType.ProbabilisticTransfer => CreateProbabilisticStrategy(temperature, alpha),
            DistillationStrategyType.VariationalInformation => CreateVariationalStrategy(temperature, alpha),
            DistillationStrategyType.FactorTransfer => CreateFactorTransferStrategy(temperature, alpha),
            DistillationStrategyType.NeuronSelectivity => CreateNeuronSelectivityStrategy(temperature, alpha),
            DistillationStrategyType.SelfDistillation => CreateResponseBasedStrategy(temperature, alpha), // Self uses response-based
            DistillationStrategyType.Hybrid => CreateHybridStrategy(temperature, alpha, strategies, strategyWeights),
            _ => throw new ArgumentException($"Unknown strategy type: {strategyType}", nameof(strategyType))
        };
    }

    private static IDistillationStrategy<Vector<T>, T> CreateResponseBasedStrategy(
        double temperature,
        double alpha)
    {
        return new DistillationLoss<T>(temperature, alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateFeatureBasedStrategy(
        Vector<string>? layerPairs,
        double featureWeight)
    {
        // FeatureDistillationStrategy doesn't implement IDistillationStrategy
        // It's a separate utility class that computes feature loss
        throw new NotSupportedException(
            "FeatureDistillationStrategy requires layer-specific feature extraction and cannot be used " +
            "through the factory. Create it directly: new FeatureDistillationStrategy<T>(layerPairs, featureWeight)");
    }

    private static IDistillationStrategy<Vector<T>, T> CreateAttentionBasedStrategy(
        Vector<string>? attentionLayers,
        double temperature,
        double alpha,
        double attentionWeight)
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

    private static IDistillationStrategy<Vector<T>, T> CreateRelationBasedStrategy(
        double temperature,
        double alpha)
    {
        return new RelationalDistillationStrategy<T>(
            distanceWeight: 0.5,
            angleWeight: 0.5,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateContrastiveStrategy(
        double temperature,
        double alpha,
        ContrastiveMode mode)
    {
        return new ContrastiveDistillationStrategy<T>(
            contrastiveWeight: 0.8,
            temperature: temperature,
            alpha: alpha,
            negativesSampleSize: 1024,
            mode: mode);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateSimilarityPreservingStrategy(
        double temperature,
        double alpha)
    {
        return new SimilarityPreservingStrategy<T>(
            similarityWeight: 0.5,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateProbabilisticStrategy(
        double temperature,
        double alpha)
    {
        return new ProbabilisticDistillationStrategy<T>(
            distributionWeight: 0.5,
            mode: ProbabilisticMode.MomentMatching,
            mmdBandwidth: 1.0,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateVariationalStrategy(
        double temperature,
        double alpha)
    {
        return new VariationalDistillationStrategy<T>(
            variationalWeight: 0.5,
            mode: VariationalMode.LatentSpaceKL,
            betaIB: 1.0,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateFactorTransferStrategy(
        double temperature,
        double alpha)
    {
        return new FactorTransferDistillationStrategy<T>(
            factorWeight: 0.5,
            mode: FactorMode.LowRankApproximation,
            numFactors: 32,
            normalizeFactors: true,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateNeuronSelectivityStrategy(
        double temperature,
        double alpha)
    {
        return new NeuronSelectivityDistillationStrategy<T>(
            selectivityWeight: 0.5,
            metric: SelectivityMetric.Variance,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateHybridStrategy(
        double temperature,
        double alpha,
        Vector<IDistillationStrategy<Vector<T>, T>>? strategies,
        Vector<double>? strategyWeights)
    {
        // Create tuple array from strategies and weights
        (IDistillationStrategy<Vector<T>, T> Strategy, double Weight)[] strategyTuples;

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

    /// <summary>
    /// Creates a strategy with custom parameters using a fluent builder pattern.
    /// </summary>
    public static StrategyBuilder<T> Configure(DistillationStrategyType strategyType)
    {
        return new StrategyBuilder<T>(strategyType);
    }

    /// <summary>
    /// Fluent builder for configuring distillation strategies with custom parameters.
    /// </summary>
    public class StrategyBuilder<TNum>
    {
        private readonly DistillationStrategyType _strategyType;
        private double _temperature = 3.0;
        private double _alpha = 0.3;
        private double? _featureWeight;
        private double? _attentionWeight;
        private ContrastiveMode? _contrastiveMode;
        private Vector<string>? _featureLayerPairs;
        private Vector<string>? _attentionLayers;
        private Vector<IDistillationStrategy<Vector<TNum>, TNum>>? _strategies;
        private Vector<double>? _strategyWeights;

        internal StrategyBuilder(DistillationStrategyType strategyType)
        {
            _strategyType = strategyType;
        }

        public StrategyBuilder<TNum> WithTemperature(double temperature)
        {
            _temperature = temperature;
            return this;
        }

        public StrategyBuilder<TNum> WithAlpha(double alpha)
        {
            _alpha = alpha;
            return this;
        }

        public StrategyBuilder<TNum> WithFeatureWeight(double weight)
        {
            _featureWeight = weight;
            return this;
        }

        public StrategyBuilder<TNum> WithAttentionWeight(double weight)
        {
            _attentionWeight = weight;
            return this;
        }

        public StrategyBuilder<TNum> WithContrastiveMode(ContrastiveMode mode)
        {
            _contrastiveMode = mode;
            return this;
        }

        public StrategyBuilder<TNum> WithFeatureLayerPairs(Vector<string> layerPairs)
        {
            _featureLayerPairs = layerPairs;
            return this;
        }

        public StrategyBuilder<TNum> WithAttentionLayers(Vector<string> layers)
        {
            _attentionLayers = layers;
            return this;
        }

        public StrategyBuilder<TNum> WithStrategies(
            Vector<IDistillationStrategy<Vector<TNum>, TNum>> strategies,
            Vector<double>? weights = null)
        {
            _strategies = strategies;
            _strategyWeights = weights;
            return this;
        }

        public IDistillationStrategy<Vector<TNum>, TNum> Build()
        {
            return CreateStrategy(
                _strategyType,
                _temperature,
                _alpha,
                _featureWeight,
                _attentionWeight,
                _contrastiveMode,
                _featureLayerPairs,
                _attentionLayers,
                _strategies,
                _strategyWeights);
        }
    }
}
