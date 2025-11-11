using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;

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
    /// <param name="contrastiveLossType">Type of contrastive loss (for contrastive strategies).</param>
    /// <param name="strategies">Array of strategies to combine (for hybrid).</param>
    /// <param name="strategyWeights">Weights for combined strategies (for hybrid).</param>
    /// <returns>A configured distillation strategy.</returns>
    public static IDistillationStrategy<Vector<T>, T> CreateStrategy(
        DistillationStrategyType strategyType,
        double temperature = 3.0,
        double alpha = 0.3,
        double? featureWeight = null,
        double? attentionWeight = null,
        ContrastiveLossType? contrastiveLossType = null,
        IDistillationStrategy<Vector<T>, T>[]? strategies = null,
        double[]? strategyWeights = null)
    {
        return strategyType switch
        {
            DistillationStrategyType.ResponseBased => CreateResponseBasedStrategy(temperature, alpha),
            DistillationStrategyType.FeatureBased => CreateFeatureBasedStrategy(temperature, alpha, featureWeight ?? 0.5),
            DistillationStrategyType.AttentionBased => CreateAttentionBasedStrategy(temperature, alpha, attentionWeight ?? 0.5),
            DistillationStrategyType.RelationBased => CreateRelationBasedStrategy(temperature, alpha),
            DistillationStrategyType.ContrastiveBased => CreateContrastiveStrategy(temperature, alpha, contrastiveLossType ?? ContrastiveLossType.InfoNCE),
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
        double temperature,
        double alpha,
        double featureWeight)
    {
        return new FeatureDistillationStrategy<T>(
            featureWeight: featureWeight,
            matchingMode: FeatureMatchingMode.MSE,
            temperature: temperature,
            alpha: alpha);
    }

    private static IDistillationStrategy<Vector<T>, T> CreateAttentionBasedStrategy(
        double temperature,
        double alpha,
        double attentionWeight)
    {
        return new AttentionDistillationStrategy<T>(
            attentionWeight: attentionWeight,
            matchingMode: AttentionMatchingMode.MSE,
            temperature: temperature,
            alpha: alpha);
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
        ContrastiveLossType lossType)
    {
        return new ContrastiveDistillationStrategy<T>(
            contrastiveWeight: 0.5,
            lossType: lossType,
            temperature: temperature,
            alpha: alpha);
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
        IDistillationStrategy<Vector<T>, T>[]? strategies,
        double[]? strategyWeights)
    {
        if (strategies == null || strategies.Length == 0)
        {
            // Default: combine response-based and feature-based
            strategies = new IDistillationStrategy<Vector<T>, T>[]
            {
                CreateResponseBasedStrategy(temperature, alpha),
                CreateFeatureBasedStrategy(temperature, alpha, 0.5)
            };
            strategyWeights = new[] { 0.5, 0.5 };
        }

        if (strategyWeights == null)
        {
            // Equal weights if not specified
            strategyWeights = new double[strategies.Length];
            double equalWeight = 1.0 / strategies.Length;
            for (int i = 0; i < strategyWeights.Length; i++)
                strategyWeights[i] = equalWeight;
        }

        return new HybridDistillationStrategy<T>(strategies, strategyWeights, temperature, alpha);
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
        private ContrastiveLossType? _contrastiveLossType;
        private IDistillationStrategy<Vector<TNum>, TNum>[]? _strategies;
        private double[]? _strategyWeights;

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

        public StrategyBuilder<TNum> WithContrastiveLossType(ContrastiveLossType lossType)
        {
            _contrastiveLossType = lossType;
            return this;
        }

        public StrategyBuilder<TNum> WithStrategies(
            IDistillationStrategy<Vector<TNum>, TNum>[] strategies,
            double[]? weights = null)
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
                _contrastiveLossType,
                _strategies,
                _strategyWeights);
        }
    }
}
