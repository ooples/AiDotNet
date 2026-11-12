using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling.
    /// Uses an attention-based meta-network to guide the sampling of sub-networks,
    /// focusing search on promising regions of the architecture space.
    ///
    /// Reference: "AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling" (CVPR 2021)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <remarks>
    /// <para><b>For Beginners:</b> AttentiveNAS improves architecture search by paying
    /// attention to which sub-networks perform well. Instead of sampling random architectures,
    /// it learns to focus on promising designs, like a smart student who concentrates on
    /// the most important study topics rather than studying everything equally.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var searchSpace = new SearchSpaceBase&lt;float&gt;();
    /// var nas = new AttentiveNAS&lt;float&gt;(searchSpace);
    /// Architecture&lt;float&gt; architecture = nas.BestArchitecture;
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.Optimization)]
    [ModelTask(ModelTask.Classification)]
    [ModelTask(ModelTask.FeatureExtraction)]
    [ModelComplexity(ModelComplexity.VeryHigh)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling", "https://arxiv.org/abs/2011.09011")]
    public partial class AttentiveNAS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly Random _random;

        // Super-network with elastic dimensions
        private readonly List<int> _elasticDepths;
        private readonly List<double> _elasticWidthMultipliers;
        private readonly List<int> _elasticKernelSizes;

        // Attention module parameters
        [AiDotNet.Attributes.TrainableParameter]
        private readonly Tensor<T> _attentionWeights;
        [AiDotNet.Attributes.TrainableParameter]
        private readonly Matrix<T> _attentionGradients;
        private readonly int _attentionHiddenSize;

        // Architecture sampling parameters
        private readonly Dictionary<string, T> _performanceMemory;

        // Hardware cost model
        private readonly HardwareCostModel<T> _hardwareCostModel;

        /// <inheritdoc />
        protected override INumericOperations<T> NumOps => _ops;
        /// <inheritdoc />
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        /// <inheritdoc />
        protected override int NasNumNodes => Math.Max(_nasSearchSpace.MaxNodes, _elasticDepths.Max());

        public AttentiveNAS(SearchSpaceBase<T> searchSpace,
            List<int>? elasticDepths = null,
            List<double>? elasticWidthMultipliers = null,
            List<int>? elasticKernelSizes = null,
            int attentionHiddenSize = 128)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _random = RandomHelper.CreateSeededRandom(42);

            _elasticDepths = elasticDepths ?? new List<int> { 2, 3, 4, 5 };
            _elasticWidthMultipliers = elasticWidthMultipliers ?? new List<double> { 0.5, 0.75, 1.0, 1.25 };
            _elasticKernelSizes = elasticKernelSizes ?? new List<int> { 3, 5, 7 };

            _attentionHiddenSize = attentionHiddenSize;

            // Initialize attention module
            int numArchitectureChoices = _elasticDepths.Count + _elasticWidthMultipliers.Count + _elasticKernelSizes.Count;
            _attentionWeights = new Tensor<T>([_attentionHiddenSize, numArchitectureChoices]);
            _attentionGradients = new Matrix<T>(_attentionHiddenSize, numArchitectureChoices);

            for (int i = 0; i < _attentionWeights.Shape[0]; i++)
            {
                for (int j = 0; j < _attentionWeights.Shape[1]; j++)
                {
                    _attentionWeights[i, j] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
            }

            _performanceMemory = new Dictionary<string, T>();

            _hardwareCostModel = new HardwareCostModel<T>();
        }

        /// <summary>
        /// Gets or sets which candidate attentive sampling trains: the predicted Pareto-best (BestUp, the default)
        /// or Pareto-worst (WorstUp) of the pool drawn under a FLOPs target (Wang et al. 2021, Sec. 3.2).
        /// </summary>
        public AttentiveSamplingStrategy SamplingStrategy { get; set; } = AttentiveSamplingStrategy.BestUp;

        private int _candidatePoolSize = 3;

        /// <summary>
        /// Gets or sets k, the number of candidates drawn under each FLOPs target before the best or worst is
        /// picked (default 3).
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException">The value is less than 1.</exception>
        public int CandidatePoolSize
        {
            get => _candidatePoolSize;
            set
            {
                if (value < 1)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), value, "The candidate pool needs at least one architecture.");
                }

                _candidatePoolSize = value;
            }
        }

        private const int MaxRejectionDraws = 100;

        /// <summary>
        /// Samples an architecture by attentive sampling (Wang et al. 2021, Sec. 3.2): draw a FLOPs target tau from
        /// the FLOPs prior of the space, draw k candidates uniformly under that target, and rank them with the
        /// performance predictor; <see cref="SamplingStrategy"/> picks the predicted best (BestUp) or worst
        /// (WorstUp).
        /// </summary>
        public AttentiveNASConfig<T> AttentiveSample(Vector<T> contextVector)
        {
            var candidates = DrawCandidates(out _);
            return SelectCandidate(candidates, contextVector);
        }

        /// <summary>
        /// Draws the FLOPs target tau as the FLOPs of a uniformly random architecture (a draw from the space's
        /// FLOPs prior), then <see cref="CandidatePoolSize"/> uniform architectures with FLOPs &lt;= tau by
        /// rejection; after <see cref="MaxRejectionDraws"/> rejected draws a candidate falls back to the
        /// smallest architecture, which is always under the target.
        /// </summary>
        internal List<AttentiveNASConfig<T>> DrawCandidates(out double flopsTarget)
        {
            double target = RelativeFlops(RandomConfig());
            flopsTarget = target;
            var smallest = new AttentiveNASConfig<T>
            {
                Depth = _elasticDepths.Min(),
                WidthMultiplier = _elasticWidthMultipliers.Min(),
                KernelSize = _elasticKernelSizes.Min()
            };

            return Enumerable.Range(0, CandidatePoolSize)
                .Select(_ => Enumerable.Range(0, MaxRejectionDraws)
                    .Select(__ => RandomConfig())
                    .FirstOrDefault(config => RelativeFlops(config) <= target)
                    ?? new AttentiveNASConfig<T>
                    {
                        Depth = smallest.Depth,
                        WidthMultiplier = smallest.WidthMultiplier,
                        KernelSize = smallest.KernelSize
                    })
                .ToList();
        }

        /// <summary>
        /// Gives every candidate its predictor features under <paramref name="contextVector"/> and returns the
        /// first candidate with the highest (BestUp) or lowest (WorstUp) predicted performance.
        /// </summary>
        internal AttentiveNASConfig<T> SelectCandidate(IReadOnlyList<AttentiveNASConfig<T>> candidates, Vector<T> contextVector)
        {
            foreach (var candidate in candidates)
            {
                candidate.Embedding = PredictorFeatures(candidate, contextVector);
            }

            var engine = AiDotNetEngine.Current;
            var predicted = new Vector<T>(candidates.Select(PredictPerformance).ToArray());
            var key = SamplingStrategy == AttentiveSamplingStrategy.WorstUp ? engine.Negate(predicted) : predicted;
            return candidates[NasSamplingHelper.ArgMax(key)];
        }

        /// <summary>A uniformly random elastic configuration (depth, then width, then kernel).</summary>
        private AttentiveNASConfig<T> RandomConfig()
        {
            return new AttentiveNASConfig<T>
            {
                Depth = _elasticDepths[_random.Next(_elasticDepths.Count)],
                WidthMultiplier = _elasticWidthMultipliers[_random.Next(_elasticWidthMultipliers.Count)],
                KernelSize = _elasticKernelSizes[_random.Next(_elasticKernelSizes.Count)]
            };
        }

        /// <summary>
        /// FLOPs of a configuration up to a constant: a k x k convolution stack scales with depth, with the square
        /// of the width multiplier (input and output channels) and with k^2.
        /// </summary>
        internal static double RelativeFlops(AttentiveNASConfig<T> config)
        {
            return config.Depth * config.WidthMultiplier * config.WidthMultiplier * config.KernelSize * config.KernelSize;
        }

        /// <summary>
        /// The predictor input for a configuration: its architecture embedding plus the sampling context over their
        /// shared length.
        /// </summary>
        private Vector<T> PredictorFeatures(AttentiveNASConfig<T> config, Vector<T> contextVector)
        {
            var features = CreateArchitectureEmbedding(config);
            int shared = Math.Min(features.Length, contextVector.Length);
            if (shared > 0)
            {
                AiDotNetEngine.Current.Add(features.GetSubVector(0, shared), contextVector.GetSubVector(0, shared))
                    .AsSpan().CopyTo(features.AsWritableSpan());
            }

            return features;
        }

        /// <summary>
        /// The attention module as a linear performance predictor: x^T (W[:, d] + W[:, w] + W[:, k]) for the
        /// configuration's features x (<see cref="AttentiveNASConfig{T}.Embedding"/>) and its chosen depth, width
        /// and kernel columns. Zero for a configuration outside the elastic choices or without features.
        /// </summary>
        internal T PredictPerformance(AttentiveNASConfig<T> config)
        {
            var columnMask = ChoiceColumnMask(config);
            var features = config.Embedding;
            int shared = features is null ? 0 : Math.Min(_attentionWeights.Shape[0], features.Length);
            if (columnMask is null || features is null || shared == 0)
            {
                return _ops.Zero;
            }

            var engine = AiDotNetEngine.Current;
            var projected = engine.TensorMatMul(
                features.GetSubVector(0, shared).AsTensor().Reshape(new[] { 1, shared }),
                _attentionWeights.Slice(0, 0, shared)).Reshape(new[] { columnMask.Length });
            return engine.TensorSum(engine.TensorMultiply(projected, columnMask));
        }

        /// <summary>
        /// A [choices] mask with ones at the configuration's depth, width and kernel columns, or null when a value
        /// is not one of the elastic choices.
        /// </summary>
        private Tensor<T>? ChoiceColumnMask(AttentiveNASConfig<T> config)
        {
            int depthIdx = _elasticDepths.IndexOf(config.Depth);
            int widthIdx = _elasticWidthMultipliers.IndexOf(config.WidthMultiplier);
            int kernelIdx = _elasticKernelSizes.IndexOf(config.KernelSize);

            if (depthIdx < 0 || widthIdx < 0 || kernelIdx < 0)
            {
                return null;
            }

            var chosenColumns = new Tensor<int>(new[] { 3 });
            chosenColumns[0] = depthIdx;
            chosenColumns[1] = _elasticDepths.Count + widthIdx;
            chosenColumns[2] = _elasticDepths.Count + _elasticWidthMultipliers.Count + kernelIdx;
            return AiDotNetEngine.Current.ReduceSum(
                AiDotNetEngine.Current.TensorOneHot<T>(chosenColumns, _attentionWeights.Shape[1]), new[] { 0 }, false);
        }

        /// <summary>
        /// Creates an embedding for an architecture configuration
        /// </summary>
        private Vector<T> CreateArchitectureEmbedding(AttentiveNASConfig<T> config)
        {
            var embedding = new Vector<T>(_attentionHiddenSize);

            // Encode configuration as embedding (simplified)
            embedding[0] = _ops.FromDouble(config.Depth / 10.0);
            embedding[1] = _ops.FromDouble(config.WidthMultiplier);
            embedding[2] = _ops.FromDouble(config.KernelSize / 10.0);

            // The remaining entries stay zero (new vectors are zero-filled).

            return embedding;
        }

        /// <summary>
        /// Updates the attention module based on architecture performance: one gradient step of the predictor on
        /// the squared error (prediction - performance)^2 / 2, i.e. W[:, c] -= lr * (prediction - performance) * x
        /// for the configuration's three columns c.
        /// </summary>
        public void UpdateAttention(AttentiveNASConfig<T> config, T performance, T learningRate)
        {
            // Store performance in memory
            string configKey = $"{config.Depth}_{config.WidthMultiplier}_{config.KernelSize}";
            _performanceMemory[configKey] = performance;

            var columnMask = ChoiceColumnMask(config);
            if (columnMask is null)
            {
                return;
            }

            var embedding = config.Embedding
                ?? throw new InvalidOperationException("Config embedding must not be null when updating attention weights.");

            int hiddenRows = _attentionWeights.Shape[0];
            int shared = Math.Min(hiddenRows, embedding.Length);
            if (shared == 0)
            {
                return;
            }

            T error = _ops.Subtract(PredictPerformance(config), performance);
            var engine = AiDotNetEngine.Current;
            var update = new Tensor<T>(new[] { hiddenRows });
            update.SetSubTensor(new[] { 0 }, engine.TensorMultiplyScalar(
                engine.TensorMultiplyScalar(embedding.GetSubVector(0, shared).AsTensor(), error), learningRate));
            engine.TensorSubtractInPlace(_attentionWeights, engine.TensorOuter(update, columnMask));
        }

        /// <summary>
        /// Creates a context vector from recent architecture performance history
        /// </summary>
        public Vector<T> CreateContextVector()
        {
            var context = new Vector<T>(_attentionHiddenSize);

            if (_performanceMemory.Count > 0)
            {
                // Simple context: average performance and recent trends
                T avgPerformance = _ops.Zero;
                foreach (var perf in _performanceMemory.Values)
                {
                    avgPerformance = _ops.Add(avgPerformance, perf);
                }
                avgPerformance = _ops.Divide(avgPerformance, _ops.FromDouble(_performanceMemory.Count));

                context[0] = avgPerformance;

                // Fill rest with random exploration
                for (int i = 1; i < context.Length; i++)
                {
                    context[i] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
            }
            else
            {
                // Initial exploration: random context
                for (int i = 0; i < context.Length; i++)
                {
                    context[i] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
            }

            return context;
        }

        /// <summary>
        /// Searches for optimal architecture using attentive sampling
        /// </summary>
        public AttentiveNASConfig<T> Search(HardwareConstraints<T> constraints,
            int inputChannels, int spatialSize, int numIterations = 100)
        {
            AttentiveNASConfig<T>? bestConfig = null;
            T bestFitness = _ops.FromDouble(double.MinValue);

            for (int iter = 0; iter < numIterations; iter++)
            {
                // Create context from history
                var context = CreateContextVector();

                // Sample architecture
                var config = AttentiveSample(context);

                // Evaluate
                var architecture = ConfigToArchitecture(config);
                var cost = _hardwareCostModel.EstimateArchitectureCost(architecture, inputChannels, spatialSize);

                // Compute fitness
                T fitness = _ops.FromDouble(config.Depth * config.WidthMultiplier * config.KernelSize);
                if (constraints.MaxLatency.HasValue && _ops.ToDouble(cost.Latency) > constraints.MaxLatency.Value)
                {
                    fitness = _ops.Subtract(fitness, _ops.FromDouble(10000.0));
                }

                // Update best
                if (_ops.GreaterThan(fitness, bestFitness))
                {
                    bestFitness = fitness;
                    bestConfig = config;
                }

                // Update attention module
                T learningRate = _ops.FromDouble(0.001);
                UpdateAttention(config, fitness, learningRate);
            }

            return bestConfig ?? new AttentiveNASConfig<T> { Depth = 3, WidthMultiplier = 1.0, KernelSize = 3, Embedding = new Vector<T>(_attentionHiddenSize) };
        }

        private Architecture<T> ConfigToArchitecture(AttentiveNASConfig<T> config)
        {
            var architecture = new Architecture<T>();
            for (int i = 0; i < config.Depth; i++)
            {
                string operation = config.KernelSize == 3 ? "conv3x3" : config.KernelSize == 5 ? "conv5x5" : "conv7x7";
                architecture.AddOperation(i + 1, i, operation);
            }
            return architecture;
        }

        /// <summary>
        /// Gets the attention weights
        /// </summary>
        public Tensor<T> GetAttentionWeights() => _attentionWeights;

        /// <summary>
        /// Gets the performance memory
        /// </summary>
        public Dictionary<string, T> GetPerformanceMemory() => _performanceMemory;

        /// <inheritdoc />
        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            var context = new Vector<T>(_attentionHiddenSize);
            var config = AttentiveSample(context);
            return ConfigToArchitecture(config);
        }
    }

}
