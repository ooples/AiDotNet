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
        /// Samples architecture using attention-based sampling strategy.
        /// The attention module learns to focus on high-performing architecture regions.
        /// </summary>
        public AttentiveNASConfig<T> AttentiveSample(Vector<T> contextVector)
        {
            // Compute attention scores for different architecture choices
            var attentionScores = ComputeAttentionScores(contextVector);

            // Sample based on attention distribution
            var config = new AttentiveNASConfig<T>();

            // Sample depth
            int depthStartIdx = 0;
            int depthEndIdx = _elasticDepths.Count;
            var depthScores = ExtractScores(attentionScores, depthStartIdx, depthEndIdx);
            config.Depth = _elasticDepths[SampleFromDistribution(depthScores)];

            // Sample width multiplier
            int widthStartIdx = depthEndIdx;
            int widthEndIdx = widthStartIdx + _elasticWidthMultipliers.Count;
            var widthScores = ExtractScores(attentionScores, widthStartIdx, widthEndIdx);
            config.WidthMultiplier = _elasticWidthMultipliers[SampleFromDistribution(widthScores)];

            // Sample kernel size
            int kernelStartIdx = widthEndIdx;
            int kernelEndIdx = kernelStartIdx + _elasticKernelSizes.Count;
            var kernelScores = ExtractScores(attentionScores, kernelStartIdx, kernelEndIdx);
            config.KernelSize = _elasticKernelSizes[SampleFromDistribution(kernelScores)];

            // Store architecture embedding for later updates
            config.Embedding = CreateArchitectureEmbedding(config);

            return config;
        }

        /// <summary>
        /// Computes attention scores using the attention module
        /// </summary>
        private Vector<T> ComputeAttentionScores(Vector<T> contextVector)
        {
            // Linear attention projection: scores_j = sum_i W[i, j] * context[i] over the shared hidden units
            var engine = AiDotNetEngine.Current;
            int numChoices = _attentionWeights.Shape[1];
            int shared = Math.Min(_attentionWeights.Shape[0], contextVector.Length);
            var scores = shared == 0
                ? new Vector<T>(numChoices)
                : engine.TensorMatMul(
                    contextVector.GetSubVector(0, shared).AsTensor().Reshape(new[] { 1, shared }),
                    _attentionWeights.Slice(0, 0, shared)).Reshape(new[] { numChoices }).ToVector();

            // Apply softmax to get probability distribution
            return Softmax(scores);
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
        /// Extracts a subset of scores for a specific architecture dimension
        /// </summary>
        private List<T> ExtractScores(Vector<T> allScores, int startIdx, int endIdx)
        {
            int count = endIdx - startIdx;
            if (count <= 0)
            {
                return new List<T>();
            }

            // Missing scores are padded with equal probabilities 1 / count.
            var scores = new Vector<T>(count);
            scores.Fill(_ops.FromDouble(1.0 / count));
            int available = Math.Max(0, Math.Min(endIdx, allScores.Length) - startIdx);
            if (available > 0)
            {
                allScores.GetSubVector(startIdx, available).AsSpan().CopyTo(scores.AsWritableSpan());
            }

            // Normalize to sum to 1
            var engine = AiDotNetEngine.Current;
            T sum = engine.Sum(scores);
            if (_ops.GreaterThan(sum, _ops.Zero))
            {
                scores = engine.Divide(scores, sum);
            }

            return new List<T>(scores.ToArray());
        }

        /// <summary>
        /// Applies softmax to a vector
        /// </summary>
        private Vector<T> Softmax(Vector<T> logits)
        {
            return AiDotNetEngine.Current.Softmax(logits);
        }

        /// <summary>
        /// Samples from a probability distribution
        /// </summary>
        private int SampleFromDistribution(List<T> probs)
        {
            double rand = _random.NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < probs.Count; i++)
            {
                cumulative += Convert.ToDouble(probs[i]);
                if (rand <= cumulative)
                    return i;
            }

            return probs.Count - 1;
        }

        /// <summary>
        /// Updates the attention module based on architecture performance.
        /// High-performing architectures increase attention to similar regions.
        /// </summary>
        public void UpdateAttention(AttentiveNASConfig<T> config, T performance, T learningRate)
        {
            // Store performance in memory
            string configKey = $"{config.Depth}_{config.WidthMultiplier}_{config.KernelSize}";
            _performanceMemory[configKey] = performance;

            int depthIdx = _elasticDepths.IndexOf(config.Depth);
            int widthIdx = _elasticWidthMultipliers.IndexOf(config.WidthMultiplier);
            int kernelIdx = _elasticKernelSizes.IndexOf(config.KernelSize);

            if (depthIdx < 0 || widthIdx < 0 || kernelIdx < 0)
            {
                return;
            }

            int depthCol = depthIdx;
            int widthCol = _elasticDepths.Count + widthIdx;
            int kernelCol = _elasticDepths.Count + _elasticWidthMultipliers.Count + kernelIdx;

            // Update attention weights based on performance gradient
            // This is a simplified update; full implementation would use policy gradients
            var embedding = config.Embedding
                ?? throw new InvalidOperationException("Config embedding must not be null when updating attention weights.");

            int hiddenRows = _attentionWeights.Shape[0];
            int shared = Math.Min(hiddenRows, embedding.Length);
            if (shared == 0)
            {
                return;
            }

            // Gradient approximation: update_i = learningRate * (performance * embedding_i), added to the three
            // chosen columns of every shared hidden row.
            var engine = AiDotNetEngine.Current;
            var update = new Tensor<T>(new[] { hiddenRows });
            update.SetSubTensor(new[] { 0 }, engine.TensorMultiplyScalar(
                engine.TensorMultiplyScalar(embedding.GetSubVector(0, shared).AsTensor(), performance), learningRate));
            var chosenColumns = new Tensor<int>(new[] { 3 });
            chosenColumns[0] = depthCol;
            chosenColumns[1] = widthCol;
            chosenColumns[2] = kernelCol;
            var columnMask = engine.ReduceSum(
                engine.TensorOneHot<T>(chosenColumns, _attentionWeights.Shape[1]), new[] { 0 }, false);
            engine.TensorAddInPlace(_attentionWeights, engine.TensorOuter(update, columnMask));
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
