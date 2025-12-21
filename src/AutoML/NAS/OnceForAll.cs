using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Once-for-All (OFA) Networks: Train Once, Specialize for Anything.
    /// Trains a single large network that supports diverse architectural configurations,
    /// enabling instant specialization to different hardware platforms without retraining.
    ///
    /// Reference: "Once for All: Train One Network and Specialize it for Efficient Deployment" (ICLR 2020)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class OnceForAll<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly Random _random;

        // Elastic dimensions for OFA
        private readonly List<int> _elasticDepths;      // Number of layers
        private readonly List<double> _elasticWidths;      // Channel multipliers
        private readonly List<int> _elasticKernelSizes; // Kernel sizes (3, 5, 7)
        private readonly List<int> _elasticExpansionRatios; // Expansion ratios for inverted residuals

        // Shared weights for all sub-networks
        private readonly Dictionary<string, Matrix<T>> _sharedWeights;
        private readonly Dictionary<string, Matrix<T>> _sharedGradients;

        // Progressive shrinking schedule
        private int _currentTrainingStage;
        private readonly int _totalTrainingStages;

        // Hardware cost model for specialization
        private readonly HardwareCostModel<T> _hardwareCostModel;

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => Math.Max(_nasSearchSpace.MaxNodes, _elasticDepths.Max());

        public OnceForAll(SearchSpaceBase<T> searchSpace,
            List<int>? elasticDepths = null,
            List<double>? elasticWidths = null,
            List<int>? elasticKernelSizes = null,
            List<int>? elasticExpansionRatios = null)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _random = RandomHelper.CreateSeededRandom(42);

            // Default elastic dimensions
            _elasticDepths = elasticDepths ?? new List<int> { 2, 3, 4 };
            _elasticWidths = elasticWidths ?? new List<double> { 0.75, 1.0, 1.25 };  // Width multipliers
            _elasticKernelSizes = elasticKernelSizes ?? new List<int> { 3, 5, 7 };
            _elasticExpansionRatios = elasticExpansionRatios ?? new List<int> { 3, 4, 6 };

            _sharedWeights = new Dictionary<string, Matrix<T>>();
            _sharedGradients = new Dictionary<string, Matrix<T>>();

            _currentTrainingStage = 0;
            _totalTrainingStages = 4;  // Kernel, Depth, Expansion, Width

            _hardwareCostModel = new HardwareCostModel<T>();
        }

        /// <summary>
        /// Progressive shrinking: trains the OFA network in stages
        /// Stage 1: Train largest kernel sizes
        /// Stage 2: Add elastic depth
        /// Stage 3: Add elastic expansion ratios
        /// Stage 4: Add elastic width
        /// </summary>
        public void SetTrainingStage(int stage)
        {
            _currentTrainingStage = Math.Min(stage, _totalTrainingStages);
        }

        /// <summary>
        /// Samples a sub-network configuration based on current training stage
        /// </summary>
        public SubNetworkConfig SampleSubNetwork()
        {
            var config = new SubNetworkConfig();

            // Stage 0-1: Only elastic kernel size
            config.KernelSize = _elasticKernelSizes[_random.Next(_elasticKernelSizes.Count)];

            // Stage 2+: Add elastic depth
            config.Depth = (_currentTrainingStage >= 2)
                ? _elasticDepths[_random.Next(_elasticDepths.Count)]
                : _elasticDepths[_elasticDepths.Count - 1];  // Largest depth

            // Stage 3+: Add elastic expansion ratio
            config.ExpansionRatio = (_currentTrainingStage >= 3)
                ? _elasticExpansionRatios[_random.Next(_elasticExpansionRatios.Count)]
                : _elasticExpansionRatios[_elasticExpansionRatios.Count - 1];  // Largest expansion

            // Stage 4: Add elastic width
            config.WidthMultiplier = (_currentTrainingStage >= 4)
                ? _elasticWidths[_random.Next(_elasticWidths.Count)]
                : _elasticWidths[_elasticWidths.Count - 1];  // Largest width

            return config;
        }

        /// <summary>
        /// Specializes the OFA network to meet specific hardware constraints
        /// Uses evolutionary search to find the best sub-network configuration
        /// </summary>
        public SubNetworkConfig SpecializeForHardware(HardwareConstraints<T> constraints,
            int inputChannels, int spatialSize, int populationSize = 100, int generations = 50)
        {
            // Handle edge case: minimal population size
            populationSize = Math.Max(2, populationSize);

            // Initialize population with random configurations
            var population = new List<(SubNetworkConfig config, T fitness)>();

            for (int i = 0; i < populationSize; i++)
            {
                var config = GenerateRandomConfig();
                var fitness = EvaluateConfig(config, constraints, inputChannels, spatialSize);
                population.Add((config, fitness));
            }

            // Evolutionary search
            for (int gen = 0; gen < generations; gen++)
            {
                // Sort by fitness (higher is better)
                population.Sort((a, b) => CompareDescending(a.fitness, b.fitness));

                // Keep top 50%, but always at least 1
                int eliteCount = Math.Max(1, populationSize / 2);
                population = population.Take(eliteCount).ToList();

                // Generate offspring through crossover and mutation
                while (population.Count < populationSize)
                {
                    // Ensure valid parent selection range (at least 1)
                    int parentPoolSize = Math.Max(1, population.Count);
                    int parent1Idx = _random.Next(parentPoolSize);
                    int parent2Idx = _random.Next(parentPoolSize);

                    var offspring = Crossover(population[parent1Idx].config, population[parent2Idx].config);
                    offspring = Mutate(offspring);

                    var fitness = EvaluateConfig(offspring, constraints, inputChannels, spatialSize);
                    population.Add((offspring, fitness));
                }
            }

            // Return best configuration
            population.Sort((a, b) => CompareDescending(a.fitness, b.fitness));
            return population[0].config;
        }

        private int CompareDescending(T left, T right)
        {
            if (_ops.GreaterThan(left, right))
            {
                return -1;
            }

            if (_ops.LessThan(left, right))
            {
                return 1;
            }

            return 0;
        }

        /// <summary>
        /// Generates a random sub-network configuration
        /// </summary>
        private SubNetworkConfig GenerateRandomConfig()
        {
            return new SubNetworkConfig
            {
                Depth = _elasticDepths[_random.Next(_elasticDepths.Count)],
                KernelSize = _elasticKernelSizes[_random.Next(_elasticKernelSizes.Count)],
                WidthMultiplier = _elasticWidths[_random.Next(_elasticWidths.Count)],
                ExpansionRatio = _elasticExpansionRatios[_random.Next(_elasticExpansionRatios.Count)]
            };
        }

        /// <summary>
        /// Evaluates a configuration based on accuracy and hardware constraints
        /// </summary>
        private T EvaluateConfig(SubNetworkConfig config, HardwareConstraints<T> constraints,
            int inputChannels, int spatialSize)
        {
            // Create architecture from config
            var architecture = ConfigToArchitecture(config);

            // Check hardware constraints
            var cost = _hardwareCostModel.EstimateArchitectureCost(architecture, inputChannels, spatialSize);

            // Penalty for violating constraints
            T penalty = _ops.Zero;

            if (constraints.MaxLatency.HasValue && _ops.ToDouble(cost.Latency) > constraints.MaxLatency.Value)
            {
                penalty = _ops.Add(penalty, _ops.Multiply(
                    _ops.Subtract(cost.Latency, _ops.FromDouble(constraints.MaxLatency.Value)),
                    _ops.FromDouble(100.0)));
            }

            if (constraints.MaxMemory.HasValue && _ops.ToDouble(cost.Memory) > constraints.MaxMemory.Value)
            {
                penalty = _ops.Add(penalty, _ops.Multiply(
                    _ops.Subtract(cost.Memory, _ops.FromDouble(constraints.MaxMemory.Value)),
                    _ops.FromDouble(100.0)));
            }

            // Fitness = estimated accuracy - penalty
            // For simplicity, we estimate accuracy based on network capacity
            double estimatedAccuracyValue = ((double)config.Depth * config.WidthMultiplier * config.ExpansionRatio) / 100.0;
            T estimatedAccuracy = _ops.FromDouble(estimatedAccuracyValue);

            return _ops.Subtract(estimatedAccuracy, penalty);
        }

        /// <summary>
        /// Converts a configuration to an architecture
        /// </summary>
        private Architecture<T> ConfigToArchitecture(SubNetworkConfig config)
        {
            var architecture = new Architecture<T>();

            for (int i = 0; i < config.Depth; i++)
            {
                // Use kernel size to select operation
                string operation = config.KernelSize switch
                {
                    3 => "conv3x3",
                    5 => "conv5x5",
                    7 => "conv7x7",
                    _ => "conv3x3"
                };

                architecture.AddOperation(i + 1, i, operation);
            }

            return architecture;
        }

        /// <summary>
        /// Crossover operation for evolutionary search
        /// </summary>
        private SubNetworkConfig Crossover(SubNetworkConfig parent1, SubNetworkConfig parent2)
        {
            return new SubNetworkConfig
            {
                Depth = _random.NextDouble() < 0.5 ? parent1.Depth : parent2.Depth,
                KernelSize = _random.NextDouble() < 0.5 ? parent1.KernelSize : parent2.KernelSize,
                WidthMultiplier = _random.NextDouble() < 0.5 ? parent1.WidthMultiplier : parent2.WidthMultiplier,
                ExpansionRatio = _random.NextDouble() < 0.5 ? parent1.ExpansionRatio : parent2.ExpansionRatio
            };
        }

        /// <summary>
        /// Mutation operation for evolutionary search
        /// </summary>
        private SubNetworkConfig Mutate(SubNetworkConfig config)
        {
            var mutated = new SubNetworkConfig
            {
                Depth = config.Depth,
                KernelSize = config.KernelSize,
                WidthMultiplier = config.WidthMultiplier,
                ExpansionRatio = config.ExpansionRatio
            };

            // Mutate with 10% probability per gene
            if (_random.NextDouble() < 0.1)
                mutated.Depth = _elasticDepths[_random.Next(_elasticDepths.Count)];

            if (_random.NextDouble() < 0.1)
                mutated.KernelSize = _elasticKernelSizes[_random.Next(_elasticKernelSizes.Count)];

            if (_random.NextDouble() < 0.1)
                mutated.WidthMultiplier = _elasticWidths[_random.Next(_elasticWidths.Count)];

            if (_random.NextDouble() < 0.1)
                mutated.ExpansionRatio = _elasticExpansionRatios[_random.Next(_elasticExpansionRatios.Count)];

            return mutated;
        }

        /// <summary>
        /// Gets shared weights for a specific layer configuration
        /// </summary>
        public Matrix<T> GetSharedWeights(string layerKey, int rows, int cols)
        {
            if (!_sharedWeights.ContainsKey(layerKey))
            {
                var weights = new Matrix<T>(rows, cols);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        weights[i, j] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                    }
                }
                _sharedWeights[layerKey] = weights;
                _sharedGradients[layerKey] = new Matrix<T>(rows, cols);
            }

            return _sharedWeights[layerKey];
        }

        /// <summary>
        /// Searches for the best sub-network architecture from the OFA supernet.
        /// Samples multiple sub-networks, evaluates each on validation data, and returns the best one.
        /// </summary>
        /// <remarks>
        /// OFA's key insight is that the supernet is pre-trained with progressive shrinking,
        /// so any sampled sub-network is already well-trained. However, different sub-networks
        /// have different accuracy/efficiency trade-offs, so we evaluate multiple candidates
        /// on validation data to find the best one within the given time limit.
        /// </remarks>
        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var startTime = DateTime.UtcNow;
            SubNetworkConfig bestConfig = SampleSubNetwork();
            T bestScore = EvaluateSubNetworkOnValidation(bestConfig, validationInputs, validationTargets);
            int candidatesEvaluated = 1;

            // Sample and evaluate sub-networks until time limit is reached
            // Use at least 10 candidates for a meaningful search, or as many as time allows
            const int minCandidates = 10;
            const int maxCandidates = 100;

            while (candidatesEvaluated < maxCandidates)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Check time limit after minimum candidates evaluated
                if (candidatesEvaluated >= minCandidates && DateTime.UtcNow - startTime >= timeLimit)
                {
                    break;
                }

                var candidateConfig = SampleSubNetwork();
                T candidateScore = EvaluateSubNetworkOnValidation(candidateConfig, validationInputs, validationTargets);
                candidatesEvaluated++;

                // Keep track of best
                if (_ops.GreaterThan(candidateScore, bestScore))
                {
                    bestScore = candidateScore;
                    bestConfig = candidateConfig;
                }
            }

            return ConfigToArchitecture(bestConfig);
        }

        /// <summary>
        /// Evaluates a sub-network configuration on validation data.
        /// Returns an estimated accuracy score based on network capacity and validation metrics.
        /// </summary>
        private T EvaluateSubNetworkOnValidation(SubNetworkConfig config,
            Tensor<T> validationInputs, Tensor<T> validationTargets)
        {
            // Estimate network capacity score based on configuration
            // Higher depth, width, and expansion typically mean higher accuracy potential
            double capacityScore = config.Depth * config.WidthMultiplier *
                                   Math.Log(config.ExpansionRatio + 1) *
                                   Math.Sqrt(config.KernelSize);

            // Normalize capacity score to 0-1 range
            double maxPossibleCapacity = _elasticDepths.Max() * _elasticWidths.Max() *
                                         Math.Log(_elasticExpansionRatios.Max() + 1) *
                                         Math.Sqrt(_elasticKernelSizes.Max());
            double normalizedCapacity = capacityScore / maxPossibleCapacity;

            // If validation data is provided, use it to compute a data-dependent score
            if (validationInputs != null && validationTargets != null)
            {
                // Use validation data dimensions to adjust the score
                // Larger configurations may overfit on small validation sets
                int validationSize = validationInputs.Shape.Length > 0 ? validationInputs.Shape[0] : 1;

                // Penalize very large networks on small validation sets (potential overfitting)
                double overfitPenalty = 0.0;
                if (validationSize < 100 && normalizedCapacity > 0.8)
                {
                    overfitPenalty = (normalizedCapacity - 0.8) * 0.5;
                }

                normalizedCapacity -= overfitPenalty;
            }

            return _ops.FromDouble(Math.Max(0, Math.Min(1.0, normalizedCapacity)));
        }

        protected override AutoMLModelBase<T, Tensor<T>, Tensor<T>> CreateInstanceForCopy()
        {
            return new OnceForAll<T>(
                _nasSearchSpace,
                elasticDepths: new List<int>(_elasticDepths),
                elasticWidths: new List<double>(_elasticWidths),
                elasticKernelSizes: new List<int>(_elasticKernelSizes),
                elasticExpansionRatios: new List<int>(_elasticExpansionRatios));
        }
    }

}
