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
    /// BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models.
    /// Combines sandwich sampling with in-place knowledge distillation to train
    /// very large super-networks that can adapt to various deployment scenarios.
    ///
    /// Reference: "BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models"
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class BigNAS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly Random _random;

        // Elastic search space dimensions (larger than OFA)
        private readonly List<int> _elasticDepths;
        private readonly List<double> _elasticWidthMultipliers;
        private readonly List<int> _elasticKernelSizes;
        private readonly List<int> _elasticExpansionRatios;
        private readonly List<int> _elasticResolutions;  // Input resolutions

        // Sandwich sampling parameters
        private readonly bool _useSandwichSampling;
        private readonly T _distillationWeight;

        // Hardware cost model
        private readonly HardwareCostModel<T> _hardwareCostModel;

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => Math.Max(_nasSearchSpace.MaxNodes, _elasticDepths.Max());

        public BigNAS(SearchSpaceBase<T> searchSpace,
            List<int>? elasticDepths = null,
            List<double>? elasticWidthMultipliers = null,
            List<int>? elasticKernelSizes = null,
            List<int>? elasticExpansionRatios = null,
            List<int>? elasticResolutions = null,
            bool useSandwichSampling = true,
            double distillationWeight = 0.5)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _random = RandomHelper.CreateSeededRandom(42);

            // BigNAS supports larger search spaces than OFA
            _elasticDepths = elasticDepths ?? new List<int> { 2, 3, 4, 5, 6 };
            _elasticWidthMultipliers = elasticWidthMultipliers ?? new List<double> { 0.5, 0.65, 0.75, 1.0, 1.2 };
            _elasticKernelSizes = elasticKernelSizes ?? new List<int> { 3, 5, 7 };
            _elasticExpansionRatios = elasticExpansionRatios ?? new List<int> { 3, 4, 6 };
            _elasticResolutions = elasticResolutions ?? new List<int> { 128, 160, 192, 224, 256 };

            _useSandwichSampling = useSandwichSampling;
            _distillationWeight = _ops.FromDouble(distillationWeight);

            _hardwareCostModel = new HardwareCostModel<T>();
        }

        /// <summary>
        /// Sandwich sampling: samples smallest, largest, and random sub-networks together
        /// This improves training efficiency and performance of all sub-networks
        /// </summary>
        public List<BigNASConfig> SandwichSample()
        {
            var samples = new List<BigNASConfig>();

            if (_useSandwichSampling)
            {
                // 1. Largest sub-network (teacher)
                samples.Add(new BigNASConfig
                {
                    Depth = _elasticDepths[_elasticDepths.Count - 1],
                    WidthMultiplier = _elasticWidthMultipliers[_elasticWidthMultipliers.Count - 1],
                    KernelSize = _elasticKernelSizes[_elasticKernelSizes.Count - 1],
                    ExpansionRatio = _elasticExpansionRatios[_elasticExpansionRatios.Count - 1],
                    Resolution = _elasticResolutions[_elasticResolutions.Count - 1],
                    IsTeacher = true
                });

                // 2. Smallest sub-network
                samples.Add(new BigNASConfig
                {
                    Depth = _elasticDepths[0],
                    WidthMultiplier = _elasticWidthMultipliers[0],
                    KernelSize = _elasticKernelSizes[0],
                    ExpansionRatio = _elasticExpansionRatios[0],
                    Resolution = _elasticResolutions[0],
                    IsTeacher = false
                });

                // 3-4. Random sub-networks
                for (int i = 0; i < 2; i++)
                {
                    samples.Add(GenerateRandomConfig());
                }
            }
            else
            {
                // Standard uniform sampling
                for (int i = 0; i < 4; i++)
                {
                    samples.Add(GenerateRandomConfig());
                }
            }

            return samples;
        }

        /// <summary>
        /// Generates a random sub-network configuration
        /// </summary>
        private BigNASConfig GenerateRandomConfig()
        {
            return new BigNASConfig
            {
                Depth = _elasticDepths[_random.Next(_elasticDepths.Count)],
                WidthMultiplier = _elasticWidthMultipliers[_random.Next(_elasticWidthMultipliers.Count)],
                KernelSize = _elasticKernelSizes[_random.Next(_elasticKernelSizes.Count)],
                ExpansionRatio = _elasticExpansionRatios[_random.Next(_elasticExpansionRatios.Count)],
                Resolution = _elasticResolutions[_random.Next(_elasticResolutions.Count)],
                IsTeacher = false
            };
        }

        /// <summary>
        /// Computes knowledge distillation loss between teacher and student networks
        /// </summary>
        public T ComputeDistillationLoss(Vector<T> teacherLogits, Vector<T> studentLogits, T temperature)
        {
            if (teacherLogits.Length != studentLogits.Length)
            {
                throw new ArgumentException("Teacher and student logits must have the same length");
            }

            // Compute soft targets from teacher using temperature scaling
            var teacherSoftTargets = NasSamplingHelper.SoftmaxWithTemperature(teacherLogits, temperature, _ops);
            var studentSoftTargets = NasSamplingHelper.SoftmaxWithTemperature(studentLogits, temperature, _ops);

            // KL divergence loss
            T loss = _ops.Zero;
            for (int i = 0; i < teacherLogits.Length; i++)
            {
                if (_ops.GreaterThan(teacherSoftTargets[i], _ops.Zero))
                {
                    T logRatio = _ops.Log(_ops.Divide(
                        _ops.Add(teacherSoftTargets[i], _ops.FromDouble(1e-10)),
                        _ops.Add(studentSoftTargets[i], _ops.FromDouble(1e-10))
                    ));
                    loss = _ops.Add(loss, _ops.Multiply(teacherSoftTargets[i], logRatio));
                }
            }

            // Scale by temperature squared (following Hinton et al.)
            T tempSquared = _ops.Multiply(temperature, temperature);
            return _ops.Multiply(loss, tempSquared);
        }

        /// <summary>
        /// Searches for optimal sub-networks for multiple hardware constraints simultaneously
        /// </summary>
        public Dictionary<string, BigNASConfig> MultiObjectiveSearch(
            List<(string name, HardwareConstraints<T> constraints)> targetDevices,
            int inputChannels, int spatialSize,
            int populationSize = 100, int generations = 50)
        {
            var results = new Dictionary<string, BigNASConfig>();

            foreach (var (name, constraints) in targetDevices)
            {
                // Run evolutionary search for each target device
                var bestConfig = EvolutionarySearch(
                    constraints,
                    inputChannels,
                    spatialSize,
                    populationSize,
                    generations,
                    deadlineUtc: DateTime.MaxValue,
                    cancellationToken: CancellationToken.None);
                results[name] = bestConfig;
            }

            return results;
        }

        /// <summary>
        /// Evolutionary search for finding optimal sub-network
        /// </summary>
        private BigNASConfig EvolutionarySearch(
            HardwareConstraints<T> constraints,
            int inputChannels,
            int spatialSize,
            int populationSize,
            int generations,
            DateTime deadlineUtc,
            CancellationToken cancellationToken)
        {
            var population = new List<(BigNASConfig config, T fitness)>();

            // Initialize population
            for (int i = 0; i < populationSize && DateTime.UtcNow < deadlineUtc; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var config = GenerateRandomConfig();
                var fitness = EvaluateConfig(config, constraints, inputChannels, spatialSize);
                population.Add((config, fitness));
            }

            // Evolve
            for (int gen = 0; gen < generations && DateTime.UtcNow < deadlineUtc; gen++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                population.Sort((a, b) => CompareDescending(a.fitness, b.fitness));
                population = population.Take(populationSize / 2).ToList();

                while (population.Count < populationSize && DateTime.UtcNow < deadlineUtc)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int p1 = _random.Next(population.Count / 2);
                    int p2 = _random.Next(population.Count / 2);

                    var offspring = Crossover(population[p1].config, population[p2].config);
                    offspring = Mutate(offspring);

                    var fitness = EvaluateConfig(offspring, constraints, inputChannels, spatialSize);
                    population.Add((offspring, fitness));
                }
            }

            if (population.Count == 0)
            {
                return GenerateRandomConfig();
            }

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
        /// Evaluates a configuration
        /// </summary>
        private T EvaluateConfig(BigNASConfig config, HardwareConstraints<T> constraints,
            int inputChannels, int spatialSize)
        {
            var architecture = ConfigToArchitecture(config);
            var cost = _hardwareCostModel.EstimateArchitectureCost(architecture, inputChannels, spatialSize);

            T penalty = _ops.Zero;
            if (constraints.MaxLatency.HasValue && _ops.ToDouble(cost.Latency) > constraints.MaxLatency.Value)
                penalty = _ops.Add(penalty, _ops.Multiply(_ops.Subtract(cost.Latency, _ops.FromDouble(constraints.MaxLatency.Value)), _ops.FromDouble(1000.0)));

            T estimatedAccuracy = _ops.FromDouble(
                config.Depth * config.WidthMultiplier * config.ExpansionRatio * config.Resolution / 10000.0);

            return _ops.Subtract(estimatedAccuracy, penalty);
        }

        private Architecture<T> ConfigToArchitecture(BigNASConfig config)
        {
            var architecture = new Architecture<T>();
            for (int i = 0; i < config.Depth; i++)
            {
                string operation = config.KernelSize == 3 ? "conv3x3" : config.KernelSize == 5 ? "conv5x5" : "conv7x7";
                architecture.AddOperation(i + 1, i, operation);
            }
            return architecture;
        }

        private BigNASConfig Crossover(BigNASConfig p1, BigNASConfig p2)
        {
            return new BigNASConfig
            {
                Depth = _random.NextDouble() < 0.5 ? p1.Depth : p2.Depth,
                WidthMultiplier = _random.NextDouble() < 0.5 ? p1.WidthMultiplier : p2.WidthMultiplier,
                KernelSize = _random.NextDouble() < 0.5 ? p1.KernelSize : p2.KernelSize,
                ExpansionRatio = _random.NextDouble() < 0.5 ? p1.ExpansionRatio : p2.ExpansionRatio,
                Resolution = _random.NextDouble() < 0.5 ? p1.Resolution : p2.Resolution,
                IsTeacher = false
            };
        }

        private BigNASConfig Mutate(BigNASConfig config)
        {
            var mutated = new BigNASConfig
            {
                Depth = config.Depth,
                WidthMultiplier = config.WidthMultiplier,
                KernelSize = config.KernelSize,
                ExpansionRatio = config.ExpansionRatio,
                Resolution = config.Resolution,
                IsTeacher = config.IsTeacher
            };

            if (_random.NextDouble() < 0.1) mutated.Depth = _elasticDepths[_random.Next(_elasticDepths.Count)];
            if (_random.NextDouble() < 0.1) mutated.WidthMultiplier = _elasticWidthMultipliers[_random.Next(_elasticWidthMultipliers.Count)];
            if (_random.NextDouble() < 0.1) mutated.KernelSize = _elasticKernelSizes[_random.Next(_elasticKernelSizes.Count)];
            if (_random.NextDouble() < 0.1) mutated.ExpansionRatio = _elasticExpansionRatios[_random.Next(_elasticExpansionRatios.Count)];
            if (_random.NextDouble() < 0.1) mutated.Resolution = _elasticResolutions[_random.Next(_elasticResolutions.Count)];

            return mutated;
        }

        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            var constraints = new HardwareConstraints<T>();

            int inputChannels = inputs.Shape.Length > 1 ? inputs.Shape[1] : 3;
            int spatialSize = inputs.Shape.Length > 2 ? inputs.Shape[2] : 224;

            var deadlineUtc = timeLimit <= TimeSpan.Zero
                ? DateTime.UtcNow
                : DateTime.UtcNow.Add(timeLimit);

            var config = EvolutionarySearch(
                constraints,
                inputChannels,
                spatialSize,
                populationSize: 50,
                generations: 20,
                deadlineUtc: deadlineUtc,
                cancellationToken: cancellationToken);
            return ConfigToArchitecture(config);
        }

        protected override AutoMLModelBase<T, Tensor<T>, Tensor<T>> CreateInstanceForCopy()
        {
            return new BigNAS<T>(
                _nasSearchSpace,
                elasticDepths: new List<int>(_elasticDepths),
                elasticWidthMultipliers: new List<double>(_elasticWidthMultipliers),
                elasticKernelSizes: new List<int>(_elasticKernelSizes),
                elasticExpansionRatios: new List<int>(_elasticExpansionRatios),
                elasticResolutions: new List<int>(_elasticResolutions),
                useSandwichSampling: _useSandwichSampling,
                distillationWeight: _ops.ToDouble(_distillationWeight));
        }
    }

}
