using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Neural Architecture<T> Search (NAS) for automatically designing neural network architectures
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class NeuralArchitectureSearch<T> : AutoMLModelBase<T, Tensor<T>, Tensor<T>>
    {
        private readonly NeuralArchitectureSearchStrategy strategy;
        private readonly int maxLayers;
        private readonly int maxNeuronsPerLayer;
        private readonly List<LayerType> availableLayerTypes;
        private readonly List<ActivationFunction> availableActivations;
        private readonly T resourceBudget;
        private readonly int populationSize;
        private readonly int generations;

        // Search space definition
        private readonly SearchSpace<T> searchSpace;

        // Best architectures found
        private readonly List<ArchitectureCandidate<T>> topArchitectures;

        // Random number generator
        private readonly Random random = new Random();

        public NeuralArchitectureSearch(
            NeuralArchitectureSearchStrategy strategy = NeuralArchitectureSearchStrategy.Evolutionary,
            int maxLayers = 10,
            int maxNeuronsPerLayer = 512,
            T? resourceBudget = null,
            int populationSize = 50,
            int generations = 20,
            string modelName = "NeuralArchitectureSearch")
        {
            this.strategy = strategy;
            this.maxLayers = maxLayers;
            this.maxNeuronsPerLayer = maxNeuronsPerLayer;
            var ops = MathHelper.GetNumericOperations<T>();
            this.resourceBudget = resourceBudget ?? ops.FromDouble(100.0);
            this.populationSize = populationSize;
            this.generations = generations;
            this.random = new Random();

            // Define available layer types and activations
            availableLayerTypes = new List<LayerType>
            {
                LayerType.FullyConnected,
                LayerType.Convolutional,
                LayerType.LSTM,
                LayerType.GRU,
                LayerType.Dropout,
                LayerType.BatchNormalization,
                LayerType.MaxPooling,
                LayerType.AveragePooling
            };

            availableActivations = new List<ActivationFunction>
            {
                ActivationFunction.ReLU,
                ActivationFunction.LeakyReLU,
                ActivationFunction.ELU,
                ActivationFunction.Tanh,
                ActivationFunction.Sigmoid,
                ActivationFunction.Swish,
                ActivationFunction.GELU
            };

            searchSpace = new SearchSpace<T>();
            topArchitectures = new List<ArchitectureCandidate<T>>();

            InitializeSearchSpace();
        }

        /// <summary>
        /// Run the neural architecture search
        /// </summary>
        private async Task SearchAsync(Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            switch (strategy)
            {
                case NeuralArchitectureSearchStrategy.Evolutionary:
                    RunEvolutionarySearch(trainData, trainLabels, valData, valLabels);
                    break;
                case NeuralArchitectureSearchStrategy.ReinforcementLearning:
                    RunReinforcementLearningSearch(trainData, trainLabels, valData, valLabels);
                    break;
                case NeuralArchitectureSearchStrategy.GradientBased:
                    RunGradientBasedSearch(trainData, trainLabels, valData, valLabels);
                    break;
                case NeuralArchitectureSearchStrategy.RandomSearch:
                    RunRandomSearch(trainData, trainLabels, valData, valLabels);
                    break;
                case NeuralArchitectureSearchStrategy.BayesianOptimization:
                    await RunBayesianOptimizationSearchAsync(trainData, trainLabels, valData, valLabels);
                    break;
            }
        }

        /// <summary>
        /// Searches for the best neural architecture configuration
        /// </summary>
        public override async Task<IFullModel<T, Tensor<T>, Tensor<T>>> SearchAsync(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken = default)
        {
            Status = AutoMLStatus.Running;
            var startTime = DateTime.UtcNow;

            try
            {
                Console.WriteLine($"Neural Architecture Search: Strategy={strategy}, Time limit={timeLimit}");

                // Run the architecture search
                await Task.Run(() =>
                {
                    Search(inputs, targets, validationInputs, validationTargets);
                }, cancellationToken);

                // Get the best architecture found
                var bestArchitecture = GetBestArchitecture();

                if (bestArchitecture == null)
                {
                    Status = AutoMLStatus.Failed;
                    throw new InvalidOperationException("No valid architecture found during search");
                }

                // Build the final model from the best architecture
                var finalModel = BuildNetworkFromCandidate(bestArchitecture);

                // Convert to IFullModel (wrap the neural network)
                var wrappedModel = new NeuralArchitectureSearchModel<T>(finalModel, bestArchitecture);

                BestModel = wrappedModel;
                BestScore = bestArchitecture.Fitness;
                Status = AutoMLStatus.Completed;

                Console.WriteLine($"Neural Architecture Search completed. Best fitness: {BestScore:F4}, Architecture layers: {bestArchitecture.Layers.Count}");

                return BestModel;
            }
            catch (OperationCanceledException)
            {
                Status = AutoMLStatus.Cancelled;
                throw;
            }
            catch (Exception ex)
            {
                Status = AutoMLStatus.Failed;
                throw new InvalidOperationException($"Neural architecture search failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Evolutionary search strategy
        /// </summary>
        private void RunEvolutionarySearch(Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            // Initialize population
            var population = InitializePopulation(populationSize);

            for (int gen = 0; gen < generations; gen++)
            {
                // Evaluate fitness of each architecture
                foreach (var candidate in population)
                {
                    if (!candidate.IsEvaluated)
                    {
                        EvaluateArchitecture(candidate, trainData, trainLabels, valData, valLabels);
                    }
                }

                // Sort by fitness
                population = population.OrderByDescending(c => c.Fitness).ToList();

                // Store top architectures
                UpdateTopArchitectures(population.Take(5).ToList());

                // Selection
                var parents = TournamentSelection(population, populationSize / 2);

                // Crossover and mutation
                var offspring = new List<ArchitectureCandidate<T>>();
                while (offspring.Count < populationSize)
                {
                    var parent1 = parents[random.Next(parents.Count)];
                    var parent2 = parents[random.Next(parents.Count)];

                    var child = Crossover(parent1, parent2);
                    child = Mutate(child);

                    offspring.Add(child);
                }

                // Replace population
                population = offspring;

                LogProgress($"Generation {gen + 1}/{generations}, Best fitness: {topArchitectures.First().Fitness:F4}");
            }
        }

        /// <summary>
        /// Reinforcement learning-based search
        /// </summary>
        private void RunReinforcementLearningSearch(Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            // Use a controller network to generate architectures
            var controller = new ControllerNetwork<T>(searchSpace);
            var rewardHistory = new List<double>();

            var ops = MathHelper.GetNumericOperations<T>();
            var resourceBudgetInt = Convert.ToInt32(resourceBudget);
            for (int episode = 0; episode < resourceBudgetInt; episode++)
            {
                // Generate architecture using controller
                var architecture = controller.GenerateArchitecture();
                var candidate = CreateCandidateFromArchitecture(architecture);

                // Evaluate architecture
                EvaluateArchitecture(candidate, trainData, trainLabels, valData, valLabels);

                // Use validation accuracy as reward
                var reward = candidate.Fitness;
                rewardHistory.Add(Convert.ToDouble(reward));

                // Update controller using REINFORCE algorithm
                controller.UpdateWithReward(reward);

                // Update top architectures
                UpdateTopArchitectures(new List<ArchitectureCandidate<T>> { candidate });

                // Calculate average reward for the last 10 episodes
                var recentRewards = rewardHistory.Count > 10
                    ? rewardHistory.Skip(rewardHistory.Count - 10).Take(10)
                    : rewardHistory;
                var avgReward = recentRewards.Average();

                LogProgress($"Episode {episode + 1}, Reward: {reward:F4}, Avg reward: {avgReward:F4}");
            }
        }

        /// <summary>
        /// Gradient-based search (DARTS-style)
        /// </summary>
        private void RunGradientBasedSearch(Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            // TODO: Gradient-based NAS requires SuperNet to implement IFullModel<T, Tensor<T>, Tensor<T>>
            // TODO: The optimizer API has changed - Step() no longer accepts parameters/gradients
            // TODO: Need to refactor SuperNet to work with the new optimizer interface

            // For now, fall back to random search
            LogProgress("Gradient-based search not yet implemented with new optimizer API. Using random search instead.");
            RunRandomSearch(trainData, trainLabels, valData, valLabels);

            // TODO: Implement DARTS (Differentiable Architecture Search) using SuperNet
            // See user story: ~/.claude/user-stories/AiDotNet/new_features/NF-001-gradient-based-nas.md
            // Requires: SuperNet implementing IFullModel<T>, refactoring for new optimizer API
            // Previous implementation available in commit history before optimizer API changes
        }

        /// <summary>
        /// Update parameters using gradient descent with momentum
        /// </summary>
        private void UpdateParametersWithMomentum(List<Tensor<T>> parameters, List<Tensor<T>> gradients,
            List<Tensor<T>> momentumBuffers, T learningRate, T momentum)
        {
            var ops = MathHelper.GetNumericOperations<T>();

            // Initialize momentum buffers if needed
            while (momentumBuffers.Count < parameters.Count)
            {
                var param = parameters[momentumBuffers.Count];
                momentumBuffers.Add(new Tensor<T>(param.Shape));
            }

            // Update each parameter
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = parameters[i];
                var grad = gradients[i];
                var momentumBuffer = momentumBuffers[i];

                // Update momentum: m = momentum * m + (1 - momentum) * grad
                for (int j = 0; j < param.Length; j++)
                {
                    var currentMomentum = ops.Multiply(momentum, momentumBuffer[j]);
                    var gradContribution = ops.Multiply(ops.Subtract(ops.One, momentum), grad[j]);
                    momentumBuffer[j] = ops.Add(currentMomentum, gradContribution);

                    // Update parameter: param = param - learning_rate * m
                    param[j] = ops.Subtract(param[j], ops.Multiply(learningRate, momentumBuffer[j]));
                }
            }
        }

        /// <summary>
        /// Random search baseline
        /// </summary>
        private void RunRandomSearch(Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            var evaluations = Convert.ToInt32(resourceBudget);

            for (int i = 0; i < evaluations; i++)
            {
                // Generate random architecture
                var candidate = GenerateRandomArchitecture();

                // Evaluate
                EvaluateArchitecture(candidate, trainData, trainLabels, valData, valLabels);

                // Update top architectures
                UpdateTopArchitectures(new List<ArchitectureCandidate<T>> { candidate });

                var bestFitness = topArchitectures.FirstOrDefault()?.Fitness;
                var bestFitnessValue = bestFitness != null ? Convert.ToDouble(bestFitness) : 0.0;
                LogProgress($"Evaluation {i + 1}/{evaluations}, Best fitness: {bestFitnessValue:F4}");
            }
        }

        /// <summary>
        /// Bayesian optimization search
        /// </summary>
        private async Task RunBayesianOptimizationSearchAsync(Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            var bayesianOptimizer = new BayesianOptimizationAutoML<T, Tensor<T>, Tensor<T>>(numInitialPoints: 10, explorationWeight: 2.0);

            // Define hyperparameter space for architectures
            var searchSpace = new Dictionary<string, ParameterRange>();
            searchSpace["num_layers"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = maxLayers
            };
            searchSpace["neurons_per_layer"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 16,
                MaxValue = maxNeuronsPerLayer
            };
            searchSpace["dropout_rate"] = new ParameterRange
            {
                Type = ParameterType.Continuous,
                MinValue = 0.0,
                MaxValue = 0.5
            };
            searchSpace["learning_rate"] = new ParameterRange
            {
                Type = ParameterType.Continuous,
                MinValue = 0.0001,
                MaxValue = 0.01,
                LogScale = true
            };

            bayesianOptimizer.SetSearchSpace(searchSpace);
            bayesianOptimizer.SetOptimizationMetric(MetricType.Accuracy, maximize: true);

            // Run Bayesian optimization
            try
            {
                var bestModel = await bayesianOptimizer.SearchAsync(
                    trainData,
                    trainLabels,
                    valData,
                    valLabels,
                    TimeSpan.FromHours(1)
                );

                // Convert results to architecture candidates
                var trialHistory = bayesianOptimizer.GetTrialHistory();
                foreach (var trial in trialHistory.Where(t => t.IsSuccessful).OrderByDescending(t => t.Score).Take(10))
                {
                    var candidate = CreateCandidateFromHyperparameters(trial.Parameters);
                    var ops = MathHelper.GetNumericOperations<T>();
                    candidate.Fitness = ops.FromDouble(trial.Score);
                    UpdateTopArchitectures(new List<ArchitectureCandidate<T>> { candidate });
                }
            }
            catch (Exception ex)
            {
                LogError($"Bayesian optimization failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Evaluate a candidate architecture
        /// </summary>
        private void EvaluateArchitecture(ArchitectureCandidate<T> candidate, Tensor<T> trainData, Tensor<T> trainLabels, Tensor<T> valData, Tensor<T> valLabels)
        {
            try
            {
                // Build the network
                var network = BuildNetworkFromCandidate(candidate);

                // Train for limited epochs (early stopping)
                // Note: In production, the optimizer would be created with the actual model
                // For now, we'll use the network's training method directly
                var maxEpochs = 10; // Quick evaluation

                for (int epoch = 0; epoch < maxEpochs; epoch++)
                {
                    // Train one epoch
                    network.Train(trainData, trainLabels);
                }

                // Evaluate on validation set
                var valAccuracy = EvaluateAccuracy(network, valData, valLabels);

                // Compute efficiency metrics
                var parameters = CountParameters(network);
                var flops = EstimateFLOPs(network, trainData.Shape);

                // Combine metrics into fitness score
                candidate.Fitness = ComputeFitnessScore(valAccuracy, parameters, flops);
                candidate.ValidationAccuracy = valAccuracy;
                candidate.Parameters = parameters;
                candidate.FLOPs = flops;
                candidate.IsEvaluated = true;
            }
            catch (Exception ex)
            {
                // Invalid architecture
                var ops = MathHelper.GetNumericOperations<T>();
                candidate.Fitness = ops.Zero;
                candidate.IsEvaluated = true;
                LogError($"Failed to evaluate architecture: {ex.Message}");
            }
        }

        /// <summary>
        /// Build a neural network from architecture candidate
        /// </summary>
        private INeuralNetworkModel<T> BuildNetworkFromCandidate(ArchitectureCandidate<T> candidate)
        {
            // For now, create a simple feedforward network
            // In production, this would create the actual architecture from the candidate
            var ops = MathHelper.GetNumericOperations<T>();
            var learningRate = ops.FromDouble(0.001);

            // Create architecture
            var architecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);

            // Create and return the network
            var network = new FeedForwardNeuralNetwork<T>(
                architecture,
                null, // optimizer will be set during training
                null, // loss function will be set during training
                Convert.ToDouble(learningRate)
            );

            return network;
        }

        /// <summary>
        /// Initialize search space
        /// </summary>
        private void InitializeSearchSpace()
        {
            var ops = MathHelper.GetNumericOperations<T>();
            searchSpace.LayerTypes = availableLayerTypes;
            searchSpace.ActivationFunctions = availableActivations;
            searchSpace.MaxLayers = maxLayers;
            searchSpace.MaxUnitsPerLayer = maxNeuronsPerLayer;
            searchSpace.MaxFilters = 512;
            searchSpace.KernelSizes = new Vector<int>(new[] { 1, 3, 5, 7 });
            searchSpace.DropoutRates = new Vector<T>(new[] {
                ops.FromDouble(0.0),
                ops.FromDouble(0.1),
                ops.FromDouble(0.2),
                ops.FromDouble(0.3),
                ops.FromDouble(0.4),
                ops.FromDouble(0.5)
            });
        }

        /// <summary>
        /// Initialize population for evolutionary search
        /// </summary>
        private List<ArchitectureCandidate<T>> InitializePopulation(int size)
        {
            var population = new List<ArchitectureCandidate<T>>();

            for (int i = 0; i < size; i++)
            {
                population.Add(GenerateRandomArchitecture());
            }

            return population;
        }

        /// <summary>
        /// Generate a random architecture
        /// </summary>
        private ArchitectureCandidate<T> GenerateRandomArchitecture()
        {
            var candidate = new ArchitectureCandidate<T>();
            var numLayers = random.Next(2, maxLayers + 1);

            for (int i = 0; i < numLayers; i++)
            {
                var layerType = availableLayerTypes[random.Next(availableLayerTypes.Count)];
                var layer = new LayerConfiguration<T>
                {
                    Type = layerType,
                    Units = random.Next(16, maxNeuronsPerLayer + 1),
                    Activation = availableActivations[random.Next(availableActivations.Count)],
                    Filters = random.Next(8, 512),
                    KernelSize = searchSpace.KernelSizes[random.Next(searchSpace.KernelSizes.Length)],
                    Stride = 1,
                    PoolSize = 2,
                    DropoutRate = searchSpace.DropoutRates[random.Next(searchSpace.DropoutRates.Length)],
                    ReturnSequences = i < numLayers - 1
                };

                candidate.Layers.Add(layer);
            }

            return candidate;
        }

        /// <summary>
        /// Tournament selection
        /// </summary>
        private List<ArchitectureCandidate<T>> TournamentSelection(List<ArchitectureCandidate<T>> population, int numSelected)
        {
            var selected = new List<ArchitectureCandidate<T>>();
            var tournamentSize = 3;

            while (selected.Count < numSelected)
            {
                var tournament = new List<ArchitectureCandidate<T>>();

                for (int i = 0; i < tournamentSize; i++)
                {
                    tournament.Add(population[random.Next(population.Count)]);
                }

                selected.Add(tournament.OrderByDescending(c => c.Fitness).First());
            }

            return selected;
        }

        /// <summary>
        /// Crossover two parent architectures
        /// </summary>
        private ArchitectureCandidate<T> Crossover(ArchitectureCandidate<T> parent1, ArchitectureCandidate<T> parent2)
        {
            var child = new ArchitectureCandidate<T>();

            // Choose crossover point
            var minLayers = Math.Min(parent1.Layers.Count, parent2.Layers.Count);
            var crossoverPoint = random.Next(1, minLayers);

            // Take layers from parent1 up to crossover point
            for (int i = 0; i < crossoverPoint; i++)
            {
                child.Layers.Add(parent1.Layers[i].Clone());
            }

            // Take remaining layers from parent2
            for (int i = crossoverPoint; i < parent2.Layers.Count; i++)
            {
                child.Layers.Add(parent2.Layers[i].Clone());
            }

            return child;
        }

        /// <summary>
        /// Mutate an architecture
        /// </summary>
        private ArchitectureCandidate<T> Mutate(ArchitectureCandidate<T> candidate)
        {
            var mutated = candidate.Clone();
            var mutationRate = 0.1;

            // Mutate layers
            foreach (var layer in mutated.Layers)
            {
                if (random.NextDouble() < mutationRate)
                {
                    // Change layer type
                    layer.Type = availableLayerTypes[random.Next(availableLayerTypes.Count)];
                }

                if (random.NextDouble() < mutationRate)
                {
                    // Change units/filters
                    layer.Units = random.Next(16, maxNeuronsPerLayer + 1);
                    layer.Filters = random.Next(8, 512);
                }

                if (random.NextDouble() < mutationRate)
                {
                    // Change activation
                    layer.Activation = availableActivations[random.Next(availableActivations.Count)];
                }
            }

            // Add/remove layers
            if (random.NextDouble() < mutationRate && mutated.Layers.Count < maxLayers)
            {
                // Add a random layer
                var newLayer = GenerateRandomArchitecture().Layers.First();
                mutated.Layers.Insert(random.Next(mutated.Layers.Count), newLayer);
            }

            if (random.NextDouble() < mutationRate && mutated.Layers.Count > 2)
            {
                // Remove a random layer
                mutated.Layers.RemoveAt(random.Next(mutated.Layers.Count));
            }

            return mutated;
        }

        /// <summary>
        /// Update top architectures list
        /// </summary>
        private void UpdateTopArchitectures(List<ArchitectureCandidate<T>> candidates)
        {
            topArchitectures.AddRange(candidates.Where(c => c.IsEvaluated));
            topArchitectures.Sort((a, b) => b.Fitness.CompareTo(a.Fitness));

            // Keep only top 10
            if (topArchitectures.Count > 10)
            {
                topArchitectures.RemoveRange(10, topArchitectures.Count - 10);
            }
        }

        /// <summary>
        /// Suggests the next hyperparameters to try
        /// </summary>
        public override async Task<Dictionary<string, object>> SuggestNextTrialAsync()
        {
            return await Task.Run(() =>
            {
                var parameters = new Dictionary<string, object>();

                // Generate random architecture parameters
                parameters["num_layers"] = random.Next(2, maxLayers + 1);
                parameters["layer_types"] = availableLayerTypes[random.Next(availableLayerTypes.Count)];
                parameters["activation"] = availableActivations[random.Next(availableActivations.Count)];
                var ops = MathHelper.GetNumericOperations<T>();
                parameters["dropout_rate"] = ops.FromDouble(random.NextDouble() * 0.5);
                parameters["neurons_per_layer"] = random.Next(16, maxNeuronsPerLayer + 1);

                return parameters;
            });
        }

        /// <summary>
        /// Get the best architecture found
        /// </summary>
        public ArchitectureCandidate<T> GetBestArchitecture()
        {
            return topArchitectures.FirstOrDefault() ?? new ArchitectureCandidate<T>();
        }

        /// <summary>
        /// Get top N architectures
        /// </summary>
        public List<ArchitectureCandidate<T>> GetTopArchitectures(int n = 5)
        {
            return topArchitectures.Take(n).ToList();
        }

        private T ComputeFitnessScore(T accuracy, int parameters, long flops)
        {
            // Multi-objective fitness: accuracy vs efficiency
            var ops = MathHelper.GetNumericOperations<T>();
            var accuracyWeight = ops.FromDouble(0.7);
            var efficiencyWeight = ops.FromDouble(0.3);

            // Normalize efficiency (fewer parameters and FLOPs is better)
            var parametersLog = Math.Log10(parameters + 1);
            var flopsLog = Math.Log10(flops + 1) / 10;
            var efficiencyScore = ops.FromDouble(1.0 / (1.0 + parametersLog + flopsLog));

            return ops.Add(
                ops.Multiply(accuracyWeight, accuracy),
                ops.Multiply(efficiencyWeight, efficiencyScore)
            );
        }

        private int CountParameters(INeuralNetworkModel<T> network)
        {
            // Count total trainable parameters
            return 1000000; // Placeholder
        }

        private long EstimateFLOPs(INeuralNetworkModel<T> network, int[] inputShape)
        {
            // Estimate floating point operations
            return 1000000000; // Placeholder
        }

        private T TrainEpoch(INeuralNetworkModel<T> network, Tensor<T> data, Tensor<T> labels, AdamOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
        {
            // Train one epoch and return loss
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.FromDouble(0.1); // Placeholder
        }

        private T EvaluateAccuracy(INeuralNetworkModel<T> network, Tensor<T> data, Tensor<T> labels)
        {
            // Evaluate accuracy on dataset
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.FromDouble(0.9); // Placeholder
        }

        private ArchitectureCandidate<T> CreateCandidateFromArchitecture(Architecture<T> architecture)
        {
            // Convert architecture representation to candidate
            return new ArchitectureCandidate<T>(); // Placeholder
        }

        private ArchitectureCandidate<T> CreateCandidateFromHyperparameters(Dictionary<string, object> hyperparameters)
        {
            // Convert hyperparameters to candidate
            return new ArchitectureCandidate<T>(); // Placeholder
        }

        private void LogProgress(string message)
        {
            Console.WriteLine($"[NAS] {message}");
        }

        private void LogError(string message)
        {
            Console.WriteLine($"[NAS ERROR] {message}");
        }

        protected override Task<IFullModel<T, Tensor<T>, Tensor<T>>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters)
        {
            throw new NotImplementedException("Neural Architecture<T> Search creates architectures, not individual models");
        }

        protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
        {
            return new Dictionary<string, ParameterRange>();
        }
    }

    /// <summary>
    /// Wrapper model for Neural Architecture Search results
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class NeuralArchitectureSearchModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
    {
        private readonly INeuralNetworkModel<T> _innerModel;
        private readonly ArchitectureCandidate<T> _architecture;

        /// <summary>
        /// Creates a new NeuralArchitectureSearchModel with the specified inner model and architecture.
        /// </summary>
        /// <param name="innerModel">The inner neural network model</param>
        /// <param name="architecture">The architecture candidate</param>
        /// <remarks>
        /// Note: This constructor signature was updated to use fully generic types.
        /// Previous signature used INeuralNetworkModel&lt;double&gt;, which limited flexibility.
        /// This change aligns with project guidelines requiring generic implementations.
        /// </remarks>
        public NeuralArchitectureSearchModel(INeuralNetworkModel<T> innerModel, ArchitectureCandidate<T> architecture)
        {
            _innerModel = innerModel ?? throw new ArgumentNullException(nameof(innerModel));
            _architecture = architecture ?? throw new ArgumentNullException(nameof(architecture));
        }

        public ModelType Type => ModelType.NeuralNetwork;

        public string[] FeatureNames { get; set; } = Array.Empty<string>();

        public int ParameterCount => _architecture.Parameters;

        public void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Training is handled during the search process
            throw new NotSupportedException("NAS models are trained during the search process. Use SearchAsync instead.");
        }

        public Tensor<T> Predict(Tensor<T> input)
        {
            // This is a simplified placeholder - actual implementation would depend on neural network predict method
            throw new NotImplementedException("Prediction needs to be implemented based on the inner neural network model");
        }

        public ModelMetaData<T> GetModelMetaData()
        {
            return new ModelMetaData<T>
            {
                Name = "NeuralArchitectureSearch",
                Description = $"Neural Architecture with {_architecture.Layers.Count} layers",
                Version = "1.0",
                TrainingDate = DateTime.UtcNow,
                Properties = new Dictionary<string, object>
                {
                    ["Architecture"] = _architecture,
                    ["Fitness"] = _architecture.Fitness,
                    ["ValidationAccuracy"] = _architecture.ValidationAccuracy,
                    ["Parameters"] = _architecture.Parameters,
                    ["FLOPs"] = _architecture.FLOPs,
                    ["NumLayers"] = _architecture.Layers.Count
                }
            };
        }

        public void SaveModel(string filePath)
        {
            throw new NotImplementedException("Model serialization not yet implemented for NAS models");
        }

        public void LoadModel(string filePath)
        {
            throw new NotImplementedException("Model deserialization not yet implemented for NAS models");
        }

        public byte[] Serialize()
        {
            throw new NotImplementedException("Model serialization not yet implemented for NAS models");
        }

        public void Deserialize(byte[] data)
        {
            throw new NotImplementedException("Model deserialization not yet implemented for NAS models");
        }

        public Vector<T> GetParameters()
        {
            throw new NotImplementedException("GetParameters not yet implemented for NAS models");
        }

        public void SetParameters(Vector<T> parameters)
        {
            throw new NotImplementedException("SetParameters not yet implemented for NAS models");
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            throw new NotImplementedException("WithParameters not yet implemented for NAS models");
        }

        public Dictionary<string, T> GetFeatureImportance()
        {
            // Neural networks typically don't have explicit feature importance
            // Return empty dictionary for consistency with interface
            return new Dictionary<string, T>();
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            // All features are typically active in neural networks
            return Enumerable.Empty<int>();
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            // All features are typically used in neural networks
            return true;
        }

        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            // Feature selection not applicable for neural networks
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            return new NeuralArchitectureSearchModel<T>(_innerModel, _architecture.Clone());
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            return Clone();
        }
    }
}
