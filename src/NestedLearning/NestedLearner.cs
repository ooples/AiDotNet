using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using AiDotNet.Interfaces;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.NestedLearning
{
    /// <summary>
    /// Implementation of Nested Learning algorithm - a paradigm that treats ML models as
    /// interconnected, multi-level learning problems optimized simultaneously.
    /// Based on Google's Nested Learning research for continual learning.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">Input data type</typeparam>
    /// <typeparam name="TOutput">Output data type</typeparam>
    public class NestedLearner<T, TInput, TOutput> : INestedLearner<T, TInput, TOutput>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>
    {
        private readonly IFullModel<T, TInput, TOutput> _model;
        private readonly ILossFunction<T> _lossFunction;
        private readonly int _numLevels;
        private readonly int[] _updateFrequencies;
        private readonly T[] _learningRates;
        private readonly IContinuumMemorySystem<T> _memorySystem;
        private readonly IContextFlow<T> _contextFlow;

        private int _globalStep;
        private readonly Dictionary<int, List<T>> _lossHistory;
        private Vector<T>? _previousTaskParameters;

        /// <summary>
        /// Initializes a new Nested Learner.
        /// </summary>
        /// <param name="model">The model to train with nested learning</param>
        /// <param name="lossFunction">Loss function for training</param>
        /// <param name="numLevels">Number of nested optimization levels</param>
        /// <param name="learningRates">Learning rates per level (fastest to slowest)</param>
        /// <param name="memoryDimension">Dimension of continuum memory</param>
        public NestedLearner(
            IFullModel<T, TInput, TOutput> model,
            ILossFunction<T> lossFunction,
            int numLevels = 3,
            T[]? learningRates = null,
            int memoryDimension = 128)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
            _numLevels = numLevels;

            // Initialize learning rates (decreasing with level)
            _learningRates = learningRates ?? CreateDefaultLearningRates(numLevels);

            // Initialize update frequencies (exponentially increasing)
            _updateFrequencies = CreateUpdateFrequencies(numLevels);

            // Initialize CMS and context flow
            _memorySystem = new ContinuumMemorySystem<T>(memoryDimension, numLevels);
            _contextFlow = new ContextFlow<T>(memoryDimension, numLevels);

            _globalStep = 0;
            _lossHistory = new Dictionary<int, List<T>>();
            for (int i = 0; i < numLevels; i++)
            {
                _lossHistory[i] = new List<T>();
            }
        }

        private T[] CreateDefaultLearningRates(int numLevels)
        {
            var rates = new T[numLevels];
            T baseLR = T.CreateChecked(0.01);

            for (int i = 0; i < numLevels; i++)
            {
                // Exponentially decreasing: 0.01, 0.001, 0.0001, ...
                rates[i] = baseLR / T.CreateChecked(Math.Pow(10, i));
            }

            return rates;
        }

        private int[] CreateUpdateFrequencies(int numLevels)
        {
            var frequencies = new int[numLevels];
            for (int i = 0; i < numLevels; i++)
            {
                // Level 0: every step, Level 1: every 10 steps, Level 2: every 100 steps, etc.
                frequencies[i] = (int)Math.Pow(10, i);
            }
            return frequencies;
        }

        /// <inheritdoc/>
        public NestedLearningStepResult<T> NestedStep(TInput input, TOutput expectedOutput, int level = 0)
        {
            var result = new NestedLearningStepResult<T>();
            _globalStep++;

            // Train the model
            _model.Train(input, expectedOutput);

            // Compute loss
            var prediction = _model.Predict(input);
            T loss = _lossFunction.ComputeLoss(prediction, expectedOutput);
            result.Loss = loss;

            // Get current parameters
            var currentParams = GetModelParameters();

            // Update each level based on frequency
            for (int lvl = 0; lvl < _numLevels; lvl++)
            {
                if (_globalStep % _updateFrequencies[lvl] == 0)
                {
                    // This level should be updated
                    result.UpdatedLevels.Add(lvl);

                    // Compute context representation
                    var context = ComputeContextRepresentation(currentParams, loss);

                    // Propagate through context flow
                    var flowedContext = _contextFlow.PropagateContext(context, lvl);

                    // Store in memory system
                    _memorySystem.Store(flowedContext, lvl);

                    // Compute level-specific loss (could use different metrics per level)
                    result.LossPerLevel[lvl] = loss;

                    // Compute and store gradients (if model supports it)
                    if (currentParams != null)
                    {
                        var gradients = ComputeLevelGradients(lvl, currentParams, loss);
                        result.GradientsPerLevel[lvl] = gradients;

                        // Apply level-specific update
                        ApplyLevelUpdate(lvl, gradients);
                    }

                    _lossHistory[lvl].Add(loss);
                }
            }

            // Periodically consolidate memory
            if (_globalStep % 100 == 0)
            {
                _memorySystem.Consolidate();
            }

            return result;
        }

        /// <inheritdoc/>
        public NestedLearningResult<T> Train(
            IEnumerable<(TInput Input, TOutput Output)> trainingData,
            int numLevels = 3,
            int maxIterations = 1000)
        {
            var result = new NestedLearningResult<T>();
            var stopwatch = Stopwatch.StartNew();

            var dataList = trainingData.ToList();
            T previousLoss = T.CreateChecked(double.MaxValue);
            int iterationsWithoutImprovement = 0;
            const int patienceThreshold = 50;

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                T epochLoss = T.Zero;
                int sampleCount = 0;

                // Train on all samples
                foreach (var (input, output) in dataList)
                {
                    var stepResult = NestedStep(input, output);
                    epochLoss += stepResult.Loss;
                    sampleCount++;
                }

                // Average loss
                T avgLoss = epochLoss / T.CreateChecked(sampleCount);

                // Check for convergence
                T improvement = T.Abs(previousLoss - avgLoss);
                if (improvement < T.CreateChecked(1e-6))
                {
                    iterationsWithoutImprovement++;
                    if (iterationsWithoutImprovement >= patienceThreshold)
                    {
                        result.Converged = true;
                        break;
                    }
                }
                else
                {
                    iterationsWithoutImprovement = 0;
                }

                previousLoss = avgLoss;
                result.FinalLoss = avgLoss;
            }

            stopwatch.Stop();
            result.Duration = stopwatch.Elapsed;
            result.Iterations = _globalStep;
            result.LossHistoryPerLevel = _lossHistory.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.ToList());

            return result;
        }

        /// <inheritdoc/>
        public NestedAdaptationResult<T> AdaptToNewTask(
            IEnumerable<(TInput Input, TOutput Output)> newTaskData,
            T preservationStrength = default)
        {
            var result = new NestedAdaptationResult<T>();

            // Store current parameters before adaptation
            _previousTaskParameters = GetModelParameters();

            // If no preservation strength specified, use default
            if (preservationStrength == default)
            {
                preservationStrength = T.CreateChecked(0.5);
            }

            var dataList = newTaskData.ToList();
            int adaptationSteps = 0;
            T newTaskLoss = T.Zero;

            // Adapt using nested learning
            foreach (var (input, output) in dataList)
            {
                var stepResult = NestedStep(input, output);
                newTaskLoss += stepResult.Loss;
                adaptationSteps++;

                // Apply preservation constraint (elastic weight consolidation style)
                if (_previousTaskParameters != null)
                {
                    var currentParams = GetModelParameters();
                    if (currentParams != null)
                    {
                        // Pull current parameters toward previous task parameters
                        var constraint = (currentParams - _previousTaskParameters) * preservationStrength;
                        SetModelParameters(currentParams - constraint);
                    }
                }
            }

            result.NewTaskLoss = newTaskLoss / T.CreateChecked(dataList.Count);
            result.AdaptationSteps = adaptationSteps;

            // Measure forgetting if we have previous task parameters
            if (_previousTaskParameters != null)
            {
                var currentParams = GetModelParameters();
                if (currentParams != null)
                {
                    // Forgetting metric: parameter drift
                    T drift = (currentParams - _previousTaskParameters).L2Norm();
                    result.ForgettingMetric = drift;
                }
            }

            result.PreviousTasksLoss = T.Zero; // Would need old task data to compute this

            return result;
        }

        /// <inheritdoc/>
        public int NumberOfLevels => _numLevels;

        /// <inheritdoc/>
        public int[] UpdateFrequencies => _updateFrequencies;

        private Tensor<T> ComputeContextRepresentation(Vector<T>? parameters, T loss)
        {
            // Simple context: combine loss with parameter statistics
            int contextDim = _memorySystem.MemoryDimension;
            var contextArray = new T[contextDim];

            // Encode loss in first dimension
            contextArray[0] = loss;

            if (parameters != null && parameters.Count > 1)
            {
                // Encode parameter statistics in remaining dimensions
                T mean = parameters.Average();
                T std = T.Sqrt(parameters.Select(p => (p - mean) * (p - mean)).Average());

                contextArray[1] = mean;
                if (contextDim > 2)
                {
                    contextArray[2] = std;
                }

                // Fill remaining with parameter samples
                for (int i = 3; i < Math.Min(contextDim, parameters.Count + 3); i++)
                {
                    contextArray[i] = parameters[i - 3];
                }
            }

            return Tensor<T>.CreateFromArray(contextArray, new[] { contextDim });
        }

        private Vector<T> ComputeLevelGradients(int level, Vector<T> parameters, T loss)
        {
            // Simplified gradient computation
            // In a full implementation, this would use automatic differentiation
            var gradients = Vector<T>.Build.Dense(parameters.Count);

            // Finite difference approximation
            T epsilon = T.CreateChecked(1e-5);

            for (int i = 0; i < Math.Min(10, parameters.Count); i++) // Sample subset for efficiency
            {
                T originalValue = parameters[i];

                parameters[i] = originalValue + epsilon;
                // Would recompute loss here with perturbed parameters

                parameters[i] = originalValue - epsilon;
                // Would recompute loss here with perturbed parameters

                parameters[i] = originalValue; // Restore

                // Approximate gradient
                gradients[i] = T.Zero; // Placeholder
            }

            return gradients;
        }

        private void ApplyLevelUpdate(int level, Vector<T> gradients)
        {
            var currentParams = GetModelParameters();
            if (currentParams == null) return;

            // Apply gradient descent with level-specific learning rate
            var update = gradients * _learningRates[level];
            SetModelParameters(currentParams - update);
        }

        private Vector<T>? GetModelParameters()
        {
            if (_model is IParametricModel<T> parametricModel)
            {
                return parametricModel.GetParameters();
            }
            return null;
        }

        private void SetModelParameters(Vector<T> parameters)
        {
            if (_model is IParametricModel<T> parametricModel)
            {
                parametricModel.UpdateParameters(parameters);
            }
        }

        /// <summary>
        /// Gets the continuum memory system for inspection.
        /// </summary>
        public IContinuumMemorySystem<T> GetMemorySystem() => _memorySystem;

        /// <summary>
        /// Gets the context flow mechanism for inspection.
        /// </summary>
        public IContextFlow<T> GetContextFlow() => _contextFlow;
    }
}
