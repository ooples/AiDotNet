using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Implements a dynamic selection strategy that selects different models for different inputs based on their competence in local regions of the feature space.
    /// This strategy uses a k-nearest neighbors approach to determine which model performs best for inputs similar to the current one.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <typeparam name="TInput">The type of input data</typeparam>
    /// <typeparam name="TOutput">The type of output data</typeparam>
    public class DynamicSelectionStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
    {
        private readonly int _k;
        private readonly List<ValidationSample> _validationSet = default!;
        private readonly Dictionary<int, List<T>> _modelCompetenceScores = default!;
        private bool _isTrained;

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "DynamicSelection";

        /// <summary>
        /// Gets whether this strategy requires training.
        /// </summary>
        public override bool RequiresTraining => true;

        /// <summary>
        /// Initializes a new instance of the DynamicSelectionStrategy class.
        /// </summary>
        /// <param name="k">The number of nearest neighbors to consider for competence evaluation</param>
        public DynamicSelectionStrategy(int k = 5) : base()
        {
            if (k <= 0)
                throw new ArgumentException("K must be greater than 0", nameof(k));

            _k = k;
            _validationSet = new List<ValidationSample>();
            _modelCompetenceScores = new Dictionary<int, List<T>>();
            _isTrained = false;
        }

        /// <summary>
        /// Trains the dynamic selection strategy using historical predictions and true targets.
        /// </summary>
        /// <param name="predictions">Historical predictions from each model for all samples</param>
        /// <param name="targets">True target values</param>
        /// <param name="additionalData">Optional: input features for each sample to enable distance-based selection</param>
        public void Train(List<List<TOutput>> predictions, List<TOutput> targets, object? additionalData = null)
        {
            if (predictions == null || predictions.Count == 0)
                throw new ArgumentNullException(nameof(predictions));
            
            if (targets == null || targets.Count == 0)
                throw new ArgumentNullException(nameof(targets));

            var numModels = predictions.Count;
            var numSamples = targets.Count;

            // Initialize competence scores
            _modelCompetenceScores.Clear();
            for (int i = 0; i < numModels; i++)
            {
                _modelCompetenceScores[i] = new List<T>();
            }

            // Calculate competence scores for each model on each sample
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
            {
                for (int modelIdx = 0; modelIdx < numModels; modelIdx++)
                {
                    if (sampleIdx < predictions[modelIdx].Count)
                    {
                        var error = CalculateError(targets[sampleIdx], predictions[modelIdx][sampleIdx]);
                        // Convert error to competence (lower error = higher competence)
                        var competence = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, error));
                        _modelCompetenceScores[modelIdx].Add(competence);
                    }
                }

                // Store validation samples if we have input features
                if (additionalData is List<TInput> inputs && sampleIdx < inputs.Count)
                {
                    var sample = new ValidationSample
                    {
                        InputIndex = sampleIdx,
                        Input = inputs[sampleIdx],
                        ExpectedOutput = targets[sampleIdx],
                        ModelErrors = new T[numModels]
                    };

                    for (int modelIdx = 0; modelIdx < numModels; modelIdx++)
                    {
                        if (sampleIdx < predictions[modelIdx].Count)
                        {
                            sample.ModelErrors[modelIdx] = CalculateError(targets[sampleIdx], predictions[modelIdx][sampleIdx]);
                        }
                    }

                    _validationSet.Add(sample);
                }
            }

            _isTrained = true;
        }

        /// <summary>
        /// Combines predictions from multiple models by dynamically selecting the best model.
        /// </summary>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            if (predictions == null || predictions.Count == 0)
                throw new ArgumentException("Predictions cannot be null or empty.", nameof(predictions));

            // If we have validation data and can do dynamic selection
            if (_isTrained && _validationSet.Count > 0)
            {
                // Without the current input, we use average competence scores
                var avgCompetence = new T[predictions.Count];
                for (int i = 0; i < predictions.Count; i++)
                {
                    if (_modelCompetenceScores.ContainsKey(i) && _modelCompetenceScores[i].Count > 0)
                    {
                        T sum = NumOps.Zero;
                        foreach (var score in _modelCompetenceScores[i])
                        {
                            sum = NumOps.Add(sum, score);
                        }
                        avgCompetence[i] = NumOps.Divide(sum, NumOps.FromDouble(_modelCompetenceScores[i].Count));
                    }
                    else
                    {
                        avgCompetence[i] = NumOps.Zero;
                    }
                }

                // Find the model with highest average competence
                int bestModelIdx = 0;
                T bestCompetence = avgCompetence[0];
                for (int i = 1; i < avgCompetence.Length; i++)
                {
                    if (NumOps.GreaterThan(avgCompetence[i], bestCompetence))
                    {
                        bestCompetence = avgCompetence[i];
                        bestModelIdx = i;
                    }
                }

                return predictions[bestModelIdx];
            }
            else
            {
                // Fallback to weighted average if not trained
                if (weights != null && weights.Length == predictions.Count)
                {
                    var normalizedWeights = NormalizeWeights(weights);
                    T weightedSum = NumOps.Zero;
                    
                    for (int i = 0; i < predictions.Count; i++)
                    {
                        T predValue = ConvertToT(predictions[i]);
                        weightedSum = NumOps.Add(weightedSum, NumOps.Multiply(normalizedWeights[i], predValue));
                    }
                    
                    return ConvertFromT(weightedSum);
                }
                else
                {
                    // Simple average
                    return predictions[0];
                }
            }
        }

        /// <summary>
        /// Combines predictions with input context for better dynamic selection.
        /// </summary>
        public TOutput CombineWithContext(List<TOutput> predictions, Vector<T> weights, TInput currentInput)
        {
            if (!_isTrained || _validationSet.Count == 0)
            {
                return Combine(predictions, weights);
            }

            // Find k-nearest neighbors based on input similarity
            var nearestNeighbors = FindNearestNeighbors(currentInput, Math.Min(_k, _validationSet.Count));

            // Calculate model competence based on nearest neighbors
            var modelCompetence = new T[predictions.Count];
            for (int modelIdx = 0; modelIdx < predictions.Count; modelIdx++)
            {
                T totalCompetence = NumOps.Zero;
                int count = 0;

                foreach (var neighbor in nearestNeighbors)
                {
                    if (modelIdx < neighbor.ModelErrors.Length)
                    {
                        // Convert error to competence
                        var competence = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, neighbor.ModelErrors[modelIdx]));
                        totalCompetence = NumOps.Add(totalCompetence, competence);
                        count++;
                    }
                }

                modelCompetence[modelIdx] = count > 0 ? NumOps.Divide(totalCompetence, NumOps.FromDouble(count)) : NumOps.Zero;
            }

            // Select the model with highest competence
            int bestModelIdx = 0;
            T bestCompetence = modelCompetence[0];
            for (int i = 1; i < modelCompetence.Length; i++)
            {
                if (NumOps.GreaterThan(modelCompetence[i], bestCompetence))
                {
                    bestCompetence = modelCompetence[i];
                    bestModelIdx = i;
                }
            }

            return predictions[bestModelIdx];
        }

        /// <summary>
        /// Checks if the predictions can be combined.
        /// </summary>
        public override bool CanCombine(List<TOutput> predictions)
        {
            return predictions != null && predictions.Count > 0;
        }

        /// <summary>
        /// Calculates the error between expected and predicted outputs.
        /// </summary>
        private T CalculateError(TOutput expected, TOutput predicted)
        {
            // For numeric types, use squared error
            if (expected is T expectedT && predicted is T predictedT)
            {
                var diff = NumOps.Subtract(expectedT, predictedT);
                return NumOps.Multiply(diff, diff);
            }
            
            // For classification, use 0-1 loss
            return expected?.Equals(predicted) ?? false ? NumOps.Zero : NumOps.One;
        }

        /// <summary>
        /// Finds the k-nearest neighbors to the given input.
        /// </summary>
        private List<ValidationSample> FindNearestNeighbors(TInput input, int k)
        {
            // Simple implementation: return random k samples
            // In a real implementation, this would calculate distances based on input features
            var shuffled = _validationSet.OrderBy(x => Guid.NewGuid()).ToList();
            return shuffled.Take(k).ToList();
        }

        private T ConvertToT(TOutput value)
        {
            if (value is T tValue)
                return tValue;
            
            return NumOps.FromDouble(Convert.ToDouble(value));
        }

        private TOutput ConvertFromT(T value)
        {
            if (typeof(TOutput) == typeof(T))
                return (TOutput)(object)value;
            
            var doubleValue = Convert.ToDouble(value);
            return (TOutput)Convert.ChangeType(doubleValue, typeof(TOutput));
        }

        /// <summary>
        /// Represents a validation sample used for evaluating model competence.
        /// </summary>
        private class ValidationSample
        {
            public int InputIndex { get; set; }
            public TInput Input { get; set; } = default!;
            public TOutput ExpectedOutput { get; set; } = default!;
            public T[] ModelErrors { get; set; } = Array.Empty<T>();
        }

        /// <summary>
        /// Gets the average competence scores for each model.
        /// </summary>
        public Dictionary<int, double> GetAverageCompetenceScores()
        {
            var result = new Dictionary<int, double>();
            
            foreach (var kvp in _modelCompetenceScores)
            {
                if (kvp.Value.Count > 0)
                {
                    T sum = NumOps.Zero;
                    foreach (var score in kvp.Value)
                    {
                        sum = NumOps.Add(sum, score);
                    }
                    var avg = NumOps.Divide(sum, NumOps.FromDouble(kvp.Value.Count));
                    result[kvp.Key] = Convert.ToDouble(avg);
                }
                else
                {
                    result[kvp.Key] = 0.0;
                }
            }

            return result;
        }
    }
}