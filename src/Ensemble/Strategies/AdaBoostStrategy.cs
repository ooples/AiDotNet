using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Implements the AdaBoost (Adaptive Boosting) algorithm for combining predictions.
    /// This boosting method iteratively trains weak learners and focuses on examples that are difficult to predict correctly.
    /// Each subsequent model is trained with adjusted sample weights that give more importance to previously misclassified examples.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <typeparam name="TInput">The type of input data</typeparam>
    /// <typeparam name="TOutput">The type of output predictions</typeparam>
    public class AdaBoostStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
        where TOutput : notnull
    {
        private List<T> _alphas = new();
        private List<T> _sampleWeights = new();
        private readonly double _learningRate;
        private readonly int _maxIterations;
        private bool _isTrained = false;
        
        /// <summary>
        /// Gets the operations for the numeric type.
        /// </summary>
        protected INumericOperations<T> Operations => NumOps;

        /// <summary>
        /// Initializes a new instance of the AdaBoostStrategy class.
        /// </summary>
        /// <param name="learningRate">The learning rate for weight updates (default: 1.0)</param>
        /// <param name="maxIterations">Maximum number of boosting iterations (default: 50)</param>
        public AdaBoostStrategy(double learningRate = 1.0, int maxIterations = 50) : base()
        {
            _learningRate = learningRate;
            _maxIterations = maxIterations;
        }

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "AdaBoost";
        
        /// <summary>
        /// Gets whether this strategy requires training.
        /// AdaBoost requires training to determine model weights and sample weights.
        /// </summary>
        public override bool RequiresTraining => true;

        /// <summary>
        /// Combines predictions using the AdaBoost weighted voting scheme.
        /// </summary>
        /// <param name="predictions">The predictions from each model</param>
        /// <param name="weights">The weights for each model (not used - AdaBoost uses trained alphas)</param>
        /// <returns>The combined prediction</returns>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            // AdaBoost uses its own trained weights (alphas), not the provided weights
            return CombinePredictionsInternal(predictions, default(TInput)!);
        }

        /// <summary>
        /// Checks if the predictions can be combined.
        /// </summary>
        /// <param name="predictions">The list of predictions to check</param>
        /// <returns>True if predictions can be combined, false otherwise</returns>
        public override bool CanCombine(List<TOutput> predictions)
        {
            return predictions != null && predictions.Count > 0;
        }

        /// <summary>
        /// Trains the AdaBoost ensemble by iteratively updating sample weights and model weights.
        /// </summary>
        /// <param name="predictions">The predictions from each model for all training samples</param>
        /// <param name="targets">The true target values</param>
        /// <param name="additionalData">Optional additional data (not used in AdaBoost)</param>
        public void Train(List<List<TOutput>> predictions, List<TOutput> targets, object? additionalData = null)
        {
            if (predictions == null || predictions.Count == 0)
                throw new ArgumentNullException(nameof(predictions));
            
            if (targets == null || targets.Count == 0)
                throw new ArgumentNullException(nameof(targets));

            var numSamples = targets.Count;
            var numModels = predictions.Count;

            // Initialize sample weights uniformly
            _sampleWeights = new List<T>(numSamples);
            var initialWeight = Operations.Divide(Operations.One, Operations.FromDouble(numSamples));
            for (int i = 0; i < numSamples; i++)
            {
                _sampleWeights.Add(initialWeight);
            }

            _alphas.Clear();

            // AdaBoost iterations
            for (int m = 0; m < Math.Min(numModels, _maxIterations); m++)
            {
                if (m >= predictions.Count)
                    break;

                var modelPredictions = predictions[m];
                
                // Calculate weighted error
                T weightedError = Operations.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    if (!AreEqual(modelPredictions[i], targets[i]))
                    {
                        weightedError = Operations.Add(weightedError, _sampleWeights[i]);
                    }
                }

                // Prevent division by zero and handle perfect/terrible predictions
                if (Operations.LessThanOrEquals(weightedError, Operations.Zero))
                {
                    // Perfect prediction
                    _alphas.Add(Operations.FromDouble(10)); // Large weight for perfect predictor
                    break; // No need to continue if we have a perfect predictor
                }
                else if (Operations.GreaterThanOrEquals(weightedError, Operations.FromDouble(0.5)))
                {
                    // Worse than random, skip this model
                    _alphas.Add(Operations.Zero);
                    continue;
                }

                // Calculate alpha (model weight)
                var oneMinusError = Operations.Subtract(Operations.One, weightedError);
                var ratio = Operations.Divide(oneMinusError, weightedError);
                var logRatio = Operations.FromDouble(Math.Log(Convert.ToDouble(ratio)));
                var alpha = Operations.Multiply(Operations.FromDouble(_learningRate), Operations.Multiply(Operations.FromDouble(0.5), logRatio));
                _alphas.Add(alpha);

                // Update sample weights
                var newWeights = new List<T>(numSamples);
                T sumWeights = Operations.Zero;

                for (int i = 0; i < numSamples; i++)
                {
                    T weight = _sampleWeights[i];
                    
                    if (AreEqual(modelPredictions[i], targets[i]))
                    {
                        // Correctly classified - decrease weight
                        var factor = Operations.FromDouble(Math.Exp(-Convert.ToDouble(alpha)));
                        weight = Operations.Multiply(weight, factor);
                    }
                    else
                    {
                        // Incorrectly classified - increase weight
                        var factor = Operations.FromDouble(Math.Exp(Convert.ToDouble(alpha)));
                        weight = Operations.Multiply(weight, factor);
                    }
                    
                    newWeights.Add(weight);
                    sumWeights = Operations.Add(sumWeights, weight);
                }

                // Normalize weights
                for (int i = 0; i < numSamples; i++)
                {
                    _sampleWeights[i] = Operations.Divide(newWeights[i], sumWeights);
                }
            }

            _isTrained = true;
        }

        /// <summary>
        /// Combines predictions using the AdaBoost weighted voting scheme.
        /// </summary>
        /// <param name="predictions">The predictions from each model</param>
        /// <param name="input">The input data (not used in this implementation)</param>
        /// <returns>The combined prediction based on weighted voting</returns>
        protected TOutput CombinePredictionsInternal(List<TOutput> predictions, TInput input)
        {
            if (!_isTrained)
                throw new InvalidOperationException("AdaBoostStrategy must be trained before making predictions.");

            if (predictions == null || predictions.Count == 0)
                throw new ArgumentNullException(nameof(predictions));

            // For regression, return weighted average
            if (typeof(TOutput) == typeof(T) || typeof(TOutput) == typeof(double) || 
                typeof(TOutput) == typeof(float) || typeof(TOutput) == typeof(decimal))
            {
                T weightedSum = Operations.Zero;
                T totalWeight = Operations.Zero;

                for (int i = 0; i < Math.Min(predictions.Count, _alphas.Count); i++)
                {
                    var alpha = _alphas[i];
                    if (Operations.GreaterThan(alpha, Operations.Zero))
                    {
                        var predValue = ConvertToT(predictions[i]);
                        weightedSum = Operations.Add(weightedSum, Operations.Multiply(alpha, predValue));
                        totalWeight = Operations.Add(totalWeight, alpha);
                    }
                }

                if (Operations.GreaterThan(totalWeight, Operations.Zero))
                {
                    var result = Operations.Divide(weightedSum, totalWeight);
                    return ConvertFromT(result);
                }
            }
            
            // For classification, use weighted voting
            var voteCounts = new Dictionary<TOutput, T>();
            
            for (int i = 0; i < Math.Min(predictions.Count, _alphas.Count); i++)
            {
                var alpha = _alphas[i];
                if (Operations.GreaterThan(alpha, Operations.Zero))
                {
                    var prediction = predictions[i];
                    if (!voteCounts.ContainsKey(prediction))
                    {
                        voteCounts[prediction] = Operations.Zero;
                    }
                    voteCounts[prediction] = Operations.Add(voteCounts[prediction], alpha);
                }
            }

            if (voteCounts.Count == 0)
            {
                // Fallback to first prediction if no valid votes
                return predictions[0];
            }

            // Return the class with highest weighted vote
            return voteCounts.OrderByDescending(kvp => Convert.ToDouble(kvp.Value)).First().Key;
        }

        /// <summary>
        /// Gets the current sample weights after training.
        /// </summary>
        public IReadOnlyList<T>? SampleWeights => _sampleWeights?.AsReadOnly();

        /// <summary>
        /// Gets the model weights (alphas) after training.
        /// </summary>
        public IReadOnlyList<T>? ModelWeights => _alphas?.AsReadOnly();

        private bool AreEqual(TOutput pred, TOutput target)
        {
            if (pred == null && target == null) return true;
            if (pred == null || target == null) return false;
            return pred.Equals(target);
        }

        private T ConvertToT(TOutput value)
        {
            if (value is T tValue)
                return tValue;
            
            return Operations.FromDouble(Convert.ToDouble(value!));
        }

        private TOutput ConvertFromT(T value)
        {
            if (typeof(TOutput) == typeof(T))
                return (TOutput)(object)value!;
            
            var doubleValue = Convert.ToDouble(value);
            return (TOutput)Convert.ChangeType(doubleValue, typeof(TOutput))!;
        }
    }
}