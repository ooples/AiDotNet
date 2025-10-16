using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Implements Bayesian Model Averaging (BMA) for combining ensemble predictions.
    /// This strategy uses probability theory to weight models based on their posterior probabilities,
    /// considering both model uncertainty and prior beliefs about model performance.
    /// </summary>
    /// <remarks>
    /// Bayesian Model Averaging combines predictions from multiple models by weighting each model's
    /// contribution according to its posterior probability given the observed data. This approach:
    /// - Accounts for model uncertainty by considering all models rather than selecting a single best model
    /// - Incorporates prior beliefs about model performance
    /// - Weights models based on their likelihood given the training data
    /// - Provides a principled probabilistic framework for ensemble combination
    /// 
    /// The posterior probability for each model is calculated using Bayes' theorem:
    /// P(Model|Data) ∝ P(Data|Model) × P(Model)
    /// where P(Data|Model) is the likelihood and P(Model) is the prior probability.
    /// </remarks>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <typeparam name="TInput">The input type for the models</typeparam>
    /// <typeparam name="TOutput">The output type for predictions</typeparam>
    public class BayesianAverageStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
        where TOutput : notnull
    {
        private readonly Dictionary<int, T> _modelPriors = default!;
        private readonly Dictionary<int, T> _modelLikelihoods = default!;
        private readonly double _priorStrength;
        private readonly bool _useUniformPrior;
        private Vector<T> _posteriorWeights = default!;

        /// <summary>
        /// Initializes a new instance of the BayesianAverageStrategy class.
        /// </summary>
        /// <param name="priorStrength">The strength of the prior beliefs (higher values give more weight to priors)</param>
        /// <param name="useUniformPrior">Whether to use uniform priors for all models</param>
        public BayesianAverageStrategy(double priorStrength = 1.0, bool useUniformPrior = true) : base()
        {
            _priorStrength = Math.Max(0.01, priorStrength);
            _useUniformPrior = useUniformPrior;
            _modelPriors = new Dictionary<int, T>();
            _modelLikelihoods = new Dictionary<int, T>();
            _posteriorWeights = null!;
        }

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "BayesianAverage";

        /// <summary>
        /// Gets whether this strategy requires training.
        /// </summary>
        public override bool RequiresTraining => true;

        /// <summary>
        /// Sets custom prior probabilities for specific models.
        /// </summary>
        /// <param name="modelIndex">The index of the model to set the prior for</param>
        /// <param name="prior">The prior probability (should be between 0 and 1)</param>
        public void SetModelPrior(int modelIndex, T prior)
        {
            if (NumOps.LessThanOrEquals(prior, NumOps.Zero) || NumOps.GreaterThan(prior, NumOps.One))
                throw new ArgumentException("Prior probability must be between 0 and 1", nameof(prior));
            
            _modelPriors[modelIndex] = prior;
        }

        /// <summary>
        /// Updates model likelihoods based on validation performance.
        /// </summary>
        /// <param name="modelIndex">The index of the model to update</param>
        /// <param name="validationError">The validation error (lower is better)</param>
        public void UpdateModelLikelihood(int modelIndex, T validationError)
        {
            // Convert error to likelihood using exponential transformation
            // Lower error results in higher likelihood
            var negError = NumOps.Negate(validationError);
            var likelihood = NumOps.FromDouble(Math.Exp(Convert.ToDouble(negError)));
            _modelLikelihoods[modelIndex] = likelihood;
        }

        /// <summary>
        /// Trains the Bayesian averaging strategy by computing posterior weights.
        /// </summary>
        /// <param name="predictions">Historical predictions from each model</param>
        /// <param name="targets">True target values</param>
        /// <param name="modelErrors">Optional validation errors for each model</param>
        public void Train(List<List<TOutput>> predictions, List<TOutput> targets, Vector<T>? modelErrors = null)
        {
            if (predictions == null || predictions.Count == 0)
                throw new ArgumentNullException(nameof(predictions));

            int modelCount = predictions.Count;

            // Update likelihoods based on model errors if provided
            if (modelErrors != null && modelErrors.Length == modelCount)
            {
                for (int i = 0; i < modelCount; i++)
                {
                    UpdateModelLikelihood(i, modelErrors[i]);
                }
            }
            else
            {
                // Calculate likelihoods from predictions vs targets
                for (int i = 0; i < modelCount; i++)
                {
                    T totalError = NumOps.Zero;
                    int sampleCount = Math.Min(predictions[i].Count, targets.Count);
                    
                    for (int j = 0; j < sampleCount; j++)
                    {
                        // Simple squared error for numeric types
                        if (predictions[i][j] is T predValue && targets[j] is T targetValue)
                        {
                            var diff = NumOps.Subtract(predValue, targetValue);
                            var squaredDiff = NumOps.Multiply(diff, diff);
                            totalError = NumOps.Add(totalError, squaredDiff);
                        }
                    }
                    
                    if (sampleCount > 0)
                    {
                        var avgError = NumOps.Divide(totalError, NumOps.FromDouble(sampleCount));
                        UpdateModelLikelihood(i, avgError);
                    }
                }
            }

            // Calculate posterior weights
            _posteriorWeights = CalculatePosteriorWeights(modelCount);
        }

        /// <summary>
        /// Combines predictions using Bayesian model averaging.
        /// </summary>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            if (!RequiresTraining || _posteriorWeights != null)
            {
                // Use posterior weights if available, otherwise use provided weights
                var finalWeights = _posteriorWeights ?? weights;
                
                if (finalWeights == null || finalWeights.Length != predictions.Count)
                {
                    // Fallback to uniform weights
                    var uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(predictions.Count));
                    var array = new T[predictions.Count];
                    for (int i = 0; i < predictions.Count; i++)
                        array[i] = uniformWeight;
                    finalWeights = new Vector<T>(array);
                }

                // Normalize weights
                finalWeights = NormalizeWeights(finalWeights);

                // Combine based on output type
                if (typeof(TOutput) == typeof(T) || typeof(TOutput) == typeof(double) || 
                    typeof(TOutput) == typeof(float) || typeof(TOutput) == typeof(decimal))
                {
                    // Regression: weighted average
                    T weightedSum = NumOps.Zero;
                    for (int i = 0; i < predictions.Count; i++)
                    {
                        T predValue = ConvertToT(predictions[i]);
                        weightedSum = NumOps.Add(weightedSum, NumOps.Multiply(finalWeights[i], predValue));
                    }
                    return ConvertFromT(weightedSum);
                }
                else
                {
                    // Classification: weighted voting
                    var voteCounts = new Dictionary<TOutput, T>();
                    
                    for (int i = 0; i < predictions.Count; i++)
                    {
                        var prediction = predictions[i];
                        if (!voteCounts.ContainsKey(prediction))
                        {
                            voteCounts[prediction] = NumOps.Zero;
                        }
                        voteCounts[prediction] = NumOps.Add(voteCounts[prediction], finalWeights[i]);
                    }

                    // Return the class with highest weighted vote
                    return voteCounts.OrderByDescending(kvp => Convert.ToDouble(kvp.Value)).First().Key;
                }
            }
            
            // Fallback to simple averaging if not trained
            return predictions[0];
        }

        /// <summary>
        /// Checks if the predictions can be combined.
        /// </summary>
        public override bool CanCombine(List<TOutput> predictions)
        {
            return predictions != null && predictions.Count > 0;
        }

        /// <summary>
        /// Calculates posterior weights for each model using Bayes' theorem.
        /// </summary>
        private Vector<T> CalculatePosteriorWeights(int modelCount)
        {
            var priors = new T[modelCount];
            var likelihoods = new T[modelCount];
            var posteriors = new T[modelCount];

            // Get priors for each model
            for (int i = 0; i < modelCount; i++)
            {
                if (_useUniformPrior || !_modelPriors.ContainsKey(i))
                {
                    priors[i] = NumOps.Divide(NumOps.One, NumOps.FromDouble(modelCount)); // Uniform prior
                }
                else
                {
                    priors[i] = _modelPriors[i];
                }
            }

            // Normalize priors to ensure they sum to 1
            var priorSum = NumOps.Zero;
            for (int i = 0; i < modelCount; i++)
            {
                priorSum = NumOps.Add(priorSum, priors[i]);
            }
            
            if (NumOps.GreaterThan(priorSum, NumOps.Zero))
            {
                for (int i = 0; i < modelCount; i++)
                {
                    priors[i] = NumOps.Divide(priors[i], priorSum);
                }
            }

            // Get likelihoods for each model
            for (int i = 0; i < modelCount; i++)
            {
                if (_modelLikelihoods.ContainsKey(i))
                {
                    likelihoods[i] = _modelLikelihoods[i];
                }
                else
                {
                    // Default likelihood if not specified
                    likelihoods[i] = NumOps.Divide(NumOps.One, NumOps.FromDouble(modelCount));
                }
            }

            // Calculate unnormalized posteriors: P(Model|Data) ∝ P(Data|Model) × P(Model)^strength
            var posteriorSum = NumOps.Zero;
            for (int i = 0; i < modelCount; i++)
            {
                var priorPower = NumOps.FromDouble(Math.Pow(Convert.ToDouble(priors[i]), _priorStrength));
                posteriors[i] = NumOps.Multiply(likelihoods[i], priorPower);
                posteriorSum = NumOps.Add(posteriorSum, posteriors[i]);
            }

            // Normalize posteriors to sum to 1
            if (NumOps.GreaterThan(posteriorSum, NumOps.Zero))
            {
                for (int i = 0; i < modelCount; i++)
                {
                    posteriors[i] = NumOps.Divide(posteriors[i], posteriorSum);
                }
            }
            else
            {
                // Fallback to uniform weights if all posteriors are zero
                for (int i = 0; i < modelCount; i++)
                {
                    posteriors[i] = NumOps.Divide(NumOps.One, NumOps.FromDouble(modelCount));
                }
            }

            return new Vector<T>(posteriors);
        }

        /// <summary>
        /// Gets the current posterior weights.
        /// </summary>
        public Vector<T> GetPosteriorWeights()
        {
            return _posteriorWeights ?? new Vector<T>(new T[0]);
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
    }
}