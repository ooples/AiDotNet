using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Combination strategy that finds weights to minimize the variance of the combined prediction.
    /// This approach minimizes prediction uncertainty by finding the optimal linear combination 
    /// of models that results in the lowest variance ensemble prediction.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <typeparam name="TInput">The type of input data</typeparam>
    /// <typeparam name="TOutput">The type of output predictions</typeparam>
    public class MinimumVarianceStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
        where TOutput : notnull
    {
        private Vector<T>? _trainedWeights;
        private readonly T _regularizationParameter = default!;
        private bool _isTrained;

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "MinimumVariance";

        /// <summary>
        /// Initializes a new instance of the <see cref="MinimumVarianceStrategy{T, TInput, TOutput}"/> class.
        /// </summary>
        /// <param name="regularizationParameter">Small positive value to ensure numerical stability (default: 1e-6)</param>
        public MinimumVarianceStrategy(double regularizationParameter = 1e-6) : base()
        {
            _regularizationParameter = NumOps.FromDouble(regularizationParameter);
            _isTrained = false;
        }

        /// <summary>
        /// Gets a value indicating whether this strategy requires training.
        /// Returns true as the minimum variance strategy needs to compute the covariance matrix from training data.
        /// </summary>
        public override bool RequiresTraining => true;

        /// <summary>
        /// Trains the strategy by computing optimal weights that minimize the variance of combined predictions.
        /// </summary>
        /// <param name="predictions">List of predictions from each model for the training data</param>
        /// <param name="targets">The true target values (optional, not used in this strategy)</param>
        /// <param name="additionalData">Optional additional data</param>
        public void Train(List<List<TOutput>> predictions, List<TOutput>? targets = null, object? additionalData = null)
        {
            if (predictions == null || predictions.Count == 0)
            {
                throw new ArgumentException("Predictions cannot be null or empty.", nameof(predictions));
            }

            var numModels = predictions.Count;
            var numSamples = predictions[0].Count;

            // Validate all prediction lists have the same length
            if (predictions.Any(p => p.Count != numSamples))
            {
                throw new ArgumentException("All prediction lists must have the same number of samples.", nameof(predictions));
            }

            // Convert predictions to numeric matrix
            var predictionMatrix = new T[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                predictionMatrix[i] = new T[numModels];
                for (int j = 0; j < numModels; j++)
                {
                    predictionMatrix[i][j] = ConvertToT(predictions[j][i]);
                }
            }

            // Compute covariance matrix
            var covarianceMatrix = ComputeCovarianceMatrix(predictionMatrix, numSamples, numModels);

            // Add regularization for numerical stability
            for (int i = 0; i < numModels; i++)
            {
                covarianceMatrix[i][i] = NumOps.Add(covarianceMatrix[i][i], _regularizationParameter);
            }

            // Solve for minimum variance weights
            // The optimal weights w* = C^(-1) * 1 / (1^T * C^(-1) * 1)
            // where C is the covariance matrix and 1 is a vector of ones
            var onesVector = new T[numModels];
            for (int i = 0; i < numModels; i++)
            {
                onesVector[i] = NumOps.One;
            }

            // Solve C * x = 1 for x (which is C^(-1) * 1)
            var numerator = SolveLinearSystem(covarianceMatrix, onesVector);
            
            // Compute denominator: 1^T * C^(-1) * 1
            T denominator = NumOps.Zero;
            for (int i = 0; i < numModels; i++)
            {
                denominator = NumOps.Add(denominator, numerator[i]);
            }

            // Compute final weights
            var weights = new T[numModels];
            for (int i = 0; i < numModels; i++)
            {
                weights[i] = NumOps.Divide(numerator[i], denominator);
            }

            _trainedWeights = new Vector<T>(weights);
            _isTrained = true;
        }

        /// <summary>
        /// Combines predictions using the trained minimum variance weights.
        /// </summary>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            Vector<T> finalWeights;
            
            if (_isTrained && _trainedWeights != null)
            {
                // Use trained weights
                finalWeights = _trainedWeights;
            }
            else
            {
                // Use provided weights or uniform weights
                if (weights != null && weights.Length == predictions.Count)
                {
                    finalWeights = NormalizeWeights(weights);
                }
                else
                {
                    // Fallback to uniform weights
                    var uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(predictions.Count));
                    var uniformWeights = new T[predictions.Count];
                    for (int i = 0; i < predictions.Count; i++)
                        uniformWeights[i] = uniformWeight;
                    finalWeights = new Vector<T>(uniformWeights);
                }
            }

            // Apply weights based on output type
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

        /// <summary>
        /// Checks if the predictions can be combined.
        /// </summary>
        public override bool CanCombine(List<TOutput> predictions)
        {
            return predictions != null && predictions.Count > 0;
        }

        /// <summary>
        /// Computes the covariance matrix of predictions.
        /// </summary>
        private T[][] ComputeCovarianceMatrix(T[][] predictions, int numSamples, int numModels)
        {
            // Compute means
            var means = new T[numModels];
            for (int j = 0; j < numModels; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    sum = NumOps.Add(sum, predictions[i][j]);
                }
                means[j] = NumOps.Divide(sum, NumOps.FromDouble(numSamples));
            }

            // Compute covariance matrix
            var covariance = new T[numModels][];
            for (int i = 0; i < numModels; i++)
            {
                covariance[i] = new T[numModels];
                for (int j = 0; j < numModels; j++)
                {
                    T cov = NumOps.Zero;
                    for (int k = 0; k < numSamples; k++)
                    {
                        var diff1 = NumOps.Subtract(predictions[k][i], means[i]);
                        var diff2 = NumOps.Subtract(predictions[k][j], means[j]);
                        cov = NumOps.Add(cov, NumOps.Multiply(diff1, diff2));
                    }
                    covariance[i][j] = NumOps.Divide(cov, NumOps.FromDouble(numSamples - 1));
                }
            }

            return covariance;
        }

        /// <summary>
        /// Solves a linear system Ax = b using Gaussian elimination.
        /// </summary>
        private T[] SolveLinearSystem(T[][] A, T[] b)
        {
            int n = b.Length;
            var augmented = new T[n][];
            
            // Create augmented matrix [A|b]
            for (int i = 0; i < n; i++)
            {
                augmented[i] = new T[n + 1];
                for (int j = 0; j < n; j++)
                {
                    augmented[i][j] = A[i][j];
                }
                augmented[i][n] = b[i];
            }
            
            // Forward elimination
            for (int i = 0; i < n; i++)
            {
                // Find pivot
                int maxRow = i;
                for (int k = i + 1; k < n; k++)
                {
                    if (NumOps.GreaterThan(NumOps.Abs(augmented[k][i]), NumOps.Abs(augmented[maxRow][i])))
                    {
                        maxRow = k;
                    }
                }
                
                // Swap rows
                var temp = augmented[maxRow];
                augmented[maxRow] = augmented[i];
                augmented[i] = temp;
                
                // Make all rows below this one 0 in current column
                for (int k = i + 1; k < n; k++)
                {
                    if (!NumOps.Equals(augmented[i][i], NumOps.Zero))
                    {
                        var factor = NumOps.Divide(augmented[k][i], augmented[i][i]);
                        for (int j = i; j <= n; j++)
                        {
                            augmented[k][j] = NumOps.Subtract(augmented[k][j], NumOps.Multiply(factor, augmented[i][j]));
                        }
                    }
                }
            }
            
            // Back substitution
            var solution = new T[n];
            for (int i = n - 1; i >= 0; i--)
            {
                solution[i] = augmented[i][n];
                for (int j = i + 1; j < n; j++)
                {
                    solution[i] = NumOps.Subtract(solution[i], NumOps.Multiply(augmented[i][j], solution[j]));
                }
                if (!NumOps.Equals(augmented[i][i], NumOps.Zero))
                {
                    solution[i] = NumOps.Divide(solution[i], augmented[i][i]);
                }
            }
            
            return solution;
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
        /// Gets the trained weights if available.
        /// </summary>
        public Vector<T>? GetTrainedWeights()
        {
            return _trainedWeights;
        }
    }
}