using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Implements the Blending combination strategy for ensemble learning.
    /// Blending is similar to stacking but uses a holdout validation set instead of cross-validation
    /// to learn optimal linear weights for combining predictions from base models.
    /// This approach is simpler than stacking but may be less robust due to using a single validation set.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <typeparam name="TInput">The input type</typeparam>
    /// <typeparam name="TOutput">The output type</typeparam>
    public class BlendingStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
    {
        private Vector<T>? _blendingWeights;
        private T _blendingBias = default!;
        private readonly double _validationSplit;
        private readonly bool _includeIntercept;
        private bool _isTrained;

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "Blending";

        /// <summary>
        /// Gets whether this strategy requires training.
        /// </summary>
        public override bool RequiresTraining => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="BlendingStrategy{T, TInput, TOutput}"/> class.
        /// </summary>
        /// <param name="validationSplit">The proportion of training data to use for validation (default: 0.2)</param>
        /// <param name="includeIntercept">Whether to include an intercept term in the blending model (default: true)</param>
        public BlendingStrategy(double validationSplit = 0.2, bool includeIntercept = true) : base()
        {
            if (validationSplit <= 0 || validationSplit >= 1)
            {
                throw new ArgumentOutOfRangeException(nameof(validationSplit), 
                    "Validation split must be between 0 and 1 (exclusive)");
            }

            _validationSplit = validationSplit;
            _includeIntercept = includeIntercept;
            _blendingBias = NumOps.Zero;
            _isTrained = false;
        }

        /// <summary>
        /// Trains the blending strategy by learning optimal weights using a holdout validation set.
        /// </summary>
        /// <param name="predictions">Historical predictions from each model for all samples</param>
        /// <param name="targets">True target values</param>
        /// <param name="additionalData">Optional additional training data</param>
        public void Train(List<List<TOutput>> predictions, List<TOutput> targets, object? additionalData = null)
        {
            if (predictions == null || predictions.Count == 0)
            {
                throw new ArgumentException("At least one model's predictions are required for blending", nameof(predictions));
            }

            if (targets == null || targets.Count == 0)
            {
                throw new ArgumentException("Training targets cannot be null or empty", nameof(targets));
            }

            var numModels = predictions.Count;
            var numSamples = targets.Count;

            // Ensure all prediction lists have the same length
            if (predictions.Any(p => p.Count != numSamples))
            {
                throw new ArgumentException("All prediction lists must have the same number of samples", nameof(predictions));
            }

            // Split data into training and validation sets
            var splitIndex = (int)(numSamples * (1 - _validationSplit));

            // Convert validation predictions to numeric matrix
            var validSamples = numSamples - splitIndex;
            var validationMatrix = new T[validSamples][];
            var validationTargets = new T[validSamples];

            for (int i = 0; i < validSamples; i++)
            {
                validationMatrix[i] = new T[numModels];
                for (int j = 0; j < numModels; j++)
                {
                    validationMatrix[i][j] = ConvertToT(predictions[j][splitIndex + i]);
                }
                validationTargets[i] = ConvertToT(targets[splitIndex + i]);
            }

            // Learn blending weights using linear regression on validation set
            _blendingWeights = LearnBlendingWeights(validationMatrix, validationTargets, numModels);
            _isTrained = true;
        }

        /// <summary>
        /// Combines predictions using the learned blending weights.
        /// </summary>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            Vector<T> finalWeights;

            if (_isTrained && _blendingWeights != null)
            {
                // Use trained blending weights
                finalWeights = _blendingWeights;
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
                    var array = new T[predictions.Count];
                    for (int i = 0; i < predictions.Count; i++)
                        array[i] = uniformWeight;
                    finalWeights = new Vector<T>(array);
                }
            }

            // Apply weights
            T result = _includeIntercept ? _blendingBias : NumOps.Zero;
            
            for (int i = 0; i < predictions.Count && i < finalWeights.Length; i++)
            {
                T predValue = ConvertToT(predictions[i]);
                result = NumOps.Add(result, NumOps.Multiply(finalWeights[i], predValue));
            }

            return ConvertFromT(result);
        }

        /// <summary>
        /// Checks if the predictions can be combined.
        /// </summary>
        public override bool CanCombine(List<TOutput> predictions)
        {
            return predictions != null && predictions.Count > 0;
        }

        /// <summary>
        /// Learns optimal blending weights using ordinary least squares.
        /// </summary>
        private Vector<T> LearnBlendingWeights(T[][] validationPredictions, T[] validationTargets, int numModels)
        {
            int numSamples = validationPredictions.Length;
            int numFeatures = _includeIntercept ? numModels + 1 : numModels;

            // Build design matrix X
            var X = new T[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                X[i] = new T[numFeatures];
                if (_includeIntercept)
                {
                    X[i][0] = NumOps.One; // Intercept term
                    for (int j = 0; j < numModels; j++)
                    {
                        X[i][j + 1] = validationPredictions[i][j];
                    }
                }
                else
                {
                    for (int j = 0; j < numModels; j++)
                    {
                        X[i][j] = validationPredictions[i][j];
                    }
                }
            }

            // Solve normal equations: (X'X)β = X'y
            var XtX = new T[numFeatures][];
            var Xty = new T[numFeatures];

            // Initialize XtX
            for (int i = 0; i < numFeatures; i++)
            {
                XtX[i] = new T[numFeatures];
                Xty[i] = NumOps.Zero;
            }

            // Compute X'X and X'y
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    for (int k = 0; k < numFeatures; k++)
                    {
                        XtX[j][k] = NumOps.Add(XtX[j][k], NumOps.Multiply(X[i][j], X[i][k]));
                    }
                    Xty[j] = NumOps.Add(Xty[j], NumOps.Multiply(X[i][j], validationTargets[i]));
                }
            }

            // Add small regularization for numerical stability
            var lambda = NumOps.FromDouble(1e-6);
            for (int i = 0; i < numFeatures; i++)
            {
                XtX[i][i] = NumOps.Add(XtX[i][i], lambda);
            }

            // Solve the system
            var beta = SolveLinearSystem(XtX, Xty);

            // Extract weights and bias
            if (_includeIntercept)
            {
                _blendingBias = beta[0];
                var weights = new T[numModels];
                for (int i = 0; i < numModels; i++)
                {
                    weights[i] = beta[i + 1];
                }
                return new Vector<T>(weights);
            }
            else
            {
                _blendingBias = NumOps.Zero;
                return new Vector<T>(beta);
            }
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
            
            // Forward elimination with partial pivoting
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
        /// Gets the learned blending weights if available.
        /// </summary>
        public Vector<T>? GetBlendingWeights()
        {
            return _blendingWeights;
        }

        /// <summary>
        /// Gets the learned bias term if available.
        /// </summary>
        public T GetBlendingBias()
        {
            return _blendingBias;
        }
    }
}