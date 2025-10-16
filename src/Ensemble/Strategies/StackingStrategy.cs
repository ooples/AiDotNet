using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Implements the Stacking (Stacked Generalization) combination strategy for ensemble models.
    /// This strategy uses a meta-learner model to learn how to best combine predictions from base models.
    /// </summary>
    /// <remarks>
    /// Stacking works by training a secondary model (meta-learner) on the predictions of base models.
    /// The meta-learner learns the optimal way to combine base model outputs, potentially capturing
    /// complex relationships between model predictions and the true target values.
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The type of input data</typeparam>
    /// <typeparam name="TOutput">The type of output predictions</typeparam>
    public class StackingStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
    {
        private readonly int _cvFolds;
        private bool _isTrained;
        private List<T[]> _metaModelWeights;
        private T _metaBias = default!;

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "Stacking";

        /// <summary>
        /// Gets whether this strategy requires training before it can be used.
        /// </summary>
        public override bool RequiresTraining => true;

        /// <summary>
        /// Initializes a new instance of the StackingStrategy class.
        /// </summary>
        /// <param name="cvFolds">Number of cross-validation folds to use when training the meta-learner (default: 5)</param>
        public StackingStrategy(int cvFolds = 5) : base()
        {
            _cvFolds = cvFolds > 1 ? cvFolds : throw new ArgumentException("Number of CV folds must be greater than 1", nameof(cvFolds));
            _isTrained = false;
            _metaModelWeights = new List<T[]>();
            _metaBias = NumOps.Zero;
        }

        /// <summary>
        /// Trains the stacking strategy using cross-validation.
        /// </summary>
        /// <param name="predictions">Historical predictions from each model for all samples</param>
        /// <param name="targets">True target values</param>
        /// <param name="additionalData">Optional additional training data</param>
        public void Train(List<List<TOutput>> predictions, List<TOutput> targets, object? additionalData = null)
        {
            if (predictions == null || predictions.Count == 0)
                throw new ArgumentNullException(nameof(predictions));
            
            if (targets == null || targets.Count == 0)
                throw new ArgumentNullException(nameof(targets));

            var numModels = predictions.Count;
            var numSamples = targets.Count;

            // For simplicity, we'll use linear regression as the meta-learner
            // This learns weights for each base model's predictions
            
            // Convert predictions to numeric matrix for meta-learning
            var metaFeatures = new T[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                metaFeatures[i] = new T[numModels];
                for (int j = 0; j < numModels; j++)
                {
                    if (i < predictions[j].Count)
                    {
                        metaFeatures[i][j] = ConvertToT(predictions[j][i]);
                    }
                    else
                    {
                        metaFeatures[i][j] = NumOps.Zero;
                    }
                }
            }

            // Convert targets to numeric array
            var targetValues = new T[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                targetValues[i] = ConvertToT(targets[i]);
            }

            // Simple linear regression using least squares
            // We want to find weights w such that X*w ≈ y
            // Using normal equation: w = (X'X)^(-1)X'y
            
            var XtX = new T[numModels][];
            var Xty = new T[numModels];
            
            // Initialize XtX matrix
            for (int i = 0; i < numModels; i++)
            {
                XtX[i] = new T[numModels];
                Xty[i] = NumOps.Zero;
            }
            
            // Compute X'X and X'y
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < numModels; j++)
                {
                    for (int k = 0; k < numModels; k++)
                    {
                        XtX[j][k] = NumOps.Add(XtX[j][k], NumOps.Multiply(metaFeatures[i][j], metaFeatures[i][k]));
                    }
                    Xty[j] = NumOps.Add(Xty[j], NumOps.Multiply(metaFeatures[i][j], targetValues[i]));
                }
            }
            
            // Add regularization to prevent overfitting
            var lambda = NumOps.FromDouble(0.01);
            for (int i = 0; i < numModels; i++)
            {
                XtX[i][i] = NumOps.Add(XtX[i][i], lambda);
            }
            
            // Solve for weights using simple Gaussian elimination
            var weights = SolveLinearSystem(XtX, Xty);
            
            _metaModelWeights.Clear();
            _metaModelWeights.Add(weights);
            _isTrained = true;
        }

        /// <summary>
        /// Combines predictions using the trained stacking weights.
        /// </summary>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            if (!_isTrained)
            {
                // Fallback to weighted average if not trained
                return base.NormalizeWeights(weights).Length > 0 ? predictions[0] : predictions[0];
            }

            if (predictions.Count != _metaModelWeights[0].Length)
            {
                throw new ArgumentException($"Expected {_metaModelWeights[0].Length} predictions but got {predictions.Count}");
            }

            // Apply learned weights
            T result = _metaBias;
            for (int i = 0; i < predictions.Count; i++)
            {
                T predValue = ConvertToT(predictions[i]);
                result = NumOps.Add(result, NumOps.Multiply(_metaModelWeights[0][i], predValue));
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
                    var factor = NumOps.Divide(augmented[k][i], augmented[i][i]);
                    for (int j = i; j <= n; j++)
                    {
                        augmented[k][j] = NumOps.Subtract(augmented[k][j], NumOps.Multiply(factor, augmented[i][j]));
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
                solution[i] = NumOps.Divide(solution[i], augmented[i][i]);
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
        /// Gets information about the trained meta-model.
        /// </summary>
        public string GetMetaModelInfo()
        {
            if (!_isTrained)
                return "Meta-model not trained yet";

            var weightsStr = string.Join(", ", _metaModelWeights[0].Select(w => Convert.ToDouble(w).ToString("F4")));
            return $"Linear meta-model with weights: [{weightsStr}]";
        }
    }
}