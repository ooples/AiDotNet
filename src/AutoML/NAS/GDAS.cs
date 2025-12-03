using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Gradient-based Differentiable Architecture Search with Gumbel-Softmax sampling.
    /// GDAS uses Gumbel-Softmax to make the architecture search fully differentiable
    /// while maintaining discrete selection during forward pass.
    ///
    /// Reference: "Searching for A Robust Neural Architecture in Four GPU Hours" (CVPR 2019)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class GDAS<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpace<T> _searchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters
        private readonly List<Matrix<T>> _architectureParams;
        private readonly List<Matrix<T>> _architectureGradients;

        // Gumbel-Softmax temperature (annealed during training)
        private T _temperature;
        private readonly T _initialTemperature;
        private readonly T _finalTemperature;

        public GDAS(SearchSpace<T> searchSpace, int numNodes = 4,
            double initialTemperature = 5.0, double finalTemperature = 0.1)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _searchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = new Random(42);

            _initialTemperature = _ops.FromDouble(initialTemperature);
            _finalTemperature = _ops.FromDouble(finalTemperature);
            _temperature = _initialTemperature;

            // Initialize architecture parameters
            _architectureParams = new List<Matrix<T>>();
            _architectureGradients = new List<Matrix<T>>();

            for (int i = 0; i < _numNodes; i++)
            {
                var alpha = new Matrix<T>(i + 1, _numOperations);
                for (int j = 0; j < alpha.Rows; j++)
                {
                    for (int k = 0; k < alpha.Columns; k++)
                    {
                        alpha[j, k] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.2);
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(i + 1, _numOperations));
            }
        }

        /// <summary>
        /// Applies Gumbel-Softmax sampling to architecture parameters.
        /// This makes the discrete sampling operation differentiable.
        /// </summary>
        public Matrix<T> GumbelSoftmax(Matrix<T> alpha, bool hard = false)
        {
            var result = new Matrix<T>(alpha.Rows, alpha.Columns);

            for (int row = 0; row < alpha.Rows; row++)
            {
                // Sample Gumbel noise
                var gumbel = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    // Gumbel(0, 1) = -log(-log(U)) where U ~ Uniform(0, 1)
                    double u = _random.NextDouble() * 0.99 + 0.005;  // Avoid log(0)
                    gumbel[col] = _ops.FromDouble(-Math.Log(-Math.Log(u)));
                }

                // Add Gumbel noise and scale by temperature
                var logits = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    logits[col] = _ops.Divide(
                        _ops.Add(alpha[row, col], gumbel[col]),
                        _temperature
                    );
                }

                // Apply softmax
                T maxLogit = logits[0];
                for (int col = 1; col < alpha.Columns; col++)
                {
                    if (_ops.GreaterThan(logits[col], maxLogit))
                        maxLogit = logits[col];
                }

                T sumExp = _ops.Zero;
                var expValues = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    expValues[col] = _ops.Exp(_ops.Subtract(logits[col], maxLogit));
                    sumExp = _ops.Add(sumExp, expValues[col]);
                }

                // Normalize
                for (int col = 0; col < alpha.Columns; col++)
                {
                    result[row, col] = _ops.Divide(expValues[col], sumExp);
                }

                // Hard Gumbel-Softmax: convert to one-hot
                if (hard)
                {
                    int maxIdx = 0;
                    T maxVal = result[row, 0];
                    for (int col = 1; col < alpha.Columns; col++)
                    {
                        if (_ops.GreaterThan(result[row, col], maxVal))
                        {
                            maxVal = result[row, col];
                            maxIdx = col;
                        }
                    }

                    // One-hot encoding
                    for (int col = 0; col < alpha.Columns; col++)
                    {
                        result[row, col] = col == maxIdx ? _ops.One : _ops.Zero;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Anneals the Gumbel-Softmax temperature during training
        /// </summary>
        public void AnnealTemperature(int currentEpoch, int maxEpochs)
        {
            // Exponential annealing
            double ratio = (double)currentEpoch / maxEpochs;
            double tempValue = Convert.ToDouble(_initialTemperature) *
                Math.Pow(Convert.ToDouble(_finalTemperature) / Convert.ToDouble(_initialTemperature), ratio);
            _temperature = _ops.FromDouble(tempValue);
        }

        /// <summary>
        /// Derives the discrete architecture by selecting the operation with highest weight
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var weights = GumbelSoftmax(alpha, hard: true);

                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    int bestOpIdx = 0;
                    T bestWeight = weights[prevNodeIdx, 0];

                    for (int opIdx = 1; opIdx < _numOperations; opIdx++)
                    {
                        if (_ops.GreaterThan(weights[prevNodeIdx, opIdx], bestWeight))
                        {
                            bestWeight = weights[prevNodeIdx, opIdx];
                            bestOpIdx = opIdx;
                        }
                    }

                    if (_searchSpace.Operations != null && bestOpIdx < _searchSpace.Operations.Count)
                    {
                        var operation = _searchSpace.Operations[bestOpIdx];
                        architecture.AddOperation(nodeIdx + 1, prevNodeIdx, operation);
                    }
                }
            }

            return architecture;
        }

        /// <summary>
        /// Gets architecture parameters for optimization
        /// </summary>
        public List<Matrix<T>> GetArchitectureParameters() => _architectureParams;

        /// <summary>
        /// Gets architecture gradients
        /// </summary>
        public List<Matrix<T>> GetArchitectureGradients() => _architectureGradients;

        /// <summary>
        /// Gets current temperature
        /// </summary>
        public T GetTemperature() => _temperature;
    }
}
