using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.
    /// Uses path binarization and latency-aware loss to search directly on the target device
    /// without requiring a proxy task or separate hardware lookup tables.
    ///
    /// Reference: "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware" (ICLR 2019)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class ProxylessNAS<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpace<T> _searchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters (path weights)
        private readonly List<Matrix<T>> _architectureParams;
        private readonly List<Matrix<T>> _architectureGradients;

        // Hardware cost model for latency-aware optimization
        private readonly HardwareCostModel<T> _hardwareCostModel;
        private readonly T _latencyWeight;

        // Path binarization parameters
        private readonly bool _useBinarization;
        private T _binarizationTemperature;

        public ProxylessNAS(SearchSpace<T> searchSpace, int numNodes = 4,
            HardwarePlatform targetPlatform = HardwarePlatform.Mobile,
            double latencyWeight = 0.1, bool useBinarization = true)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _searchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = new Random(42);

            _hardwareCostModel = new HardwareCostModel<T>(targetPlatform);
            _latencyWeight = _ops.FromDouble(latencyWeight);
            _useBinarization = useBinarization;
            _binarizationTemperature = _ops.FromDouble(1.0);

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
        /// Applies path binarization for memory-efficient single-path sampling.
        /// Only one operation is active at a time during forward pass.
        /// </summary>
        public Matrix<T> BinarizePaths(Matrix<T> alpha)
        {
            if (!_useBinarization)
                return ApplySoftmax(alpha);

            var result = new Matrix<T>(alpha.Rows, alpha.Columns);

            for (int row = 0; row < alpha.Rows; row++)
            {
                // Compute softmax probabilities
                var probs = new T[alpha.Columns];
                T maxVal = alpha[row, 0];
                for (int col = 1; col < alpha.Columns; col++)
                {
                    if (_ops.GreaterThan(alpha[row, col], maxVal))
                        maxVal = alpha[row, col];
                }

                T sumExp = _ops.Zero;
                var expValues = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    expValues[col] = _ops.Exp(_ops.Subtract(alpha[row, col], maxVal));
                    sumExp = _ops.Add(sumExp, expValues[col]);
                }

                for (int col = 0; col < alpha.Columns; col++)
                {
                    probs[col] = _ops.Divide(expValues[col], sumExp);
                }

                // Sample one path based on probabilities (binarization)
                double rand = _random.NextDouble();
                double cumulative = 0.0;
                int selectedPath = 0;

                for (int col = 0; col < alpha.Columns; col++)
                {
                    cumulative += Convert.ToDouble(probs[col]);
                    if (rand <= cumulative)
                    {
                        selectedPath = col;
                        break;
                    }
                }

                // Binary encoding: only selected path is 1, others are 0
                for (int col = 0; col < alpha.Columns; col++)
                {
                    result[row, col] = col == selectedPath ? _ops.One : _ops.Zero;
                }
            }

            return result;
        }

        /// <summary>
        /// Standard softmax for non-binarized mode
        /// </summary>
        private Matrix<T> ApplySoftmax(Matrix<T> alpha)
        {
            var result = new Matrix<T>(alpha.Rows, alpha.Columns);

            for (int row = 0; row < alpha.Rows; row++)
            {
                T maxVal = alpha[row, 0];
                for (int col = 1; col < alpha.Columns; col++)
                {
                    if (_ops.GreaterThan(alpha[row, col], maxVal))
                        maxVal = alpha[row, col];
                }

                T sumExp = _ops.Zero;
                var expValues = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    expValues[col] = _ops.Exp(_ops.Subtract(alpha[row, col], maxVal));
                    sumExp = _ops.Add(sumExp, expValues[col]);
                }

                for (int col = 0; col < alpha.Columns; col++)
                {
                    result[row, col] = _ops.Divide(expValues[col], sumExp);
                }
            }

            return result;
        }

        /// <summary>
        /// Computes the expected latency cost of the architecture
        /// </summary>
        public T ComputeExpectedLatency(int inputChannels, int spatialSize)
        {
            T totalLatency = _ops.Zero;

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var probs = ApplySoftmax(alpha);

                for (int row = 0; row < alpha.Rows; row++)
                {
                    for (int col = 0; col < alpha.Columns; col++)
                    {
                        if (_searchSpace.Operations != null && col < _searchSpace.Operations.Count)
                        {
                            var operation = _searchSpace.Operations[col];
                            var cost = _hardwareCostModel.EstimateOperationCost(operation, inputChannels, inputChannels, spatialSize);

                            // Weighted latency by operation probability
                            T weightedLatency = _ops.Multiply(probs[row, col], cost.Latency);
                            totalLatency = _ops.Add(totalLatency, weightedLatency);
                        }
                    }
                }
            }

            return totalLatency;
        }

        /// <summary>
        /// Computes the total loss including task loss and latency regularization
        /// </summary>
        public T ComputeTotalLoss(T taskLoss, int inputChannels, int spatialSize)
        {
            T latencyLoss = ComputeExpectedLatency(inputChannels, spatialSize);
            return _ops.Add(taskLoss, _ops.Multiply(_latencyWeight, latencyLoss));
        }

        /// <summary>
        /// Derives the final discrete architecture by selecting operations with highest weights
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var weights = ApplySoftmax(alpha);

                // Select top-2 predecessors for each node
                var edgeScores = new List<(int prevNode, int opIdx, T score)>();

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

                    edgeScores.Add((prevNodeIdx, bestOpIdx, bestWeight));
                }

                // Sort by score and keep top-2
                edgeScores.Sort((a, b) => _ops.Compare(b.score, a.score));
                int numEdgesToKeep = Math.Min(2, edgeScores.Count);

                for (int i = 0; i < numEdgesToKeep; i++)
                {
                    var (prevNode, opIdx, _) = edgeScores[i];
                    if (_searchSpace.Operations != null && opIdx < _searchSpace.Operations.Count)
                    {
                        var operation = _searchSpace.Operations[opIdx];
                        architecture.AddOperation(nodeIdx + 1, prevNode, operation);
                    }
                }
            }

            return architecture;
        }

        /// <summary>
        /// Estimates the final architecture's hardware cost
        /// </summary>
        public HardwareCost<T> EstimateArchitectureCost(int inputChannels, int spatialSize)
        {
            var architecture = DeriveArchitecture();
            return _hardwareCostModel.EstimateArchitectureCost(architecture, inputChannels, spatialSize);
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
        /// Sets the binarization temperature
        /// </summary>
        public void SetBinarizationTemperature(double temperature)
        {
            _binarizationTemperature = _ops.FromDouble(temperature);
        }
    }
}
