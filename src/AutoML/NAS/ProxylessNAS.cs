using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
    public class ProxylessNAS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters (path weights)
        private readonly List<Matrix<T>> _architectureParams;
        private readonly List<Matrix<T>> _architectureGradients;

        // Hardware cost model for latency-aware optimization
        private readonly HardwarePlatform _targetPlatform;
        private readonly HardwareCostModel<T> _hardwareCostModel;
        private readonly T _latencyWeight;

        // Path binarization parameters
        private readonly bool _useBinarization;
        private T _binarizationTemperature;

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => _numNodes;

        public ProxylessNAS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            HardwarePlatform targetPlatform = HardwarePlatform.Mobile,
            double latencyWeight = 0.1, bool useBinarization = true)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(42);

            _targetPlatform = targetPlatform;
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
                return NasSamplingHelper.SoftmaxRowsWithTemperature(alpha, _binarizationTemperature, _ops);

            var result = new Matrix<T>(alpha.Rows, alpha.Columns);
            var probabilities = NasSamplingHelper.SoftmaxRowsWithTemperature(alpha, _binarizationTemperature, _ops);

            for (int row = 0; row < alpha.Rows; row++)
            {
                // Sample one path based on probabilities (binarization)
                double rand = _random.NextDouble();
                double cumulative = 0.0;
                int selectedPath = 0;

                for (int col = 0; col < alpha.Columns; col++)
                {
                    cumulative += _ops.ToDouble(probabilities[row, col]);
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
        /// Computes the expected latency cost of the architecture
        /// </summary>
        public T ComputeExpectedLatency(int inputChannels, int spatialSize)
        {
            T totalLatency = _ops.Zero;

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var probs = NasSamplingHelper.SoftmaxRowsWithTemperature(alpha, _binarizationTemperature, _ops);

                for (int row = 0; row < alpha.Rows; row++)
                {
                    for (int col = 0; col < alpha.Columns; col++)
                    {
                        if (_nasSearchSpace.Operations != null && col < _nasSearchSpace.Operations.Count)
                        {
                            var operation = _nasSearchSpace.Operations[col];
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
                var weights = NasSamplingHelper.SoftmaxRowsWithTemperature(alpha, _binarizationTemperature, _ops);

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
                edgeScores.Sort((a, b) => CompareDescending(a.score, b.score));
                int numEdgesToKeep = Math.Min(2, edgeScores.Count);

                for (int i = 0; i < numEdgesToKeep; i++)
                {
                    var (prevNode, opIdx, _) = edgeScores[i];
                    if (_nasSearchSpace.Operations != null && opIdx < _nasSearchSpace.Operations.Count)
                    {
                        var operation = _nasSearchSpace.Operations[opIdx];
                        architecture.AddOperation(nodeIdx + 1, prevNode, operation);
                    }
                }
            }

            return architecture;
        }

        private int CompareDescending(T left, T right)
        {
            if (_ops.GreaterThan(left, right))
            {
                return -1;
            }

            if (_ops.LessThan(left, right))
            {
                return 1;
            }

            return 0;
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

        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            return DeriveArchitecture();
        }

        protected override AutoMLModelBase<T, Tensor<T>, Tensor<T>> CreateInstanceForCopy()
        {
            return new ProxylessNAS<T>(
                _nasSearchSpace,
                _numNodes,
                targetPlatform: _targetPlatform,
                latencyWeight: _ops.ToDouble(_latencyWeight),
                useBinarization: _useBinarization);
        }
    }
}
