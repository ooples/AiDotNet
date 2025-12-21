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
    /// Partial Channel Connections for Memory-Efficient Differentiable Architecture Search.
    /// PC-DARTS reduces memory consumption by sampling only a subset of channels during the search,
    /// making it more scalable to larger search spaces and datasets.
    ///
    /// Reference: "PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search" (ICLR 2020)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class PCDARTS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters
        private readonly List<Matrix<T>> _architectureParams;
        private readonly List<Matrix<T>> _architectureGradients;

        // PC-DARTS specific: channel sampling ratio
        private readonly double _channelSamplingRatio;
        private readonly bool _useEdgeNormalization;

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => _numNodes;

        public PCDARTS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            double channelSamplingRatio = 0.25, bool useEdgeNormalization = true)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(42);

            _channelSamplingRatio = channelSamplingRatio;
            _useEdgeNormalization = useEdgeNormalization;

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
        /// Samples a subset of channels for partial channel connections
        /// </summary>
        public List<int> SampleChannels(int totalChannels)
        {
            int numSampledChannels = Math.Max(1, (int)(totalChannels * _channelSamplingRatio));
            var allChannels = Enumerable.Range(0, totalChannels).ToList();

            for (int i = allChannels.Count - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (allChannels[i], allChannels[j]) = (allChannels[j], allChannels[i]);
            }

            var sampledChannels = allChannels.Take(numSampledChannels).ToList();
            sampledChannels.Sort();
            return sampledChannels;
        }

        /// <summary>
        /// Applies edge normalization to prevent operation collapse
        /// </summary>
        public Matrix<T> ApplyEdgeNormalization(Matrix<T> alpha)
        {
            if (!_useEdgeNormalization)
                return NasSamplingHelper.SoftmaxRows(alpha, _ops);

            var result = NasSamplingHelper.SoftmaxRows(alpha, _ops);

            // Apply edge normalization: scale by 1/sqrt(num_edges)
            T edgeNormalizationFactor = _ops.FromDouble(1.0 / Math.Sqrt(alpha.Rows));
            for (int row = 0; row < result.Rows; row++)
            {
                for (int col = 0; col < result.Columns; col++)
                {
                    result[row, col] = _ops.Multiply(result[row, col], edgeNormalizationFactor);
                }
            }

            return result;
        }

        /// <summary>
        /// Derives the discrete architecture
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var weights = ApplyEdgeNormalization(alpha);

                // Select top-2 edges for each intermediate node (following DARTS convention)
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

                // Select top-2 edges
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
        /// Gets the memory savings ratio compared to standard DARTS
        /// </summary>
        public double GetMemorySavingsRatio()
        {
            return 1.0 - _channelSamplingRatio;
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
        /// Gets the channel sampling ratio
        /// </summary>
        public double GetChannelSamplingRatio() => _channelSamplingRatio;

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
            return new PCDARTS<T>(
                _nasSearchSpace,
                _numNodes,
                channelSamplingRatio: _channelSamplingRatio,
                useEdgeNormalization: _useEdgeNormalization);
        }
    }
}
