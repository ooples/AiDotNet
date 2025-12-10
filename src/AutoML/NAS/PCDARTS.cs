using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

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
    public class PCDARTS<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpace<T> _searchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters
        private readonly List<Matrix<T>> _architectureParams;
        private readonly List<Matrix<T>> _architectureGradients;

        // PC-DARTS specific: channel sampling ratio
        private readonly double _channelSamplingRatio;
        private readonly bool _useEdgeNormalization;

        public PCDARTS(SearchSpace<T> searchSpace, int numNodes = 4,
            double channelSamplingRatio = 0.25, bool useEdgeNormalization = true)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _searchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = new Random(42);

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

            // Randomly sample channels
            var sampledChannels = new List<int>();
            for (int i = 0; i < numSampledChannels; i++)
            {
                int idx = _random.Next(allChannels.Count);
                sampledChannels.Add(allChannels[idx]);
                allChannels.RemoveAt(idx);
            }

            return sampledChannels.OrderBy(x => x).ToList();
        }

        /// <summary>
        /// Applies edge normalization to prevent operation collapse
        /// </summary>
        public Matrix<T> ApplyEdgeNormalization(Matrix<T> alpha)
        {
            if (!_useEdgeNormalization)
                return ApplySoftmax(alpha);

            var result = new Matrix<T>(alpha.Rows, alpha.Columns);

            // For each edge (row), normalize across operations
            for (int row = 0; row < alpha.Rows; row++)
            {
                // Compute softmax for this edge
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
        /// Standard softmax operation
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
                edgeScores.Sort((a, b) => _ops.Compare(b.score, a.score));
                int numEdgesToKeep = Math.Min(2, edgeScores.Count);

                for (int i = 0; i < numEdgesToKeep; i++)
                {
                    var (prevNode, opIdx, _) = edgeScores[i];
                    if (_searchSpace.Operations != null && opIdx < _searchSpace.Operations.Count)
                    {
                        var operation = _searchSpace.Operations[opIdx];
                        // Skip identity operations in final architecture
                        if (operation != "identity")
                        {
                            architecture.AddOperation(nodeIdx + 1, prevNode, operation);
                        }
                    }
                }
            }

            return architecture;
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
    }
}
