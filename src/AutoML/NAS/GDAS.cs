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
    /// Gradient-based Differentiable Architecture Search with Gumbel-Softmax sampling.
    /// GDAS uses Gumbel-Softmax to make the architecture search fully differentiable
    /// while maintaining discrete selection during forward pass.
    ///
    /// Reference: "Searching for A Robust Neural Architecture in Four GPU Hours" (CVPR 2019)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class GDAS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
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

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => _numNodes;

        public GDAS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            double initialTemperature = 5.0, double finalTemperature = 0.1)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(42);

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
            return NasSamplingHelper.GumbelSoftmaxRows(alpha, _temperature, _ops, _random, hard);
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

                    if (_nasSearchSpace.Operations != null && bestOpIdx < _nasSearchSpace.Operations.Count)
                    {
                        var operation = _nasSearchSpace.Operations[bestOpIdx];
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
            return new GDAS<T>(
                _nasSearchSpace,
                _numNodes,
                initialTemperature: _ops.ToDouble(_initialTemperature),
                finalTemperature: _ops.ToDouble(_finalTemperature));
        }
    }
}
