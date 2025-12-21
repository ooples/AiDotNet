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
    /// FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search.
    /// Uses Gumbel-Softmax with hardware latency constraints to find efficient architectures
    /// optimized for specific target devices.
    ///
    /// Reference: "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS" (CVPR 2019)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class FBNet<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly int _numLayers;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters (theta in paper)
        private readonly List<Vector<T>> _architectureParams;  // Per-layer operation selection
        private readonly List<Vector<T>> _architectureGradients;

        // Hardware cost model
        private readonly HardwarePlatform _targetPlatform;
        private readonly HardwareCostModel<T> _hardwareCostModel;
        private readonly HardwareConstraints<T> _hardwareConstraints;

        // FBNet-specific parameters
        private readonly double _initialTemperature;
        private T _temperature;
        private readonly T _latencyWeight;
        private readonly int _inputChannels;
        private readonly int _spatialSize;

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => _numLayers;

        public FBNet(SearchSpaceBase<T> searchSpace, int numLayers = 20,
            HardwarePlatform targetPlatform = HardwarePlatform.Mobile,
            double latencyWeight = 0.2, double initialTemperature = 5.0,
            int inputChannels = 16, int spatialSize = 224)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numLayers = numLayers;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(42);

            _targetPlatform = targetPlatform;
            _hardwareCostModel = new HardwareCostModel<T>(targetPlatform);
            _latencyWeight = _ops.FromDouble(latencyWeight);
            _initialTemperature = initialTemperature;
            _temperature = _ops.FromDouble(initialTemperature);
            _inputChannels = inputChannels;
            _spatialSize = spatialSize;

            // Default hardware constraints (can be customized)
            _hardwareConstraints = new HardwareConstraints<T>
            {
                MaxLatency = 100.0,  // 100ms
                MaxMemory = 100.0,   // 100MB
                MaxEnergy = 500.0    // 500mJ
            };

            // Initialize architecture parameters (one per layer)
            _architectureParams = new List<Vector<T>>();
            _architectureGradients = new List<Vector<T>>();

            for (int i = 0; i < _numLayers; i++)
            {
                var theta = new Vector<T>(_numOperations);
                for (int j = 0; j < theta.Length; j++)
                {
                    theta[j] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.2);
                }
                _architectureParams.Add(theta);
                _architectureGradients.Add(new Vector<T>(_numOperations));
            }
        }

        /// <summary>
        /// Applies Gumbel-Softmax to layer-wise architecture parameters
        /// </summary>
        public Vector<T> GumbelSoftmax(Vector<T> theta, bool hard = false)
        {
            return NasSamplingHelper.GumbelSoftmax(theta, _temperature, _ops, _random, hard);
        }

        /// <summary>
        /// Computes the expected latency cost for the entire architecture
        /// </summary>
        public T ComputeExpectedLatency()
        {
            T totalLatency = _ops.Zero;

            for (int layerIdx = 0; layerIdx < _numLayers; layerIdx++)
            {
                var theta = _architectureParams[layerIdx];
                var probs = GumbelSoftmax(theta, hard: false);

                // Compute expected latency for this layer
                for (int opIdx = 0; opIdx < _numOperations; opIdx++)
                {
                    if (_nasSearchSpace.Operations != null && opIdx < _nasSearchSpace.Operations.Count)
                    {
                        var operation = _nasSearchSpace.Operations[opIdx];
                        var cost = _hardwareCostModel.EstimateOperationCost(
                            operation, _inputChannels, _inputChannels, _spatialSize);

                        T weightedLatency = _ops.Multiply(probs[opIdx], cost.Latency);
                        totalLatency = _ops.Add(totalLatency, weightedLatency);
                    }
                }
            }

            return totalLatency;
        }

        /// <summary>
        /// Computes the total loss with latency regularization
        /// Loss = Cross-Entropy + λ * log(Latency)
        /// Using log(latency) makes the loss more sensitive to changes when latency is small
        /// </summary>
        public T ComputeTotalLoss(T taskLoss)
        {
            T latency = ComputeExpectedLatency();

            // Avoid log(0) by adding small epsilon
            T logLatency = _ops.Log(_ops.Add(latency, _ops.FromDouble(1e-8)));

            T latencyLoss = _ops.Multiply(_latencyWeight, logLatency);
            return _ops.Add(taskLoss, latencyLoss);
        }

        /// <summary>
        /// Derives the discrete architecture by selecting the operation with highest probability
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int layerIdx = 0; layerIdx < _numLayers; layerIdx++)
            {
                var theta = _architectureParams[layerIdx];
                int selectedOp = 0;
                T maxLogit = theta[0];
                for (int opIdx = 1; opIdx < _numOperations; opIdx++)
                {
                    if (_ops.GreaterThan(theta[opIdx], maxLogit))
                    {
                        maxLogit = theta[opIdx];
                        selectedOp = opIdx;
                    }
                }

                if (_nasSearchSpace.Operations != null && selectedOp < _nasSearchSpace.Operations.Count)
                {
                    var operation = _nasSearchSpace.Operations[selectedOp];
                    // In FBNet, each layer is a node connecting to the previous layer
                    architecture.AddOperation(layerIdx + 1, layerIdx, operation);
                }
            }

            return architecture;
        }

        /// <summary>
        /// Checks if the derived architecture meets hardware constraints
        /// </summary>
        public bool MeetsConstraints()
        {
            var architecture = DeriveArchitecture();
            return _hardwareCostModel.MeetsConstraints(architecture, _hardwareConstraints, _inputChannels, _spatialSize);
        }

        /// <summary>
        /// Gets the final architecture's hardware cost breakdown
        /// </summary>
        public HardwareCost<T> GetArchitectureCost()
        {
            var architecture = DeriveArchitecture();
            return _hardwareCostModel.EstimateArchitectureCost(architecture, _inputChannels, _spatialSize);
        }

        /// <summary>
        /// Anneals the temperature during training
        /// </summary>
        public void AnnealTemperature(int currentEpoch, int maxEpochs)
        {
            // Exponential decay from initial temperature to near 0
            double ratio = (double)currentEpoch / maxEpochs;
            double newTemp = _initialTemperature * Math.Pow(0.1, ratio);
            _temperature = _ops.FromDouble(Math.Max(newTemp, 0.1));
        }

        /// <summary>
        /// Sets hardware constraints for the search
        /// </summary>
        public void SetConstraints(HardwareConstraints<T> constraints)
        {
            if (constraints.MaxLatency != null)
                _hardwareConstraints.MaxLatency = constraints.MaxLatency;
            if (constraints.MaxMemory != null)
                _hardwareConstraints.MaxMemory = constraints.MaxMemory;
            if (constraints.MaxEnergy != null)
                _hardwareConstraints.MaxEnergy = constraints.MaxEnergy;
        }

        /// <summary>
        /// Gets architecture parameters for optimization
        /// </summary>
        public List<Vector<T>> GetArchitectureParameters() => _architectureParams;

        /// <summary>
        /// Gets architecture gradients
        /// </summary>
        public List<Vector<T>> GetArchitectureGradients() => _architectureGradients;

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
            return new FBNet<T>(
                _nasSearchSpace,
                _numLayers,
                targetPlatform: _targetPlatform,
                latencyWeight: _ops.ToDouble(_latencyWeight),
                initialTemperature: _initialTemperature,
                inputChannels: _inputChannels,
                spatialSize: _spatialSize);
        }
    }
}
