using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
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
    /// <remarks>
    /// <para><b>For Beginners:</b> FBNet (Facebook Net) finds architectures that are both
    /// accurate and fast on specific hardware. It uses differentiable search with real
    /// hardware latency measurements as constraints. Think of it as an architect who designs
    /// buildings that are both beautiful and structurally efficient for a specific site,
    /// rather than designing in the abstract.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var searchSpace = new SearchSpaceBase&lt;float&gt;();
    /// var fbnet = new FBNet&lt;float&gt;(searchSpace, targetPlatform: HardwarePlatform.Mobile);
    /// Architecture&lt;float&gt; architecture = fbnet.DeriveArchitecture();
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.ConvolutionalNetwork)]
    [ModelCategory(ModelCategory.Optimization)]
    [ModelTask(ModelTask.Classification)]
    [ModelTask(ModelTask.FeatureExtraction)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS", "https://arxiv.org/abs/1812.03443")]
    public partial class FBNet<T> : NasAutoMLModelBase<T>
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

        /// <inheritdoc />
        protected override INumericOperations<T> NumOps => _ops;
        /// <inheritdoc />
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        /// <inheritdoc />
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
            var operations = _nasSearchSpace.Operations;
            int pricedOps = operations == null ? 0 : Math.Min(_numOperations, operations.Count);
            var opLatency = pricedOps == 0 || operations == null
                ? new Vector<T>(0)
                : new Vector<T>(operations.Take(pricedOps)
                    .Select(op => _hardwareCostModel.EstimateOperationCost(op, _inputChannels, _inputChannels, _spatialSize).Latency));

            var engine = AiDotNetEngine.Current;
            for (int layerIdx = 0; layerIdx < _numLayers; layerIdx++)
            {
                // LAT(a) = sum_l sum_i m_l,i * LAT(b_l,i) with Gumbel-softmax masks m (FBNet, Eq. 4). Every layer
                // still draws its mask in order, so the seeded noise stream is unchanged.
                var probs = GumbelSoftmax(_architectureParams[layerIdx], hard: false);
                if (pricedOps > 0)
                {
                    totalLatency = _ops.Add(totalLatency, engine.DotProduct(probs.GetSubVector(0, pricedOps), opLatency));
                }
            }

            return totalLatency;
        }

        /// <summary>
        /// Computes the FBNet loss CE * alpha * log(LAT)^beta (Wu et al. 2019, Sec. 3.2), with alpha the latency
        /// weight and beta <see cref="LatencyExponent"/>. LAT is in milliseconds; log(1 + LAT) keeps the base
        /// positive for sub-millisecond latencies, where log(LAT)^beta is undefined, and equals log(LAT) to within
        /// 1/LAT otherwise.
        /// </summary>
        public T ComputeTotalLoss(T taskLoss)
        {
            T latency = ComputeExpectedLatency();

            T logLatency = _ops.Log(_ops.Add(latency, _ops.One));

            // FBNet (Wu et al. 2019, Sec. 3.2): L(a, w_a) = CE(a, w_a) * alpha * log(LAT(a))^beta. The latency term
            // scales the task loss multiplicatively; alpha is the latency weight and beta the latency exponent.
            T latencyFactor = _ops.Multiply(_latencyWeight, _ops.Power(logLatency, _ops.FromDouble(LatencyExponent)));
            return _ops.Multiply(taskLoss, latencyFactor);
        }

        /// <summary>
        /// Gets or sets the exponent beta of the latency term in the FBNet loss CE * alpha * log(LAT)^beta
        /// (default 0.6, the paper's value).
        /// </summary>
        public double LatencyExponent { get; set; } = 0.6;

        /// <summary>
        /// Derives the discrete architecture by selecting the operation with highest probability
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int layerIdx = 0; layerIdx < _numLayers; layerIdx++)
            {
                int selectedOp = NasSamplingHelper.ArgMax(_architectureParams[layerIdx]);

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

        /// <inheritdoc />
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
    }
}
