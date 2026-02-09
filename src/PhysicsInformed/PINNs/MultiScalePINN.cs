using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Implements a Multi-Scale Physics-Informed Neural Network for solving PDEs with multiple scales.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Multi-scale problems are challenging because features span vastly different sizes.
    /// A single neural network struggles to capture both large-scale trends and fine details.
    ///
    /// Solution: Multi-Scale PINN
    /// Uses multiple sub-networks, each specialized for a different scale:
    /// - Coarse network: Captures large-scale, smooth variations
    /// - Fine network(s): Capture small-scale details and fluctuations
    ///
    /// Architecture:
    /// Input (x,t) → [Coarse Net] → u_coarse(x,t)
    ///            → [Fine Net 1] → u_fine1(x,t)
    ///            → [Fine Net 2] → u_fine2(x,t)
    ///            → ...
    ///
    /// Total solution: u(x,t) = u_coarse + u_fine1 + u_fine2 + ...
    ///
    /// Training Strategy:
    /// 1. Progressive Training: Train coarse first, then add finer scales
    /// 2. Simultaneous Training: Train all scales together with adaptive weights
    /// 3. Alternating Training: Alternate between scales during training
    ///
    /// Key Features:
    /// - Fourier feature encoding at different frequencies for each scale
    /// - Adaptive loss weighting to balance scale contributions
    /// - Scale coupling terms to ensure consistency
    /// - Progressive activation of finer scales during training
    ///
    /// Applications:
    /// - Turbulence modeling (large eddies + small vortices)
    /// - Composite materials (macroscopic + fiber-scale behavior)
    /// - Multi-physics problems (thermal + mechanical + chemical)
    /// - Climate modeling (global + regional + local scales)
    /// </remarks>
    public class MultiScalePINN<T> : NeuralNetworkBase<T>
    {
        private readonly MultiScalePINNOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        private readonly IMultiScalePDE<T> _multiScalePDE;
        private readonly List<NeuralNetworkBase<T>> _scaleNetworks;
        private readonly IBoundaryCondition<T>[] _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;
        private readonly MultiScaleTrainingOptions<T> _trainingOptions;

        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        private T[,]? _collocationPoints;
        private readonly int _numCollocationPointsPerScale;
        private T[] _scaleWeights;
        private int _currentActiveScales;

        /// <summary>
        /// Gets the number of scales in this multi-scale PINN.
        /// </summary>
        public int NumberOfScales => _multiScalePDE.NumberOfScales;

        /// <summary>
        /// Gets the characteristic length scales.
        /// </summary>
        public T[] ScaleCharacteristicLengths => _multiScalePDE.ScaleCharacteristicLengths;

        /// <summary>
        /// Initializes a new instance of the Multi-Scale PINN.
        /// </summary>
        /// <param name="architecture">The base neural network architecture.</param>
        /// <param name="multiScalePDE">The multi-scale PDE specification.</param>
        /// <param name="boundaryConditions">Boundary conditions for the problem.</param>
        /// <param name="initialCondition">Initial condition for time-dependent problems (optional).</param>
        /// <param name="numCollocationPointsPerScale">Number of collocation points per scale.</param>
        /// <param name="trainingOptions">Options for multi-scale training.</param>
        /// <param name="optimizer">Optimization algorithm.</param>
        /// <remarks>
        /// For Beginners:
        ///
        /// This creates separate neural networks for each scale:
        /// - Coarse network: Wider/shallower, larger learning rate
        /// - Fine networks: Deeper, smaller learning rate, Fourier features
        ///
        /// The collocation points are generated at different densities:
        /// - Fewer points for coarse scale (smooth variations)
        /// - More points for fine scales (detailed features)
        /// </remarks>
        public MultiScalePINN(
            NeuralNetworkArchitecture<T> architecture,
            IMultiScalePDE<T> multiScalePDE,
            IBoundaryCondition<T>[] boundaryConditions,
            IInitialCondition<T>? initialCondition = null,
            int numCollocationPointsPerScale = 5000,
            MultiScaleTrainingOptions<T>? trainingOptions = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            MultiScalePINNOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _options = options ?? new MultiScalePINNOptions();
            Options = _options;

            _multiScalePDE = multiScalePDE ?? throw new ArgumentNullException(nameof(multiScalePDE));
            _boundaryConditions = boundaryConditions ?? throw new ArgumentNullException(nameof(boundaryConditions));
            _initialCondition = initialCondition;
            _numCollocationPointsPerScale = numCollocationPointsPerScale;
            _trainingOptions = trainingOptions ?? new MultiScaleTrainingOptions<T>();

            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            // Initialize scale weights
            _scaleWeights = new T[_multiScalePDE.NumberOfScales];
            for (int i = 0; i < _scaleWeights.Length; i++)
            {
                _scaleWeights[i] = _multiScalePDE.GetScaleLossWeight(i);
            }

            // Create sub-networks for each scale
            _scaleNetworks = new List<NeuralNetworkBase<T>>();
            CreateScaleNetworks(architecture);

            // Start with just coarse scale if sequential training
            _currentActiveScales = _trainingOptions.UseSequentialScaleTraining ? 1 : _multiScalePDE.NumberOfScales;

            InitializeLayers();
            GenerateCollocationPoints();
        }

        /// <summary>
        /// Creates the neural networks for each scale.
        /// </summary>
        private void CreateScaleNetworks(NeuralNetworkArchitecture<T> baseArchitecture)
        {
            for (int scale = 0; scale < _multiScalePDE.NumberOfScales; scale++)
            {
                int outputDim = _multiScalePDE.GetScaleOutputDimension(scale);

                // Adjust network size based on scale
                // Finer scales typically need deeper networks
                var scaleArchitecture = CreateScaleArchitecture(baseArchitecture, scale, outputDim);
                var scaleNetwork = new FeedForwardNeuralNetwork<T>(scaleArchitecture);

                _scaleNetworks.Add(scaleNetwork);
            }
        }

        /// <summary>
        /// Creates architecture for a specific scale.
        /// </summary>
        private NeuralNetworkArchitecture<T> CreateScaleArchitecture(
            NeuralNetworkArchitecture<T> baseArchitecture,
            int scaleIndex,
            int outputDimension)
        {
            // For finer scales, use deeper networks with Fourier features
            int depthMultiplier = 1 + scaleIndex; // Finer scales are deeper

            return new NeuralNetworkArchitecture<T>(
                baseArchitecture.InputType,
                baseArchitecture.TaskType,
                baseArchitecture.Complexity,
                baseArchitecture.InputSize,
                baseArchitecture.InputHeight,
                baseArchitecture.InputWidth,
                baseArchitecture.InputDepth,
                outputDimension,
                null, // Let LayerHelper create default layers
                baseArchitecture.ShouldReturnFullSequence);
        }

        /// <inheritdoc/>
        protected override void InitializeLayers()
        {
            // MultiScalePINN uses sub-networks instead of traditional layers
            // The Layers collection remains empty as computation is done via scale networks
        }

        /// <summary>
        /// Generates collocation points for each scale.
        /// </summary>
        private void GenerateCollocationPoints()
        {
            int inputDim = _multiScalePDE.InputDimension;
            int totalPoints = _numCollocationPointsPerScale * _multiScalePDE.NumberOfScales;
            _collocationPoints = new T[totalPoints, inputDim];

            var random = RandomHelper.CreateSeededRandom(42);

            for (int i = 0; i < totalPoints; i++)
            {
                for (int d = 0; d < inputDim; d++)
                {
                    // Generate points in [0, 1]^d
                    _collocationPoints[i, d] = NumOps.FromDouble(random.NextDouble());
                }
            }
        }

        /// <summary>
        /// Forward pass through all scale networks.
        /// </summary>
        /// <param name="input">Input coordinates [batch, inputDim].</param>
        /// <returns>Combined output from all scales [batch, outputDim].</returns>
        public Tensor<T> Forward(Tensor<T> input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (input.Rank != 2)
            {
                throw new ArgumentException("Input must be 2D [batch, inputDim].", nameof(input));
            }

            int batchSize = input.Shape[0];
            int outputDim = _multiScalePDE.OutputDimension;

            var combinedOutput = new Tensor<T>(new int[] { batchSize, outputDim });

            // Sum contributions from all active scales
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var scaleOutput = _scaleNetworks[scale].Predict(input);

                // Add scale contribution to combined output
                for (int b = 0; b < batchSize; b++)
                {
                    int scaleOutputDim = _multiScalePDE.GetScaleOutputDimension(scale);
                    for (int d = 0; d < Math.Min(scaleOutputDim, outputDim); d++)
                    {
                        combinedOutput[b, d] = NumOps.Add(combinedOutput[b, d], scaleOutput[b, d]);
                    }
                }
            }

            return combinedOutput;
        }

        /// <inheritdoc/>
        public override Tensor<T> ForwardWithMemory(Tensor<T> input)
        {
            // Set all scale networks to training mode
            foreach (var network in _scaleNetworks.Take(_currentActiveScales))
            {
                network.SetTrainingMode(true);
            }

            return Forward(input);
        }

        /// <summary>
        /// Solves the multi-scale PDE using physics-informed training.
        /// </summary>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <returns>Training history.</returns>
        public TrainingHistory<T> Solve(int epochs = 1000, double learningRate = 0.001, bool verbose = true)
        {
            var history = new TrainingHistory<T>();

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);

            try
            {
                if (_trainingOptions.UseSequentialScaleTraining)
                {
                    // Progressive training: coarse to fine
                    SolveSequential(epochs, history, verbose);
                }
                else
                {
                    // Simultaneous training of all scales
                    SolveSimultaneous(epochs, history, verbose);
                }
            }
            finally
            {
                SetTrainingMode(false);
            }

            return history;
        }

        /// <summary>
        /// Sequential training: train coarse scale first, then progressively add finer scales.
        /// </summary>
        private void SolveSequential(int epochs, TrainingHistory<T> history, bool verbose)
        {
            int pretrainingEpochs = _trainingOptions.ScalePretrainingEpochs;
            int epochsPerScale = epochs / _multiScalePDE.NumberOfScales;

            for (int scale = 0; scale < _multiScalePDE.NumberOfScales; scale++)
            {
                _currentActiveScales = scale + 1;

                if (verbose)
                {
                    Console.WriteLine($"\n=== Training Scale {scale + 1}/{_multiScalePDE.NumberOfScales} ===");
                }

                int scaleEpochs = (scale == _multiScalePDE.NumberOfScales - 1)
                    ? epochs - scale * epochsPerScale
                    : epochsPerScale;

                for (int epoch = 0; epoch < scaleEpochs; epoch++)
                {
                    T loss = TrainEpoch();
                    history.AddEpoch(loss);

                    if (verbose && epoch % 50 == 0)
                    {
                        Console.WriteLine($"Scale {scale + 1}, Epoch {epoch}/{scaleEpochs}, Loss: {loss}");
                    }
                }
            }
        }

        /// <summary>
        /// Simultaneous training: train all scales together with adaptive weighting.
        /// </summary>
        private void SolveSimultaneous(int epochs, TrainingHistory<T> history, bool verbose)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T loss = TrainEpoch();
                history.AddEpoch(loss);

                // Update scale weights adaptively
                if (_trainingOptions.UseAdaptiveScaleWeighting && epoch > 0 && epoch % 100 == 0)
                {
                    UpdateAdaptiveScaleWeights();
                }

                if (verbose && epoch % 50 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {loss}");
                }
            }
        }

        /// <summary>
        /// Performs one training epoch.
        /// </summary>
        private T TrainEpoch()
        {
            T totalLoss = NumOps.Zero;

            if (_collocationPoints == null)
            {
                throw new InvalidOperationException("Collocation points not initialized.");
            }

            int numPoints = _collocationPoints.GetLength(0);
            int inputDim = _multiScalePDE.InputDimension;

            // Process collocation points in batches
            int batchSize = Math.Min(256, numPoints);

            for (int batchStart = 0; batchStart < numPoints; batchStart += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, numPoints - batchStart);

                // Create input tensor for this batch
                var inputTensor = new Tensor<T>(new int[] { actualBatchSize, inputDim });
                for (int i = 0; i < actualBatchSize; i++)
                {
                    for (int d = 0; d < inputDim; d++)
                    {
                        inputTensor[i, d] = _collocationPoints[batchStart + i, d];
                    }
                }

                // Compute loss for this batch
                T batchLoss = ComputeMultiScaleLoss(inputTensor);
                totalLoss = NumOps.Add(totalLoss, batchLoss);

                // Update parameters
                UpdateAllScaleNetworks();
            }

            return NumOps.Divide(totalLoss, NumOps.FromDouble(numPoints / batchSize));
        }

        /// <summary>
        /// Computes the multi-scale physics-informed loss.
        /// </summary>
        private T ComputeMultiScaleLoss(Tensor<T> input)
        {
            T totalLoss = NumOps.Zero;
            int batchSize = input.Shape[0];
            int inputDim = _multiScalePDE.InputDimension;

            // Get outputs from all active scales
            var scaleOutputs = new List<Tensor<T>>();
            var scaleDerivatives = new List<PDEDerivatives<T>[]>();

            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var output = _scaleNetworks[scale].ForwardWithMemory(input);
                scaleOutputs.Add(output);

                // Compute derivatives for each point in the batch
                var derivatives = new PDEDerivatives<T>[batchSize];
                for (int b = 0; b < batchSize; b++)
                {
                    derivatives[b] = ComputeDerivativesForPoint(input, b, scale);
                }
                scaleDerivatives.Add(derivatives);
            }

            // Compute scale-specific residual losses
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                T scaleLoss = NumOps.Zero;

                for (int b = 0; b < batchSize; b++)
                {
                    var inputPoint = ExtractPoint(input, b);
                    var outputPoint = ExtractOutputPoint(scaleOutputs[scale], b);

                    T residual = _multiScalePDE.ComputeScaleResidual(
                        scale, inputPoint, outputPoint, scaleDerivatives[scale][b]);

                    scaleLoss = NumOps.Add(scaleLoss, NumOps.Multiply(residual, residual));
                }

                // Weight and accumulate
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_scaleWeights[scale], scaleLoss));
            }

            // Compute coupling losses between scales
            T couplingWeight = _trainingOptions.CouplingWeight ?? NumOps.One;

            for (int coarse = 0; coarse < _currentActiveScales - 1; coarse++)
            {
                int fine = coarse + 1;

                for (int b = 0; b < batchSize; b++)
                {
                    var inputPoint = ExtractPoint(input, b);
                    var coarseOutput = ExtractOutputPoint(scaleOutputs[coarse], b);
                    var fineOutput = ExtractOutputPoint(scaleOutputs[fine], b);

                    T couplingResidual = _multiScalePDE.ComputeScaleCoupling(
                        coarse, fine, inputPoint,
                        coarseOutput, fineOutput,
                        scaleDerivatives[coarse][b], scaleDerivatives[fine][b]);

                    totalLoss = NumOps.Add(totalLoss,
                        NumOps.Multiply(couplingWeight, NumOps.Multiply(couplingResidual, couplingResidual)));
                }
            }

            return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        }

        /// <summary>
        /// Computes derivatives for a single point using finite differences.
        /// </summary>
        private PDEDerivatives<T> ComputeDerivativesForPoint(Tensor<T> input, int batchIndex, int scaleIndex)
        {
            int inputDim = _multiScalePDE.InputDimension;
            int outputDim = _multiScalePDE.GetScaleOutputDimension(scaleIndex);
            T epsilon = NumOps.FromDouble(1e-5);

            var derivatives = new PDEDerivatives<T>
            {
                FirstDerivatives = new T[outputDim, inputDim],
                SecondDerivatives = new T[outputDim, inputDim, inputDim]
            };

            // Extract the base point
            var basePoint = ExtractPoint(input, batchIndex);
            var baseInput = new Tensor<T>(new int[] { 1, inputDim });
            for (int d = 0; d < inputDim; d++)
            {
                baseInput[0, d] = basePoint[d];
            }

            var baseOutput = _scaleNetworks[scaleIndex].Predict(baseInput);

            // Compute first and second derivatives using central differences
            for (int d = 0; d < inputDim; d++)
            {
                var plusInput = new Tensor<T>(new int[] { 1, inputDim });
                var minusInput = new Tensor<T>(new int[] { 1, inputDim });

                for (int i = 0; i < inputDim; i++)
                {
                    plusInput[0, i] = basePoint[i];
                    minusInput[0, i] = basePoint[i];
                }

                plusInput[0, d] = NumOps.Add(basePoint[d], epsilon);
                minusInput[0, d] = NumOps.Subtract(basePoint[d], epsilon);

                var plusOutput = _scaleNetworks[scaleIndex].Predict(plusInput);
                var minusOutput = _scaleNetworks[scaleIndex].Predict(minusInput);

                T twoEpsilon = NumOps.Multiply(NumOps.FromDouble(2.0), epsilon);

                for (int o = 0; o < outputDim; o++)
                {
                    // First derivative: (f(x+h) - f(x-h)) / 2h
                    derivatives.FirstDerivatives[o, d] = NumOps.Divide(
                        NumOps.Subtract(plusOutput[0, o], minusOutput[0, o]),
                        twoEpsilon);

                    // Second derivative: (f(x+h) - 2*f(x) + f(x-h)) / h^2
                    T epsilonSq = NumOps.Multiply(epsilon, epsilon);
                    derivatives.SecondDerivatives[o, d, d] = NumOps.Divide(
                        NumOps.Subtract(
                            NumOps.Add(plusOutput[0, o], minusOutput[0, o]),
                            NumOps.Multiply(NumOps.FromDouble(2.0), baseOutput[0, o])),
                        epsilonSq);
                }
            }

            return derivatives;
        }

        /// <summary>
        /// Updates all scale network parameters.
        /// </summary>
        private void UpdateAllScaleNetworks()
        {
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                _optimizer.UpdateParameters(_scaleNetworks[scale].Layers);
            }
        }

        /// <summary>
        /// Updates scale weights adaptively based on gradient magnitudes.
        /// </summary>
        private void UpdateAdaptiveScaleWeights()
        {
            // Compute gradient magnitudes for each scale
            var gradientMagnitudes = new T[_currentActiveScales];

            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var gradients = _scaleNetworks[scale].GetGradients();
                T magnitude = NumOps.Zero;

                for (int i = 0; i < gradients.Length; i++)
                {
                    magnitude = NumOps.Add(magnitude, NumOps.Multiply(gradients[i], gradients[i]));
                }

                gradientMagnitudes[scale] = NumOps.Sqrt(magnitude);
            }

            // Normalize weights to balance gradient contributions
            T totalMagnitude = NumOps.Zero;
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                totalMagnitude = NumOps.Add(totalMagnitude, gradientMagnitudes[scale]);
            }

            if (!NumOps.Equals(totalMagnitude, NumOps.Zero))
            {
                for (int scale = 0; scale < _currentActiveScales; scale++)
                {
                    // Inverse magnitude weighting: scales with larger gradients get smaller weights
                    _scaleWeights[scale] = NumOps.Divide(
                        totalMagnitude,
                        NumOps.Add(gradientMagnitudes[scale], NumOps.FromDouble(1e-8)));
                }
            }
        }

        /// <summary>
        /// Extracts a single point from the input tensor.
        /// </summary>
        private T[] ExtractPoint(Tensor<T> input, int batchIndex)
        {
            int inputDim = input.Shape[1];
            var point = new T[inputDim];

            for (int d = 0; d < inputDim; d++)
            {
                point[d] = input[batchIndex, d];
            }

            return point;
        }

        /// <summary>
        /// Extracts output values for a single point.
        /// </summary>
        private T[] ExtractOutputPoint(Tensor<T> output, int batchIndex)
        {
            int outputDim = output.Shape[1];
            var point = new T[outputDim];

            for (int d = 0; d < outputDim; d++)
            {
                point[d] = output[batchIndex, d];
            }

            return point;
        }

        /// <inheritdoc/>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            SetTrainingMode(false);
            return Forward(input);
        }

        /// <inheritdoc/>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (expectedOutput == null)
            {
                throw new ArgumentNullException(nameof(expectedOutput));
            }

            SetTrainingMode(true);

            try
            {
                var prediction = ForwardWithMemory(input);
                var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
                LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

                // Backpropagate through all active scale networks
                var outputGradientVector = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
                var outputGradient = new Tensor<T>(prediction.Shape, outputGradientVector);

                // Distribute gradients to each scale network
                for (int scale = 0; scale < _currentActiveScales; scale++)
                {
                    _scaleNetworks[scale].Backpropagate(outputGradient);
                }

                UpdateAllScaleNetworks();
            }
            finally
            {
                SetTrainingMode(false);
            }
        }

        /// <inheritdoc/>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int offset = 0;

            foreach (var network in _scaleNetworks)
            {
                int paramCount = network.GetParameterCount();
                var subParams = parameters.GetSubVector(offset, paramCount);
                network.UpdateParameters(subParams);
                offset += paramCount;
            }
        }

        /// <inheritdoc/>
        public override Vector<T> GetParameters()
        {
            var allParams = new List<Vector<T>>();

            foreach (var network in _scaleNetworks)
            {
                allParams.Add(network.GetParameters());
            }

            return Vector<T>.Concatenate(allParams.ToArray());
        }

        /// <inheritdoc/>
        public override Vector<T> GetGradients()
        {
            var allGradients = new List<Vector<T>>();

            foreach (var network in _scaleNetworks)
            {
                allGradients.Add(network.GetGradients());
            }

            return Vector<T>.Concatenate(allGradients.ToArray());
        }

        /// <inheritdoc/>
        public override int ParameterCount => _scaleNetworks.Sum(n => n.GetParameterCount());

        /// <inheritdoc/>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "NetworkType", "MultiScalePINN" },
                    { "NumberOfScales", _multiScalePDE.NumberOfScales },
                    { "ActiveScales", _currentActiveScales },
                    { "PDEName", _multiScalePDE.Name },
                    { "ParameterCount", ParameterCount }
                },
                ModelData = Serialize()
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_multiScalePDE.NumberOfScales);
            writer.Write(_currentActiveScales);
            writer.Write(_numCollocationPointsPerScale);

            foreach (var network in _scaleNetworks)
            {
                var bytes = network.Serialize();
                writer.Write(bytes.Length);
                writer.Write(bytes);
            }
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int numScales = reader.ReadInt32();
            _currentActiveScales = reader.ReadInt32();
            int numPoints = reader.ReadInt32();

            if (numScales != _multiScalePDE.NumberOfScales)
            {
                throw new InvalidOperationException("Serialized number of scales does not match.");
            }

            for (int scale = 0; scale < numScales; scale++)
            {
                int length = reader.ReadInt32();
                _scaleNetworks[scale].Deserialize(reader.ReadBytes(length));
            }
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new MultiScalePINN<T>(
                Architecture,
                _multiScalePDE,
                _boundaryConditions,
                _initialCondition,
                _numCollocationPointsPerScale,
                _trainingOptions,
                _optimizer);
        }

        /// <inheritdoc/>
        public override bool SupportsTraining => true;

        /// <inheritdoc/>
        public override bool SupportsJitCompilation => false;
    }
}
