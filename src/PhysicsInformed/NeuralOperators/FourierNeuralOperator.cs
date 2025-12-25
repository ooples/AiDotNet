using System;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed;
using AiDotNet.Tensors.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.PhysicsInformed.NeuralOperators
{
    /// <summary>
    /// Implements the Fourier Neural Operator (FNO) for learning operators between function spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// A Neural Operator learns mappings between entire functions, not just inputs to outputs.
    ///
    /// Traditional Neural Networks:
    /// - Learn: point → point mappings
    /// - Input: a vector (x, y, z)
    /// - Output: a vector (u, v, w)
    /// - Example: (temperature, pressure) → (velocity)
    ///
    /// Neural Operators:
    /// - Learn: function → function mappings
    /// - Input: an entire function a(x)
    /// - Output: an entire function u(x)
    /// - Example: initial condition → solution after time T
    ///
    /// Why This Matters:
    /// Many problems in physics involve operators:
    /// - PDE solution operator: (initial/boundary conditions) → (solution)
    /// - Green's function: (source) → (response)
    /// - Transfer function: (input signal) → (output signal)
    ///
    /// Traditionally, you'd need to solve the PDE from scratch for each new set of conditions.
    /// With neural operators, you train once, then can instantly evaluate for new conditions!
    ///
    /// Fourier Neural Operator (FNO):
    /// The key innovation is doing computations in Fourier space.
    ///
    /// How FNO Works:
    /// 1. Lift: Embed input function into higher-dimensional space
    /// 2. Fourier Layers (repeated):
    ///    a) Apply FFT to transform to frequency domain
    ///    b) Linear transformation in frequency space (learn weights)
    ///    c) Apply inverse FFT to return to physical space
    ///    d) Add skip connection and activation
    /// 3. Project: Map back to output function
    ///
    /// Why Fourier Space?
    /// - Many PDEs have simple form in frequency domain
    /// - Derivatives → multiplication (∂/∂x in physical space = ik in Fourier space)
    /// - Captures global information efficiently
    /// - Natural for periodic problems
    /// - Computational efficiency (FFT is O(n log n))
    ///
    /// Key Advantages:
    /// 1. Resolution-invariant: Train at one resolution, evaluate at another
    /// 2. Fast: Instant evaluation after training (vs. solving PDE each time)
    /// 3. Mesh-free: No discretization needed
    /// 4. Generalizes well: Works for different parameter values
    /// 5. Captures long-range dependencies naturally
    ///
    /// Applications:
    /// - Fluid dynamics (Navier-Stokes)
    /// - Climate modeling (weather prediction)
    /// - Material science (stress-strain)
    /// - Seismic imaging
    /// - Quantum chemistry (electron density)
    ///
    /// Example Use Case:
    /// Problem: Solve 2D Navier-Stokes for different initial vorticity fields
    /// Traditional: Solve PDE numerically for each initial condition (slow)
    /// FNO: Train once on many examples, then instantly predict solution for new initial conditions
    ///
    /// Historical Context:
    /// FNO was introduced by Li et al. in 2021 and has achieved remarkable success
    /// in learning solution operators for PDEs, often matching or exceeding traditional
    /// numerical methods in accuracy while being orders of magnitude faster.
    /// </remarks>
    public class FourierNeuralOperator<T> : NeuralNetworkBase<T>
    {
        private readonly int _modes; // Number of Fourier modes to keep
        private readonly int _width; // Channel width of the network
        private readonly int[] _spatialDimensions;
        private readonly List<FourierLayer<T>> _fourierLayers;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        /// <summary>
        /// Initializes a new instance of the Fourier Neural Operator.
        /// </summary>
        /// <param name="architecture">The network architecture.</param>
        /// <param name="modes">Number of Fourier modes to retain (higher = more detail, but more computation).</param>
        /// <param name="width">Channel width of the network (similar to hidden layer size).</param>
        /// <param name="spatialDimensions">Dimensions of the input function domain (e.g., [64, 64] for 64x64 grid).</param>
        /// <param name="numLayers">Number of Fourier layers.</param>
        /// <remarks>
        /// For Beginners:
        ///
        /// Parameters Explained:
        ///
        /// modes: How many Fourier modes to keep
        /// - Low modes = low frequency information (smooth, large-scale features)
        /// - High modes = high frequency information (fine details, sharp features)
        /// - Typical: 12-32 modes
        /// - Trade-off: accuracy vs. computational cost
        ///
        /// width: Number of channels in the network
        /// - Like hidden layer size in regular networks
        /// - More width = more capacity, but slower
        /// - Typical: 32-128
        ///
        /// spatialDimensions: Size of the discretized function
        /// - For 1D: [N] (function sampled at N points)
        /// - For 2D: [Nx, Ny] (function on Nx × Ny grid)
        /// - For 3D: [Nx, Ny, Nz]
        /// - FNO can handle different resolutions at test time!
        ///
        /// numLayers: Depth of the network
        /// - More layers = more expressive, but diminishing returns
        /// - Typical: 4-8 layers
        /// </remarks>
        public FourierNeuralOperator(
            NeuralNetworkArchitecture<T> architecture,
            int modes = 16,
            int width = 64,
            int[]? spatialDimensions = null,
            int numLayers = 4,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _modes = modes;
            _width = width;
            _spatialDimensions = spatialDimensions ?? new int[] { 64, 64 }; // Default 2D
            _fourierLayers = new List<FourierLayer<T>>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            InitializeLayers();
            InitializeFourierLayers(numLayers);
        }

        protected override void InitializeLayers()
        {
            // Lifting layer: embed input to higher dimension
            var liftLayer = new NeuralNetworks.Layers.DenseLayer<T>(
                Architecture.InputSize,
                _width,
                new GELUActivation<T>() as IActivationFunction<T>);
            Layers.Add(liftLayer);

            // Projection layer will be added after Fourier layers
        }

        /// <summary>
        /// Initializes the Fourier layers.
        /// </summary>
        private void InitializeFourierLayers(int numLayers)
        {
            for (int i = 0; i < numLayers; i++)
            {
                var fourierLayer = new FourierLayer<T>(_width, _modes, _spatialDimensions);
                _fourierLayers.Add(fourierLayer);
            }

            // Projection layer: map back to output dimension
            var projectLayer = new NeuralNetworks.Layers.DenseLayer<T>(
                _width,
                Architecture.OutputSize,
                NeuralNetworkHelper<T>.GetDefaultActivationFunction(Architecture.TaskType));
            Layers.Add(projectLayer);
        }

        /// <summary>
        /// Forward pass through the FNO.
        /// </summary>
        /// <param name="input">Input function (discretized on a grid).</param>
        /// <returns>Output function (solution).</returns>
        /// <remarks>
        /// For Beginners:
        /// The forward pass consists of:
        /// 1. Lift: input channels → width channels
        /// 2. Apply Fourier layers (multiple times)
        /// 3. Project: width channels → output channels
        ///
        /// Each Fourier layer does:
        /// - FFT to frequency domain
        /// - Learned linear transformation
        /// - Inverse FFT back to physical space
        /// - Add skip connection
        /// - Apply activation
        /// </remarks>
        public Tensor<T> Forward(Tensor<T> input)
        {
            return ForwardInternal(input);
        }

        public override Tensor<T> ForwardWithMemory(Tensor<T> input)
        {
            if (!SupportsTraining)
            {
                throw new InvalidOperationException("This network does not support training mode");
            }

            return ForwardInternal(input);
        }

        private Tensor<T> ForwardInternal(Tensor<T> input)
        {
            ValidateInputShape(input);

            if (Layers.Count < 2)
            {
                throw new InvalidOperationException("FourierNeuralOperator requires lift and projection layers.");
            }

            var liftLayer = Layers[0] as NeuralNetworks.Layers.DenseLayer<T>;
            var projectLayer = Layers[Layers.Count - 1] as NeuralNetworks.Layers.DenseLayer<T>;

            if (liftLayer == null || projectLayer == null)
            {
                throw new InvalidOperationException("FourierNeuralOperator expects DenseLayer lift and projection layers.");
            }

            Tensor<T> x = ApplyPointwiseDense(input, liftLayer);

            foreach (var fourierLayer in _fourierLayers)
            {
                x = fourierLayer.Forward(x);
            }

            x = ApplyPointwiseDense(x, projectLayer);

            return x;
        }

        /// <summary>
        /// Trains the FNO on input-output function pairs.
        /// </summary>
        /// <param name="inputFunctions">Training input functions.</param>
        /// <param name="outputFunctions">Training output functions (solutions).</param>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate.</param>
        /// <returns>Training history.</returns>
        /// <remarks>
        /// For Beginners:
        /// Training an FNO is like training a regular network, but:
        /// - Inputs are functions (represented as discretized grids)
        /// - Outputs are functions
        /// - Loss measures difference between predicted and true output functions
        ///
        /// Example:
        /// - Input: Initial temperature distribution T(x, y, t=0)
        /// - Output: Temperature distribution at later time T(x, y, t=1)
        /// - Loss: ||FNO(T_initial) - T_final||²
        ///
        /// After training, you can:
        /// - Give it a new initial condition
        /// - Instantly get the solution (no PDE solving!)
        /// - Even evaluate at different resolutions
        /// </remarks>
        public TrainingHistory<T> Train(
            Tensor<T>[] inputFunctions,
            Tensor<T>[] outputFunctions,
            int epochs = 100,
            double learningRate = 0.001)
        {
            var history = new TrainingHistory<T>();
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();

            if (inputFunctions.Length != outputFunctions.Length)
            {
                throw new ArgumentException("Number of input and output functions must match.");
            }

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }
            foreach (var layer in _fourierLayers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    T totalLoss = NumOps.Zero;

                    for (int i = 0; i < inputFunctions.Length; i++)
                    {
                        var prediction = ForwardWithMemory(inputFunctions[i]);
                        var target = outputFunctions[i];

                        var loss = lossFunction.CalculateLoss(prediction.ToVector(), target.ToVector());
                        totalLoss = NumOps.Add(totalLoss, loss);

                        var outputGradientVector = lossFunction.CalculateDerivative(prediction.ToVector(), target.ToVector());
                        var outputGradient = new Tensor<T>(prediction.Shape, outputGradientVector);

                        Backpropagate(outputGradient);

                        var gradients = GetGradients();
                        var parameters = GetParameters();
                        if (parameters.Length > 0)
                        {
                            var updatedParameters = _optimizer.UpdateParameters(parameters, gradients);
                            UpdateParameters(updatedParameters);
                        }

                        ClearGradients();
                    }

                    T avgLoss = inputFunctions.Length > 0
                        ? NumOps.Divide(totalLoss, NumOps.FromDouble(inputFunctions.Length))
                        : NumOps.Zero;

                    history.AddEpoch(avgLoss);
                    if (epoch % 10 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {avgLoss}");
                    }
                }
            }
            finally
            {
                foreach (var layer in _fourierLayers)
                {
                    layer.SetTrainingMode(false);
                }
                foreach (var layer in Layers)
                {
                    layer.SetTrainingMode(false);
                }
                SetTrainingMode(false);
            }

            return history;
        }

        private T ComputeAverageLoss(Tensor<T>[] inputFunctions, Tensor<T>[] outputFunctions)
        {
            T totalLoss = NumOps.Zero;

            for (int i = 0; i < inputFunctions.Length; i++)
            {
                Tensor<T> prediction = Forward(inputFunctions[i]);
                T loss = ComputeMSE(prediction, outputFunctions[i]);
                totalLoss = NumOps.Add(totalLoss, loss);
            }

            return inputFunctions.Length > 0
                ? NumOps.Divide(totalLoss, NumOps.FromDouble(inputFunctions.Length))
                : NumOps.Zero;
        }

        private T ComputeMSE(Tensor<T> prediction, Tensor<T> target)
        {
            if (!prediction.Shape.SequenceEqual(target.Shape))
            {
                throw new ArgumentException("Prediction and target shapes must match.");
            }

            T sumSquaredError = NumOps.Zero;
            int count = prediction.Length;

            for (int i = 0; i < prediction.Length; i++)
            {
                T error = NumOps.Subtract(prediction[i], target[i]);
                sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
            }

            return count > 0
                ? NumOps.Divide(sumSquaredError, NumOps.FromDouble(count))
                : NumOps.Zero;
        }

        private void ValidateInputShape(Tensor<T> input)
        {
            int expectedRank = _spatialDimensions.Length + 2;
            if (input.Rank != expectedRank)
            {
                throw new ArgumentException(
                    $"Expected input rank {expectedRank} [batch, channels, spatial...], got {input.Rank}.");
            }

            if (input.Shape[1] != Architecture.InputSize)
            {
                throw new ArgumentException(
                    $"Expected input channels {Architecture.InputSize}, got {input.Shape[1]}.");
            }
        }

        private Tensor<T> ApplyPointwiseDense(Tensor<T> input, NeuralNetworks.Layers.DenseLayer<T> layer)
        {
            int batchSize = input.Shape[0];
            int[] spatialShape = input.Shape.Skip(2).ToArray();
            var flattened = FlattenPointwiseInput(input, spatialShape);
            var projected = layer.Forward(flattened);
            return UnflattenPointwiseOutput(projected, batchSize, spatialShape);
        }

        private Tensor<T> FlattenPointwiseInput(Tensor<T> input, int[] spatialShape)
        {
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);
            int[] spatialStrides = ComputeStrides(spatialShape);

            var flattened = new Tensor<T>(new int[] { batchSize * spatialSize, channels });
            var inputIndices = new int[input.Rank];

            for (int b = 0; b < batchSize; b++)
            {
                inputIndices[0] = b;
                for (int s = 0; s < spatialSize; s++)
                {
                    FillSpatialIndices(s, spatialShape, spatialStrides, inputIndices, 2);
                    int row = b * spatialSize + s;
                    for (int c = 0; c < channels; c++)
                    {
                        inputIndices[1] = c;
                        flattened[row, c] = input[inputIndices];
                    }
                }
            }

            return flattened;
        }

        private Tensor<T> UnflattenPointwiseOutput(Tensor<T> flattened, int batchSize, int[] spatialShape)
        {
            int channels = flattened.Shape[1];
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);
            int[] spatialStrides = ComputeStrides(spatialShape);

            int[] outputShape = new int[spatialShape.Length + 2];
            outputShape[0] = batchSize;
            outputShape[1] = channels;
            Array.Copy(spatialShape, 0, outputShape, 2, spatialShape.Length);

            var output = new Tensor<T>(outputShape);
            var outputIndices = new int[output.Rank];

            for (int b = 0; b < batchSize; b++)
            {
                outputIndices[0] = b;
                for (int s = 0; s < spatialSize; s++)
                {
                    FillSpatialIndices(s, spatialShape, spatialStrides, outputIndices, 2);
                    int row = b * spatialSize + s;
                    for (int c = 0; c < channels; c++)
                    {
                        outputIndices[1] = c;
                        output[outputIndices] = flattened[row, c];
                    }
                }
            }

            return output;
        }

        private Tensor<T> ApplyVectorActivation(Tensor<T> input, IVectorActivationFunction<T> activation)
        {
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int spatialRank = input.Rank - 2;
            int[] spatialShape = input.Shape.Skip(2).ToArray();
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);
            int[] spatialStrides = ComputeStrides(spatialShape);

            var output = new Tensor<T>(input.Shape);
            var inputIndices = new int[input.Rank];
            var outputIndices = new int[input.Rank];
            var channelVector = new Vector<T>(channels);

            for (int b = 0; b < batchSize; b++)
            {
                inputIndices[0] = b;
                outputIndices[0] = b;

                for (int s = 0; s < spatialSize; s++)
                {
                    FillSpatialIndices(s, spatialShape, spatialStrides, inputIndices, 2);
                    FillSpatialIndices(s, spatialShape, spatialStrides, outputIndices, 2);

                    for (int c = 0; c < channels; c++)
                    {
                        inputIndices[1] = c;
                        channelVector[c] = input[inputIndices];
                    }

                    var activated = ActivationHelper.ApplyActivation(activation, channelVector, AiDotNetEngine.Current);
                    for (int c = 0; c < channels; c++)
                    {
                        outputIndices[1] = c;
                        output[outputIndices] = activated[c];
                    }
                }
            }

            return output;
        }

        private static int[] ComputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            return strides;
        }

        private static void FillSpatialIndices(int linearIndex, int[] shape, int[] strides, int[] indices, int offset)
        {
            int remaining = linearIndex;
            for (int i = 0; i < shape.Length; i++)
            {
                int coord = remaining / strides[i];
                remaining %= strides[i];
                indices[offset + i] = coord;
            }
        }

        /// <summary>
        /// Makes a prediction using the FNO.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <returns>Predicted output tensor.</returns>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            bool wasTraining = IsTrainingMode;
            SetTrainingMode(false);

            try
            {
                return Forward(input);
            }
            finally
            {
                SetTrainingMode(wasTraining);
            }
        }

        /// <summary>
        /// Updates the trainable parameters from a flattened vector.
        /// </summary>
        /// <param name="parameters">Parameter vector.</param>
        public override void UpdateParameters(Vector<T> parameters)
        {
            if (parameters.Length != ParameterCount)
            {
                throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.");
            }

            int index = 0;
            foreach (var layer in Layers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    Vector<T> layerParameters = parameters.GetSubVector(index, layerParameterCount);
                    layer.UpdateParameters(layerParameters);
                    index += layerParameterCount;
                }
            }

            foreach (var layer in _fourierLayers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    Vector<T> layerParameters = parameters.GetSubVector(index, layerParameterCount);
                    layer.UpdateParameters(layerParameters);
                    index += layerParameterCount;
                }
            }
        }

        public override Tensor<T> Backpropagate(Tensor<T> outputGradients)
        {
            if (!IsTrainingMode)
            {
                throw new InvalidOperationException("Cannot backpropagate when network is not in training mode");
            }

            if (!SupportsTraining)
            {
                throw new InvalidOperationException("This network does not support backpropagation");
            }

            if (Layers.Count < 2)
            {
                throw new InvalidOperationException("FourierNeuralOperator requires lift and projection layers.");
            }

            var liftLayer = Layers[0] as NeuralNetworks.Layers.DenseLayer<T>;
            var projectLayer = Layers[Layers.Count - 1] as NeuralNetworks.Layers.DenseLayer<T>;

            if (liftLayer == null || projectLayer == null)
            {
                throw new InvalidOperationException("FourierNeuralOperator expects DenseLayer lift and projection layers.");
            }

            int batchSize = outputGradients.Shape[0];
            int[] spatialShape = outputGradients.Shape.Skip(2).ToArray();

            var flatOutputGradients = FlattenPointwiseInput(outputGradients, spatialShape);
            var projectionGradients = projectLayer.Backward(flatOutputGradients);
            var currentGradient = UnflattenPointwiseOutput(projectionGradients, batchSize, spatialShape);

            for (int i = _fourierLayers.Count - 1; i >= 0; i--)
            {
                currentGradient = _fourierLayers[i].Backward(currentGradient);
            }

            var flatLiftGradients = FlattenPointwiseInput(currentGradient, spatialShape);
            var liftGradients = liftLayer.Backward(flatLiftGradients);
            return UnflattenPointwiseOutput(liftGradients, batchSize, spatialShape);
        }

        /// <summary>
        /// Gets the trainable parameters as a flattened vector.
        /// </summary>
        public override Vector<T> GetParameters()
        {
            var parameters = new Vector<T>(ParameterCount);
            int index = 0;

            foreach (var layer in Layers)
            {
                var layerParameters = layer.GetParameters();
                for (int i = 0; i < layerParameters.Length; i++)
                {
                    parameters[index + i] = layerParameters[i];
                }

                index += layerParameters.Length;
            }

            foreach (var layer in _fourierLayers)
            {
                var layerParameters = layer.GetParameters();
                for (int i = 0; i < layerParameters.Length; i++)
                {
                    parameters[index + i] = layerParameters[i];
                }

                index += layerParameters.Length;
            }

            return parameters;
        }

        public override Vector<T> GetGradients()
        {
            var gradients = new Vector<T>(ParameterCount);
            int index = 0;

            foreach (var layer in Layers)
            {
                var layerGradients = layer.GetParameterGradients();
                for (int i = 0; i < layerGradients.Length; i++)
                {
                    gradients[index + i] = layerGradients[i];
                }

                index += layerGradients.Length;
            }

            foreach (var layer in _fourierLayers)
            {
                var layerGradients = layer.GetParameterGradients();
                for (int i = 0; i < layerGradients.Length; i++)
                {
                    gradients[index + i] = layerGradients[i];
                }

                index += layerGradients.Length;
            }

            return gradients;
        }

        private void ClearGradients()
        {
            foreach (var layer in Layers)
            {
                layer.ClearGradients();
            }

            foreach (var layer in _fourierLayers)
            {
                layer.ClearGradients();
            }
        }

        /// <summary>
        /// Gets the total parameter count for lift, Fourier, and projection layers.
        /// </summary>
        public override int ParameterCount =>
            Layers.Sum(layer => layer.ParameterCount) + _fourierLayers.Sum(layer => layer.ParameterCount);

        /// <summary>
        /// Performs a basic supervised training step using MSE loss.
        /// </summary>
        /// <param name="input">Training input tensor.</param>
        /// <param name="expectedOutput">Expected output tensor.</param>
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
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }
            foreach (var layer in _fourierLayers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                var prediction = ForwardWithMemory(input);
                if (!prediction.Shape.SequenceEqual(expectedOutput.Shape))
                {
                    throw new ArgumentException("Expected output tensor must match prediction shape.", nameof(expectedOutput));
                }

                var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
                LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

                var outputGradientVector = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
                var outputGradient = new Tensor<T>(prediction.Shape, outputGradientVector);

                Backpropagate(outputGradient);

                var gradients = GetGradients();
                var parameters = GetParameters();
                if (parameters.Length > 0)
                {
                    var updatedParameters = _optimizer.UpdateParameters(parameters, gradients);
                    UpdateParameters(updatedParameters);
                }

                ClearGradients();
            }
            finally
            {
                foreach (var layer in _fourierLayers)
                {
                    layer.SetTrainingMode(false);
                }
                foreach (var layer in Layers)
                {
                    layer.SetTrainingMode(false);
                }
                SetTrainingMode(false);
            }
        }

        /// <summary>
        /// Gets metadata about the FNO model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "Modes", _modes },
                    { "Width", _width },
                    { "FourierLayers", _fourierLayers.Count },
                    { "SpatialDimensions", _spatialDimensions },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes FNO-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_modes);
            writer.Write(_width);
            writer.Write(_fourierLayers.Count);
            writer.Write(_spatialDimensions.Length);
            for (int i = 0; i < _spatialDimensions.Length; i++)
            {
                writer.Write(_spatialDimensions[i]);
            }

            foreach (var layer in _fourierLayers)
            {
                SerializationHelper<T>.SerializeVector(writer, layer.GetParameters());
            }
        }

        /// <summary>
        /// Deserializes FNO-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedModes = reader.ReadInt32();
            int storedWidth = reader.ReadInt32();
            int storedLayerCount = reader.ReadInt32();
            int storedSpatialDims = reader.ReadInt32();

            if (storedModes != _modes || storedWidth != _width || storedLayerCount != _fourierLayers.Count)
            {
                throw new InvalidOperationException("Serialized FNO configuration does not match the current instance.");
            }

            if (storedSpatialDims != _spatialDimensions.Length)
            {
                throw new InvalidOperationException("Serialized spatial dimensions do not match the current instance.");
            }

            for (int i = 0; i < storedSpatialDims; i++)
            {
                int storedDim = reader.ReadInt32();
                if (storedDim != _spatialDimensions[i])
                {
                    throw new InvalidOperationException("Serialized spatial dimensions do not match the current instance.");
                }
            }

            foreach (var layer in _fourierLayers)
            {
                layer.SetParameters(SerializationHelper<T>.DeserializeVector(reader));
            }
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New FNO instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new FourierNeuralOperator<T>(
                Architecture,
                _modes,
                _width,
                _spatialDimensions.ToArray(),
                _fourierLayers.Count);
        }

        public override bool SupportsTraining => true;
        public override bool SupportsJitCompilation => false;
    }

    /// <summary>
    /// Represents a single Fourier layer in the FNO.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// This layer is the heart of the FNO. It performs:
    /// 1. FFT: Convert to frequency domain
    /// 2. Spectral convolution: Multiply by learned weights (in Fourier space)
    /// 3. IFFT: Convert back to physical space
    /// 4. Add local convolution (via 1x1 convolution)
    /// 5. Apply activation function
    ///
    /// Why This Works:
    /// - In Fourier space, convolution becomes multiplication (very efficient!)
    /// - We learn which frequencies are important
    /// - Captures both global (low frequency) and local (high frequency) information
    ///
    /// The spectral convolution is key: it's a global operation that couples
    /// all spatial points, allowing the network to capture long-range dependencies.
    /// </remarks>
    public class FourierLayer<T> : NeuralNetworks.Layers.LayerBase<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly INumericOperations<Complex<T>> _complexOps;
        private readonly int _width;
        private readonly int _modes;
        private readonly int[] _spatialDimensions;
        private readonly int[] _modeSizes;
        private readonly IActivationFunction<T> _activation;
        private Tensor<Complex<T>> _spectralWeights;
        private Tensor<T> _pointwiseWeights;
        private Vector<T> _pointwiseBias;
        private Tensor<T>? _lastInput;
        private Tensor<T>? _lastPreActivation;
        private Tensor<Complex<T>>? _spectralWeightsGradient;
        private Tensor<T>? _pointwiseWeightsGradient;
        private Vector<T>? _pointwiseBiasGradient;

        public FourierLayer(int width, int modes, int[] spatialDimensions, IActivationFunction<T>? activation = null)
            : base(new[] { width }, new[] { width })
        {
            if (spatialDimensions == null || spatialDimensions.Length == 0)
            {
                throw new ArgumentException("Spatial dimensions must be provided.", nameof(spatialDimensions));
            }

            _numOps = MathHelper.GetNumericOperations<T>();
            _complexOps = MathHelper.GetNumericOperations<Complex<T>>();
            _width = width;
            _modes = modes;
            _spatialDimensions = spatialDimensions.ToArray();
            _modeSizes = _spatialDimensions.Select(dim => Math.Min(_modes, dim)).ToArray();
            _activation = activation ?? new GELUActivation<T>();

            _spectralWeights = new Tensor<Complex<T>>(new[] { _width, _width }.Concat(_modeSizes).ToArray());
            _pointwiseWeights = new Tensor<T>(new[] { _width, _width });
            _pointwiseBias = new Vector<T>(_width);

            InitializeSpectralWeights();
            InitializePointwiseWeights();
        }

        /// <inheritdoc/>
        public override bool SupportsJitCompilation => false;

        public override bool SupportsTraining => true;

        public override void UpdateParameters(T learningRate)
        {
            if (_spectralWeightsGradient == null || _pointwiseWeightsGradient == null || _pointwiseBiasGradient == null)
            {
                throw new InvalidOperationException("Backward pass must be called before updating parameters.");
            }

            var lrComplex = new Complex<T>(learningRate, _numOps.Zero);
            for (int i = 0; i < _spectralWeights.Length; i++)
            {
                var update = _complexOps.Multiply(_spectralWeightsGradient[i], lrComplex);
                _spectralWeights[i] = _complexOps.Subtract(_spectralWeights[i], update);
            }

            for (int i = 0; i < _pointwiseWeights.Length; i++)
            {
                _pointwiseWeights[i] = _numOps.Subtract(
                    _pointwiseWeights[i],
                    _numOps.Multiply(_pointwiseWeightsGradient[i], learningRate));
            }

            for (int i = 0; i < _pointwiseBias.Length; i++)
            {
                _pointwiseBias[i] = _numOps.Subtract(
                    _pointwiseBias[i],
                    _numOps.Multiply(_pointwiseBiasGradient[i], learningRate));
            }
        }

        public override Tensor<T> Forward(Tensor<T> input)
        {
            int expectedRank = _spatialDimensions.Length + 2;
            if (input.Rank != expectedRank)
            {
                throw new ArgumentException(
                    $"FourierLayer expects rank {expectedRank} [batch, channels, spatial...], got {input.Rank}.");
            }

            if (input.Shape[1] != _width)
            {
                throw new ArgumentException($"Expected channel width {_width}, got {input.Shape[1]}.");
            }

            _lastInput = input;
            var spectral = ApplySpectralConvolution(input);
            var local = ApplyPointwiseMixing(input);
            var combined = AddTensors(spectral, local);

            _lastPreActivation = combined;
            return _activation.Activate(combined);
        }

        public override Tensor<T> Backward(Tensor<T> outputGradient)
        {
            if (_lastInput == null || _lastPreActivation == null)
            {
                throw new InvalidOperationException("Forward pass must be called before backward pass.");
            }

            var activationGradient = _activation.Backward(_lastPreActivation, outputGradient);
            var pointwiseInputGradient = ComputePointwiseGradients(_lastInput, activationGradient);
            var spectralInputGradient = ComputeSpectralGradients(_lastInput, activationGradient);

            return AddTensors(pointwiseInputGradient, spectralInputGradient);
        }

        private Tensor<T> ComputePointwiseGradients(Tensor<T> input, Tensor<T> activationGradient)
        {
            int batchSize = input.Shape[0];
            int spatialRank = input.Rank - 2;
            int[] spatialShape = input.Shape.Skip(2).ToArray();
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);
            int[] spatialStrides = ComputeStrides(spatialShape);

            _pointwiseWeightsGradient = new Tensor<T>(new int[] { _width, _width });
            _pointwiseWeightsGradient.Fill(_numOps.Zero);
            _pointwiseBiasGradient = new Vector<T>(_width);
            _pointwiseBiasGradient.Fill(_numOps.Zero);
            var inputGradient = new Tensor<T>(input.Shape);
            inputGradient.Fill(_numOps.Zero);

            var inputIndices = new int[input.Rank];
            var outputIndices = new int[input.Rank];

            for (int b = 0; b < batchSize; b++)
            {
                inputIndices[0] = b;
                outputIndices[0] = b;

                for (int s = 0; s < spatialSize; s++)
                {
                    FillSpatialIndices(s, spatialShape, spatialStrides, inputIndices, 2);
                    FillSpatialIndices(s, spatialShape, spatialStrides, outputIndices, 2);

                    for (int outCh = 0; outCh < _width; outCh++)
                    {
                        outputIndices[1] = outCh;
                        T gradValue = activationGradient[outputIndices];
                        _pointwiseBiasGradient[outCh] = _numOps.Add(_pointwiseBiasGradient[outCh], gradValue);

                        for (int inCh = 0; inCh < _width; inCh++)
                        {
                            inputIndices[1] = inCh;
                            _pointwiseWeightsGradient[outCh, inCh] = _numOps.Add(
                                _pointwiseWeightsGradient[outCh, inCh],
                                _numOps.Multiply(gradValue, input[inputIndices]));

                            inputGradient[inputIndices] = _numOps.Add(
                                inputGradient[inputIndices],
                                _numOps.Multiply(gradValue, _pointwiseWeights[outCh, inCh]));
                        }
                    }
                }
            }

            return inputGradient;
        }

        private Tensor<T> ComputeSpectralGradients(Tensor<T> input, Tensor<T> activationGradient)
        {
            var inputSpectrum = ForwardFFT(input);
            var gradSpectrum = ForwardFFT(activationGradient);
            var inputGradSpectrum = new Tensor<Complex<T>>(inputSpectrum.Shape);
            _spectralWeightsGradient = new Tensor<Complex<T>>(_spectralWeights.Shape);

            for (int i = 0; i < inputGradSpectrum.Length; i++)
            {
                inputGradSpectrum[i] = _complexOps.Zero;
            }

            for (int i = 0; i < _spectralWeightsGradient.Length; i++)
            {
                _spectralWeightsGradient[i] = _complexOps.Zero;
            }

            int batchSize = input.Shape[0];
            int spatialRank = _spatialDimensions.Length;
            int[] spatialShape = input.Shape.Skip(2).ToArray();
            int[][] modeIndices = BuildModeIndices(spatialShape);

            int[] freqIndices = new int[spatialRank];
            int[] spectrumIndices = new int[spatialRank + 2];
            int[] gradIndices = new int[spatialRank + 2];
            int[] weightIndices = new int[spatialRank + 2];

            for (int b = 0; b < batchSize; b++)
            {
                spectrumIndices[0] = b;
                gradIndices[0] = b;

                IterateModeIndices(modeIndices, 0, freqIndices, () =>
                {
                    for (int d = 0; d < spatialRank; d++)
                    {
                        spectrumIndices[2 + d] = freqIndices[d];
                        gradIndices[2 + d] = freqIndices[d];
                    }

                    for (int outCh = 0; outCh < _width; outCh++)
                    {
                        gradIndices[1] = outCh;
                        var gradOut = gradSpectrum[gradIndices];
                        weightIndices[0] = outCh;

                        for (int inCh = 0; inCh < _width; inCh++)
                        {
                            spectrumIndices[1] = inCh;
                            weightIndices[1] = inCh;

                            bool valid = true;
                            for (int d = 0; d < spatialRank; d++)
                            {
                                int modeIndex = MapModeIndex(freqIndices[d], spatialShape[d], _modeSizes[d]);
                                if (modeIndex < 0)
                                {
                                    valid = false;
                                    break;
                                }

                                weightIndices[2 + d] = modeIndex;
                            }

                            if (!valid)
                            {
                                continue;
                            }

                            var inputValue = inputSpectrum[spectrumIndices];
                            var weight = _spectralWeights[weightIndices];

                            _spectralWeightsGradient[weightIndices] = _complexOps.Add(
                                _spectralWeightsGradient[weightIndices],
                                _complexOps.Multiply(inputValue.Conjugate(), gradOut));

                            inputGradSpectrum[spectrumIndices] = _complexOps.Add(
                                inputGradSpectrum[spectrumIndices],
                                _complexOps.Multiply(gradOut, weight.Conjugate()));
                        }
                    }
                });
            }

            var inputGradComplex = InverseFFT(inputGradSpectrum);
            var inputGradient = new Tensor<T>(input.Shape);
            for (int i = 0; i < inputGradient.Length; i++)
            {
                inputGradient[i] = inputGradComplex[i].Real;
            }

            return inputGradient;
        }

        /// <inheritdoc/>
        public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        {
            throw new NotSupportedException(
                "FourierLayer does not support computation graph export yet for spectral convolution.");
        }

        public override Vector<T> GetParameters()
        {
            int spectralCount = _spectralWeights.Length;
            int pointwiseCount = _pointwiseWeights.Length;
            int biasCount = _pointwiseBias.Length;

            var parameters = new Vector<T>(spectralCount * 2 + pointwiseCount + biasCount);
            int index = 0;

            for (int i = 0; i < spectralCount; i++)
            {
                parameters[index++] = _spectralWeights[i].Real;
            }

            for (int i = 0; i < spectralCount; i++)
            {
                parameters[index++] = _spectralWeights[i].Imaginary;
            }

            for (int i = 0; i < pointwiseCount; i++)
            {
                parameters[index++] = _pointwiseWeights[i];
            }

            for (int i = 0; i < biasCount; i++)
            {
                parameters[index++] = _pointwiseBias[i];
            }

            return parameters;
        }

        public override void SetParameters(Vector<T> parameters)
        {
            if (parameters.Length != ParameterCount)
            {
                throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.");
            }

            int spectralCount = _spectralWeights.Length;
            int pointwiseCount = _pointwiseWeights.Length;
            int biasCount = _pointwiseBias.Length;
            int index = 0;

            var realParts = new T[spectralCount];
            for (int i = 0; i < spectralCount; i++)
            {
                realParts[i] = parameters[index++];
            }

            for (int i = 0; i < spectralCount; i++)
            {
                _spectralWeights[i] = new Complex<T>(realParts[i], parameters[index++]);
            }

            for (int i = 0; i < pointwiseCount; i++)
            {
                _pointwiseWeights[i] = parameters[index++];
            }

            for (int i = 0; i < biasCount; i++)
            {
                _pointwiseBias[i] = parameters[index++];
            }
        }

        public override Vector<T> GetParameterGradients()
        {
            if (_spectralWeightsGradient == null || _pointwiseWeightsGradient == null || _pointwiseBiasGradient == null)
            {
                return new Vector<T>(ParameterCount);
            }

            int spectralCount = _spectralWeightsGradient.Length;
            int pointwiseCount = _pointwiseWeightsGradient.Length;
            int biasCount = _pointwiseBiasGradient.Length;

            var gradients = new Vector<T>(spectralCount * 2 + pointwiseCount + biasCount);
            int index = 0;

            for (int i = 0; i < spectralCount; i++)
            {
                gradients[index++] = _spectralWeightsGradient[i].Real;
            }

            for (int i = 0; i < spectralCount; i++)
            {
                gradients[index++] = _spectralWeightsGradient[i].Imaginary;
            }

            for (int i = 0; i < pointwiseCount; i++)
            {
                gradients[index++] = _pointwiseWeightsGradient[i];
            }

            for (int i = 0; i < biasCount; i++)
            {
                gradients[index++] = _pointwiseBiasGradient[i];
            }

            return gradients;
        }

        public override void ClearGradients()
        {
            if (_spectralWeightsGradient != null)
            {
                for (int i = 0; i < _spectralWeightsGradient.Length; i++)
                {
                    _spectralWeightsGradient[i] = _complexOps.Zero;
                }
            }

            if (_pointwiseWeightsGradient != null)
            {
                _pointwiseWeightsGradient.Fill(_numOps.Zero);
            }

            if (_pointwiseBiasGradient != null)
            {
                _pointwiseBiasGradient.Fill(_numOps.Zero);
            }
        }

        public override int ParameterCount
        {
            get
            {
                int spectralCount = _spectralWeights.Length;
                int pointwiseCount = _pointwiseWeights.Length;
                int biasCount = _pointwiseBias.Length;
                return spectralCount * 2 + pointwiseCount + biasCount;
            }
        }

        public override void ResetState()
        {
            _lastInput = null;
            _lastPreActivation = null;
            _spectralWeightsGradient = null;
            _pointwiseWeightsGradient = null;
            _pointwiseBiasGradient = null;
        }

        private void InitializeSpectralWeights()
        {
            var random = RandomHelper.CreateSeededRandom(42);
            double scale = 1.0 / Math.Max(1, _width);
            T scaleValue = _numOps.FromDouble(scale);

            for (int i = 0; i < _spectralWeights.Length; i++)
            {
                T real = _numOps.Multiply(_numOps.FromDouble(random.NextDouble() * 2.0 - 1.0), scaleValue);
                T imag = _numOps.Multiply(_numOps.FromDouble(random.NextDouble() * 2.0 - 1.0), scaleValue);
                _spectralWeights[i] = new Complex<T>(real, imag);
            }
        }

        private void InitializePointwiseWeights()
        {
            var random = RandomHelper.CreateSeededRandom(1337);
            double scale = 1.0 / Math.Max(1, _width);
            T scaleValue = _numOps.FromDouble(scale);

            for (int i = 0; i < _pointwiseWeights.Length; i++)
            {
                _pointwiseWeights[i] = _numOps.Multiply(
                    _numOps.FromDouble(random.NextDouble() * 2.0 - 1.0),
                    scaleValue);
            }

            for (int i = 0; i < _pointwiseBias.Length; i++)
            {
                _pointwiseBias[i] = _numOps.Zero;
            }
        }

        private Tensor<T> ApplySpectralConvolution(Tensor<T> input)
        {
            var spectrum = ForwardFFT(input);
            var outputSpectrum = new Tensor<Complex<T>>(spectrum.Shape);

            for (int i = 0; i < outputSpectrum.Length; i++)
            {
                outputSpectrum[i] = _complexOps.Zero;
            }

            int batchSize = input.Shape[0];
            int spatialRank = _spatialDimensions.Length;
            int[] spatialShape = input.Shape.Skip(2).ToArray();
            int[][] modeIndices = BuildModeIndices(spatialShape);

            int[] freqIndices = new int[spatialRank];
            int[] spectrumIndices = new int[spatialRank + 2];
            int[] outputIndices = new int[spatialRank + 2];
            int[] weightIndices = new int[spatialRank + 2];

            for (int b = 0; b < batchSize; b++)
            {
                spectrumIndices[0] = b;
                outputIndices[0] = b;

                IterateModeIndices(modeIndices, 0, freqIndices, () =>
                {
                    for (int d = 0; d < spatialRank; d++)
                    {
                        spectrumIndices[2 + d] = freqIndices[d];
                        outputIndices[2 + d] = freqIndices[d];
                    }

                    for (int outCh = 0; outCh < _width; outCh++)
                    {
                        Complex<T> sum = _complexOps.Zero;
                        outputIndices[1] = outCh;
                        weightIndices[0] = outCh;

                        for (int inCh = 0; inCh < _width; inCh++)
                        {
                            spectrumIndices[1] = inCh;
                            weightIndices[1] = inCh;

                            for (int d = 0; d < spatialRank; d++)
                            {
                                int modeIndex = MapModeIndex(freqIndices[d], spatialShape[d], _modeSizes[d]);
                                weightIndices[2 + d] = modeIndex;
                            }

                            var weight = _spectralWeights[weightIndices];
                            var value = spectrum[spectrumIndices];
                            sum = _complexOps.Add(sum, _complexOps.Multiply(value, weight));
                        }

                        outputSpectrum[outputIndices] = sum;
                    }
                });
            }

            var spatialComplex = InverseFFT(outputSpectrum);
            var output = new Tensor<T>(input.Shape);
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = spatialComplex[i].Real;
            }

            return output;
        }

        private Tensor<T> ApplyPointwiseMixing(Tensor<T> input)
        {
            int batchSize = input.Shape[0];
            int spatialRank = input.Rank - 2;
            int[] spatialShape = input.Shape.Skip(2).ToArray();
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);
            int[] spatialStrides = ComputeStrides(spatialShape);

            var output = new Tensor<T>(input.Shape);
            var inputIndices = new int[input.Rank];
            var outputIndices = new int[input.Rank];

            for (int b = 0; b < batchSize; b++)
            {
                inputIndices[0] = b;
                outputIndices[0] = b;

                for (int s = 0; s < spatialSize; s++)
                {
                    FillSpatialIndices(s, spatialShape, spatialStrides, inputIndices, 2);
                    FillSpatialIndices(s, spatialShape, spatialStrides, outputIndices, 2);

                    for (int outCh = 0; outCh < _width; outCh++)
                    {
                        T sum = _pointwiseBias[outCh];
                        for (int inCh = 0; inCh < _width; inCh++)
                        {
                            inputIndices[1] = inCh;
                            sum = _numOps.Add(sum, _numOps.Multiply(_pointwiseWeights[outCh, inCh], input[inputIndices]));
                        }

                        outputIndices[1] = outCh;
                        output[outputIndices] = sum;
                    }
                }
            }

            return output;
        }

        private Tensor<T> AddTensors(Tensor<T> left, Tensor<T> right)
        {
            if (!left.Shape.SequenceEqual(right.Shape))
            {
                throw new ArgumentException("Tensor shapes must match for addition.");
            }

            var output = new Tensor<T>(left.Shape);
            for (int i = 0; i < left.Length; i++)
            {
                output[i] = _numOps.Add(left[i], right[i]);
            }

            return output;
        }

        private Tensor<Complex<T>> ForwardFFT(Tensor<T> input)
        {
            var complex = new Tensor<Complex<T>>(input.Shape);
            for (int i = 0; i < input.Length; i++)
            {
                complex[i] = new Complex<T>(input[i], _numOps.Zero);
            }

            ApplyFft(complex, inverse: false);
            return complex;
        }

        private Tensor<Complex<T>> InverseFFT(Tensor<Complex<T>> input)
        {
            var output = input.Clone();
            ApplyFft(output, inverse: true);
            return output;
        }

        private void ApplyFft(Tensor<Complex<T>> data, bool inverse)
        {
            for (int axis = 2; axis < data.Rank; axis++)
            {
                ApplyFftAlongAxis(data, axis, inverse);
            }
        }

        private void ApplyFftAlongAxis(Tensor<Complex<T>> data, int axis, bool inverse)
        {
            int rank = data.Rank;
            int axisSize = data.Shape[axis];
            if (axisSize <= 1)
            {
                return;
            }

            var otherDims = new int[rank - 1];
            var otherDimIndices = new int[rank - 1];
            int idx = 0;
            for (int d = 0; d < rank; d++)
            {
                if (d == axis)
                {
                    continue;
                }

                otherDims[idx] = data.Shape[d];
                otherDimIndices[idx] = d;
                idx++;
            }

            int[] otherStrides = ComputeStrides(otherDims);
            int outerSize = otherDims.Aggregate(1, (a, b) => a * b);
            var indices = new int[rank];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int remaining = outer;
                for (int i = 0; i < otherDims.Length; i++)
                {
                    int coord = remaining / otherStrides[i];
                    remaining %= otherStrides[i];
                    indices[otherDimIndices[i]] = coord;
                }

                var slice = new Vector<Complex<T>>(axisSize);
                for (int i = 0; i < axisSize; i++)
                {
                    indices[axis] = i;
                    slice[i] = data[indices];
                }

                var transformed = inverse ? InverseFft1D(slice) : ForwardFft1D(slice);
                for (int i = 0; i < axisSize; i++)
                {
                    indices[axis] = i;
                    data[indices] = transformed[i];
                }
            }
        }

        private Vector<Complex<T>> ForwardFft1D(Vector<Complex<T>> input)
        {
            if (!IsPowerOfTwo(input.Length))
            {
                return Dft(input, inverse: false);
            }

            return FFTInternal(input, inverse: false);
        }

        private Vector<Complex<T>> InverseFft1D(Vector<Complex<T>> input)
        {
            Vector<Complex<T>> output = IsPowerOfTwo(input.Length)
                ? FFTInternal(input, inverse: true)
                : Dft(input, inverse: true);

            T scale = _numOps.FromDouble(input.Length);
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = new Complex<T>(
                    _numOps.Divide(output[i].Real, scale),
                    _numOps.Divide(output[i].Imaginary, scale));
            }

            return output;
        }

        private Vector<Complex<T>> FFTInternal(Vector<Complex<T>> input, bool inverse)
        {
            int n = input.Length;
            if (n <= 1)
            {
                return input;
            }

            var even = new Vector<Complex<T>>(n / 2);
            var odd = new Vector<Complex<T>>(n / 2);

            for (int i = 0; i < n / 2; i++)
            {
                even[i] = input[2 * i];
                odd[i] = input[2 * i + 1];
            }

            even = FFTInternal(even, inverse);
            odd = FFTInternal(odd, inverse);

            var output = new Vector<Complex<T>>(n);
            T angleSign = inverse ? _numOps.One : _numOps.Negate(_numOps.One);

            for (int k = 0; k < n / 2; k++)
            {
                T angle = _numOps.Multiply(angleSign,
                    _numOps.Multiply(_numOps.FromDouble(2 * Math.PI * k), _numOps.FromDouble(1.0 / n)));
                var twiddle = Complex<T>.FromPolarCoordinates(_numOps.One, angle);
                var t = _complexOps.Multiply(twiddle, odd[k]);
                output[k] = _complexOps.Add(even[k], t);
                output[k + n / 2] = _complexOps.Subtract(even[k], t);
            }

            return output;
        }

        private Vector<Complex<T>> Dft(Vector<Complex<T>> input, bool inverse)
        {
            int n = input.Length;
            var output = new Vector<Complex<T>>(n);
            T sign = inverse ? _numOps.One : _numOps.Negate(_numOps.One);

            for (int k = 0; k < n; k++)
            {
                Complex<T> sum = _complexOps.Zero;
                for (int t = 0; t < n; t++)
                {
                    double angleValue = 2.0 * Math.PI * k * t / n;
                    T angle = _numOps.Multiply(sign, _numOps.FromDouble(angleValue));
                    var twiddle = Complex<T>.FromPolarCoordinates(_numOps.One, angle);
                    sum = _complexOps.Add(sum, _complexOps.Multiply(input[t], twiddle));
                }
                output[k] = sum;
            }

            return output;
        }

        private int[][] BuildModeIndices(int[] spatialShape)
        {
            var modeIndices = new int[spatialShape.Length][];
            for (int d = 0; d < spatialShape.Length; d++)
            {
                int modeSize = Math.Min(_modeSizes[d], spatialShape[d]);
                modeIndices[d] = BuildModeIndicesForDim(spatialShape[d], modeSize);
            }

            return modeIndices;
        }

        private static int[] BuildModeIndicesForDim(int dimSize, int modeSize)
        {
            var indices = new HashSet<int>();
            for (int i = 0; i < modeSize; i++)
            {
                indices.Add(i);
            }

            int start = dimSize - modeSize;
            if (start > 0)
            {
                for (int i = start; i < dimSize; i++)
                {
                    indices.Add(i);
                }
            }

            return indices.ToArray();
        }

        private static int MapModeIndex(int freqIndex, int dimSize, int modeSize)
        {
            if (freqIndex < modeSize)
            {
                return freqIndex;
            }

            int start = dimSize - modeSize;
            if (freqIndex >= start)
            {
                return freqIndex - start;
            }

            return -1;
        }

        private static void IterateModeIndices(int[][] modeIndices, int depth, int[] current, Action callback)
        {
            if (depth == modeIndices.Length)
            {
                callback();
                return;
            }

            foreach (int index in modeIndices[depth])
            {
                current[depth] = index;
                IterateModeIndices(modeIndices, depth + 1, current, callback);
            }
        }

        private static int[] ComputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            return strides;
        }

        private static void FillSpatialIndices(int linearIndex, int[] shape, int[] strides, int[] indices, int offset)
        {
            int remaining = linearIndex;
            for (int i = 0; i < shape.Length; i++)
            {
                int coord = remaining / strides[i];
                remaining %= strides[i];
                indices[offset + i] = coord;
            }
        }

        private static bool IsPowerOfTwo(int value)
        {
            return value > 0 && (value & (value - 1)) == 0;
        }
    }
}
