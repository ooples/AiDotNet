using AiDotNet.LearningRateSchedulers;
using System;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;

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
    /// <example>
    /// <code>
    /// // Each row is one sampled function: the operator learns a mapping between whole
    /// // functions, not between individual points.
    /// var inputFunctions = Tensor&lt;float&gt;.CreateRandom(8, 64);
    /// var outputFunctions = Tensor&lt;float&gt;.CreateRandom(8, 64);
    ///
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new FourierNeuralOperator&lt;float&gt;())
    ///     .Build(inputFunctions, outputFunctions);
    ///
    /// var newInputFunction = Tensor&lt;float&gt;.CreateRandom(1, 64);
    /// Tensor&lt;float&gt; prediction = result.Predict(newInputFunction);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.VeryHigh)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Fourier Neural Operator for Parametric Partial Differential Equations", "https://doi.org/10.48550/arXiv.2010.08895", Year = 2021, Authors = "Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001, DecayRate = 0.5,
                    StepSize = 100, Schedule = LearningRateSchedulerType.Step,
                    ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                    Source = "Li et al. 2021, Sec. 5: the Adam optimizer trained for 500 epochs with an "
                            + "initial learning rate of 0.001 that is halved every 100 epochs. The step size "
                            + "is in epochs, which the step mode records.")]
    public partial class FourierNeuralOperator<T> : NeuralNetworkBase<T>
    {
        private readonly FourierNeuralOperatorOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        private readonly int _modes; // Number of Fourier modes to keep
        private readonly int _width; // Channel width of the network
        private readonly int[] _spatialDimensions;
        private readonly List<FourierLayer<T>> _fourierLayers;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        /// <summary>
        /// Initializes a new instance of the Fourier Neural Operator with default configuration.
        /// </summary>
        public FourierNeuralOperator()
            : this(new NeuralNetworkArchitecture<T>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 64,
                outputSize: 64))
        {
        }

        /// <summary>
        /// Initializes a new instance of the Fourier Neural Operator.
        /// </summary>
        /// <param name="architecture">The network architecture.</param>
        /// <param name="spatialDimensions">Dimensions of the input function domain (e.g., [64, 64] for 64x64 grid).</param>
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
        public FourierNeuralOperator(NeuralNetworkArchitecture<T> architecture,
            int[]? spatialDimensions = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            FourierNeuralOperatorOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            options ??= new FourierNeuralOperatorOptions();
            _options = options;
            Options = _options;

            _modes = options.Modes;
            _width = options.Width;
            _spatialDimensions = spatialDimensions ?? new int[] { 64, 64 }; // Default 2D
            _fourierLayers = new List<FourierLayer<T>>();
            _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            InitializeLayers();
            InitializeFourierLayers(options.NumLayers);
        }

        protected override void InitializeLayers()
        {
            // Lifting layer: embed input to higher dimension
            var liftLayer = new NeuralNetworks.Layers.DenseLayer<T>(
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
                // Also register in base Layers so TrainWithTape discovers their parameters
                Layers.Add(fourierLayer);
            }

            // Projection layer: map back to output dimension
            var projectLayer = new NeuralNetworks.Layers.DenseLayer<T>(
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

        /// <summary>
        /// Forward pass used by tape-based training. Must go through the same pointwise-reshape
        /// path as ForwardInternal so that the gradient tape records correct operations on the
        /// lift, Fourier, and projection layers.
        /// </summary>
        public override Tensor<T> ForwardForTraining(Tensor<T> input)
        {
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
                        Train(inputFunctions[i], outputFunctions[i]);
                        totalLoss = NumOps.Add(totalLoss, LastLoss ?? NumOps.Zero);
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
            if (!prediction._shape.SequenceEqual(target._shape))
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
            int[] spatialShape = input._shape.Skip(2).ToArray();
            var flattened = FlattenPointwiseInput(input, spatialShape);
            var projected = layer.Forward(flattened);
            return UnflattenPointwiseOutput(projected, batchSize, spatialShape);
        }

        /// <summary>
        /// Flattens a channel-first tensor <c>[B, C, d_1, ..., d_N]</c> into the
        /// <c>[B * d_1 * ... * d_N, C]</c> row-major layout that the pointwise
        /// DenseLayer consumes. Implemented as <c>TensorPermute + Reshape</c> so
        /// every op records on the gradient tape — the previous element-by-element
        /// copy loop bypassed the tape and blocked tape-based training for the FNO.
        /// </summary>
        private Tensor<T> FlattenPointwiseInput(Tensor<T> input, int[] spatialShape)
        {
            int rank = input.Rank;
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);

            // Permute [B, C, d_1, ..., d_N] → [B, d_1, ..., d_N, C].
            // axes: [0, 2, 3, ..., rank-1, 1]
            int[] perm = new int[rank];
            perm[0] = 0;
            for (int d = 0; d < rank - 2; d++) perm[1 + d] = 2 + d;
            perm[rank - 1] = 1;
            var permuted = Engine.TensorPermute(input, perm);

            return Engine.Reshape(permuted, new[] { batchSize * spatialSize, channels });
        }

        /// <summary>
        /// Inverse of <see cref="FlattenPointwiseInput"/>. Reshapes the row-major
        /// <c>[B * spatialSize, C]</c> tensor back to <c>[B, C, d_1, ..., d_N]</c>
        /// using <c>Reshape + TensorPermute</c> so the op chain stays on the
        /// gradient tape.
        /// </summary>
        private Tensor<T> UnflattenPointwiseOutput(Tensor<T> flattened, int batchSize, int[] spatialShape)
        {
            int channels = flattened.Shape[1];
            int spatialRank = spatialShape.Length;

            // Reshape [B * spatialSize, C] → [B, d_1, ..., d_N, C].
            int[] unflattenShape = new int[spatialRank + 2];
            unflattenShape[0] = batchSize;
            for (int d = 0; d < spatialRank; d++) unflattenShape[1 + d] = spatialShape[d];
            unflattenShape[spatialRank + 1] = channels;
            var reshaped = Engine.Reshape(flattened, unflattenShape);

            // Permute [B, d_1, ..., d_N, C] → [B, C, d_1, ..., d_N].
            // axes: [0, spatialRank+1, 1, 2, ..., spatialRank]
            int[] perm = new int[spatialRank + 2];
            perm[0] = 0;
            perm[1] = spatialRank + 1;
            for (int d = 0; d < spatialRank; d++) perm[2 + d] = 1 + d;
            return Engine.TensorPermute(reshaped, perm);
        }

        /// <summary>
        /// Makes a prediction using the FNO.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <returns>Predicted output tensor.</returns>
        protected override Tensor<T> PredictCore(Tensor<T> input)
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

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
        // GetParameters was overridden here to walk Layers and copy each layer's vector into a flat
        // buffer sized from ParameterCount. That is the base implementation, restated. The lift,
        // Fourier and projection layers are all registered into Layers at construction, and the
        // spectral and pointwise weights belong to FourierLayer, which registers them itself -- so
        // there was never anything here the base walk could not reach.

        public override Vector<T> GetGradients()
        {
            var gradients = new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
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

            return gradients;
        }

        private void ClearGradients()
        {
            foreach (var layer in Layers)
            {
                layer.ClearGradients();
            }
        }

        // ParameterCount was overridden here as Layers.Sum(layer => layer.ParameterCount) -- exactly
        // what the base already computes. Deleted along with the matching GetParameters above.

        /// <inheritdoc/>
        /// <remarks>The mean squared error of the lifted, Fourier-propagated, projected field (Li et al. 2021).</remarks>
        protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
            => TapeTrainStep(input, expectedOutput);

        /// <summary>
        /// Tape-based FNO training step. Runs the full lift → FourierLayers →
        /// project pipeline under a <see cref="GradientTape{T}"/>, computes MSE
        /// loss, walks the recorded ops to compute parameter gradients, and
        /// applies an SGD update in place.
        /// </summary>
        /// <remarks>
        /// The stale comment on the old ManualTrainStep said "the FNO's custom
        /// spatial reshape breaks the gradient tape, so we use explicit
        /// backpropagation" — that was true before <see cref="FlattenPointwiseInput"/>
        /// / <see cref="UnflattenPointwiseOutput"/> were rewritten as
        /// <c>TensorPermute + Reshape</c> and before <see cref="FourierLayer"/>'s
        /// spectral conv moved to <c>Engine.FFT2D</c> / <c>Engine.FFT</c>. With
        /// both of those in place every op in the forward pass records on the
        /// tape, so backward just falls out of <c>tape.ComputeGradients</c>.
        /// </remarks>
        // Built on the first step and kept, so its state (none for plain SGD, but a schedule's position) persists.
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainingOptimizer;

        private IReadOnlyList<Training.TrainingObjective<T>> TapeTrainStep(Tensor<T> input, Tensor<T> expectedOutput)
        {
            ValidateInputShape(input);

            var liftLayer = Layers[0] as NeuralNetworks.Layers.DenseLayer<T>;
            var projectLayer = Layers[Layers.Count - 1] as NeuralNetworks.Layers.DenseLayer<T>;

            if (liftLayer == null || projectLayer == null)
            {
                throw new InvalidOperationException("FNO requires DenseLayer lift and projection layers.");
            }

            // Lift -> Fourier stack -> project with an MSE objective, as one step on the shared fused session: the
            // compiled plan when it applies (CPU or GPU), the eager tape otherwise. Plain SGD at 0.001, the update
            // this step has always applied. _fourierLayers are registered in Layers, so collecting Layers alone
            // trains each Fourier parameter once; collecting both would apply its update twice per step.
            Tensor<T> Forward(Tensor<T> inp)
            {
                var spatial = inp._shape.Skip(2).ToArray();
                var lifted = UnflattenPointwiseOutput(liftLayer.Forward(FlattenPointwiseInput(inp, spatial)), inp.Shape[0], spatial);
                Tensor<T> y = lifted;
                foreach (var fl in _fourierLayers) y = fl.Forward(y);
                return UnflattenPointwiseOutput(projectLayer.Forward(FlattenPointwiseInput(y, spatial)), inp.Shape[0], spatial);
            }

            Tensor<T> MeanSquaredError(Tensor<T> pred, Tensor<T> tgt)
            {
                var diff = Engine.TensorSubtract(pred, tgt);
                return Engine.TensorMultiplyScalar(
                    Engine.ReduceSum(Engine.TensorMultiply(diff, diff), null),
                    NumOps.Divide(NumOps.One, NumOps.FromDouble(pred.Length)));
            }

            _trainingOptimizer ??= new StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(
                this, new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 0.001 });
            return new[]
            {
                new Training.TrainingObjective<T>(
                    new[] { input, expectedOutput },
                    data => MeanSquaredError(Forward(data[0]), data[1]))
                {
                    Optimizer = _trainingOptimizer,
                    Parameters = () => Training.TapeTrainingStep<T>.CollectParameters(Layers),
                },
            };
        }

        /// <summary>
        /// Gets metadata about the FNO model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "Modes", _modes },
                    { "Width", _width },
                    { "FourierLayers", _fourierLayers.Count },
                    { "SpatialDimensions", _spatialDimensions },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes FNO-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes FNO-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        public override bool SupportsTraining => true;
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
    // SHAPE-PRESERVING, taken from ForwardTraced rather than from the FFT's reputation. The spectral
    // branch truncates to _modeSizes in frequency space, but must come back full-size: the very next
    // line is `Engine.TensorAdd(spectral, local)` against the pointwise branch, which is input-sized,
    // and the activation after it is elementwise. The channel axis survives because _pointwiseWeights
    // is [_width, _width] and ForwardTraced rejects any input whose Shape[1] is not _width.
    //
    // Three ranks are declared because the RANK is configuration here - ForwardTraced computes
    // `expectedRank = _spatialDimensions.Length + 2` and throws on anything else - so the type accepts
    // rank 3, 4 or 5 for 1-D, 2-D and 3-D problems respectively. OutputAxesFor below narrows that back
    // to the single rank THIS instance accepts; the attributes cannot, being per-type.
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Width,
        Direction = TensorLayoutDirection.Input,
        Note = "1-D problem: [batch, channels, x].")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Width,
        Direction = TensorLayoutDirection.Output)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Input,
        Note = "2-D problem: the fused Engine.FFT2D fast path.")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Output)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Input,
        Note = "3-D problem: the separable 1-D Engine.FFT loop.")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Output)]
    [AutoParameters]
    public partial class FourierLayer<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
    {
        /// <inheritdoc />
        /// <remarks>
        /// <para>
        /// Every relation is <c>Same</c>, so this looks like a case the generator could have covered
        /// from the layouts alone. It is written by hand for the RANK GUARD, not for the relations: a
        /// generated body would answer for all three declared ranks, while a given instance accepts
        /// exactly one - <c>_spatialDimensions.Length + 2</c>, which is the condition ForwardTraced
        /// throws on. Answering for a rank this instance rejects is the same defect as declaring one.
        /// </para>
        /// </remarks>
        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        {
            if (inputRank != _spatialDimensions.Length + 2) return null;

            // Spatial axes, outermost first, matching the layouts above: 1-D names its single axis
            // Width, 2-D adds Height ahead of it, 3-D adds Depth ahead of that.
            TensorAxis[] spatial = _spatialDimensions.Length switch
            {
                1 => new[] { TensorAxis.Width },
                2 => new[] { TensorAxis.Height, TensorAxis.Width },
                3 => new[] { TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width },
                // Higher-dimensional problems run fine through the separable FFT loop, but this
                // contract has no roles left to name their axes with. Declining is honest.
                _ => Array.Empty<TensorAxis>(),
            };

            if (spatial.Length == 0) return null;

            var axes = new List<OutputAxisContract>
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
            };

            foreach (var axis in spatial)
            {
                axes.Add(new OutputAxisContract(axis, AxisRelation.Same(axis)));
            }

            return axes;
        }

        private readonly INumericOperations<T> _numOps;
        private readonly int _width;
        private readonly int _modes;
        private readonly int[] _spatialDimensions;
        private readonly int[] _modeSizes;
        private readonly IActivationFunction<T> _activation;

        // Split-complex spectral weights stored as two real tensors so they participate directly in the gradient
        // tape alongside the DFT-matmul spectral transform. Registered as trainable parameters below in the
        // constructor.
        [TrainableParameter(Role = PersistentTensorRole.Weights)]
        private Tensor<T> _spectralWeightsReal;

        [TrainableParameter(Role = PersistentTensorRole.Weights)]
        private Tensor<T> _spectralWeightsImag;

        [TrainableParameter(Role = PersistentTensorRole.Weights)]
        private Tensor<T> _pointwiseWeights;

        [TrainableParameter(Role = PersistentTensorRole.Biases)]
        private Tensor<T> _pointwiseBias;

        public FourierLayer(int width, int modes, int[] spatialDimensions, IActivationFunction<T>? activation = null)
            : base(new[] { width }, new[] { width })
        {
            if (spatialDimensions == null || spatialDimensions.Length == 0)
            {
                throw new ArgumentException("Spatial dimensions must be provided.", nameof(spatialDimensions));
            }

            _numOps = MathHelper.GetNumericOperations<T>();
            _width = width;
            _modes = modes;
            _spatialDimensions = spatialDimensions.ToArray();
            _modeSizes = _spatialDimensions.Select(dim => Math.Min(_modes, dim)).ToArray();
            _activation = activation ?? new GELUActivation<T>();

            var spectralShape = new[] { _width, _width }.Concat(_modeSizes).ToArray();
            _spectralWeightsReal = new Tensor<T>(spectralShape);
            _spectralWeightsImag = new Tensor<T>(spectralShape);
            _pointwiseWeights = new Tensor<T>(new[] { _width, _width });
            _pointwiseBias = new Tensor<T>(new[] { _width });

            InitializeSpectralWeights();
            InitializePointwiseWeights();

            RegisterTrainableParameter(_spectralWeightsReal, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_spectralWeightsImag, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_pointwiseWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_pointwiseBias, PersistentTensorRole.Biases);
        }

        public override bool SupportsTraining => true;

        /// <summary>
        /// Legacy scalar-learning-rate parameter update. The tape training path
        /// (FNO.TapeTrainStep → tape.ComputeGradients → in-place SGD on the
        /// tensor spans) bypasses this method entirely, and the hand-rolled
        /// backward that used to populate the private <c>_*Gradient</c> fields
        /// was removed along with those fields. Kept as a no-op so the abstract
        /// base contract is still satisfied.
        /// </summary>
        public override void UpdateParameters(T learningRate)
        {
            // No-op: tape-based training updates weights via SetParameters on
            // the split-complex _spectralWeightsReal / _spectralWeightsImag
            // tensors (and _pointwiseWeights / _pointwiseBias) directly.
        }

        protected override Tensor<T> ForwardTraced(Tensor<T> input)
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

            // Spectral convolution through dense DFT matmuls (see ApplySeparableFft), the same for every spatial
            // rank, so the spectral weights are on the gradient tape and in a compiled plan.
            var spectral = ApplySpectralConvolutionNDTape(input);
            var local = ApplyPointwiseMixing(input);
            var combined = Engine.TensorAdd(spectral, local);

            return _activation.Activate(combined);
        }

        public override void ClearGradients()
        {
            // No-op: see GetParameterGradients — no persistent gradient buffers.
        }

        public override void ResetState()
        {
            // No cached state — tape owns the gradient graph for the current forward.
        }

        private void InitializeSpectralWeights()
        {
            var random = RandomHelper.CreateSeededRandom(42);
            double scale = 1.0 / Math.Max(1, _width);
            T scaleValue = _numOps.FromDouble(scale);

            var realSpan = _spectralWeightsReal.Data.Span;
            var imagSpan = _spectralWeightsImag.Data.Span;
            for (int i = 0; i < realSpan.Length; i++)
            {
                realSpan[i] = _numOps.Multiply(_numOps.FromDouble(random.NextDouble() * 2.0 - 1.0), scaleValue);
                imagSpan[i] = _numOps.Multiply(_numOps.FromDouble(random.NextDouble() * 2.0 - 1.0), scaleValue);
            }
        }

        private void InitializePointwiseWeights()
        {
            var random = RandomHelper.CreateSeededRandom(1337);
            double scale = 1.0 / Math.Max(1, _width);
            T scaleValue = _numOps.FromDouble(scale);

            var weightsSpan = _pointwiseWeights.Data.Span;
            for (int i = 0; i < weightsSpan.Length; i++)
            {
                weightsSpan[i] = _numOps.Multiply(
                    _numOps.FromDouble(random.NextDouble() * 2.0 - 1.0),
                    scaleValue);
            }

            var biasSpan = _pointwiseBias.Data.Span;
            for (int i = 0; i < biasSpan.Length; i++)
            {
                biasSpan[i] = _numOps.Zero;
            }
        }


        /// <summary>
        /// Applies compact
        /// per-frequency weights <c>[C_out, C_in, m_1, ..., m_N]</c> to an input
        /// block <c>[B, C_in, m_1, ..., m_N]</c> producing <c>[B, C_out, m_1, ..., m_N]</c>
        /// by treating the <c>N</c> spatial location axes as a flattened batch for
        /// <c>TensorBatchMatMul</c>.
        /// </summary>
        private Tensor<T> PerLocationMatMulND(
            Tensor<T> input, Tensor<T> weights,
            int batchSize, int inChannels, int outChannels,
            int[] modeShape)
        {
            int nSpatial = modeShape.Length;
            int rank = nSpatial + 2;
            int modeTotal = 1;
            for (int d = 0; d < nSpatial; d++) modeTotal *= modeShape[d];

            // Permute input [B, C_in, m_1, ..., m_N] → [m_1, ..., m_N, B, C_in]
            // axes: [2, 3, ..., rank-1, 0, 1]
            int[] inputPerm = new int[rank];
            for (int d = 0; d < nSpatial; d++) inputPerm[d] = 2 + d;
            inputPerm[nSpatial] = 0;
            inputPerm[nSpatial + 1] = 1;
            var inputPermuted = Engine.TensorPermute(input, inputPerm);
            var inputBatched = Engine.Reshape(inputPermuted, new[] { modeTotal, batchSize, inChannels });

            // Permute weights [C_out, C_in, m_1, ..., m_N] → [m_1, ..., m_N, C_in, C_out]
            // axes: [2, 3, ..., rank-1, 1, 0]
            int[] weightPerm = new int[rank];
            for (int d = 0; d < nSpatial; d++) weightPerm[d] = 2 + d;
            weightPerm[nSpatial] = 1;
            weightPerm[nSpatial + 1] = 0;
            var weightsPermuted = Engine.TensorPermute(weights, weightPerm);
            var weightsBatched = Engine.Reshape(weightsPermuted, new[] { modeTotal, inChannels, outChannels });

            // Batched matmul: [modeTotal, B, C_in] @ [modeTotal, C_in, C_out] → [modeTotal, B, C_out]
            var resultBatched = Engine.TensorBatchMatMul(inputBatched, weightsBatched);

            // Reshape [modeTotal, B, C_out] → [m_1, ..., m_N, B, C_out]
            int[] unbatchedShape = new int[rank];
            for (int d = 0; d < nSpatial; d++) unbatchedShape[d] = modeShape[d];
            unbatchedShape[nSpatial] = batchSize;
            unbatchedShape[nSpatial + 1] = outChannels;
            var resultUnbatched = Engine.Reshape(resultBatched, unbatchedShape);

            // Permute back: [m_1, ..., m_N, B, C_out] → [B, C_out, m_1, ..., m_N]
            // axes: [rank-2, rank-1, 0, 1, ..., nSpatial-1]
            int[] outPerm = new int[rank];
            outPerm[0] = nSpatial;
            outPerm[1] = nSpatial + 1;
            for (int d = 0; d < nSpatial; d++) outPerm[2 + d] = d;
            return Engine.TensorPermute(resultUnbatched, outPerm);
        }

        /// <summary>
        /// The N-D forward or inverse discrete Fourier transform over the spatial axes, as one dense DFT matmul per
        /// axis. The inverse carries the 1/N factor, as <c>Engine.IFFT</c> does.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The split-complex <c>Engine.FFT</c> / <c>Engine.IFFT</c> / <c>Engine.FFT2D</c> primitives record nothing
        /// on the gradient tape (their out-parameter outputs cannot carry a node), so a spectral convolution built
        /// on them gave its spectral weights no gradient at all: FNO trained only its lift, pointwise and
        /// projection weights. A DFT along an axis is a matmul with a fixed basis, so writing it as one is
        /// differentiable everywhere, replays in a compiled plan, and is exact at every length. (<c>Engine.FFT</c>
        /// zero-pads a length that is not a power of two to the next one, a different transform.)
        /// </para>
        /// <para>
        /// Along an axis of length N, with C[n, k] = cos(2 pi n k / N) and S[n, k] = sin(2 pi n k / N):
        /// forward (re + i im)(C - iS) = (re C + im S) + i(im C - re S);
        /// inverse (re + i im)(C + iS) / N = (re C - im S) / N + i(im C + re S) / N.
        /// </para>
        /// </remarks>
        internal (Tensor<T> real, Tensor<T> imag) ApplySeparableFft(
            Tensor<T> inputReal, Tensor<T> inputImag, bool inverse)
        {
            int rank = inputReal.Rank;
            Tensor<T> re = inputReal;
            Tensor<T> im = inputImag;

            for (int axis = 2; axis < rank; axis++)
            {
                // Move xis to the last position, keeping the other axes in order.
                int[] perm = new int[rank];
                int[] invPerm = new int[rank];
                int idx = 0;
                for (int d = 0; d < rank; d++)
                {
                    if (d != axis)
                    {
                        perm[idx++] = d;
                    }
                }
                perm[rank - 1] = axis;
                for (int d = 0; d < rank; d++)
                {
                    invPerm[perm[d]] = d;
                }

                var rePerm = Engine.TensorPermute(re, perm);
                var imPerm = Engine.TensorPermute(im, perm);
                int n = rePerm.Shape[rank - 1];
                int rows = rePerm.Length / n;
                var permShape = rePerm._shape;
                var reFlat = Engine.Reshape(rePerm, new[] { rows, n });
                var imFlat = Engine.Reshape(imPerm, new[] { rows, n });
                var (cos, sin) = DftBasis(n);

                var reC = Engine.TensorMatMul(reFlat, cos);
                var imC = Engine.TensorMatMul(imFlat, cos);
                var reS = Engine.TensorMatMul(reFlat, sin);
                var imS = Engine.TensorMatMul(imFlat, sin);
                Tensor<T> newRe, newIm;
                if (inverse)
                {
                    var invN = _numOps.FromDouble(1.0 / n);
                    newRe = Engine.TensorMultiplyScalar(Engine.TensorSubtract(reC, imS), invN);
                    newIm = Engine.TensorMultiplyScalar(Engine.TensorAdd(imC, reS), invN);
                }
                else
                {
                    newRe = Engine.TensorAdd(reC, imS);
                    newIm = Engine.TensorSubtract(imC, reS);
                }

                re = Engine.TensorPermute(Engine.Reshape(newRe, permShape), invPerm);
                im = Engine.TensorPermute(Engine.Reshape(newIm, permShape), invPerm);
            }

            return (re, im);
        }

        // The cosine and sine DFT bases for a length, [N, N], built once per length.
        [Scratch]
        private readonly Dictionary<int, (Tensor<T> Cos, Tensor<T> Sin)> _dftBases = new();

        private (Tensor<T> Cos, Tensor<T> Sin) DftBasis(int n)
        {
            if (_dftBases.TryGetValue(n, out var cached))
                return cached;
            var cos = new Tensor<T>(new[] { n, n });
            var sin = new Tensor<T>(new[] { n, n });
            for (int r = 0; r < n; r++)
            {
                for (int k = 0; k < n; k++)
                {
                    // Reduce n*k mod N first so the angle stays small and exact for long axes.
                    double angle = 2.0 * Math.PI * ((long)r * k % n) / n;
                    cos[r, k] = _numOps.FromDouble(Math.Cos(angle));
                    sin[r, k] = _numOps.FromDouble(Math.Sin(angle));
                }
            }
            var basis = (cos, sin);
            _dftBases[n] = basis;
            return basis;
        }

        /// <summary>
        /// Tape-tracked spectral convolution for any spatial rank: the separable DFT of <see cref="ApplySeparableFft"/>
        /// over each spatial axis, then the <c>2^N</c> mode corners, each sliced out, multiplied by the compact
        /// complex weights (four real matmuls) and scattered back into a zero-filled output spectrum.
        /// </summary>
        private Tensor<T> ApplySpectralConvolutionNDTape(Tensor<T> input)
        {
            int rank = input.Rank;
            int nSpatial = rank - 2;
            int batchSize = input.Shape[0];
            int[] spatialShape = input._shape.Skip(2).ToArray();
            int[] modeShape = new int[nSpatial];
            for (int d = 0; d < nSpatial; d++)
            {
                modeShape[d] = Math.Min(_modeSizes[d], spatialShape[d]);
            }

            // Real input: a zero imaginary companion for the split-complex transform.
            var inputImag = new Tensor<T>(input._shape);
            inputImag.Fill(_numOps.Zero);

            var (specRe, specIm) = ApplySeparableFft(input, inputImag, inverse: false);

            var outSpecRe = new Tensor<T>(specRe._shape);
            var outSpecIm = new Tensor<T>(specIm._shape);
            outSpecRe.Fill(_numOps.Zero);
            outSpecIm.Fill(_numOps.Zero);

            // 2^nSpatial mode corners. For each corner a bit i chooses the low
            // frequencies [0..modes-1] on axis i when 0 or the high frequencies
            // [dim-modes..dim-1] when 1. This mirrors the legacy BuildModeIndices
            // behavior exactly — it keeps both positive and negative frequencies.
            int cornerCount = 1 << nSpatial;
            for (int corner = 0; corner < cornerCount; corner++)
            {
                int[] sliceStart = new int[rank];
                int[] sliceSize = new int[rank];
                sliceStart[0] = 0;
                sliceStart[1] = 0;
                sliceSize[0] = batchSize;
                sliceSize[1] = _width;

                bool validCorner = true;
                for (int d = 0; d < nSpatial; d++)
                {
                    bool useHigh = ((corner >> d) & 1) == 1;
                    int start = useHigh ? spatialShape[d] - modeShape[d] : 0;
                    if (start < 0 || start + modeShape[d] > spatialShape[d])
                    {
                        validCorner = false;
                        break;
                    }

                    sliceStart[2 + d] = start;
                    sliceSize[2 + d] = modeShape[d];
                }

                if (!validCorner) continue;

                var inBlockRe = Engine.TensorSlice(specRe, sliceStart, sliceSize);
                var inBlockIm = Engine.TensorSlice(specIm, sliceStart, sliceSize);

                var arWr = PerLocationMatMulND(inBlockRe, _spectralWeightsReal, batchSize, _width, _width, modeShape);
                var aiWi = PerLocationMatMulND(inBlockIm, _spectralWeightsImag, batchSize, _width, _width, modeShape);
                var arWi = PerLocationMatMulND(inBlockRe, _spectralWeightsImag, batchSize, _width, _width, modeShape);
                var aiWr = PerLocationMatMulND(inBlockIm, _spectralWeightsReal, batchSize, _width, _width, modeShape);

                var outBlockRe = Engine.TensorSubtract(arWr, aiWi);
                var outBlockIm = Engine.TensorAdd(arWi, aiWr);

                outSpecRe = Engine.TensorSetSlice(outSpecRe, outBlockRe, sliceStart);
                outSpecIm = Engine.TensorSetSlice(outSpecIm, outBlockIm, sliceStart);
            }

            var (spatialRe, _) = ApplySeparableFft(outSpecRe, outSpecIm, inverse: true);
            return spatialRe;
        }

        /// <summary>
        /// Tape-tracked 1×1 pointwise mixing across the channel axis.
        /// Implements <c>output[b, oc, ...spatial] = bias[oc] + sum_ic(W[oc, ic] * input[b, ic, ...spatial])</c>
        /// as permute → reshape → MatMul(W^T) → broadcast-add(bias) → reshape → permute,
        /// so every op records on the gradient tape. Previous implementation was an
        /// element-by-element triple loop that allocated a detached output tensor and
        /// broke the gradient tape connection to the pointwise weights and bias.
        /// </summary>
        private Tensor<T> ApplyPointwiseMixing(Tensor<T> input)
        {
            int rank = input.Rank;
            int batchSize = input.Shape[0];
            int[] spatialShape = input._shape.Skip(2).ToArray();
            int spatialSize = spatialShape.Aggregate(1, (a, b) => a * b);
            int flatRows = batchSize * spatialSize;

            // [B, C_in, d_1, ..., d_N] → [B, d_1, ..., d_N, C_in]
            // perm axes: [0, 2, 3, ..., rank-1, 1]
            int[] toFlatPerm = new int[rank];
            toFlatPerm[0] = 0;
            for (int d = 0; d < rank - 2; d++) toFlatPerm[1 + d] = 2 + d;
            toFlatPerm[rank - 1] = 1;
            var permuted = Engine.TensorPermute(input, toFlatPerm);

            // [B, d_1, ..., d_N, C_in] → [B * spatialSize, C_in]
            var flat = Engine.Reshape(permuted, new[] { flatRows, _width });

            // Weights stored as [C_out, C_in] — need [C_in, C_out] for the matmul.
            var weightsT = Engine.TensorPermute(_pointwiseWeights, new[] { 1, 0 });

            // [B*S, C_in] @ [C_in, C_out] → [B*S, C_out]
            var matmul = Engine.TensorMatMul(flat, weightsT);

            // Bias shape [C_out] broadcasts across the B*S rows.
            var biased = Engine.TensorAdd(matmul, _pointwiseBias);

            // [B*S, C_out] → [B, d_1, ..., d_N, C_out]
            int[] unflatShape = new int[rank];
            unflatShape[0] = batchSize;
            for (int d = 0; d < rank - 2; d++) unflatShape[1 + d] = spatialShape[d];
            unflatShape[rank - 1] = _width;
            var unflat = Engine.Reshape(biased, unflatShape);

            // [B, d_1, ..., d_N, C_out] → [B, C_out, d_1, ..., d_N]
            // perm axes: [0, rank-1, 1, 2, ..., rank-2]
            int[] fromFlatPerm = new int[rank];
            fromFlatPerm[0] = 0;
            fromFlatPerm[1] = rank - 1;
            for (int d = 0; d < rank - 2; d++) fromFlatPerm[2 + d] = 1 + d;
            return Engine.TensorPermute(unflat, fromFlatPerm);
        }
    }
}
