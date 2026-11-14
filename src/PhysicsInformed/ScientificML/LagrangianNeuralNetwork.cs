using AiDotNet.LearningRateSchedulers;
using System;
using System.IO;
using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.PhysicsInformed.Options;

namespace AiDotNet.PhysicsInformed.ScientificML
{
    /// <summary>
    /// Implements Lagrangian Neural Networks (LNN) for learning mechanical systems.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Lagrangian Neural Networks learn physics using the Lagrangian formulation of mechanics.
    ///
    /// Lagrangian Mechanics:
    /// Alternative to Hamiltonian mechanics, uses the Lagrangian:
    /// L(q, q̇) = T - V = Kinetic Energy - Potential Energy
    ///
    /// Equations of Motion (Euler-Lagrange):
    /// d/dt(∂L/∂q̇) - ∂L/∂q = 0
    ///
    /// Where:
    /// - q = generalized coordinates (positions)
    /// - q̇ = generalized velocities
    /// - T = kinetic energy (usually ½m q̇²)
    /// - V = potential energy (depends on q)
    ///
    /// Why Lagrangian vs. Hamiltonian?
    /// - Lagrangian: Uses (q, q̇) - position and velocity
    /// - Hamiltonian: Uses (q, p) - position and momentum
    /// - Lagrangian often more intuitive for mechanical systems
    /// - Both give same physics, different formulations
    ///
    /// How LNN Works:
    /// 1. Neural network learns L(q, q̇)
    /// 2. Compute ∂L/∂q and ∂L/∂q̇ using automatic differentiation
    /// 3. Apply Euler-Lagrange equation to get acceleration q̈
    /// 4. Guaranteed to conserve energy and satisfy principle of least action
    ///
    /// Applications:
    /// - Robotics (manipulator dynamics)
    /// - Biomechanics (human motion)
    /// - Aerospace (satellite dynamics)
    /// - Any mechanical system
    /// </remarks>
    /// <example>
    /// <code>
    /// var lnn = new LagrangianNeuralNetwork&lt;float&gt;();
    /// float[] acceleration = lnn.ComputeAcceleration(
    ///     new float[] { 1.0f, 0.5f }, new float[] { 0.1f, -0.2f });
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Lagrangian Neural Networks", "https://doi.org/10.48550/arXiv.2003.04630", Year = 2020, Authors = "Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, Shirley Ho")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    [PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 0.001,
                    Source = "Cranmer et al. 2020, Sec. 4: a decaying learning rate starting at 1e-3. No "
                            + "decay interval or endpoint is given, so no schedule is declared, and the "
                            + "optimizer is left unspecified because the paper names none -- its single "
                            + "occurrence of Adam is the acknowledgment of Adam Burrows.")]
    public partial class LagrangianNeuralNetwork<T> : NeuralNetworkBase<T>, ITrainingObjectiveProvider<T>
    {
        private readonly LagrangianNeuralNetworkOptions _options;

        private readonly int _configurationDim; // Dimension of configuration space (q)
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        /// <summary>
        /// Creates a default LagrangianNeuralNetwork with a 2D configuration space.
        /// </summary>
        public LagrangianNeuralNetwork()
            : this(new NeuralNetworkArchitecture<T>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4,
                outputSize: 1), configurationDim: 2) { }

        public LagrangianNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            int configurationDim,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            LagrangianNeuralNetworkOptions? options = null)
            : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
                NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _options = options ?? new LagrangianNeuralNetworkOptions();
            Options = _options;

            if (configurationDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(configurationDim), "Configuration dimension must be positive.");
            }

            if (Architecture.OutputSize != 1)
            {
                throw new ArgumentException("Lagrangian networks require a scalar output (OutputSize = 1).", nameof(architecture));
            }

            int expectedInputSize = 2 * configurationDim;
            if (Architecture.CalculatedInputSize != expectedInputSize)
            {
                throw new ArgumentException(
                    $"Lagrangian network input size ({Architecture.CalculatedInputSize}) must be {expectedInputSize}.",
                    nameof(architecture));
            }

            _configurationDim = configurationDim;
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultLagrangianLayers(Architecture));
            }
        }


        /// <summary>
        /// Computes the Lagrangian L(q, q̇) = T - V.
        /// </summary>
        public T ComputeLagrangian(Vector<T> q, Vector<T> qDot)
        {
            return Forward(StateRow(q, qDot)).AsSpan()[0];
        }

        /// <summary>
        /// Computes acceleration using Euler-Lagrange equation.
        /// </summary>
        public Vector<T> ComputeAcceleration(Vector<T> q, Vector<T> qDot)
        {
            return Accelerations(StateRow(q, qDot)).ToVector();
        }

        /// <inheritdoc />
        public TrainingObjectiveKind TrainingObjectiveKind => TrainingObjectiveKind.LagrangianDynamics;

        /// <inheritdoc />
        Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        {
            int count = input.Length / (2 * _configurationDim);
            if (proposedTarget.Length != count * _configurationDim)
            {
                throw new ArgumentException(
                    $"Lagrangian dynamics targets are the accelerations [batch, {_configurationDim}]; " +
                    $"got [{string.Join(",", proposedTarget.Shape)}].",
                    nameof(proposedTarget));
            }

            return Engine.Reshape(proposedTarget, new[] { count, _configurationDim });
        }

        /// <inheritdoc />
        T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
            => EvaluateTrainingObjective(input, target);

        /// <summary>
        /// The accelerations q̈ [N, n] the learned Lagrangian implies at states [q, q̇] (Cranmer et al. 2020, Eq. 6).
        /// </summary>
        /// <remarks>
        /// <para>
        /// The Euler-Lagrange equation gives q̈ = (∇q̇∇q̇ᵀL)⁻¹ [∇qL - (∇q∇q̇ᵀL) q̇]. The gradient and Hessian of L come
        /// from the gradient tape, so the accelerations stay differentiable in the weights and training fits them
        /// to observed accelerations, the paper's objective.
        /// </para>
        /// <para>
        /// The old code asked for first derivatives only, so the Hessian was missing and every call fell back to
        /// finite differences; it added a fixed 1e-6 to the mass matrix and returned zero acceleration when the
        /// solve failed.
        /// </para>
        /// </remarks>
        private Tensor<T> Accelerations(Tensor<T> states)
        {
            int count = states.Shape[0];
            int n = _configurationDim;
            PinnAutodiff<T>.Evaluate(Forward, states, 2, out var derivatives);
            var first = derivatives.FirstDerivatives
                ?? throw new InvalidOperationException("The tape produced no input gradient.");
            var second = derivatives.SecondDerivatives
                ?? throw new InvalidOperationException("The tape produced no input Hessian.");

            var gradient = Engine.Reshape(first, new[] { count, 2 * n });
            var hessian = Engine.Reshape(second, new[] { count, 2 * n, 2 * n });
            var dLdq = Engine.TensorSlice(gradient, new[] { 0, 0 }, new[] { count, n });
            var mass = Engine.TensorSlice(hessian, new[] { 0, n, n }, new[] { count, n, n });
            var cross = Engine.TensorSlice(hessian, new[] { 0, n, 0 }, new[] { count, n, n });
            var qDot = Engine.Reshape(Engine.TensorSlice(states, new[] { 0, n }, new[] { count, n }), new[] { count, n, 1 });
            var coriolis = Engine.Reshape(Engine.TensorBatchMatMul(cross, qDot), new[] { count, n });
            return SolveBatched(mass, Engine.TensorSubtract(dLdq, coriolis));
        }

        /// <summary>
        /// Solves M x = b for a batch of systems M [N, n, n], b [N, n] by Gauss-Jordan elimination in engine
        /// operations, so the solution stays on the gradient tape.
        /// </summary>
        /// <remarks>
        /// No pivoting: the mass matrix ∇q̇∇q̇ᵀL of a physical Lagrangian is positive definite, so every pivot is
        /// positive. A singular mass matrix gives non-finite accelerations rather than a silent zero.
        /// </remarks>
        private Tensor<T> SolveBatched(Tensor<T> matrix, Tensor<T> rhs)
        {
            int count = matrix.Shape[0];
            int n = matrix.Shape[1];
            var rows = new Tensor<T>[n];
            for (int i = 0; i < n; i++)
            {
                var row = Engine.Reshape(Engine.TensorSlice(matrix, new[] { 0, i, 0 }, new[] { count, 1, n }), new[] { count, n });
                var value = Engine.TensorSlice(rhs, new[] { 0, i }, new[] { count, 1 });
                rows[i] = Engine.TensorConcatenate(new[] { row, value }, 1);
            }

            for (int k = 0; k < n; k++)
            {
                var pivot = Engine.TensorSlice(rows[k], new[] { 0, k }, new[] { count, 1 });
                rows[k] = Engine.TensorDivide(rows[k], pivot);
                for (int i = 0; i < n; i++)
                {
                    if (i != k)
                    {
                        var factor = Engine.TensorSlice(rows[i], new[] { 0, k }, new[] { count, 1 });
                        rows[i] = Engine.TensorSubtract(rows[i], Engine.TensorMultiply(factor, rows[k]));
                    }
                }
            }

            var solution = new Tensor<T>[n];
            for (int i = 0; i < n; i++)
            {
                solution[i] = Engine.TensorSlice(rows[i], new[] { 0, n }, new[] { count, 1 });
            }

            return Engine.TensorConcatenate(solution, 1);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The accelerations the Lagrangian implies, the quantity Cranmer et al. (2020) fit to data. The public
        /// inference output remains the scalar Lagrangian.
        /// </remarks>
        public override Tensor<T> ForwardForTraining(Tensor<T> input)
        {
            int width = 2 * _configurationDim;
            if (input.Rank < 1 || input.Shape[^1] != width)
            {
                throw new ArgumentException(
                    $"The final input dimension must equal the state dimension {width}; got [{string.Join(",", input.Shape)}].",
                    nameof(input));
            }

            return Accelerations(Engine.Reshape(input, new[] { input.Length / width, width }));
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The accelerations differentiate L twice with respect to its inputs inside the step, a nested backward
        /// the compile-once plan cannot replay; the eager tape re-runs it every step.
        /// </remarks>
        protected override bool SupportsFusedCompiledTraining => false;

        private Tensor<T> StateRow(Vector<T> q, Vector<T> qDot)
        {
            ValidateStateVectors(q, qDot);
            var state = Vector<T>.Concatenate(q, qDot);
            return Engine.Reshape(Tensor<T>.FromVector(state), new[] { 1, state.Length });
        }

        /// <summary>
        /// Performs a forward pass through the network.
        /// </summary>
        /// <param name="input">Input tensor for evaluation.</param>
        /// <returns>Network output tensor.</returns>
        public Tensor<T> Forward(Tensor<T> input)
        {
            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        /// <summary>
        /// Makes a prediction using the Lagrangian network.
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
        /// <summary>
        /// Gets metadata about the Lagrangian network.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "ConfigurationDimension", _configurationDim },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes Lagrangian-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes Lagrangian-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;

        private void ValidateStateVectors(Vector<T> q, Vector<T> qDot)
        {
            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }

            if (qDot == null)
            {
                throw new ArgumentNullException(nameof(qDot));
            }

            if (q.Length != _configurationDim || qDot.Length != _configurationDim)
            {
                throw new ArgumentException("q and qDot must match the configuration dimension.");
            }
        }
    }
}
