using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.PhysicsInformed.Options;

namespace AiDotNet.PhysicsInformed.ScientificML
{
    /// <summary>
    /// Implements Hamiltonian Neural Networks (HNN) for learning conservative dynamical systems.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Hamiltonian Neural Networks learn the laws of physics by respecting conservation principles.
    ///
    /// Classical Mechanics - Hamiltonian Formulation:
    /// Many physical systems are described by Hamilton's equations:
    /// - dq/dt = ∂H/∂p  (position changes with momentum gradient)
    /// - dp/dt = -∂H/∂q (momentum changes with negative position gradient)
    ///
    /// Where:
    /// - q = position coordinates
    /// - p = momentum coordinates
    /// - H(q,p) = Hamiltonian (total energy of the system)
    ///
    /// Key Property: Energy Conservation
    /// For conservative systems, H(q,p) = constant (energy is conserved)
    ///
    /// Traditional Neural Networks vs. HNN:
    ///
    /// Traditional NN:
    /// - Learn dynamics directly: (q,p) → (dq/dt, dp/dt)
    /// - Can violate physics laws
    /// - May not conserve energy
    /// - Can accumulate errors over time
    ///
    /// HNN:
    /// - Learn the Hamiltonian: (q,p) → H
    /// - Compute dynamics from H: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    /// - Automatically conserves energy (by construction!)
    /// - More accurate long-term predictions
    ///
    /// How It Works:
    /// 1. Neural network learns H(q,p)
    /// 2. Use automatic differentiation to get ∂H/∂q and ∂H/∂p
    /// 3. Apply Hamilton's equations to get dynamics
    /// 4. Guaranteed to preserve Hamiltonian structure
    ///
    /// Applications:
    /// - Planetary motion (gravitational systems)
    /// - Molecular dynamics (particle interactions)
    /// - Robotics (mechanical systems)
    /// - Quantum mechanics (Schrödinger equation)
    /// - Any conservative system
    ///
    /// Example - Pendulum:
    /// H(q,p) = p²/(2m) + mgl(1 - cos(q))
    /// - q = angle, p = angular momentum
    /// - HNN learns this from data without knowing the formula!
    ///
    /// Key Benefit:
    /// By encoding physics structure (Hamiltonian formulation), the network
    /// learns faster, generalizes better, and makes physically consistent predictions.
    /// </remarks>
    /// <example>
    /// <code>
    /// var hnn = new HamiltonianNeuralNetwork&lt;float&gt;();
    /// float[] derivatives = hnn.ComputeTimeDerivative(new float[] { 1.0f, 0.0f, 0.0f, 1.0f });
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Hamiltonian Neural Networks", "https://doi.org/10.48550/arXiv.1906.01563", Year = 2019, Authors = "Sam Greydanus, Misko Dzamba, Jason Yosinski")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    public partial class HamiltonianNeuralNetwork<T> : NeuralNetworkBase<T>, ITrainingObjectiveProvider<T>
    {
        private readonly HamiltonianNeuralNetworkOptions _options;

        private readonly int _stateDim; // Dimension of state space (q and p together)
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    private const int DefaultStateDim = 2;

    public HamiltonianNeuralNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            // The network input IS the full phase-space state vector (q, p), so
            // its size must equal stateDim — the ctor enforces
            // CalculatedInputSize == stateDim. The previous `DefaultStateDim * 2`
            // made the input 4 while stateDim stayed 2, so the parameterless ctor
            // always threw "input size (4) must match state dimension (2)".
            // DefaultStateDim = 2 is the canonical 1-DOF Hamiltonian phase space
            // (one position q + one momentum p).
            inputSize: DefaultStateDim,
            outputSize: 1),
            stateDim: DefaultStateDim)
    {
    }

    public HamiltonianNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            int stateDim,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            HamiltonianNeuralNetworkOptions? options = null)
            : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
                NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _options = options ?? new HamiltonianNeuralNetworkOptions();
            Options = _options;
            _options.Validate();

            if (stateDim <= 0 || stateDim % 2 != 0)
            {
                throw new ArgumentException("stateDim must be a positive even number (equal parts q and p).", nameof(stateDim));
            }

            if (Architecture.OutputSize != 1)
            {
                throw new ArgumentException("Hamiltonian networks require a scalar output (OutputSize = 1).", nameof(architecture));
            }

            if (Architecture.CalculatedInputSize != stateDim)
            {
                throw new ArgumentException(
                    $"Hamiltonian network input size ({Architecture.CalculatedInputSize}) must match state dimension ({stateDim}).",
                    nameof(architecture));
            }

            _stateDim = stateDim;
            _optimizer = optimizer ?? CreateFixedAdamOptimizer(
                _options.LearningRate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                l2RegularizationStrength: _options.WeightDecay);
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
                // Chain-resolve lazy layers from the architecture's declared
                // input shape. The tape derivative of H
                // walks the layer stack, which
                // reads `weights.Shape[0]` to size its per-layer activation
                // buffers — on a lazy DenseLayer that read returns 0 and the
                // derivative computation throws "Layer input size mismatch.
                // Expected 0, got <stateDim>" on the very first probe (this
                // is the HamiltonianNN_ComputeTimeDerivative /
                // HamiltonianNN_Simulate failure in the Integration P-Q
                // shard). The else-branch's CreateDefaultHamiltonianLayers
                // already chain-resolves via LayerHelper's private
                // ChainResolveLazyLayers; we inline the same walk here so
                // user-supplied layer stacks also report concrete weight
                // shapes before any Forward call.
                var rootShape = Architecture.GetInputShape();
                if (rootShape != null && rootShape.Length > 0
                    && System.Linq.Enumerable.All(rootShape, d => d > 0))
                {
                    int[] running = rootShape;
                    foreach (var layer in Layers)
                    {
                        if (layer is AiDotNet.NeuralNetworks.Layers.LayerBase<T> lb
                            && !lb.IsShapeResolved)
                        {
                            try { lb.ResolveFromShape(running); }
                            catch (Exception ex)
                            {
                                // Don't silently swallow: a failure here means the shape chain is
                                // wrong, which would surface as an opaque error at first Forward.
                                System.Diagnostics.Trace.TraceWarning(
                                    "HamiltonianNeuralNetwork: failed pre-resolving {0} with shape [{1}] - {2}",
                                    lb.GetType().Name, string.Join(",", running), ex.Message);
                                throw;
                            }
                        }
                        running = layer.GetOutputShape();
                    }
                }
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultHamiltonianLayers(
                    Architecture,
                    _options.HiddenLayerCount,
                    _options.HiddenDimension));
            }
        }

        /// <inheritdoc />
        public TrainingObjectiveKind TrainingObjectiveKind => TrainingObjectiveKind.HamiltonianDynamics;

        /// <inheritdoc />
        Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(
            Tensor<T> input,
            Tensor<T> proposedTarget)
        {
            if (!input.Shape.ToArray().SequenceEqual(proposedTarget.Shape.ToArray()))
            {
                throw new ArgumentException(
                    $"Hamiltonian dynamics targets must match the phase-space input shape " +
                    $"[{string.Join(",", input.Shape)}]; got [{string.Join(",", proposedTarget.Shape)}].",
                    nameof(proposedTarget));
            }

            return proposedTarget;
        }

        /// <inheritdoc />
        T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
            => EvaluateTrainingObjective(input, target);

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
        /// Produces the symplectic time-derivative field optimized by the cited HNN paper.
        /// </summary>
        /// <remarks>
        /// The public inference output remains the scalar Hamiltonian. Training instead compares
        /// (dH/dp, -dH/dq) with observed state derivatives, as Equation 3 requires. The input gradient
        /// comes from the gradient tape, so the field stays differentiable in the weights.
        /// </remarks>
        public override Tensor<T> ForwardForTraining(Tensor<T> input)
        {
            if (input.Rank < 1 || input.Shape[^1] != _stateDim)
            {
                throw new ArgumentException(
                    $"The final input dimension must equal the phase-space dimension {_stateDim}; " +
                    $"got [{string.Join(",", input.Shape)}].",
                    nameof(input));
            }

            var rows = Engine.Reshape(input, new[] { input.Length / _stateDim, _stateDim });
            return Engine.Reshape(SymplecticGradient(rows), input.Shape.ToArray());
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The training output differentiates H with respect to its inputs inside the step, a nested backward the
        /// compile-once plan cannot replay; the eager tape re-runs it every step.
        /// </remarks>
        protected override bool SupportsFusedCompiledTraining => false;

        /// <summary>
        /// The symplectic gradient (dH/dp, -dH/dq) [N, stateDim] of the learned Hamiltonian at states [N, stateDim].
        /// </summary>
        /// <remarks>
        /// Greydanus et al. (2019), Eq. 2-3, take dH/dq and dH/dp by automatic differentiation of the network. The
        /// gradient comes from the tape with createGraph, so a loss on this field reaches the weights. It was a
        /// central difference of two perturbed forwards, which carries an O(h^2) error and a cancellation error
        /// that grows as the step shrinks.
        /// </remarks>
        private Tensor<T> SymplecticGradient(Tensor<T> states)
        {
            PinnAutodiff<T>.Evaluate(Forward, states, 1, out var derivatives);
            var first = derivatives.FirstDerivatives
                ?? throw new InvalidOperationException("The tape produced no input gradient.");
            int count = states.Shape[0];
            int n = _stateDim / 2;
            var gradient = Engine.Reshape(first, new[] { count, _stateDim });
            var dHdq = Engine.TensorSlice(gradient, new[] { 0, 0 }, new[] { count, n });
            var dHdp = Engine.TensorSlice(gradient, new[] { 0, n }, new[] { count, n });
            return Engine.TensorConcatenate(new[] { dHdp, Engine.TensorNegate(dHdq) }, 1);
        }

        /// <summary>
        /// Computes the Hamiltonian (energy) for a given state.
        /// </summary>
        /// <param name="state">State vector [q₁, ..., qₙ, p₁, ..., pₙ].</param>
        /// <returns>Hamiltonian value (energy).</returns>
        public T ComputeHamiltonian(Vector<T> state)
        {
            ValidateState(state, nameof(state));
            return Forward(AsRow(state)).AsSpan()[0];
        }

        /// <summary>
        /// Computes the time derivative of the state using Hamilton's equations.
        /// </summary>
        /// <param name="state">Current state [q, p].</param>
        /// <returns>Time derivative [dq/dt, dp/dt].</returns>
        /// <remarks>
        /// For Beginners:
        /// This is where the physics happens!
        /// Instead of predicting the derivative directly, we:
        /// 1. Compute H(q,p) using the network
        /// 2. Compute ∂H/∂p (derivative of H with respect to momentum)
        /// 3. Compute ∂H/∂q (derivative of H with respect to position)
        /// 4. Apply Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        ///
        /// This guarantees that energy is conserved!
        /// </remarks>
        public Vector<T> ComputeTimeDerivative(Vector<T> state)
        {
            ValidateState(state, nameof(state));
            return SymplecticGradient(AsRow(state)).ToVector();
        }

        /// <summary>
        /// Simulates the system forward in time.
        /// </summary>
        /// <param name="initialState">Initial state [q₀, p₀].</param>
        /// <param name="dt">Time step.</param>
        /// <param name="numSteps">Number of time steps.</param>
        /// <returns>Trajectory [numSteps + 1, stateDim].</returns>
        /// <remarks>
        /// Symplectic Euler: p is advanced with -dH/dq at (q, p), then q with dH/dp at (q, p_new), which keeps the
        /// flow symplectic.
        /// </remarks>
        public Tensor<T> Simulate(Vector<T> initialState, T dt, int numSteps)
        {
            ValidateState(initialState, nameof(initialState));
            if (numSteps <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numSteps), "Number of steps must be positive.");
            }

            int n = _stateDim / 2;
            var current = AsRow(initialState);
            var trajectory = new List<Tensor<T>>(numSteps + 1) { current };
            for (int step = 0; step < numSteps; step++)
            {
                var q = Engine.TensorSlice(current, new[] { 0, 0 }, new[] { 1, n });
                var p = Engine.TensorSlice(current, new[] { 0, n }, new[] { 1, n });
                var momentumRate = Engine.TensorSlice(SymplecticGradient(current), new[] { 0, n }, new[] { 1, n });
                p = Engine.TensorAdd(p, Engine.TensorMultiplyScalar(momentumRate, dt));

                var half = Engine.TensorConcatenate(new[] { q, p }, 1);
                var positionRate = Engine.TensorSlice(SymplecticGradient(half), new[] { 0, 0 }, new[] { 1, n });
                q = Engine.TensorAdd(q, Engine.TensorMultiplyScalar(positionRate, dt));

                current = Engine.TensorConcatenate(new[] { q, p }, 1);
                trajectory.Add(current);
            }

            return Engine.TensorConcatenate(trajectory.ToArray(), 0);
        }

        private Tensor<T> AsRow(Vector<T> state)
        {
            return Engine.Reshape(Tensor<T>.FromVector(state), new[] { 1, state.Length });
        }

        /// <summary>
        /// Makes a prediction using the Hamiltonian network.
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
        /// Gets metadata about the Hamiltonian network.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "StateDimension", _stateDim },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes Hamiltonian-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes Hamiltonian-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;

        private void ValidateState(Vector<T> state, string paramName)
        {
            if (state == null)
            {
                throw new ArgumentNullException(paramName);
            }

            if (state.Length != _stateDim)
            {
                throw new ArgumentException($"State length must be {_stateDim}.", paramName);
            }
        }
    }
}
