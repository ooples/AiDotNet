using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
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
    public class HamiltonianNeuralNetwork<T> : NeuralNetworkBase<T>
    {
        private readonly HamiltonianNeuralNetworkOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        private readonly int _stateDim; // Dimension of state space (q and p together)
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

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
                Layers.AddRange(LayerHelper<T>.CreateDefaultHamiltonianLayers(Architecture));
            }
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
        /// Computes the Hamiltonian (energy) for a given state.
        /// </summary>
        /// <param name="state">State vector [q₁, ..., qₙ, p₁, ..., pₙ].</param>
        /// <returns>Hamiltonian value (energy).</returns>
        public T ComputeHamiltonian(T[] state)
        {
            ValidateState(state, nameof(state));

            var inputTensor = new Tensor<T>(new int[] { 1, state.Length });
            for (int i = 0; i < state.Length; i++)
            {
                inputTensor[0, i] = state[i];
            }

            var output = Forward(inputTensor);
            return output[0, 0];
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
        public T[] ComputeTimeDerivative(T[] state)
        {
            ValidateState(state, nameof(state));
            int n = _stateDim / 2; // Half are positions, half are momenta

            // Compute gradient of H with respect to state
            T[,] gradient = ComputeHamiltonianGradient(state);

            T[] derivative = new T[_stateDim];

            // Hamilton's equations:
            // dq/dt = ∂H/∂p
            // dp/dt = -∂H/∂q

            for (int i = 0; i < n; i++)
            {
                derivative[i] = gradient[0, n + i];      // dq_i/dt = ∂H/∂p_i
                derivative[n + i] = NumOps.Negate(gradient[0, i]);     // dp_i/dt = -∂H/∂q_i
            }

            return derivative;
        }

        /// <summary>
        /// Simulates the system forward in time.
        /// </summary>
        /// <param name="initialState">Initial state [q₀, p₀].</param>
        /// <param name="dt">Time step.</param>
        /// <param name="numSteps">Number of time steps.</param>
        /// <returns>Trajectory [numSteps + 1, stateDim].</returns>
        public T[,] Simulate(T[] initialState, T dt, int numSteps)
        {
            ValidateState(initialState, nameof(initialState));
            if (numSteps <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numSteps), "Number of steps must be positive.");
            }

            T[,] trajectory = new T[numSteps + 1, _stateDim];

            // Set initial state
            for (int i = 0; i < _stateDim; i++)
            {
                trajectory[0, i] = initialState[i];
            }

            // Integrate using symplectic Euler (preserves Hamiltonian structure)
            for (int step = 0; step < numSteps; step++)
            {
                T[] currentState = new T[_stateDim];
                for (int i = 0; i < _stateDim; i++)
                {
                    currentState[i] = trajectory[step, i];
                }

                int n = _stateDim / 2;

                // Step 1: update momentum using current position
                T[] derivative = ComputeTimeDerivative(currentState);
                for (int i = n; i < _stateDim; i++)
                {
                    currentState[i] = NumOps.Add(currentState[i], NumOps.Multiply(dt, derivative[i]));
                }

                // Step 2: update position using updated momentum
                derivative = ComputeTimeDerivative(currentState);
                for (int i = 0; i < n; i++)
                {
                    currentState[i] = NumOps.Add(currentState[i], NumOps.Multiply(dt, derivative[i]));
                }

                for (int i = 0; i < _stateDim; i++)
                {
                    trajectory[step + 1, i] = currentState[i];
                }
            }

            return trajectory;
        }

        /// <summary>
        /// Makes a prediction using the Hamiltonian network.
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
        /// Updates the network parameters from a flattened vector.
        /// </summary>
        /// <param name="parameters">Parameter vector.</param>
        public override void UpdateParameters(Vector<T> parameters)
        {
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
        }

        /// <summary>
        /// Performs a backward pass through the network to calculate gradients.
        /// </summary>
        /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
        /// <returns>The gradient of the loss with respect to the network's input.</returns>
        public Tensor<T> Backward(Tensor<T> outputGradient)
        {
            Tensor<T> gradient = outputGradient;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                gradient = Layers[i].Backward(gradient);
            }

            return gradient;
        }

        /// <summary>
        /// Trains the Hamiltonian neural network using the provided input and expected output.
        /// </summary>
        /// <param name="input">The input tensor for training (state vectors [q, p]).</param>
        /// <param name="expectedOutput">The expected output tensor (Hamiltonian values).</param>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method trains the network to predict the correct Hamiltonian (energy)
        /// for given state vectors. The training follows the standard neural network pattern:
        /// forward pass → loss calculation → backward pass → parameter update.
        /// </para>
        /// </remarks>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            IsTrainingMode = true;

            var prediction = Forward(input);

            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
            var primaryLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            // Check for auxiliary losses from layers that support them
            T auxiliaryLoss = NumOps.Zero;
            foreach (var auxLayer in Layers.OfType<IAuxiliaryLossLayer<T>>().Where(l => l.UseAuxiliaryLoss))
            {
                var layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
                var weightedAuxLoss = NumOps.Multiply(layerAuxLoss, auxLayer.AuxiliaryLossWeight);
                auxiliaryLoss = NumOps.Add(auxiliaryLoss, weightedAuxLoss);
            }

            LastLoss = NumOps.Add(primaryLoss, auxiliaryLoss);

            var outputGradient = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = Tensor<T>.FromVector(outputGradient).Reshape(prediction.Shape);

            Backward(outputGradientTensor);
            _optimizer.UpdateParameters(Layers);

            IsTrainingMode = false;
        }

        /// <summary>
        /// Gets metadata about the Hamiltonian network.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "StateDimension", _stateDim },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes Hamiltonian-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_stateDim);
        }

        /// <summary>
        /// Deserializes Hamiltonian-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedStateDim = reader.ReadInt32();
            if (storedStateDim != _stateDim)
            {
                throw new InvalidOperationException("Serialized Hamiltonian configuration does not match the current instance.");
            }
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New Hamiltonian network instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new HamiltonianNeuralNetwork<T>(Architecture, _stateDim, _optimizer);
        }

        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;

        private void ValidateState(T[] state, string paramName)
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

        private T[,] ComputeHamiltonianGradient(T[] state)
        {
            try
            {
                return NeuralNetworkDerivatives<T>.ComputeGradient(this, state, 1);
            }
            catch (InvalidOperationException)
            {
                return ComputeGradientFiniteDifference(state);
            }
            catch (NotSupportedException)
            {
                return ComputeGradientFiniteDifference(state);
            }
        }

        private T[,] ComputeGradientFiniteDifference(T[] state)
        {
            T epsilon = NumOps.FromDouble(1e-5);
            T twoEpsilon = NumOps.FromDouble(2e-5);
            var gradient = new T[1, _stateDim];
            var perturbed = new T[state.Length];
            Array.Copy(state, perturbed, state.Length);

            for (int i = 0; i < _stateDim; i++)
            {
                T original = perturbed[i];
                perturbed[i] = NumOps.Add(original, epsilon);
                T plus = ComputeHamiltonian(perturbed);
                perturbed[i] = NumOps.Subtract(original, epsilon);
                T minus = ComputeHamiltonian(perturbed);
                gradient[0, i] = NumOps.Divide(NumOps.Subtract(plus, minus), twoEpsilon);
                perturbed[i] = original;
            }

            return gradient;
        }
    }
}



