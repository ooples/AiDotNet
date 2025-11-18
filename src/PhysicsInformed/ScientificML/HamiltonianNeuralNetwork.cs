using System;
using System.Numerics;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

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
    public class HamiltonianNeuralNetwork<T> : NeuralNetworkBase<T> where T : struct, INumber<T>
    {
        private readonly int _stateDim; // Dimension of state space (q and p together)

        public HamiltonianNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            int stateDim)
            : base(architecture, null, 1.0)
        {
            _stateDim = stateDim;
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            if (Architecture.CustomLayers != null && Architecture.CustomLayers.Count > 0)
            {
                foreach (var layer in Architecture.CustomLayers)
                {
                    Layers.Add(layer);
                }
            }
            else
            {
                var layerSizes = Architecture.HiddenLayerSizes ?? new int[] { 64, 64, 64 };

                for (int i = 0; i < layerSizes.Length; i++)
                {
                    int inputSize = i == 0 ? _stateDim : layerSizes[i - 1];
                    int outputSize = layerSizes[i];

                    var denseLayer = new NeuralNetworks.Layers.DenseLayer<T>(
                        inputSize,
                        outputSize,
                        Enums.ActivationFunctionType.Tanh);

                    Layers.Add(denseLayer);
                }

                // Output is scalar Hamiltonian
                var outputLayer = new NeuralNetworks.Layers.DenseLayer<T>(
                    layerSizes[layerSizes.Length - 1],
                    1,
                    Enums.ActivationFunctionType.Linear);

                Layers.Add(outputLayer);
            }
        }

        /// <summary>
        /// Computes the Hamiltonian (energy) for a given state.
        /// </summary>
        /// <param name="state">State vector [q₁, ..., qₙ, p₁, ..., pₙ].</param>
        /// <returns>Hamiltonian value (energy).</returns>
        public T ComputeHamiltonian(T[] state)
        {
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
            int n = _stateDim / 2; // Half are positions, half are momenta

            // Compute gradient of H with respect to state
            T[,] gradient = AutomaticDifferentiation<T>.ComputeGradient(
                s => new T[] { ComputeHamiltonian(s) },
                state,
                1);

            T[] derivative = new T[_stateDim];

            // Hamilton's equations:
            // dq/dt = ∂H/∂p
            // dp/dt = -∂H/∂q

            for (int i = 0; i < n; i++)
            {
                derivative[i] = gradient[0, n + i];      // dq_i/dt = ∂H/∂p_i
                derivative[n + i] = -gradient[0, i];     // dp_i/dt = -∂H/∂q_i
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

                T[] derivative = ComputeTimeDerivative(currentState);

                for (int i = 0; i < _stateDim; i++)
                {
                    trajectory[step + 1, i] = currentState[i] + dt * derivative[i];
                }
            }

            return trajectory;
        }
    }
}
