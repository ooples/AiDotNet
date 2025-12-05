using System;
using System.Numerics;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.PhysicsInformed.ScientificML
{
    /// <summary>
    /// Implements Universal Differential Equations (UDEs) - ODEs with neural network components.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Universal Differential Equations combine known physics with machine learning.
    ///
    /// Traditional ODEs:
    /// dx/dt = f(x, t, θ) where f is a known function with parameters θ
    /// Example: dx/dt = -kx (exponential decay, k is known)
    ///
    /// Pure Neural ODEs:
    /// dx/dt = NN(x, t, θ) where NN is a neural network
    /// - Very flexible, can learn any dynamics
    /// - But ignores known physics
    /// - May violate physical laws
    ///
    /// Universal Differential Equations (UDEs):
    /// dx/dt = f_known(x, t) + NN(x, t, θ)
    /// - Combines known physics (f_known) with learned corrections (NN)
    /// - Best of both worlds!
    ///
    /// Key Idea:
    /// Use neural networks to model UNKNOWN parts of the physics while keeping
    /// KNOWN parts as explicit equations.
    ///
    /// Example - Epidemic Model:
    /// Known: dS/dt = -βSI, dI/dt = βSI - γI (basic SIR model)
    /// Unknown: How β (infection rate) varies with temperature, policy, etc.
    /// UDE: dS/dt = -β(T, P)SI where β(T, P) = NN(temperature, policy)
    ///
    /// Applications:
    /// - Climate modeling (known physics + unknown feedback loops)
    /// - Epidemiology (known disease spread + unknown interventions)
    /// - Chemistry (known reactions + unknown catalysis effects)
    /// - Biology (known population dynamics + unknown environmental factors)
    /// - Engineering (known mechanics + unknown friction/damping)
    /// </remarks>
    public class UniversalDifferentialEquation<T> : NeuralNetworkBase<T> where T : struct, INumber<T>
    {
        private readonly Func<T[], T, T[]> _knownDynamics;
        private readonly int _stateDim;

        public UniversalDifferentialEquation(
            NeuralNetworkArchitecture<T> architecture,
            int stateDim,
            Func<T[], T, T[]>? knownDynamics = null)
            : base(architecture, null, 1.0)
        {
            _stateDim = stateDim;
            _knownDynamics = knownDynamics ?? ((x, t) => new T[stateDim]); // Default to zero
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
                var layerSizes = Architecture.HiddenLayerSizes ?? new int[] { 32, 32 };

                for (int i = 0; i < layerSizes.Length; i++)
                {
                    int inputSize = i == 0 ? _stateDim + 1 : layerSizes[i - 1]; // +1 for time
                    int outputSize = layerSizes[i];

                    var denseLayer = new NeuralNetworks.Layers.DenseLayer<T>(
                        inputSize,
                        outputSize,
                        Enums.ActivationFunctionType.Tanh);

                    Layers.Add(denseLayer);
                }

                var outputLayer = new NeuralNetworks.Layers.DenseLayer<T>(
                    layerSizes[layerSizes.Length - 1],
                    _stateDim,
                    Enums.ActivationFunctionType.Linear);

                Layers.Add(outputLayer);
            }
        }

        /// <summary>
        /// Computes dx/dt = f_known(x, t) + NN(x, t).
        /// </summary>
        public T[] ComputeDerivative(T[] state, T time)
        {
            // Known physics component
            T[] knownPart = _knownDynamics(state, time);

            // Neural network component (learned unknown physics)
            T[] input = new T[_stateDim + 1];
            Array.Copy(state, input, _stateDim);
            input[_stateDim] = time;

            var inputTensor = new Tensor<T>(new int[] { 1, input.Length });
            for (int i = 0; i < input.Length; i++)
            {
                inputTensor[0, i] = input[i];
            }

            var output = Forward(inputTensor);
            T[] learnedPart = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                learnedPart[i] = output[0, i];
            }

            // Combine: total derivative = known + learned
            T[] totalDerivative = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                totalDerivative[i] = knownPart[i] + learnedPart[i];
            }

            return totalDerivative;
        }

        /// <summary>
        /// Simulates the UDE forward in time.
        /// </summary>
        public T[,] Simulate(T[] initialState, T tStart, T tEnd, int numSteps)
        {
            T dt = (tEnd - tStart) / T.CreateChecked(numSteps);
            T[,] trajectory = new T[numSteps + 1, _stateDim];

            for (int i = 0; i < _stateDim; i++)
            {
                trajectory[0, i] = initialState[i];
            }

            for (int step = 0; step < numSteps; step++)
            {
                T[] currentState = new T[_stateDim];
                for (int i = 0; i < _stateDim; i++)
                {
                    currentState[i] = trajectory[step, i];
                }

                T currentTime = tStart + T.CreateChecked(step) * dt;
                T[] derivative = ComputeDerivative(currentState, currentTime);

                for (int i = 0; i < _stateDim; i++)
                {
                    trajectory[step + 1, i] = currentState[i] + dt * derivative[i];
                }
            }

            return trajectory;
        }
    }
}
