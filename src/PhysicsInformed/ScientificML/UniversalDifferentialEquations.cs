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
    /// Supported integration schemes for UDE simulation.
    /// </summary>
    public enum OdeIntegrationMethod
    {
        Euler,
        RungeKutta4
    }

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
    public class UniversalDifferentialEquation<T> : NeuralNetworkBase<T>
    {
        private readonly UniversalDifferentialEquationsOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        private readonly Func<T[], T, T[]> _knownDynamics;
        private readonly int _stateDim;
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        public UniversalDifferentialEquation(
            NeuralNetworkArchitecture<T> architecture,
            int stateDim,
            Func<T[], T, T[]>? knownDynamics = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            UniversalDifferentialEquationsOptions? options = null)
            : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
                NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _options = options ?? new UniversalDifferentialEquationsOptions();
            Options = _options;

            if (stateDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(stateDim), "State dimension must be positive.");
            }

            if (Architecture.OutputSize != stateDim)
            {
                throw new ArgumentException(
                    $"UDE output size ({Architecture.OutputSize}) must match state dimension ({stateDim}).",
                    nameof(architecture));
            }

            int expectedInputSize = stateDim + 1;
            if (Architecture.CalculatedInputSize != expectedInputSize)
            {
                throw new ArgumentException(
                    $"UDE input size ({Architecture.CalculatedInputSize}) must be {expectedInputSize}.",
                    nameof(architecture));
            }

            _stateDim = stateDim;
            _knownDynamics = knownDynamics ?? ((x, t) => CreateZeroState());
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultUniversalDELayers(Architecture));
            }
        }

        private T[] CreateZeroState()
        {
            var zeros = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                zeros[i] = NumOps.Zero;
            }

            return zeros;
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
        /// Performs a backward pass through the network (backpropagation).
        /// </summary>
        /// <param name="outputGradient">Gradient of the loss with respect to network output.</param>
        /// <returns>Gradient of the loss with respect to input.</returns>
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
        /// Computes dx/dt = f_known(x, t) + NN(x, t).
        /// </summary>
        public T[] ComputeDerivative(T[] state, T time)
        {
            ValidateState(state, nameof(state));

            // Known physics component
            T[] knownPart = _knownDynamics(state, time);
            if (knownPart == null || knownPart.Length != _stateDim)
            {
                throw new InvalidOperationException("Known dynamics must return a vector matching the state dimension.");
            }

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
                totalDerivative[i] = NumOps.Add(knownPart[i], learnedPart[i]);
            }

            return totalDerivative;
        }

        /// <summary>
        /// Simulates the UDE forward in time.
        /// </summary>
        public T[,] Simulate(
            T[] initialState,
            T tStart,
            T tEnd,
            int numSteps,
            OdeIntegrationMethod method = OdeIntegrationMethod.Euler)
        {
            if (initialState == null)
            {
                throw new ArgumentNullException(nameof(initialState));
            }

            if (initialState.Length != _stateDim)
            {
                throw new ArgumentException(
                    $"Initial state length ({initialState.Length}) must match state dimension ({_stateDim}).",
                    nameof(initialState));
            }

            if (numSteps <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numSteps), "Number of steps must be positive.");
            }

            T dt = NumOps.Divide(NumOps.Subtract(tEnd, tStart), NumOps.FromDouble(numSteps));
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

                T currentTime = NumOps.Add(tStart, NumOps.Multiply(NumOps.FromDouble(step), dt));
                T[] nextState = method == OdeIntegrationMethod.RungeKutta4
                    ? StepRungeKutta4(currentState, currentTime, dt)
                    : StepEuler(currentState, currentTime, dt);

                for (int i = 0; i < _stateDim; i++)
                {
                    trajectory[step + 1, i] = nextState[i];
                }
            }

            return trajectory;
        }

        private T[] StepEuler(T[] state, T time, T dt)
        {
            T[] derivative = ComputeDerivative(state, time);
            T[] nextState = new T[_stateDim];

            for (int i = 0; i < _stateDim; i++)
            {
                nextState[i] = NumOps.Add(state[i], NumOps.Multiply(dt, derivative[i]));
            }

            return nextState;
        }

        private T[] StepRungeKutta4(T[] state, T time, T dt)
        {
            T half = NumOps.Divide(dt, NumOps.FromDouble(2.0));
            T sixth = NumOps.Divide(dt, NumOps.FromDouble(6.0));

            T[] k1 = ComputeDerivative(state, time);

            T[] state2 = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                state2[i] = NumOps.Add(state[i], NumOps.Multiply(half, k1[i]));
            }

            T[] k2 = ComputeDerivative(state2, NumOps.Add(time, half));

            T[] state3 = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                state3[i] = NumOps.Add(state[i], NumOps.Multiply(half, k2[i]));
            }

            T[] k3 = ComputeDerivative(state3, NumOps.Add(time, half));

            T[] state4 = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                state4[i] = NumOps.Add(state[i], NumOps.Multiply(dt, k3[i]));
            }

            T[] k4 = ComputeDerivative(state4, NumOps.Add(time, dt));

            T[] nextState = new T[_stateDim];
            for (int i = 0; i < _stateDim; i++)
            {
                T sum = NumOps.Add(k1[i], NumOps.Multiply(NumOps.FromDouble(2.0), k2[i]));
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.FromDouble(2.0), k3[i]));
                sum = NumOps.Add(sum, k4[i]);
                nextState[i] = NumOps.Add(state[i], NumOps.Multiply(sixth, sum));
            }

            return nextState;
        }

        /// <summary>
        /// Makes a prediction using the UDE model.
        /// </summary>
        /// <param name="input">Input tensor with state and time.</param>
        /// <returns>Predicted derivative tensor.</returns>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            if (input.Rank != 2 || input.Shape[1] != _stateDim + 1)
            {
                throw new ArgumentException($"Expected input shape [batch, {_stateDim + 1}].");
            }

            bool wasTraining = IsTrainingMode;
            SetTrainingMode(false);

            try
            {
                Tensor<T> learned = Forward(input);
                var known = BuildKnownDynamicsTensor(input);
                return Engine.TensorAdd(known, learned);
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
        /// Performs a supervised training step against derivative targets.
        /// </summary>
        /// <param name="input">Input tensor with state and time.</param>
        /// <param name="expectedOutput">Expected derivative tensor.</param>
        /// <remarks>
        /// Uses standard backpropagation like all other neural networks.
        /// The network learns to predict the derivative (dx/dt) at each state.
        /// </remarks>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (input.Rank != 2 || input.Shape[1] != _stateDim + 1)
            {
                throw new ArgumentException($"Expected input shape [batch, {_stateDim + 1}].");
            }

            if (expectedOutput == null)
            {
                throw new ArgumentNullException(nameof(expectedOutput));
            }

            if (expectedOutput.Rank != 2 || expectedOutput.Shape[0] != input.Shape[0] || expectedOutput.Shape[1] != _stateDim)
            {
                throw new ArgumentException($"Expected output shape [batch, {_stateDim}].", nameof(expectedOutput));
            }

            SetTrainingMode(true);

            // Step 1: Forward pass - compute learned corrections
            Tensor<T> learned = Forward(input);
            var known = BuildKnownDynamicsTensor(input);
            var prediction = Engine.TensorAdd(known, learned);

            // Step 2: Calculate loss
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
            LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            // Step 3: Backward pass - compute gradients
            var outputGradient = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);
            Backward(outputGradientTensor);

            // Step 4: Update parameters
            _optimizer.UpdateParameters(Layers);

            SetTrainingMode(false);
        }

        /// <summary>
        /// Gets metadata about the UDE model.
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
        /// Serializes UDE-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_stateDim);
        }

        /// <summary>
        /// Deserializes UDE-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedStateDim = reader.ReadInt32();
            if (storedStateDim != _stateDim)
            {
                throw new InvalidOperationException("Serialized UDE configuration does not match the current instance.");
            }
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New UDE instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new UniversalDifferentialEquation<T>(Architecture, _stateDim, _knownDynamics, _optimizer);
        }

        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;
        public override bool SupportsJitCompilation => false;

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

        private Tensor<T> BuildKnownDynamicsTensor(Tensor<T> input)
        {
            int batchSize = input.Shape[0];
            var known = new Tensor<T>(new int[] { batchSize, _stateDim });

            for (int i = 0; i < batchSize; i++)
            {
                var state = new T[_stateDim];
                for (int j = 0; j < _stateDim; j++)
                {
                    state[j] = input[i, j];
                }

                T time = input[i, _stateDim];
                T[] knownPart = _knownDynamics(state, time);
                if (knownPart == null || knownPart.Length != _stateDim)
                {
                    throw new InvalidOperationException("Known dynamics must return a vector matching the state dimension.");
                }

                for (int j = 0; j < _stateDim; j++)
                {
                    known[i, j] = knownPart[j];
                }
            }

            return known;
        }
    }
}


