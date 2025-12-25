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
using AiDotNet.PhysicsInformed.Interfaces;

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
    public class LagrangianNeuralNetwork<T> : NeuralNetworkBase<T>
    {
        private readonly int _configurationDim; // Dimension of configuration space (q)
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        public LagrangianNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            int configurationDim,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
            : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
                NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
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
        public T ComputeLagrangian(T[] q, T[] qDot)
        {
            ValidateStateVectors(q, qDot);

            T[] state = new T[2 * _configurationDim];
            Array.Copy(q, 0, state, 0, _configurationDim);
            Array.Copy(qDot, 0, state, _configurationDim, _configurationDim);

            return ComputeLagrangianFromState(state);
        }

        /// <summary>
        /// Computes acceleration using Euler-Lagrange equation.
        /// </summary>
        public T[] ComputeAcceleration(T[] q, T[] qDot)
        {
            ValidateStateVectors(q, qDot);

            T[] state = new T[2 * _configurationDim];
            Array.Copy(q, 0, state, 0, _configurationDim);
            Array.Copy(qDot, 0, state, _configurationDim, _configurationDim);

            var derivatives = ComputeDerivativesWithFallback(state);
            if (derivatives.FirstDerivatives == null || derivatives.SecondDerivatives == null)
            {
                throw new InvalidOperationException("Failed to compute derivatives for the Lagrangian.");
            }

            var gradient = derivatives.FirstDerivatives;
            var hessian = derivatives.SecondDerivatives;

            T[] dLdq = new T[_configurationDim];
            for (int i = 0; i < _configurationDim; i++)
            {
                dLdq[i] = gradient[0, i];
            }

            T[,] mass = new T[_configurationDim, _configurationDim];
            T[,] cross = new T[_configurationDim, _configurationDim];
            for (int i = 0; i < _configurationDim; i++)
            {
                for (int j = 0; j < _configurationDim; j++)
                {
                    mass[i, j] = hessian[0, _configurationDim + i, _configurationDim + j];
                    cross[i, j] = hessian[0, _configurationDim + i, j];
                }
            }

            T[] coriolis = new T[_configurationDim];
            for (int i = 0; i < _configurationDim; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < _configurationDim; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(cross[i, j], qDot[j]));
                }
                coriolis[i] = sum;
            }

            T[] rhs = new T[_configurationDim];
            for (int i = 0; i < _configurationDim; i++)
            {
                rhs[i] = NumOps.Subtract(dLdq[i], coriolis[i]);
            }

            var regularization = NumOps.FromDouble(1e-6);
            for (int i = 0; i < _configurationDim; i++)
            {
                mass[i, i] = NumOps.Add(mass[i, i], regularization);
            }

            var massMatrix = new Matrix<T>(mass);
            var rhsVector = new Vector<T>(rhs);

            try
            {
                var solver = new LuDecomposition<T>(massMatrix);
                var solution = solver.Solve(rhsVector);
                var acceleration = new T[_configurationDim];
                for (int i = 0; i < _configurationDim; i++)
                {
                    acceleration[i] = solution[i];
                }

                return acceleration;
            }
            catch (InvalidOperationException)
            {
                var acceleration = new T[_configurationDim];
                for (int i = 0; i < _configurationDim; i++)
                {
                    acceleration[i] = NumOps.Zero;
                }

                return acceleration;
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
        /// Makes a prediction using the Lagrangian network.
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
        /// Trains the Lagrangian neural network using the provided input and expected output.
        /// </summary>
        /// <param name="input">The input tensor for training (state vectors [q, q̇]).</param>
        /// <param name="expectedOutput">The expected output tensor (Lagrangian values).</param>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method trains the network to predict the correct Lagrangian (kinetic
        /// minus potential energy) for given state vectors. The training follows the standard neural network
        /// pattern: forward pass → loss calculation → backward pass → parameter update.
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
        /// Gets metadata about the Lagrangian network.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "ConfigurationDimension", _configurationDim },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes Lagrangian-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_configurationDim);
        }

        /// <summary>
        /// Deserializes Lagrangian-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedConfigDim = reader.ReadInt32();
            if (storedConfigDim != _configurationDim)
            {
                throw new InvalidOperationException("Serialized Lagrangian configuration does not match the current instance.");
            }
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New Lagrangian network instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new LagrangianNeuralNetwork<T>(Architecture, _configurationDim, _optimizer);
        }

        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;

        private void ValidateStateVectors(T[] q, T[] qDot)
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

        private T ComputeLagrangianFromState(T[] state)
        {
            var inputTensor = new Tensor<T>(new int[] { 1, state.Length });
            for (int i = 0; i < state.Length; i++)
            {
                inputTensor[0, i] = state[i];
            }

            var output = Forward(inputTensor);
            return output[0, 0];
        }

        private PDEDerivatives<T> ComputeDerivativesWithFallback(T[] state)
        {
            try
            {
                return NeuralNetworkDerivatives<T>.ComputeDerivatives(this, state, 1);
            }
            catch (InvalidOperationException)
            {
                return ComputeDerivativesFiniteDifference(state);
            }
            catch (NotSupportedException)
            {
                return ComputeDerivativesFiniteDifference(state);
            }
        }

        private PDEDerivatives<T> ComputeDerivativesFiniteDifference(T[] state)
        {
            int dim = state.Length;
            var derivatives = new PDEDerivatives<T>
            {
                FirstDerivatives = new T[1, dim],
                SecondDerivatives = new T[1, dim, dim]
            };

            T epsilon = NumOps.FromDouble(1e-5);
            T twoEpsilon = NumOps.FromDouble(2e-5);
            T epsSquared = NumOps.Multiply(epsilon, epsilon);
            T fourEpsSquared = NumOps.Multiply(NumOps.FromDouble(4.0), epsSquared);

            var perturbed = new T[dim];
            Array.Copy(state, perturbed, dim);
            T baseValue = ComputeLagrangianFromState(state);

            for (int i = 0; i < dim; i++)
            {
                T original = perturbed[i];
                perturbed[i] = NumOps.Add(original, epsilon);
                T plus = ComputeLagrangianFromState(perturbed);
                perturbed[i] = NumOps.Subtract(original, epsilon);
                T minus = ComputeLagrangianFromState(perturbed);
                derivatives.FirstDerivatives[0, i] = NumOps.Divide(NumOps.Subtract(plus, minus), twoEpsilon);

                T second = NumOps.Divide(
                    NumOps.Add(NumOps.Subtract(plus, NumOps.Multiply(NumOps.FromDouble(2.0), baseValue)), minus),
                    epsSquared);
                derivatives.SecondDerivatives[0, i, i] = second;
                perturbed[i] = original;
            }

            for (int i = 0; i < dim; i++)
            {
                for (int j = i + 1; j < dim; j++)
                {
                    T originalI = perturbed[i];
                    T originalJ = perturbed[j];

                    perturbed[i] = NumOps.Add(originalI, epsilon);
                    perturbed[j] = NumOps.Add(originalJ, epsilon);
                    T plusPlus = ComputeLagrangianFromState(perturbed);

                    perturbed[j] = NumOps.Subtract(originalJ, epsilon);
                    T plusMinus = ComputeLagrangianFromState(perturbed);

                    perturbed[i] = NumOps.Subtract(originalI, epsilon);
                    perturbed[j] = NumOps.Add(originalJ, epsilon);
                    T minusPlus = ComputeLagrangianFromState(perturbed);

                    perturbed[j] = NumOps.Subtract(originalJ, epsilon);
                    T minusMinus = ComputeLagrangianFromState(perturbed);

                    T cross = NumOps.Divide(
                        NumOps.Add(NumOps.Subtract(plusPlus, plusMinus), NumOps.Subtract(minusMinus, minusPlus)),
                        fourEpsSquared);

                    derivatives.SecondDerivatives[0, i, j] = cross;
                    derivatives.SecondDerivatives[0, j, i] = cross;

                    perturbed[i] = originalI;
                    perturbed[j] = originalJ;
                }
            }

            return derivatives;
        }
    }
}








