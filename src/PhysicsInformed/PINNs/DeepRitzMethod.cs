using System;
using System.IO;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Implements the Deep Ritz Method for solving variational problems and PDEs.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Deep Ritz Method is a variational approach to solving PDEs using neural networks.
    /// Instead of minimizing the PDE residual directly (like standard PINNs), it minimizes
    /// an energy functional.
    ///
    /// The Ritz Method (Classical):
    /// Many PDEs can be reformulated as minimization problems. For example:
    /// - Poisson equation: -∇²u = f is equivalent to minimizing E(u) = ½∫|∇u|² dx - ∫fu dx
    /// - This is called the "variational formulation"
    /// - The solution minimizes the energy functional
    ///
    /// Deep Ritz (Modern):
    /// - Use a neural network to represent u(x)
    /// - Compute the energy functional using automatic differentiation
    /// - Train the network to minimize the energy
    /// - Naturally incorporates boundary conditions
    ///
    /// Advantages over Standard PINNs:
    /// 1. More stable training (minimizing energy vs. residual)
    /// 2. Natural framework for problems with variational structure
    /// 3. Often converges faster
    /// 4. Physical interpretation (energy minimization)
    ///
    /// Applications:
    /// - Elasticity (minimize strain energy)
    /// - Electrostatics (minimize electrostatic energy)
    /// - Fluid dynamics (minimize dissipation)
    /// - Quantum mechanics (minimize expected energy)
    /// - Optimal control problems
    ///
    /// Key Difference from PINNs:
    /// PINN: Minimize ||PDE residual||²
    /// Deep Ritz: Minimize ∫ Energy(u, ∇u) dx
    ///
    /// Both solve the same PDE, but Deep Ritz uses the variational (energy) formulation,
    /// which can be more natural and stable for certain problems.
    /// </remarks>
    public class DeepRitzMethod<T> : NeuralNetworkBase<T>
    {
        private readonly Func<T[], T[], T[,], T> _energyFunctional;
        private readonly Func<T[], bool>? _boundaryCheck;
        private readonly Func<T[], T[]>? _boundaryValue;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;
        private T[,]? _quadraturePoints;
        private T[]? _quadratureWeights;
        private readonly int _numQuadraturePoints;

        /// <summary>
        /// Initializes a new instance of the Deep Ritz Method.
        /// </summary>
        /// <param name="architecture">The neural network architecture.</param>
        /// <param name="energyFunctional">The energy functional to minimize: E(x, u, ∇u).</param>
        /// <param name="boundaryCheck">Function to check if a point is on the boundary.</param>
        /// <param name="boundaryValue">Function returning the boundary value at a point.</param>
        /// <param name="numQuadraturePoints">Number of quadrature points for numerical integration.</param>
        /// <remarks>
        /// For Beginners:
        /// The energy functional should encode the physics of your problem.
        ///
        /// Example - Poisson Equation (-∇²u = f):
        /// Energy: E(u) = ½∫|∇u|² dx - ∫fu dx
        /// Implementation: energyFunctional = (x, u, grad_u) => 0.5 * ||grad_u||² - f(x) * u
        ///
        /// Example - Linear Elasticity:
        /// Energy: E(u) = ∫ strain_energy(∇u) dx
        /// Implementation: energyFunctional = (x, u, grad_u) => compute_strain_energy(grad_u)
        ///
        /// The method will integrate this over the domain using quadrature points.
        /// </remarks>
        public DeepRitzMethod(
            NeuralNetworkArchitecture<T> architecture,
            Func<T[], T[], T[,], T> energyFunctional,
            Func<T[], bool>? boundaryCheck = null,
            Func<T[], T[]>? boundaryValue = null,
            int numQuadraturePoints = 10000)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _energyFunctional = energyFunctional ?? throw new ArgumentNullException(nameof(energyFunctional));
            _boundaryCheck = boundaryCheck;
            _boundaryValue = boundaryValue;
            _numQuadraturePoints = numQuadraturePoints;
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = true;

            InitializeLayers();
            GenerateQuadraturePoints(_numQuadraturePoints, architecture.InputSize);
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultDeepRitzLayers(Architecture));
            }
        }

        /// <summary>
        /// Generates quadrature points for numerical integration.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Numerical integration approximates ∫f(x)dx ≈ Σw_i * f(x_i)
        /// where x_i are quadrature points and w_i are weights.
        ///
        /// This method uses Monte Carlo integration:
        /// - Randomly sample points in the domain
        /// - Each point has equal weight = volume / numPoints
        /// - Simple but effective for high-dimensional problems
        ///
        /// Advanced users might want to use:
        /// - Gauss quadrature (for 1D problems)
        /// - Sparse grids (for moderate dimensions)
        /// - Quasi-Monte Carlo (better coverage than random)
        /// </remarks>
        private void GenerateQuadraturePoints(int numPoints, int dimension)
        {
            _quadraturePoints = new T[numPoints, dimension];
            _quadratureWeights = new T[numPoints];

            var random = RandomHelper.CreateSeededRandom(42);
            T weight = NumOps.Divide(NumOps.One, NumOps.FromDouble(numPoints)); // Uniform weights

            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    _quadraturePoints[i, j] = NumOps.FromDouble(random.NextDouble());
                }
                _quadratureWeights[i] = weight;
            }
        }

        /// <summary>
        /// Computes the total energy functional by integrating over the domain.
        /// </summary>
        /// <returns>The total energy value.</returns>
        /// <remarks>
        /// For Beginners:
        /// This is the key method that computes ∫E(u, ∇u)dx numerically.
        ///
        /// Steps:
        /// 1. For each quadrature point x_i:
        ///    a) Evaluate u(x_i) using the network
        ///    b) Compute ∇u(x_i) using automatic differentiation
        ///    c) Evaluate the energy density E(x_i, u(x_i), ∇u(x_i))
        /// 2. Sum weighted energies: Total = Σ w_i * E_i
        /// 3. Add boundary penalty if needed
        ///
        /// The gradient of this total energy with respect to network parameters
        /// tells us how to update the network to minimize energy.
        /// </remarks>
        public T ComputeTotalEnergy()
        {
            if (_quadraturePoints == null || _quadratureWeights == null)
            {
                throw new InvalidOperationException("Quadrature points not initialized.");
            }

            T totalEnergy = NumOps.Zero;

            // Integrate energy over the domain
            for (int i = 0; i < _quadraturePoints.GetLength(0); i++)
            {
                T[] point = new T[_quadraturePoints.GetLength(1)];
                for (int j = 0; j < point.Length; j++)
                {
                    point[j] = _quadraturePoints[i, j];
                }

                // Skip boundary points if boundary conditions are specified
                if (_boundaryCheck != null && _boundaryCheck(point))
                {
                    continue;
                }

                // Evaluate network
                T[] u = EvaluateAtPoint(point);

                // Compute gradient
                T[,] gradU = NeuralNetworkDerivatives<T>.ComputeGradient(
                    this,
                    point,
                    Architecture.OutputSize);

                // Evaluate energy density
                T energyDensity = _energyFunctional(point, u, gradU);

                // Weighted sum
                totalEnergy = NumOps.Add(totalEnergy, NumOps.Multiply(_quadratureWeights[i], energyDensity));
            }

            // Add boundary penalty if needed
            if (_boundaryCheck != null && _boundaryValue != null)
            {
                T boundaryPenalty = ComputeBoundaryPenalty();
                totalEnergy = NumOps.Add(totalEnergy, boundaryPenalty);
            }

            return totalEnergy;
        }

        /// <summary>
        /// Computes penalty for violating boundary conditions.
        /// </summary>
        private T ComputeBoundaryPenalty()
        {
            if (_quadraturePoints == null || _boundaryCheck == null || _boundaryValue == null)
            {
                return NumOps.Zero;
            }

            T penalty = NumOps.Zero;
            T penaltyWeight = NumOps.FromDouble(100.0); // Large weight to enforce BC

            for (int i = 0; i < _quadraturePoints.GetLength(0); i++)
            {
                T[] point = new T[_quadraturePoints.GetLength(1)];
                for (int j = 0; j < point.Length; j++)
                {
                    point[j] = _quadraturePoints[i, j];
                }

                if (_boundaryCheck(point))
                {
                    T[] u = EvaluateAtPoint(point);
                    T[] uBC = _boundaryValue(point);

                    for (int k = 0; k < u.Length; k++)
                    {
                        T error = NumOps.Subtract(u[k], uBC[k]);
                        penalty = NumOps.Add(penalty, NumOps.Multiply(penaltyWeight, NumOps.Multiply(error, error)));
                    }
                }
            }

            return penalty;
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

        private Tensor<T> BuildEvaluationTensor(T[,] basePoints, int start, int count, T step)
        {
            int inputDim = basePoints.GetLength(1);
            int evalPerBase = 1 + (2 * inputDim);
            var tensor = new Tensor<T>(new int[] { count * evalPerBase, inputDim });

            int row = 0;
            for (int b = 0; b < count; b++)
            {
                int baseRow = start + b;
                CopyPoint(basePoints, baseRow, tensor, row++);

                for (int dim = 0; dim < inputDim; dim++)
                {
                    CopyPoint(basePoints, baseRow, tensor, row);
                    tensor[row, dim] = NumOps.Add(tensor[row, dim], step);
                    row++;

                    CopyPoint(basePoints, baseRow, tensor, row);
                    tensor[row, dim] = NumOps.Subtract(tensor[row, dim], step);
                    row++;
                }
            }

            return tensor;
        }

        private static void CopyPoint(T[,] source, int sourceRow, Tensor<T> target, int targetRow)
        {
            int inputDim = source.GetLength(1);
            for (int dim = 0; dim < inputDim; dim++)
            {
                target[targetRow, dim] = source[sourceRow, dim];
            }
        }

        private (T Energy, T[] OutputGradients, T[,] GradientGradients) ComputeEnergyGradient(
            T[] point,
            T[] outputs,
            T[,] gradients)
        {
            int outputDim = outputs.Length;
            int inputDim = gradients.GetLength(1);

            var outputGradients = new T[outputDim];
            var gradientGradients = new T[outputDim, inputDim];

            T energy = _energyFunctional(point, outputs, gradients);
            T eps = NumOps.FromDouble(1e-4);
            T invTwoEps = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), eps));

            var outputsCopy = new T[outputDim];
            Array.Copy(outputs, outputsCopy, outputDim);

            var gradientsCopy = new T[outputDim, inputDim];
            Array.Copy(gradients, gradientsCopy, gradients.Length);

            for (int outIdx = 0; outIdx < outputDim; outIdx++)
            {
                T original = outputsCopy[outIdx];
                outputsCopy[outIdx] = NumOps.Add(original, eps);
                T plus = _energyFunctional(point, outputsCopy, gradientsCopy);
                outputsCopy[outIdx] = NumOps.Subtract(original, eps);
                T minus = _energyFunctional(point, outputsCopy, gradientsCopy);
                outputsCopy[outIdx] = original;

                outputGradients[outIdx] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
            }

            for (int outIdx = 0; outIdx < outputDim; outIdx++)
            {
                for (int dim = 0; dim < inputDim; dim++)
                {
                    T original = gradientsCopy[outIdx, dim];
                    gradientsCopy[outIdx, dim] = NumOps.Add(original, eps);
                    T plus = _energyFunctional(point, outputsCopy, gradientsCopy);
                    gradientsCopy[outIdx, dim] = NumOps.Subtract(original, eps);
                    T minus = _energyFunctional(point, outputsCopy, gradientsCopy);
                    gradientsCopy[outIdx, dim] = original;

                    gradientGradients[outIdx, dim] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
                }
            }

            return (energy, outputGradients, gradientGradients);
        }

        private T[] EvaluateAtPoint(T[] inputs)
        {
            var inputTensor = new Tensor<T>(new int[] { 1, inputs.Length });
            for (int i = 0; i < inputs.Length; i++)
            {
                inputTensor[0, i] = inputs[i];
            }

            var outputTensor = Forward(inputTensor);

            T[] result = new T[outputTensor.Shape[1]];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = outputTensor[0, i];
            }

            return result;
        }

        /// <summary>
        /// Trains the network to minimize the energy functional.
        /// </summary>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate for optimization.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <param name="batchSize">Number of quadrature points per batch.</param>
        /// <param name="derivativeStep">Finite-difference step size for input derivatives.</param>
        public TrainingHistory<T> Solve(
            int epochs = 1000,
            double learningRate = 0.001,
            bool verbose = true,
            int batchSize = 256,
            double derivativeStep = 1e-4)
        {
            if (_quadraturePoints == null || _quadratureWeights == null)
            {
                throw new InvalidOperationException("Quadrature points not initialized.");
            }

            if (batchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
            }

            if (derivativeStep <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(derivativeStep), "Derivative step must be positive.");
            }

            int inputDim = Architecture.InputSize;
            int outputDim = Architecture.OutputSize;
            int totalCount = _quadraturePoints.GetLength(0);

            if (totalCount == 0)
            {
                throw new InvalidOperationException("Quadrature points are empty.");
            }

            var history = new TrainingHistory<T>();
            T step = NumOps.FromDouble(derivativeStep);
            int evalPerBase = 1 + (2 * inputDim);

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    LearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    T epochEnergy = NumOps.Zero;

                    for (int batchStart = 0; batchStart < totalCount; batchStart += batchSize)
                    {
                        int batchCount = Math.Min(batchSize, totalCount - batchStart);
                        var evaluationInputs = BuildEvaluationTensor(_quadraturePoints, batchStart, batchCount, step);
                        var outputs = ForwardWithMemory(evaluationInputs);

                        if (outputs.Shape[1] != outputDim)
                        {
                            throw new InvalidOperationException(
                                $"Expected {outputDim} outputs from the network, got {outputs.Shape[1]}.");
                        }
                        var outputGradients = new Tensor<T>(outputs.Shape);

                        T invTwoStep = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), step));
                        T batchScale = NumOps.Divide(NumOps.FromDouble(totalCount), NumOps.FromDouble(batchCount));
                        T penaltyWeight = NumOps.FromDouble(100.0);

                        var point = new T[inputDim];
                        var baseOutput = new T[outputDim];
                        var gradU = new T[outputDim, inputDim];

                        for (int b = 0; b < batchCount; b++)
                        {
                            int baseRow = batchStart + b;
                            int baseOffset = b * evalPerBase;

                            for (int j = 0; j < inputDim; j++)
                            {
                                point[j] = _quadraturePoints[baseRow, j];
                            }

                            for (int j = 0; j < outputDim; j++)
                            {
                                baseOutput[j] = outputs[baseOffset, j];
                            }

                            for (int outIdx = 0; outIdx < outputDim; outIdx++)
                            {
                                for (int dim = 0; dim < inputDim; dim++)
                                {
                                    int plusIndex = baseOffset + 1 + (2 * dim);
                                    int minusIndex = plusIndex + 1;

                                    T plus = outputs[plusIndex, outIdx];
                                    T minus = outputs[minusIndex, outIdx];
                                    gradU[outIdx, dim] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoStep);
                                }
                            }

                            if (_boundaryCheck != null && _boundaryCheck(point))
                            {
                                if (_boundaryValue != null)
                                {
                                    T[] uBC = _boundaryValue(point);
                                    for (int outIdx = 0; outIdx < outputDim; outIdx++)
                                    {
                                        T error = NumOps.Subtract(baseOutput[outIdx], uBC[outIdx]);
                                        T penalty = NumOps.Multiply(penaltyWeight, NumOps.Multiply(error, error));
                                        epochEnergy = NumOps.Add(epochEnergy, penalty);

                                        T grad = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(penaltyWeight, error));
                                        T scaledGrad = NumOps.Multiply(grad, batchScale);
                                        outputGradients[baseOffset, outIdx] = NumOps.Add(outputGradients[baseOffset, outIdx], scaledGrad);
                                    }
                                }

                                continue;
                            }

                            var (energy, outputGrad, gradUGrad) = ComputeEnergyGradient(point, baseOutput, gradU);
                            T weight = _quadratureWeights[baseRow];
                            epochEnergy = NumOps.Add(epochEnergy, NumOps.Multiply(weight, energy));
                            T scaledWeight = NumOps.Multiply(weight, batchScale);

                            for (int outIdx = 0; outIdx < outputDim; outIdx++)
                            {
                                T outputContribution = NumOps.Multiply(scaledWeight, outputGrad[outIdx]);
                                outputGradients[baseOffset, outIdx] = NumOps.Add(outputGradients[baseOffset, outIdx], outputContribution);

                                for (int dim = 0; dim < inputDim; dim++)
                                {
                                    int plusIndex = baseOffset + 1 + (2 * dim);
                                    int minusIndex = plusIndex + 1;

                                    T gradContribution = NumOps.Multiply(scaledWeight, gradUGrad[outIdx, dim]);
                                    T delta = NumOps.Multiply(gradContribution, invTwoStep);
                                    outputGradients[plusIndex, outIdx] = NumOps.Add(outputGradients[plusIndex, outIdx], delta);
                                    outputGradients[minusIndex, outIdx] = NumOps.Subtract(outputGradients[minusIndex, outIdx], delta);
                                }
                            }
                        }

                        Backpropagate(outputGradients);
                        _optimizer.UpdateParameters(Layers);
                    }

                    LastLoss = epochEnergy;
                    history.AddEpoch(epochEnergy);

                    if (verbose && epoch % 100 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Energy: {epochEnergy}");
                    }
                }
            }
            finally
            {
                foreach (var layer in Layers)
                {
                    layer.SetTrainingMode(false);
                }

                SetTrainingMode(false);
            }

            return history;
        }

        /// <summary>
        /// Gets the solution at a specific point.
        /// </summary>
        public T[] GetSolution(T[] point)
        {
            return EvaluateAtPoint(point);
        }

        /// <summary>
        /// Makes a prediction using the Deep Ritz network.
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
        /// Performs a basic supervised training step using MSE loss.
        /// </summary>
        /// <param name="input">Training input tensor.</param>
        /// <param name="expectedOutput">Expected output tensor.</param>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            SetTrainingMode(true);

            var prediction = Forward(input);
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
            LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            var outputGradient = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = Tensor<T>.FromVector(outputGradient).Reshape(prediction.Shape);

            Backpropagate(outputGradientTensor);
            _optimizer.UpdateParameters(Layers);

            SetTrainingMode(false);
        }

        /// <summary>
        /// Gets metadata about the Deep Ritz model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "InputSize", Architecture.InputSize },
                    { "OutputSize", Architecture.OutputSize },
                    { "QuadraturePoints", _numQuadraturePoints },
                    { "HasBoundaryConditions", _boundaryCheck != null && _boundaryValue != null },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes Deep Ritz-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_numQuadraturePoints);
            writer.Write(Architecture.InputSize);
        }

        /// <summary>
        /// Deserializes Deep Ritz-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedNumPoints = reader.ReadInt32();
            int storedDimension = reader.ReadInt32();

            if (storedNumPoints != _numQuadraturePoints || storedDimension != Architecture.InputSize)
            {
                throw new InvalidOperationException("Serialized Deep Ritz configuration does not match the current instance.");
            }

            GenerateQuadraturePoints(storedNumPoints, storedDimension);
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New Deep Ritz instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new DeepRitzMethod<T>(
                Architecture,
                _energyFunctional,
                _boundaryCheck,
                _boundaryValue,
                _numQuadraturePoints);
        }

        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;
    }
}


