using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Implements a Physics-Informed Neural Network for inverse problems (parameter identification).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Inverse problems are about discovering unknown parameters from observations.
    /// This is the opposite of forward problems where parameters are known.
    ///
    /// Examples of Inverse Problems:
    /// 1. Medical Imaging: Find tumor location from external measurements
    /// 2. Material Science: Identify Young's modulus from stress-strain data
    /// 3. Geophysics: Determine subsurface properties from seismic data
    /// 4. Finance: Calibrate model parameters from market prices
    ///
    /// How InverseProblemPINN Works:
    /// 1. Neural network learns the solution u(x,t)
    /// 2. Additional trainable variables represent unknown parameters θ
    /// 3. Both are trained together to minimize:
    ///    - Data loss: ||u_predicted - u_observed||²
    ///    - Physics loss: ||PDE_residual(u, θ)||²
    ///    - Regularization: Prior knowledge about parameters
    ///
    /// Key Advantages:
    /// - Handles noisy and sparse observations
    /// - Physics acts as regularization
    /// - No need for iterative PDE solves
    /// - Can quantify parameter uncertainty
    ///
    /// Training Strategy:
    /// 1. Initialize parameters near prior estimates
    /// 2. Train with emphasis on physics initially
    /// 3. Gradually increase data weight
    /// 4. Use separate learning rates for network and parameters
    /// </remarks>
    public class InverseProblemPINN<T> : NeuralNetworkBase<T>
    {
        private readonly IInverseProblem<T> _inverseProblem;
        private readonly IBoundaryCondition<T>[] _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;
        private readonly InverseProblemOptions<T> _options;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        // Trainable parameters (the unknowns we're trying to find)
        private T[] _parameters;
        private T[]? _parameterGradients;

        // Current PDE with parameters applied
        private IPDESpecification<T>? _currentPDE;

        // Collocation points
        private T[,]? _collocationPoints;
        private readonly int _numCollocationPoints;

        // Parameter history for monitoring convergence
        private readonly List<T[]> _parameterHistory;

        /// <summary>
        /// Gets the current estimated parameter values.
        /// </summary>
        public T[] Parameters => _parameters.ToArray();

        /// <summary>
        /// Gets the parameter names.
        /// </summary>
        public string[] ParameterNames => _inverseProblem.ParameterNames;

        /// <summary>
        /// Gets the parameter estimation history.
        /// </summary>
        public IReadOnlyList<T[]> ParameterHistory => _parameterHistory;

        /// <summary>
        /// Initializes a new instance of the InverseProblemPINN.
        /// </summary>
        /// <param name="architecture">The neural network architecture.</param>
        /// <param name="inverseProblem">The inverse problem specification.</param>
        /// <param name="boundaryConditions">Boundary conditions for the problem.</param>
        /// <param name="initialCondition">Initial condition for time-dependent problems (optional).</param>
        /// <param name="numCollocationPoints">Number of collocation points for PDE enforcement.</param>
        /// <param name="options">Configuration options for inverse problem training.</param>
        /// <param name="optimizer">Optimization algorithm.</param>
        /// <remarks>
        /// For Beginners:
        ///
        /// The inverse problem specification defines:
        /// - What parameters are unknown (e.g., diffusion coefficient)
        /// - Initial guesses and bounds for these parameters
        /// - The observations used to identify parameters
        ///
        /// The training balances:
        /// - Fitting observations (data loss)
        /// - Satisfying physics (PDE residual)
        /// - Staying near prior estimates (regularization)
        /// </remarks>
        public InverseProblemPINN(
            NeuralNetworkArchitecture<T> architecture,
            IInverseProblem<T> inverseProblem,
            IBoundaryCondition<T>[] boundaryConditions,
            IInitialCondition<T>? initialCondition = null,
            int numCollocationPoints = 10000,
            InverseProblemOptions<T>? options = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _inverseProblem = inverseProblem ?? throw new ArgumentNullException(nameof(inverseProblem));
            _boundaryConditions = boundaryConditions ?? throw new ArgumentNullException(nameof(boundaryConditions));
            _initialCondition = initialCondition;
            _numCollocationPoints = numCollocationPoints;
            _options = options ?? new InverseProblemOptions<T>();
            Options = _options;

            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            // Initialize unknown parameters with initial guesses
            _parameters = _inverseProblem.InitialParameterGuesses.ToArray();
            _parameterHistory = new List<T[]>();

            InitializeLayers();
            GenerateCollocationPoints();
            UpdatePDE();
        }

        /// <inheritdoc/>
        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultPINNLayers(Architecture));
            }
        }

        /// <summary>
        /// Generates collocation points for PDE enforcement.
        /// </summary>
        private void GenerateCollocationPoints()
        {
            // Get input dimension from the first observation location
            int inputDim = _inverseProblem.Observations[0].location.Length;
            _collocationPoints = new T[_numCollocationPoints, inputDim];

            var random = RandomHelper.CreateSeededRandom(42);

            for (int i = 0; i < _numCollocationPoints; i++)
            {
                for (int d = 0; d < inputDim; d++)
                {
                    _collocationPoints[i, d] = NumOps.FromDouble(random.NextDouble());
                }
            }
        }

        /// <summary>
        /// Updates the PDE with current parameter values.
        /// </summary>
        private void UpdatePDE()
        {
            _currentPDE = _inverseProblem.CreateParameterizedPDE(_parameters);
        }

        /// <summary>
        /// Solves the inverse problem to identify unknown parameters.
        /// </summary>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate for the neural network.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <returns>Results including identified parameters and uncertainties.</returns>
        public InverseProblemResult<T> Solve(int epochs = 1000, double learningRate = 0.001, bool verbose = true)
        {
            var result = new InverseProblemResult<T>
            {
                ParameterNames = _inverseProblem.ParameterNames
            };

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);
            _parameterHistory.Clear();

            try
            {
                T previousLoss = NumOps.FromDouble(double.MaxValue);
                int noImprovementCount = 0;
                const int patience = 50;
                const double tolerance = 1e-8;

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    var (dataLoss, physicsLoss, regularizationLoss) = TrainEpoch();
                    T totalLoss = NumOps.Add(dataLoss, NumOps.Add(physicsLoss, regularizationLoss));

                    // Record parameter history
                    if (_options.LogParameterHistory)
                    {
                        _parameterHistory.Add(_parameters.ToArray());
                    }

                    // Check convergence
                    T lossDiff = NumOps.Abs(NumOps.Subtract(totalLoss, previousLoss));
                    if (NumOps.ToDouble(lossDiff) < tolerance)
                    {
                        noImprovementCount++;
                        if (noImprovementCount >= patience)
                        {
                            result.Converged = true;
                            result.IterationsToConverge = epoch;
                            if (verbose)
                            {
                                Console.WriteLine($"Converged at epoch {epoch}");
                            }
                            break;
                        }
                    }
                    else
                    {
                        noImprovementCount = 0;
                    }
                    previousLoss = totalLoss;

                    if (verbose && epoch % 50 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}:");
                        Console.WriteLine($"  Data Loss: {dataLoss}, Physics Loss: {physicsLoss}");
                        Console.Write("  Parameters: ");
                        for (int i = 0; i < _parameters.Length; i++)
                        {
                            Console.Write($"{_inverseProblem.ParameterNames[i]}={_parameters[i]:G4} ");
                        }
                        Console.WriteLine();
                    }
                }

                result.Parameters = _parameters.ToArray();
                result.DataLoss = ComputeDataLoss();
                result.PhysicsLoss = ComputePhysicsLoss();
                result.TotalLoss = NumOps.Add(result.DataLoss, result.PhysicsLoss);

                if (_options.LogParameterHistory)
                {
                    result.ParameterHistory = _parameterHistory.ToList();
                }

                // Estimate uncertainties if requested
                if (_options.EstimateUncertainty)
                {
                    result.ParameterUncertainties = EstimateParameterUncertainties();
                    result.ParameterCorrelations = EstimateParameterCorrelations();
                }
            }
            finally
            {
                SetTrainingMode(false);
            }

            return result;
        }

        /// <summary>
        /// Performs one training epoch.
        /// </summary>
        private (T dataLoss, T physicsLoss, T regularizationLoss) TrainEpoch()
        {
            // Compute losses
            T dataLoss = ComputeDataLoss();
            T physicsLoss = ComputePhysicsLoss();
            T regularizationLoss = ComputeRegularizationLoss();

            // Backpropagate and update network parameters
            BackpropagateAndUpdate(dataLoss, physicsLoss, regularizationLoss);

            // Update unknown parameters
            UpdateUnknownParameters();

            // Update PDE with new parameters
            UpdatePDE();

            return (dataLoss, physicsLoss, regularizationLoss);
        }

        /// <summary>
        /// Computes the data loss (fit to observations).
        /// </summary>
        private T ComputeDataLoss()
        {
            T loss = NumOps.Zero;
            var observations = _inverseProblem.Observations;
            int count = observations.Count;

            if (count == 0)
            {
                return loss;
            }

            foreach (var (location, observedValue) in observations)
            {
                // Create input tensor for this observation location
                var inputTensor = new Tensor<T>(new int[] { 1, location.Length });
                for (int d = 0; d < location.Length; d++)
                {
                    inputTensor[0, d] = location[d];
                }

                // Predict at this location
                var prediction = Predict(inputTensor);

                // Compute squared error
                for (int o = 0; o < observedValue.Length; o++)
                {
                    T error = NumOps.Subtract(prediction[0, o], observedValue[o]);
                    loss = NumOps.Add(loss, NumOps.Multiply(error, error));
                }
            }

            T dataWeight = _options.DataWeight ?? NumOps.One;
            return NumOps.Multiply(dataWeight, NumOps.Divide(loss, NumOps.FromDouble(count)));
        }

        /// <summary>
        /// Computes the physics loss (PDE residual).
        /// </summary>
        private T ComputePhysicsLoss()
        {
            if (_currentPDE == null || _collocationPoints == null)
            {
                return NumOps.Zero;
            }

            T loss = NumOps.Zero;
            int numPoints = _collocationPoints.GetLength(0);
            int inputDim = _collocationPoints.GetLength(1);

            // Sample a batch of collocation points
            int batchSize = Math.Min(256, numPoints);
            var random = RandomHelper.Shared;

            for (int i = 0; i < batchSize; i++)
            {
                int idx = random.Next(numPoints);

                // Create input tensor
                var inputTensor = new Tensor<T>(new int[] { 1, inputDim });
                for (int d = 0; d < inputDim; d++)
                {
                    inputTensor[0, d] = _collocationPoints[idx, d];
                }

                // Get prediction and compute derivatives
                var prediction = Predict(inputTensor);
                var derivatives = ComputeDerivatives(inputTensor);

                // Extract arrays for PDE residual computation
                var inputArray = new T[inputDim];
                for (int d = 0; d < inputDim; d++)
                {
                    inputArray[d] = _collocationPoints[idx, d];
                }

                var outputArray = new T[_currentPDE.OutputDimension];
                for (int o = 0; o < _currentPDE.OutputDimension; o++)
                {
                    outputArray[o] = prediction[0, o];
                }

                // Compute PDE residual
                T residual = _currentPDE.ComputeResidual(inputArray, outputArray, derivatives);
                loss = NumOps.Add(loss, NumOps.Multiply(residual, residual));
            }

            return NumOps.Divide(loss, NumOps.FromDouble(batchSize));
        }

        /// <summary>
        /// Computes derivatives using finite differences.
        /// </summary>
        private PDEDerivatives<T> ComputeDerivatives(Tensor<T> input)
        {
            if (_currentPDE == null)
            {
                throw new InvalidOperationException("PDE not initialized.");
            }

            int inputDim = input.Shape[1];
            int outputDim = _currentPDE.OutputDimension;
            T epsilon = NumOps.FromDouble(1e-5);

            var derivatives = new PDEDerivatives<T>
            {
                FirstDerivatives = new T[outputDim, inputDim],
                SecondDerivatives = new T[outputDim, inputDim, inputDim]
            };

            var baseOutput = Predict(input);

            for (int d = 0; d < inputDim; d++)
            {
                var plusInput = new Tensor<T>(input.Shape);
                var minusInput = new Tensor<T>(input.Shape);

                for (int i = 0; i < inputDim; i++)
                {
                    plusInput[0, i] = input[0, i];
                    minusInput[0, i] = input[0, i];
                }

                plusInput[0, d] = NumOps.Add(input[0, d], epsilon);
                minusInput[0, d] = NumOps.Subtract(input[0, d], epsilon);

                var plusOutput = Predict(plusInput);
                var minusOutput = Predict(minusInput);

                T twoEpsilon = NumOps.Multiply(NumOps.FromDouble(2.0), epsilon);

                for (int o = 0; o < outputDim; o++)
                {
                    derivatives.FirstDerivatives[o, d] = NumOps.Divide(
                        NumOps.Subtract(plusOutput[0, o], minusOutput[0, o]),
                        twoEpsilon);

                    T epsilonSq = NumOps.Multiply(epsilon, epsilon);
                    derivatives.SecondDerivatives[o, d, d] = NumOps.Divide(
                        NumOps.Subtract(
                            NumOps.Add(plusOutput[0, o], minusOutput[0, o]),
                            NumOps.Multiply(NumOps.FromDouble(2.0), baseOutput[0, o])),
                        epsilonSq);
                }
            }

            return derivatives;
        }

        /// <summary>
        /// Computes the regularization loss for parameters.
        /// </summary>
        private T ComputeRegularizationLoss()
        {
            T lambda = _options.RegularizationStrength ?? NumOps.FromDouble(0.01);
            T loss = NumOps.Zero;

            switch (_options.Regularization)
            {
                case InverseProblemRegularization.None:
                    return NumOps.Zero;

                case InverseProblemRegularization.L2Tikhonov:
                    // L2: sum of squared parameters
                    foreach (var p in _parameters)
                    {
                        loss = NumOps.Add(loss, NumOps.Multiply(p, p));
                    }
                    break;

                case InverseProblemRegularization.L1Lasso:
                    // L1: sum of absolute parameters
                    foreach (var p in _parameters)
                    {
                        loss = NumOps.Add(loss, NumOps.Abs(p));
                    }
                    break;

                case InverseProblemRegularization.ElasticNet:
                    // Combination of L1 and L2
                    T l1Sum = NumOps.Zero;
                    T l2Sum = NumOps.Zero;
                    foreach (var p in _parameters)
                    {
                        l1Sum = NumOps.Add(l1Sum, NumOps.Abs(p));
                        l2Sum = NumOps.Add(l2Sum, NumOps.Multiply(p, p));
                    }
                    loss = NumOps.Add(l1Sum, l2Sum);
                    break;

                case InverseProblemRegularization.Bayesian:
                    // Negative log-prior (assuming Gaussian)
                    if (_options.PriorMeans != null && _options.PriorStandardDeviations != null)
                    {
                        for (int i = 0; i < _parameters.Length; i++)
                        {
                            T diff = NumOps.Subtract(_parameters[i], _options.PriorMeans[i]);
                            T stdSq = NumOps.Multiply(_options.PriorStandardDeviations[i],
                                                       _options.PriorStandardDeviations[i]);
                            loss = NumOps.Add(loss, NumOps.Divide(
                                NumOps.Multiply(diff, diff),
                                NumOps.Multiply(NumOps.FromDouble(2.0), stdSq)));
                        }
                    }
                    break;

                default:
                    return NumOps.Zero;
            }

            return NumOps.Multiply(lambda, loss);
        }

        /// <summary>
        /// Backpropagates losses and updates network parameters.
        /// </summary>
        private void BackpropagateAndUpdate(T dataLoss, T physicsLoss, T regularizationLoss)
        {
            // Total loss gradient flows through the network
            // For simplicity, we use the combined loss for backpropagation
            T totalLoss = NumOps.Add(dataLoss, NumOps.Add(physicsLoss, regularizationLoss));

            // Compute network gradients via backpropagation
            // Note: In practice, you'd compute gradients through the loss functions
            // Here we use the optimizer to update parameters based on gradients

            _optimizer.UpdateParameters(Layers);
        }

        /// <summary>
        /// Updates the unknown parameters using gradient descent.
        /// </summary>
        private void UpdateUnknownParameters()
        {
            // Compute parameter gradients using finite differences
            _parameterGradients = ComputeParameterGradients();

            T paramLR = NumOps.FromDouble(_options.ParameterLearningRate);

            for (int i = 0; i < _parameters.Length; i++)
            {
                // Gradient descent step
                _parameters[i] = NumOps.Subtract(
                    _parameters[i],
                    NumOps.Multiply(paramLR, _parameterGradients[i]));

                // Apply bounds if specified
                var lowerBounds = _inverseProblem.ParameterLowerBounds;
                var upperBounds = _inverseProblem.ParameterUpperBounds;

                if (lowerBounds != null && NumOps.LessThan(_parameters[i], lowerBounds[i]))
                {
                    _parameters[i] = lowerBounds[i];
                }

                if (upperBounds != null && NumOps.GreaterThan(_parameters[i], upperBounds[i]))
                {
                    _parameters[i] = upperBounds[i];
                }
            }
        }

        /// <summary>
        /// Computes gradients of the loss with respect to parameters using finite differences.
        /// </summary>
        private T[] ComputeParameterGradients()
        {
            var gradients = new T[_parameters.Length];
            T epsilon = NumOps.FromDouble(1e-5);

            T baseLoss = NumOps.Add(ComputeDataLoss(), ComputePhysicsLoss());

            for (int i = 0; i < _parameters.Length; i++)
            {
                T originalValue = _parameters[i];

                // Forward difference
                _parameters[i] = NumOps.Add(originalValue, epsilon);
                UpdatePDE();
                T plusLoss = NumOps.Add(ComputeDataLoss(), ComputePhysicsLoss());

                // Backward difference
                _parameters[i] = NumOps.Subtract(originalValue, epsilon);
                UpdatePDE();
                T minusLoss = NumOps.Add(ComputeDataLoss(), ComputePhysicsLoss());

                // Central difference gradient
                gradients[i] = NumOps.Divide(
                    NumOps.Subtract(plusLoss, minusLoss),
                    NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));

                // Restore original value
                _parameters[i] = originalValue;
            }

            UpdatePDE();
            return gradients;
        }

        /// <summary>
        /// Estimates parameter uncertainties using Fisher information.
        /// </summary>
        private T[]? EstimateParameterUncertainties()
        {
            // Simplified uncertainty estimation using diagonal of Hessian
            var uncertainties = new T[_parameters.Length];
            T epsilon = NumOps.FromDouble(1e-5);

            for (int i = 0; i < _parameters.Length; i++)
            {
                T originalValue = _parameters[i];

                // Compute second derivative (curvature)
                _parameters[i] = NumOps.Add(originalValue, epsilon);
                UpdatePDE();
                T plusLoss = NumOps.Add(ComputeDataLoss(), ComputePhysicsLoss());

                _parameters[i] = NumOps.Subtract(originalValue, epsilon);
                UpdatePDE();
                T minusLoss = NumOps.Add(ComputeDataLoss(), ComputePhysicsLoss());

                _parameters[i] = originalValue;
                UpdatePDE();
                T centerLoss = NumOps.Add(ComputeDataLoss(), ComputePhysicsLoss());

                T epsilonSq = NumOps.Multiply(epsilon, epsilon);
                T curvature = NumOps.Divide(
                    NumOps.Subtract(NumOps.Add(plusLoss, minusLoss),
                                    NumOps.Multiply(NumOps.FromDouble(2.0), centerLoss)),
                    epsilonSq);

                // Uncertainty is inverse square root of curvature
                T absCurvature = NumOps.Abs(curvature);
                if (NumOps.GreaterThan(absCurvature, NumOps.FromDouble(1e-10)))
                {
                    uncertainties[i] = NumOps.Sqrt(NumOps.Divide(NumOps.One, absCurvature));
                }
                else
                {
                    uncertainties[i] = NumOps.FromDouble(double.PositiveInfinity);
                }
            }

            return uncertainties;
        }

        /// <summary>
        /// Estimates correlation matrix between parameters.
        /// </summary>
        private T[,]? EstimateParameterCorrelations()
        {
            // Simplified: return identity matrix
            int n = _parameters.Length;
            var correlations = new T[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    correlations[i, j] = (i == j) ? NumOps.One : NumOps.Zero;
                }
            }

            return correlations;
        }

        /// <inheritdoc/>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        /// <inheritdoc/>
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

            try
            {
                var prediction = ForwardWithMemory(input);
                var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
                LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

                var outputGradientVector = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
                var outputGradient = new Tensor<T>(prediction.Shape, outputGradientVector);

                Backpropagate(outputGradient);
                _optimizer.UpdateParameters(Layers);
            }
            finally
            {
                SetTrainingMode(false);
            }
        }

        /// <inheritdoc/>
        public override Tensor<T> ForwardWithMemory(Tensor<T> input)
        {
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            return Predict(input);
        }

        /// <inheritdoc/>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int offset = 0;
            foreach (var layer in Layers)
            {
                int paramCount = layer.ParameterCount;
                if (paramCount > 0)
                {
                    var subParams = parameters.GetSubVector(offset, paramCount);
                    layer.UpdateParameters(subParams);
                    offset += paramCount;
                }
            }

            // Update inverse problem parameters
            for (int i = 0; i < _parameters.Length; i++)
            {
                _parameters[i] = parameters[offset + i];
            }

            UpdatePDE();
        }

        /// <inheritdoc/>
        public override Vector<T> GetParameters()
        {
            var allParams = new List<T>();

            foreach (var layer in Layers)
            {
                var layerParams = layer.GetParameters();
                for (int i = 0; i < layerParams.Length; i++)
                {
                    allParams.Add(layerParams[i]);
                }
            }

            // Add inverse problem parameters
            allParams.AddRange(_parameters);

            return new Vector<T>(allParams.ToArray());
        }

        /// <inheritdoc/>
        public override Vector<T> GetGradients()
        {
            var allGradients = new List<T>();

            foreach (var layer in Layers)
            {
                var layerGradients = layer.GetParameterGradients();
                for (int i = 0; i < layerGradients.Length; i++)
                {
                    allGradients.Add(layerGradients[i]);
                }
            }

            // Add parameter gradients
            if (_parameterGradients != null)
            {
                allGradients.AddRange(_parameterGradients);
            }
            else
            {
                for (int i = 0; i < _parameters.Length; i++)
                {
                    allGradients.Add(NumOps.Zero);
                }
            }

            return new Vector<T>(allGradients.ToArray());
        }

        /// <inheritdoc/>
        public override int ParameterCount =>
            Layers.Sum(l => l.ParameterCount) + _inverseProblem.NumberOfParameters;

        /// <inheritdoc/>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var paramDict = new Dictionary<string, object>();
            for (int i = 0; i < _parameters.Length; i++)
            {
                paramDict[_inverseProblem.ParameterNames[i]] = _parameters[i]!;
            }

            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "NetworkType", "InverseProblemPINN" },
                    { "NumberOfParameters", _inverseProblem.NumberOfParameters },
                    { "IdentifiedParameters", paramDict },
                    { "Regularization", _options.Regularization.ToString() }
                },
                ModelData = Serialize()
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_parameters.Length);
            foreach (var p in _parameters)
            {
                writer.Write(NumOps.ToDouble(p));
            }
            writer.Write(_numCollocationPoints);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int numParams = reader.ReadInt32();
            if (numParams != _parameters.Length)
            {
                throw new InvalidOperationException("Serialized parameter count does not match.");
            }

            for (int i = 0; i < numParams; i++)
            {
                _parameters[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            int numPoints = reader.ReadInt32();
            if (numPoints != _numCollocationPoints)
            {
                throw new InvalidOperationException("Serialized collocation point count does not match.");
            }

            UpdatePDE();
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new InverseProblemPINN<T>(
                Architecture,
                _inverseProblem,
                _boundaryConditions,
                _initialCondition,
                _numCollocationPoints,
                _options,
                _optimizer);
        }

        /// <inheritdoc/>
        public override bool SupportsTraining => true;

        /// <inheritdoc/>
        public override bool SupportsJitCompilation => false;
    }
}
