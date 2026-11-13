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
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;

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
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional, NeuralNetworkTaskType.Regression,
    ///     inputSize: 2, outputSize: 1);
    ///
    /// // What you measured: a temperature at each (x, t) you probed.
    /// var observations = new[]
    /// {
    ///     (location: new[] { 0.25, 0.1 }, value: new[] { 0.82 }),
    ///     (location: new[] { 0.50, 0.1 }, value: new[] { 0.95 }),
    ///     (location: new[] { 0.75, 0.1 }, value: new[] { 0.80 })
    /// };
    ///
    /// // Recover the conductivity that explains them, starting the search at 0.5.
    /// var problem = new HeatConductivityInverseProblem&lt;double&gt;(observations, initialGuess: 0.5);
    ///
    /// // Insulated at both ends: no heat crosses the boundary.
    /// var boundaryConditions = new IBoundaryCondition&lt;double&gt;[]
    /// {
    ///     new NeumannBoundaryCondition&lt;double&gt;(flux: 0.0, lowerBound: 0.0, upperBound: 1.0)
    /// };
    ///
    /// var pinn = new InverseProblemPINN&lt;double&gt;(architecture, problem, boundaryConditions);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelTask(ModelTask.CausalInference)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", "https://doi.org/10.1016/j.jcp.2018.10.045", Year = 2019, Authors = "M. Raissi, P. Perdikaris, G.E. Karniadakis")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    public partial class InverseProblemPINN<T> : NeuralNetworkBase<T>
    {
        private readonly IInverseProblem<T> _inverseProblem;
        private readonly IBoundaryCondition<T>[] _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;
        private readonly InverseProblemOptions<T> _options;

        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        // The unknown physical parameters being identified. A buffer: saved and counted, but updated by
        // Solve's own finite-difference step, never handed to the network's gradient tape or optimizer.
        [AiDotNet.Attributes.Buffer]
        private Vector<T> _parameters;

        // Solve's per-step finite-difference gradient of those parameters - workspace, rewritten every
        // step, so [Scratch]. As a [Buffer] it was counted as model state (ParameterCount changed after
        // the first Solve step), and while still null it left the layout FitDeferred, so every
        // parameter read threw before the model had ever been solved.
        [AiDotNet.Attributes.Scratch]
        private Vector<T>? _parameterGradients;

        // Current PDE with parameters applied
        private PDEs.PDESpecificationBase<T> _currentPDE;

        // Collocation points [N, inputDim] in the unit cube, and the observations as tensors.
        private readonly Tensor<T> _collocationPoints;
        private readonly int _numCollocationPoints;
        private readonly Tensor<T> _observationInputs;
        private readonly Tensor<T> _observationTargets;
        private readonly Random _batchSampler;

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
            InverseProblemOptions<T>? options = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            Guard.NotNull(inverseProblem);
            _inverseProblem = inverseProblem;
            Guard.NotNull(boundaryConditions);
            _boundaryConditions = boundaryConditions;
            _initialCondition = initialCondition;
            _options = options ?? new InverseProblemOptions<T>();
            _numCollocationPoints = _options.NumCollocationPoints;
            Options = _options;

            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            // Initialize unknown parameters with initial guesses
            _parameters = new Vector<T>(_inverseProblem.InitialParameterGuesses.ToArray());
            _parameterHistory = new List<T[]>();

            var observations = _inverseProblem.Observations;
            if (observations.Count == 0)
            {
                throw new ArgumentException("The inverse problem has no observations.", nameof(inverseProblem));
            }

            int inputDim = observations[0].location.Length;
            int outputDim = observations[0].value.Length;
            _observationInputs = new Tensor<T>(observations.SelectMany(o => o.location).ToArray(), new[] { observations.Count, inputDim });
            _observationTargets = new Tensor<T>(observations.SelectMany(o => o.value).ToArray(), new[] { observations.Count, outputDim });

            var random = RandomHelper.CreateSeededRandom(42);
            _collocationPoints = Engine.TensorRandomUniformRange<T>(
                new[] { _numCollocationPoints, inputDim }, NumOps.Zero, NumOps.One, random.Next());
            _batchSampler = RandomHelper.CreateSeededRandom(random.Next());

            InitializeLayers();
            _currentPDE = BuildPDE(_parameters);
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
        /// Updates the PDE with current parameter values.
        /// </summary>
        private void UpdatePDE()
        {
            _currentPDE = BuildPDE(_parameters);
        }

        private PDEs.PDESpecificationBase<T> BuildPDE(Vector<T> parameters)
        {
            var pde = _inverseProblem.CreateParameterizedPDE(parameters.ToArray());
            return pde as PDEs.PDESpecificationBase<T>
                ?? throw new NotSupportedException(
                    $"PDE '{pde.Name}' does not derive from PDESpecificationBase, so it has no tape residual to train on.");
        }

        /// <summary>
        /// Solves the inverse problem to identify unknown parameters.
        /// </summary>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate for the neural network.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <returns>Results including identified parameters and uncertainties.</returns>
        /// <remarks>
        /// Raissi et al. (2019), Sec. 4, train the network and the unknown coefficients together on
        /// MSE_u + MSE_f. Each epoch takes one gradient step for the network on that loss at a batch of
        /// collocation points, with the input derivatives from the gradient tape, then one step for the
        /// coefficients on MSE_f plus the regularization at the same points. The network step used to run the
        /// optimizer with no backward pass, so the network never trained, and the coefficient gradient was a
        /// finite difference of a loss resampled from a shared random source on every call.
        /// </remarks>
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
                T previousLoss = NumOps.MaxValue;
                int noImprovementCount = 0;
                const int patience = 50;

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    var (dataLoss, physicsLoss, totalLoss) = TrainEpoch();

                    if (_options.LogParameterHistory)
                    {
                        _parameterHistory.Add(_parameters.ToArray());
                    }

                    if (IsStalled(totalLoss, previousLoss))
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
                        Console.WriteLine("  Parameters: " + string.Join(" ",
                            _inverseProblem.ParameterNames.Zip(_parameters.ToArray(), (name, value) => $"{name}={value:G4}")));
                    }
                }

                var points = SampleBatch();
                var outputs = PinnAutodiff<T>.Evaluate(Forward, points, PdeOrder, out var derivatives);
                result.Parameters = _parameters.ToArray();
                result.DataLoss = Scalar(DataLoss());
                result.PhysicsLoss = Scalar(PhysicsLoss(_currentPDE, points, outputs, derivatives));
                result.TotalLoss = NumOps.Add(result.DataLoss, result.PhysicsLoss);

                if (_options.LogParameterHistory)
                {
                    result.ParameterHistory = _parameterHistory.ToList();
                }

                if (_options.EstimateUncertainty)
                {
                    var hessian = ParameterHessian(points, outputs, derivatives);
                    result.ParameterUncertainties = EstimateParameterUncertainties(hessian);
                    result.ParameterCorrelations = EstimateParameterCorrelations(hessian);
                }
            }
            finally
            {
                SetTrainingMode(false);
            }

            return result;
        }

        /// <summary>
        /// Whether the loss moved by less than 1e-8 since the previous epoch.
        /// </summary>
        private bool IsStalled(T totalLoss, T previousLoss)
        {
            return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(totalLoss, previousLoss)), NumOps.FromDouble(1e-8));
        }

        private int PdeOrder => Math.Max(1, _currentPDE.DerivativeOrder);

        private static T Scalar(Tensor<T> value) => value.AsSpan()[0];

        private Tensor<T> Forward(Tensor<T> input)
        {
            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        /// <summary>
        /// A batch of min(256, N) collocation points drawn from the seeded sampler.
        /// </summary>
        private Tensor<T> SampleBatch()
        {
            int numPoints = _collocationPoints.Shape[0];
            int batchSize = Math.Min(256, numPoints);
            int start = numPoints == batchSize ? 0 : _batchSampler.Next(numPoints - batchSize + 1);
            return Engine.TensorSlice(_collocationPoints, new[] { start, 0 }, new[] { batchSize, _collocationPoints.Shape[1] });
        }

        /// <summary>
        /// Performs one training epoch: a network step on MSE_u + MSE_f, then a coefficient step.
        /// </summary>
        /// <returns>The data loss, the physics loss and the total loss including the regularization.</returns>
        private (T dataLoss, T physicsLoss, T totalLoss) TrainEpoch()
        {
            var points = SampleBatch();
            var pde = _currentPDE;
            TrainCustomTapeStep(points, points, x => x, (x, _) =>
            {
                var outputs = PinnAutodiff<T>.Evaluate(Forward, x, PdeOrder, out var derivatives);
                return Engine.TensorAdd(DataLoss(), PhysicsLoss(pde, x, outputs, derivatives));
            }, _optimizer);

            var detachedOutputs = PinnAutodiff<T>.Evaluate(Forward, points, PdeOrder, out var detached);
            T dataLoss = Scalar(DataLoss());
            T physicsLoss = Scalar(PhysicsLoss(pde, points, detachedOutputs, detached));
            T regularizationLoss = Scalar(RegularizationLoss(Tensor<T>.FromVector(_parameters)));

            UpdateUnknownParameters(points, detachedOutputs, detached);
            UpdatePDE();

            return (dataLoss, physicsLoss, NumOps.Add(dataLoss, NumOps.Add(physicsLoss, regularizationLoss)));
        }

        /// <summary>
        /// The weighted data loss DataWeight * mean |u(x_obs) - u_obs|^2.
        /// </summary>
        private Tensor<T> DataLoss()
        {
            var diff = Engine.TensorSubtract(Forward(_observationInputs), _observationTargets);
            var mse = Engine.ReduceMean(Engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            return Engine.TensorMultiplyScalar(mse, _options.DataWeight ?? NumOps.One);
        }

        /// <summary>
        /// The physics loss mean R^2 of <paramref name="pde"/> at the collocation batch.
        /// </summary>
        private Tensor<T> PhysicsLoss(PDEs.PDESpecificationBase<T> pde, Tensor<T> points, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            var residual = pde.ComputeTapeResidual(points, outputs, derivatives);
            return Engine.ReduceMean(Engine.TensorMultiply(residual, residual), Enumerable.Range(0, residual.Rank).ToArray(), keepDims: false);
        }

        /// <summary>
        /// The regularization loss lambda * R(p) of the coefficient vector <paramref name="parameters"/>.
        /// </summary>
        private Tensor<T> RegularizationLoss(Tensor<T> parameters)
        {
            T lambda = _options.RegularizationStrength ?? NumOps.FromDouble(0.01);
            Tensor<T> penalty;
            switch (_options.Regularization)
            {
                case InverseProblemRegularization.L2Tikhonov:
                    penalty = Engine.TensorMultiply(parameters, parameters);
                    break;
                case InverseProblemRegularization.L1Lasso:
                    penalty = Engine.TensorAbs(parameters);
                    break;
                case InverseProblemRegularization.ElasticNet:
                    penalty = Engine.TensorAdd(Engine.TensorAbs(parameters), Engine.TensorMultiply(parameters, parameters));
                    break;
                case InverseProblemRegularization.Bayesian when _options.PriorMeans != null && _options.PriorStandardDeviations != null:
                    // Gaussian negative log-prior 1/2 ((p - mu) / sigma)^2, scaled before squaring.
                    var means = new Tensor<T>(_options.PriorMeans, new[] { _options.PriorMeans.Length });
                    var deviations = new Tensor<T>(_options.PriorStandardDeviations, new[] { _options.PriorStandardDeviations.Length });
                    var z = Engine.TensorDivide(Engine.TensorSubtract(parameters, means), deviations);
                    penalty = Engine.TensorMultiplyScalar(Engine.TensorMultiply(z, z), NumOps.FromDouble(0.5));
                    break;
                default:
                    return new Tensor<T>(new[] { 1 });
            }

            return Engine.TensorMultiplyScalar(Engine.ReduceSum(penalty, new[] { 0 }, keepDims: true), lambda);
        }

        /// <summary>
        /// The coefficient objective MSE_f(p) + lambda R(p) at fixed network outputs: the network does not depend
        /// on p, so only the PDE built from p changes.
        /// </summary>
        private T CoefficientLoss(Tensor<T> parameters, Tensor<T> points, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            var pde = BuildPDE(new Vector<T>(parameters.ToArray()));
            return Scalar(Engine.TensorAdd(PhysicsLoss(pde, points, outputs, derivatives), RegularizationLoss(parameters)));
        }

        /// <summary>
        /// Step sizes h_i = 1e-3 * max(1, |p_i|) for the central differences in the coefficients.
        /// </summary>
        private Tensor<T> StepSizes(Tensor<T> parameters)
        {
            return Engine.TensorMultiplyScalar(Engine.TensorMax(Engine.TensorAbs(parameters), NumOps.One), NumOps.FromDouble(1e-3));
        }

        private Tensor<T> Basis(Tensor<T> steps, int index)
        {
            int n = steps.Shape[0];
            var row = Engine.Reshape(Engine.TensorSlice(Engine.TensorEye<T>(n), new[] { index, 0 }, new[] { 1, n }), new[] { n });
            return Engine.TensorMultiply(row, steps);
        }

        /// <summary>
        /// Takes one gradient step on the coefficients, then clamps them to the problem's bounds.
        /// </summary>
        private void UpdateUnknownParameters(Tensor<T> points, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            var parameters = Tensor<T>.FromVector(_parameters);
            var steps = StepSizes(parameters);
            var plus = new List<T>(_parameters.Length);
            var minus = new List<T>(_parameters.Length);
            for (int i = 0; i < _parameters.Length; i++)
            {
                var offset = Basis(steps, i);
                plus.Add(CoefficientLoss(Engine.TensorAdd(parameters, offset), points, outputs, derivatives));
                minus.Add(CoefficientLoss(Engine.TensorSubtract(parameters, offset), points, outputs, derivatives));
            }

            var difference = Engine.TensorSubtract(Tensor<T>.FromVector(Vector<T>.FromList(plus)), Tensor<T>.FromVector(Vector<T>.FromList(minus)));
            var gradient = Engine.TensorDivide(difference, Engine.TensorMultiplyScalar(steps, NumOps.FromDouble(2.0)));
            _parameterGradients = gradient.ToVector();

            var updated = Engine.TensorSubtract(parameters, Engine.TensorMultiplyScalar(gradient, NumOps.FromDouble(_options.ParameterLearningRate)));
            if (_inverseProblem.ParameterLowerBounds is { } lower)
            {
                updated = Engine.TensorMax(updated, new Tensor<T>(lower, new[] { lower.Length }));
            }

            if (_inverseProblem.ParameterUpperBounds is { } upper)
            {
                updated = Engine.TensorMin(updated, new Tensor<T>(upper, new[] { upper.Length }));
            }

            _parameters = updated.ToVector();
        }

        /// <summary>
        /// The central-difference Hessian [n, n] of the coefficient objective at the current coefficients.
        /// </summary>
        private Tensor<T> ParameterHessian(Tensor<T> points, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            var parameters = Tensor<T>.FromVector(_parameters);
            var steps = StepSizes(parameters);
            int n = _parameters.Length;
            var corners = new List<T>(4 * n * n);
            for (int i = 0; i < n; i++)
            {
                var di = Basis(steps, i);
                for (int j = 0; j < n; j++)
                {
                    var dj = Basis(steps, j);
                    var up = Engine.TensorAdd(parameters, di);
                    var down = Engine.TensorSubtract(parameters, di);
                    corners.Add(CoefficientLoss(Engine.TensorAdd(up, dj), points, outputs, derivatives));
                    corners.Add(CoefficientLoss(Engine.TensorSubtract(up, dj), points, outputs, derivatives));
                    corners.Add(CoefficientLoss(Engine.TensorAdd(down, dj), points, outputs, derivatives));
                    corners.Add(CoefficientLoss(Engine.TensorSubtract(down, dj), points, outputs, derivatives));
                }
            }

            // H_ij = (L(++) - L(+-) - L(-+) + L(--)) / (4 h_i h_j).
            var values = Engine.Reshape(Tensor<T>.FromVector(Vector<T>.FromList(corners)), new[] { n * n, 4 });
            var signs = new Tensor<T>(new[] { NumOps.One, NumOps.Negate(NumOps.One), NumOps.Negate(NumOps.One), NumOps.One }, new[] { 1, 4 });
            var numerator = Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(values, signs), new[] { 1 }, keepDims: false), new[] { n, n });
            var column = Engine.Reshape(steps, new[] { n, 1 });
            var row = Engine.Reshape(steps, new[] { 1, n });
            var denominator = Engine.TensorMultiplyScalar(Engine.TensorMultiply(column, row), NumOps.FromDouble(4.0));
            return Engine.TensorDivide(numerator, denominator);
        }

        /// <summary>
        /// Laplace-approximation standard deviations 1 / sqrt(|H_ii|); a coefficient with zero curvature is
        /// unidentified and gets MaxValue.
        /// </summary>
        private T[] EstimateParameterUncertainties(Tensor<T> hessian)
        {
            var curvature = Engine.TensorAbs(Engine.TensorDiagonal(hessian));
            var zero = Engine.TensorEquals(curvature, NumOps.Zero);
            var ones = Tensor<T>.CreateDefault(curvature.Shape.ToArray(), NumOps.One);
            var deviations = Engine.TensorReciprocal(Engine.TensorSqrt(Engine.TensorWhere(zero, ones, curvature)));
            var unidentified = Tensor<T>.CreateDefault(curvature.Shape.ToArray(), NumOps.MaxValue);
            return Engine.TensorWhere(zero, unidentified, deviations).ToArray();
        }

        /// <summary>
        /// The coefficient correlations of the Laplace approximation: C = H^-1, corr_ij = C_ij / sqrt(C_ii C_jj).
        /// </summary>
        /// <remarks>
        /// This returned the identity, which claimed every pair of coefficients uncorrelated whatever the data.
        /// </remarks>
        private Tensor<T>? EstimateParameterCorrelations(Tensor<T> hessian)
        {
            int n = hessian.Shape[0];
            Tensor<T> covariance;
            try
            {
                covariance = Tensor<T>.FromMatrix(hessian.ToMatrix().Inverse());
            }
            catch (InvalidOperationException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"The coefficient Hessian is singular; no correlations: {ex}");
                return null;
            }

            var scale = Engine.TensorSqrt(Engine.TensorAbs(Engine.TensorDiagonal(covariance)));
            var outer = Engine.TensorMultiply(Engine.Reshape(scale, new[] { n, 1 }), Engine.Reshape(scale, new[] { 1, n }));
            return Engine.TensorDivide(covariance, outer);
        }

        /// <inheritdoc/>
        protected override Tensor<T> PredictCore(Tensor<T> input)
        {
            return Forward(input);
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

        // ParameterCount, GetParameters and UpdateParameters were all overridden here and are all
        // gone. Each walked Layers by hand and then appended or wrote _parameters; the base does
        // exactly that walk, and the GetExtraTrainableTensors below declares _parameters as part of
        // it -- layers first, then the coefficients, the same order GetGradients still uses.
        // Restoring the coefficients is now SetParameters' job, which is what serialization calls --
        // the deleted UpdateParameters was the only path that handled them, so a saved model came
        // back with its physical coefficients silently reset to their initial guesses.

        /// <inheritdoc/>
        public override Vector<T> GetGradients()
        {
            var parts = Layers.Select(layer => layer.GetParameterGradients()).ToList();
            parts.Add(_parameterGradients ?? new Vector<T>(_parameters.Length));
            return Vector<T>.Concatenate(parts.ToArray());
        }

        /// <inheritdoc />
        /// <remarks>
        /// The PDE is BUILT from the coefficients rather than reading them live, so restoring them
        /// without rebuilding it leaves the model computing with the old physics while reporting
        /// the new weights.
        /// <para>
        /// This used to live in an <c>UpdateParameters</c> override that reimplemented restore in
        /// full -- re-slicing the vector per layer, writing <c>_parameters</c> by hand, then
        /// rebuilding. <c>SetParameters</c>, which is what serialization actually calls, had no
        /// override at all: it walked only <c>Layers</c>, so a round-trip silently dropped every
        /// coefficient and left the PDE stale. One declaration above and this one rebuild replace
        /// all of it, and both restore paths now do the same thing because there is only one.
        /// </para>
        /// </remarks>
        protected override void OnParametersRestored() => UpdatePDE();

        /// <inheritdoc/>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "NetworkType", "InverseProblemPINN" },
                    { "NumberOfParameters", _inverseProblem.NumberOfParameters },
                    { "IdentifiedParameters", GetIdentifiedParameterSnapshot() },
                    { "Regularization", _options.Regularization.ToString() }
                }
            };

            metadata.SetModelDataProvider(() =>
            {
                byte[] modelData = Serialize();
                metadata.AdditionalInfo["IdentifiedParameters"] = GetIdentifiedParameterSnapshot();
                return modelData;
            });
            return metadata;
        }

        private Dictionary<string, object> GetIdentifiedParameterSnapshot()
        {
            var paramDict = new Dictionary<string, object>();
            var values = _parameters.ToArray();
            for (int i = 0; i < values.Length; i++)
            {
                object? value = values[i];
                if (value is null)
                {
                    throw new InvalidOperationException(
                        $"Identified parameter '{_inverseProblem.ParameterNames[i]}' cannot be null.");
                }
                paramDict[_inverseProblem.ParameterNames[i]] = value;
            }

            return paramDict;
        }

        /// <inheritdoc/>


        /// <inheritdoc/>


        /// <inheritdoc/>
        public override bool SupportsTraining => true;
    }
}
