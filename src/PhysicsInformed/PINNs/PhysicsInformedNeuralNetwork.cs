#pragma warning disable CS0649, CS0414, CS0169
using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Represents a Physics-Informed Neural Network (PINN) for solving PDEs.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// A Physics-Informed Neural Network (PINN) is a neural network that learns to solve
    /// Partial Differential Equations (PDEs) by incorporating physical laws directly into
    /// the training process.
    ///
    /// Traditional Approach (Finite Elements/Differences):
    /// - Discretize the domain into a grid
    /// - Approximate derivatives using neighboring points
    /// - Solve a large system of equations
    /// - Works well but can be slow for complex geometries
    ///
    /// PINN Approach:
    /// - Neural network represents the solution u(x,t)
    /// - Use automatic differentiation to compute ∂u/∂x, ∂²u/∂x², etc.
    /// - Train the network to minimize:
    ///   * PDE residual (how much the PDE is violated)
    ///   * Boundary condition errors
    ///   * Initial condition errors
    ///   * Data fitting errors (if measurements are available)
    ///
    /// Key Advantages:
    /// 1. Meshless: No need to discretize the domain
    /// 2. Data-efficient: Can work with sparse or noisy data
    /// 3. Flexible: Easy to handle complex geometries and boundary conditions
    /// 4. Interpolation: Get solution at any point by evaluating the network
    /// 5. Inverse problems: Can discover unknown parameters in the PDE
    ///
    /// Key Challenges:
    /// 1. Training can be difficult (multiple objectives to balance)
    /// 2. May require careful tuning of loss weights
    /// 3. Network architecture affects accuracy
    /// 4. Computational cost during training (many derivative evaluations)
    ///
    /// Applications:
    /// - Fluid dynamics (Navier-Stokes equations)
    /// - Heat transfer
    /// - Structural mechanics
    /// - Quantum mechanics
    /// - Financial modeling (Black-Scholes PDE)
    /// - Climate and weather modeling
    ///
    /// Historical Context:
    /// PINNs were introduced by Raissi, Perdikaris, and Karniadakis in 2019.
    /// They've revolutionized scientific machine learning by showing that deep learning
    /// can be guided by physics rather than just data.
    /// </remarks>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.Regression,
    ///     inputSize: 2, outputSize: 1);
    /// var pde = new HeatEquation&lt;float&gt;();
    /// // No IBoundaryCondition implementation ships with the library: write your own with
    /// // IsOnBoundary, ComputeBoundaryResidual and Name, then put instances in this array.
    /// var bc = Array.Empty&lt;IBoundaryCondition&lt;float&gt;&gt;();
    /// var pinn = new PhysicsInformedNeuralNetwork&lt;float&gt;(architecture, pde, bc);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", "https://doi.org/10.1016/j.jcp.2018.10.045", Year = 2019, Authors = "M. Raissi, P. Perdikaris, G.E. Karniadakis")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    public partial class PhysicsInformedNeuralNetwork<T> : NeuralNetworkBase<T>
    {
        private readonly PhysicsInformedNeuralNetworkOptions _options;

        /// <summary>
        /// The PDE specification that defines the physics constraints.
        /// Protected to allow derived classes (e.g., MultiFidelityPINN) to evaluate residuals on custom solutions.
        /// </summary>
        protected readonly IPDESpecification<T> _pdeSpecification;
        private readonly IBoundaryCondition<T>[] _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;
        private readonly PhysicsInformedLoss<T> _physicsLoss;

        // The boundary and initial-condition terms alone, for their per-point values and gradients.
        private readonly PhysicsInformedLoss<T> _constraintLoss;

        /// <summary>
        /// The weighted PINN loss terms; derived PINNs build their objectives from it.
        /// </summary>
        protected PhysicsInformedLoss<T> PhysicsLoss => _physicsLoss;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        // Collocation points [numPoints, inputDim] where the PDE residual is enforced
        private Tensor<T> _collocationPoints;
        private readonly int _numCollocationPoints;

        /// <summary>
        /// Initializes a new instance of the PINN class.
        /// </summary>
        /// <param name="architecture">The neural network architecture (typically a deep feedforward network).</param>
        /// <param name="pdeSpecification">The PDE that the solution must satisfy.</param>
        /// <param name="boundaryConditions">Boundary conditions for the problem.</param>
        /// <param name="initialCondition">Initial condition for time-dependent problems (optional).</param>
        /// <param name="optimizer">Optimization algorithm (Adam is recommended for PINNs).</param>
        /// <param name="options">The loss weights and collocation settings.</param>
        /// <remarks>
        /// For Beginners:
        /// When creating a PINN, you need to specify:
        /// 1. Network architecture: Usually a deep network (5-10 hidden layers, 20-50 neurons each)
        ///    - Activation: tanh or sin often work well for smooth solutions
        ///    - Input: spatial coordinates (x, y, z) and possibly time (t)
        ///    - Output: solution values u(x,t)
        ///
        /// 2. PDE specification: Defines the physics (e.g., Heat Equation, Navier-Stokes)
        ///
        /// 3. Boundary conditions: What happens at the edges of your domain
        ///
        /// 4. Collocation points: Where to enforce the PDE
        ///    - More points = better accuracy but slower training
        ///    - Typically 10,000-100,000 points
        ///    - Can use random sampling or quasi-random (Sobol, Latin hypercube)
        ///
        /// 5. Loss weights: Balance between different objectives
        ///    - Start with all weights = 1.0
        ///    - If PDE residual is large, increase pdeWeight
        ///    - If boundary conditions are violated, increase boundaryWeight
        ///    - This is often the trickiest part of PINN training!
        /// </remarks>
        public PhysicsInformedNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            IPDESpecification<T> pdeSpecification,
            IBoundaryCondition<T>[] boundaryConditions,
            IInitialCondition<T>? initialCondition = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            PhysicsInformedNeuralNetworkOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _options = options ?? new PhysicsInformedNeuralNetworkOptions();
            _options.Validate();
            Options = _options;

            Guard.NotNull(pdeSpecification);
            _pdeSpecification = pdeSpecification;
            Guard.NotNull(boundaryConditions);
            _boundaryConditions = boundaryConditions;
            _initialCondition = initialCondition;
            _numCollocationPoints = _options.NumCollocationPoints;

            // Create the physics-informed loss function
            _physicsLoss = new PhysicsInformedLoss<T>(
                _pdeSpecification,
                _boundaryConditions,
                _initialCondition,
                _options.DataWeight,
                _options.PdeWeight,
                _options.BoundaryWeight,
                _options.InitialWeight);
            _constraintLoss = new PhysicsInformedLoss<T>(
                null,
                _boundaryConditions,
                _initialCondition,
                0.0,
                0.0,
                _options.BoundaryWeight,
                _options.InitialWeight);

            if (optimizer == null)
            {
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
                _usesDefaultOptimizer = true;
            }
            else
            {
                _optimizer = optimizer;
                _usesDefaultOptimizer = false;
            }

            InitializeLayers();
            _collocationPoints = GenerateCollocationPoints();
        }

        /// <summary>
        /// Initializes the neural network layers.
        /// </summary>
        protected override void InitializeLayers()
        {
            // Use custom layers if provided in architecture
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                // Create default deep feedforward architecture
                // For PINNs, we typically want deeper networks with moderate width
                Layers.AddRange(LayerHelper<T>.CreateDefaultPINNLayers(Architecture));
            }
        }

        /// <summary>
        /// Generates collocation points for enforcing the PDE in the domain.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Collocation points are locations in the domain where we enforce the PDE.
        /// Think of them as "checkpoints" where the neural network must satisfy the physics.
        ///
        /// Sampling Strategies:
        /// 1. Uniform grid: Simple but can miss important regions
        /// 2. Random sampling: Used here - good coverage, easy to implement
        /// 3. Latin Hypercube: Better space-filling properties
        /// 4. Adaptive sampling: Add more points where error is high
        ///
        /// For this implementation, we use random sampling in the unit hypercube [0,1]^d.
        /// Use <see cref="SetCollocationPoints"/> to sample your actual domain (e.g., x in [-1, 1], t in [0, 10]).
        /// </remarks>
        private Tensor<T> GenerateCollocationPoints()
        {
            int inputDim = _pdeSpecification.InputDimension;
            var random = RandomHelper.CreateSeededRandom(42); // Fixed seed for reproducibility
            return Engine.TensorRandomUniformRange<T>(
                new[] { _numCollocationPoints, inputDim }, NumOps.Zero, NumOps.One, random.Next());
        }

        /// <summary>
        /// Sets custom collocation points (for advanced users who want specific sampling).
        /// </summary>
        /// <param name="points">Collocation points [numPoints, inputDim].</param>
        public void SetCollocationPoints(Tensor<T> points)
        {
            Guard.NotNull(points);
            if (points.Rank != 2 || points.Shape[1] != _pdeSpecification.InputDimension)
            {
                throw new ArgumentException(
                    $"Collocation points must be a [numPoints, {_pdeSpecification.InputDimension}] tensor.");
            }

            _collocationPoints = points;
        }

        /// <summary>
        /// Gets the collocation points the PDE residual is enforced at, [numPoints, inputDim].
        /// </summary>
        public Tensor<T> CollocationPoints => _collocationPoints;

        /// <summary>
        /// The input-derivative order the PDE residual needs.
        /// </summary>
        protected int PdeDerivativeOrder =>
            _pdeSpecification is PDEs.PDESpecificationBase<T> tapeSpec ? tapeSpec.DerivativeOrder : 2;

        /// <inheritdoc/>
        /// <remarks>
        /// <para>
        /// Raissi et al. (2019), Eq. 4: the loss is MSE_u + MSE_f. MSE_u fits the network to the training data passed
        /// to <c>Train</c> (the paper's initial and boundary samples, and any measurements), and MSE_f is the mean
        /// squared PDE residual at the collocation points, with the derivatives of u taken through the gradient
        /// tape. Both terms are weighted by the options' DataWeight and PdeWeight. The configured boundary and initial
        /// conditions add their residual terms, weighted by BoundaryWeight and InitialWeight, at every data or
        /// collocation point they constrain (see <see cref="ComputeConstraintLoss"/>).
        /// </para>
        /// <para>
        /// The PDE term needs a residual written with engine operations
        /// (<see cref="PDEs.PDESpecificationBase{T}.ComputeTapeResidual"/>); every PDE in the library has one.
        /// </para>
        /// </remarks>
        /// <summary>
        /// Replaces the default Adam optimizer with one at <paramref name="learningRate"/>. A caller-supplied optimizer is
        /// kept unchanged.
        /// </summary>
        private protected void UseDefaultOptimizerLearningRate(double learningRate)
        {
            if (double.IsNaN(learningRate) || double.IsInfinity(learningRate) || learningRate <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be finite and positive.");
            }

            if (_usesDefaultOptimizer)
            {
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                    this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = learningRate });
            }
        }

        protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> target)
        {
            if (_pdeSpecification is not PDEs.PDESpecificationBase<T>)
            {
                throw new NotSupportedException(
                    $"PINN training needs the PDE residual on the gradient tape; '{_pdeSpecification.Name}' does not " +
                    "derive from PDESpecificationBase, so it has no ComputeTapeResidual.");
            }

            return new[]
            {
                new Training.TrainingObjective<T>(
                    new[] { input, target, _collocationPoints },
                    data => ComputeTrainingLoss(data[0], data[1], data[2]))
            };
        }

        /// <summary>
        /// The PINN objective MSE_u + MSE_f on the gradient tape.
        /// </summary>
        /// <param name="dataInputs">The data points [N_u, inputDim].</param>
        /// <param name="dataTargets">The data values [N_u, outputDim].</param>
        /// <param name="collocationPoints">The collocation points [N_f, inputDim].</param>
        /// <returns>The scalar loss.</returns>
        protected virtual Tensor<T> ComputeTrainingLoss(Tensor<T> dataInputs, Tensor<T> dataTargets, Tensor<T> collocationPoints)
        {
            var loss = _physicsLoss.ComputeTapeLoss(Forward(dataInputs), dataTargets);
            var constraints = ComputeConstraintLoss(dataInputs, collocationPoints);
            if (constraints is not null)
            {
                loss = Engine.TensorAdd(loss, Engine.Reshape(constraints, loss.Shape.ToArray()));
            }

            if (_options.PdeWeight == 0.0)
            {
                return loss;
            }

            var outputs = PinnAutodiff<T>.Evaluate(Forward, collocationPoints, PdeDerivativeOrder, out var derivatives);
            return Engine.TensorAdd(loss, _physicsLoss.ComputeTapePdeLoss(collocationPoints, outputs, derivatives));
        }

        /// <summary>
        /// The configured boundary and initial-condition terms, BoundaryWeight * mean(r_bc^2) + InitialWeight *
        /// mean((u - u_0)^2), over the data and collocation points that lie on a boundary or at the initial time, on the
        /// gradient tape; null when no condition is configured or no point is constrained.
        /// </summary>
        /// <remarks>
        /// <para>
        /// A condition's residual is a host callback over one point's values (see <see cref="IBoundaryCondition{T}"/>),
        /// so it cannot be recorded itself. Its value and its gradient with respect to the point's outputs and
        /// derivatives come from <see cref="PhysicsInformedLoss{T}.ComputePhysicsLossGradients"/> (the condition's
        /// analytic gradient, or central differences of the callback), and the tape term is the first-order expansion
        /// <c>sum_p [l_p + g_p . (y_p - y_p*)]</c> around the current values <c>y_p*</c>: its value is the loss, and its
        /// gradient through the outputs and the tape derivatives reaches the weights. For the Dirichlet, Neumann and
        /// initial conditions the residual is linear in y, so the expansion is exact.
        /// </para>
        /// <para>
        /// Raissi et al. (2019) fit the initial and boundary data as part of MSE_u; the objective used to drop these
        /// conditions, so BoundaryWeight and InitialWeight had no effect.
        /// </para>
        /// </remarks>
        private Tensor<T>? ComputeConstraintLoss(Tensor<T> dataInputs, Tensor<T> collocationPoints)
        {
            if (_boundaryConditions.Length == 0 && _initialCondition is null)
            {
                return null;
            }

            var candidates = Engine.TensorConcatenate(new[] { dataInputs, collocationPoints }, 0);
            int inputDim = candidates.Shape[1];
            var hostPoints = candidates.ToArray();
            var rows = new List<int>();
            for (int p = 0; p < candidates.Shape[0]; p++)
            {
                var point = new Vector<T>(new ArraySegment<T>(hostPoints, p * inputDim, inputDim).ToArray());
                if (_boundaryConditions.Any(bc => bc.IsOnBoundary(point))
                    || (_initialCondition is not null && _initialCondition.IsAtInitialTime(point)))
                {
                    rows.Add(p);
                }
            }

            if (rows.Count == 0)
            {
                return null;
            }

            var points = Engine.TensorIndexSelect(candidates, Vector<int>.Wrap(rows.ToArray()).AsTensor(), 0);
            int order = Math.Min(2, Math.Max(1, PdeDerivativeOrder));
            var outputs = PinnAutodiff<T>.Evaluate(Forward, points, order, out var derivatives);
            int count = rows.Count;
            int outputDim = outputs.Shape[1];
            var first = derivatives.FirstDerivatives;
            var second = order >= 2 ? derivatives.SecondDerivatives : null;

            var hostInputs = points.ToArray();
            var hostOutputs = outputs.ToArray();
            var hostFirst = first?.ToArray();
            var hostSecond = second?.ToArray();
            int firstSize = outputDim * inputDim;
            int secondSize = firstSize * inputDim;
            var outputGradients = new T[hostOutputs.Length];
            var firstGradients = hostFirst is null ? null : new T[hostFirst.Length];
            var secondGradients = hostSecond is null ? null : new T[hostSecond.Length];
            var losses = new T[count];
            for (int p = 0; p < count; p++)
            {
                var pointDerivatives = new PDEDerivatives<T>
                {
                    FirstDerivatives = hostFirst is null ? null : new Tensor<T>(
                        new ArraySegment<T>(hostFirst, p * firstSize, firstSize).ToArray(), new[] { outputDim, inputDim }),
                    SecondDerivatives = hostSecond is null ? null : new Tensor<T>(
                        new ArraySegment<T>(hostSecond, p * secondSize, secondSize).ToArray(),
                        new[] { outputDim, inputDim, inputDim }),
                };
                var gradient = _constraintLoss.ComputePhysicsLossGradients(
                    new Vector<T>(new ArraySegment<T>(hostOutputs, p * outputDim, outputDim).ToArray()),
                    null,
                    pointDerivatives,
                    new Vector<T>(new ArraySegment<T>(hostInputs, p * inputDim, inputDim).ToArray()));
                losses[p] = gradient.Loss;
                gradient.OutputGradients.ToArray().CopyTo(outputGradients, p * outputDim);
                if (firstGradients is not null)
                {
                    gradient.FirstDerivatives.ToArray().CopyTo(firstGradients, p * firstSize);
                }

                if (secondGradients is not null)
                {
                    gradient.SecondDerivatives.ToArray().CopyTo(secondGradients, p * secondSize);
                }
            }

            // sum_p g_p . y_p on the tape, and the constant sum_p (l_p - g_p . y_p*) that makes its value the loss.
            var outputWeights = new Tensor<T>(outputGradients, outputs.Shape.ToArray());
            var linear = Engine.ReduceSum(Engine.TensorMultiply(outputWeights, outputs), null);
            var anchor = Engine.ReduceSum(Engine.TensorMultiply(outputWeights, new Tensor<T>(hostOutputs, outputs.Shape.ToArray())), null);
            if (first is not null && hostFirst is not null && firstGradients is not null)
            {
                var weights = new Tensor<T>(firstGradients, first.Shape.ToArray());
                linear = Engine.TensorAdd(linear, Engine.ReduceSum(Engine.TensorMultiply(weights, first), null));
                anchor = Engine.TensorAdd(anchor, Engine.ReduceSum(
                    Engine.TensorMultiply(weights, new Tensor<T>(hostFirst, first.Shape.ToArray())), null));
            }

            if (second is not null && hostSecond is not null && secondGradients is not null)
            {
                var weights = new Tensor<T>(secondGradients, second.Shape.ToArray());
                linear = Engine.TensorAdd(linear, Engine.ReduceSum(Engine.TensorMultiply(weights, second), null));
                anchor = Engine.TensorAdd(anchor, Engine.ReduceSum(
                    Engine.TensorMultiply(weights, new Tensor<T>(hostSecond, second.Shape.ToArray())), null));
            }

            var lossSum = Engine.ReduceSum(new Tensor<T>(losses, new[] { count }), null);
            var total = Engine.TensorAdd(linear, Engine.TensorSubtract(lossSum, anchor));
            return Engine.TensorMultiplyScalar(total, NumOps.FromDouble(1.0 / count));
        }

        /// <summary>
        /// The weighted residual term pde_weight * mean(f^2) of this network at <paramref name="points"/>.
        /// </summary>
        internal Tensor<T> ComputePdeLossAt(Tensor<T> points)
        {
            var outputs = PinnAutodiff<T>.Evaluate(Forward, points, PdeDerivativeOrder, out var derivatives);
            return _physicsLoss.ComputeTapePdeLoss(points, outputs, derivatives);
        }

        /// <summary>
        /// The PDE residual f of this network at <paramref name="points"/>, [N], on the gradient tape.
        /// </summary>
        internal Tensor<T> ComputeResidualAt(Tensor<T> points, out Tensor<T> outputs)
        {
            if (_pdeSpecification is not PDEs.PDESpecificationBase<T> tapeSpec)
            {
                throw new NotSupportedException(
                    $"'{_pdeSpecification.Name}' has no tape-differentiable residual (PDESpecificationBase.ComputeTapeResidual).");
            }

            outputs = PinnAutodiff<T>.Evaluate(Forward, points, tapeSpec.DerivativeOrder, out var derivatives);
            return tapeSpec.ComputeTapeResidual(points, outputs, derivatives);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The PDE term differentiates the network with respect to its inputs inside the step, a nested backward the
        /// compile-once plan cannot replay; the eager tape re-runs it every step.
        /// </remarks>
        protected override bool SupportsFusedCompiledTraining => false;

        /// <summary>
        /// Performs a forward pass through the network.
        /// </summary>
        /// <param name="input">Input tensor for evaluation.</param>
        /// <returns>Network output tensor.</returns>
        /// <remarks>
        /// Virtual so a composite PINN (DomainDecompositionPINN) can define its solution once and have
        /// Predict, GetSolution and the PDE residual all evaluate it.
        /// </remarks>
        public virtual Tensor<T> Forward(Tensor<T> input)
        {
            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        /// <summary>
        /// Gets the solution at a specific point in the domain.
        /// </summary>
        /// <param name="point">The point coordinates (x, y, t, etc.).</param>
        /// <returns>The solution value(s) at that point.</returns>
        public Vector<T> GetSolution(Vector<T> point)
        {
            Guard.NotNull(point);
            var output = Forward(Engine.Reshape(Tensor<T>.FromVector(point), new[] { 1, point.Length }));
            return output.ToVector();
        }

        /// <summary>
        /// Evaluates the PDE residual at a point (for validation).
        /// </summary>
        /// <param name="point">The point coordinates.</param>
        /// <returns>The PDE residual (should be close to zero for a good solution).</returns>
        /// <remarks>The derivatives of the network are exact, taken through the gradient tape.</remarks>
        public T EvaluatePDEResidual(Vector<T> point)
        {
            Guard.NotNull(point);
            var batch = Engine.Reshape(Tensor<T>.FromVector(point), new[] { 1, point.Length });
            var outputs = PinnAutodiff<T>.Evaluate(Forward, batch, PdeDerivativeOrder, out var derivatives);
            return _pdeSpecification.ComputeResidual(point, outputs.ToVector(), PointDerivatives(derivatives));
        }

        /// <summary>
        /// The derivatives of one point from a batch of one: drops the leading batch axis.
        /// </summary>
        /// <param name="batched">Batched derivatives with a batch of one.</param>
        /// <returns>The point-wise derivatives ([out, in], [out, in, in], ...).</returns>
        protected PDEDerivatives<T> PointDerivatives(PDEDerivatives<T> batched)
        {
            return new PDEDerivatives<T>
            {
                FirstDerivatives = DropBatchAxis(batched.FirstDerivatives),
                SecondDerivatives = DropBatchAxis(batched.SecondDerivatives),
                ThirdDerivatives = DropBatchAxis(batched.ThirdDerivatives),
                HigherDerivatives = batched.HigherDerivatives
            };
        }

        private Tensor<T>? DropBatchAxis(Tensor<T>? tensor)
        {
            if (tensor is null)
            {
                return null;
            }

            if (tensor.Rank < 2 || tensor.Shape[0] != 1)
            {
                throw new ArgumentException("Expected derivatives for a batch of one point.");
            }

            return Engine.Reshape(tensor, tensor.Shape.ToArray().Skip(1).ToArray());
        }

        /// <summary>
        /// Sums two sets of PDE derivatives element-wise.
        /// Used by derived classes (e.g., MultiFidelityPINN) to compute derivatives of combined solutions.
        /// </summary>
        /// <param name="a">First set of derivatives.</param>
        /// <param name="b">Second set of derivatives.</param>
        /// <returns>Combined derivatives where each element is the sum of corresponding elements.</returns>
        protected PDEDerivatives<T> SumDerivatives(PDEDerivatives<T> a, PDEDerivatives<T> b)
        {
            return new PDEDerivatives<T>
            {
                FirstDerivatives = SumOrEither(a.FirstDerivatives, b.FirstDerivatives, "First"),
                SecondDerivatives = SumOrEither(a.SecondDerivatives, b.SecondDerivatives, "Second"),
                ThirdDerivatives = SumOrEither(a.ThirdDerivatives, b.ThirdDerivatives, "Third")
            };
        }

        private Tensor<T>? SumOrEither(Tensor<T>? a, Tensor<T>? b, string order)
        {
            if (a is null || b is null)
            {
                return a ?? b;
            }

            if (!a.Shape.ToArray().SequenceEqual(b.Shape.ToArray()))
            {
                throw new ArgumentException(
                    $"{order} derivatives dimension mismatch: a has [{string.Join(", ", a.Shape.ToArray())}], " +
                    $"b has [{string.Join(", ", b.Shape.ToArray())}].");
            }

            return Engine.TensorAdd(a, b);
        }

        /// <summary>
        /// Makes a prediction using the PINN.
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
        /// Gets metadata about the PINN model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "PDE", _pdeSpecification.Name },
                    { "InputDimension", _pdeSpecification.InputDimension },
                    { "OutputDimension", _pdeSpecification.OutputDimension },
                    { "BoundaryConditionCount", _boundaryConditions.Length },
                    { "HasInitialCondition", _initialCondition != null },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes PINN-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes PINN-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        /// <summary>
        /// Indicates whether this PINN supports training.
        /// </summary>
        public override bool SupportsTraining => true;
    }

}


