#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.LearningRateSchedulers;
using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Implements Variational Physics-Informed Neural Networks (VPINNs).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Variational PINNs (VPINNs) use the weak (variational) formulation of PDEs instead of
    /// the strong form. This is similar to finite element methods (FEM).
    ///
    /// Strong vs. Weak Formulation:
    ///
    /// Strong Form (standard PINN):
    /// - PDE must hold pointwise: PDE(u) = 0 at every point
    /// - Example: -∇²u = f everywhere
    /// - Requires computing second derivatives
    /// - Solution must be twice differentiable
    ///
    /// Weak Form (VPINN):
    /// - PDE holds "on average" against test functions
    /// - ∫∇u·∇v dx = ∫fv dx for all test functions v
    /// - Integration by parts reduces derivative order
    /// - Solution only needs to be once differentiable
    /// - More stable numerically
    ///
    /// Key Advantages:
    /// 1. Lower derivative requirements (better numerical stability)
    /// 2. Natural incorporation of boundary conditions (through integration by parts)
    /// 3. Can handle discontinuities and rough solutions better
    /// 4. Closer to FEM (well-understood mathematical theory)
    /// 5. Often better convergence properties
    ///
    /// How VPINNs Work:
    /// 1. Choose test functions (often neural networks themselves)
    /// 2. Multiply PDE by test function and integrate
    /// 3. Use integration by parts to reduce derivative order
    /// 4. Minimize the residual in the weak sense
    ///
    /// Example - Poisson Equation:
    /// Strong: -∇²u = f
    /// Weak: ∫∇u·∇v dx = ∫fv dx (after integration by parts)
    ///
    /// VPINNs train the network u(x) to satisfy the weak form for all test functions v.
    ///
    /// Applications:
    /// - Same as PINNs, but particularly useful for:
    ///   * Problems with rough solutions
    ///   * Conservation laws
    ///   * Problems where weak solutions are more natural
    ///   * High-order PDEs (where reducing derivative order helps)
    ///
    /// Comparison with Standard PINNs:
    /// - VPINN: More stable, lower derivative requirements, closer to FEM
    /// - Standard PINN: Simpler to implement, direct enforcement of PDE
    ///
    /// The variational formulation often provides better training dynamics and accuracy.
    /// </remarks>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.Regression,
    ///     inputSize: 2, outputSize: 1);
    /// var vpinn = new VariationalPINN&lt;float&gt;(architecture, VariationalWeakForm&lt;float&gt;.Poisson(1.0f));
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Variational Physics-Informed Neural Networks For Solving Partial Differential Equations", "https://doi.org/10.48550/arXiv.1912.00873", Year = 2020, Authors = "Ehsan Kharazmi, Zhongqiang Zhang, George Em Karniadakis")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001,
                    Source = "Kharazmi et al. 2019, parameter table: the optimizer is Adam with a learning "
                            + "rate of 1e-3.")]
    public partial class VariationalPINN<T> : NeuralNetworkBase<T>
    {
        private readonly VariationalPINNOptions _options;

        private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> _weakFormResidual;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;
        private readonly Tensor<T> _quadraturePoints;
        private readonly Tensor<T> _quadratureWeights;
        [AiDotNet.Attributes.Scratch]
        private readonly Tensor<T>[] _testValues;
        [AiDotNet.Attributes.Scratch]
        private readonly Tensor<T>[] _testGradients;
        private readonly int _numTestFunctions;
        private readonly int _numQuadraturePoints;

        /// <summary>
        /// Initializes a new instance of the Variational PINN.
        /// </summary>
        /// <param name="architecture">The neural network architecture for the solution.</param>
        /// <param name="weakFormResidual">
        /// The weak form integrand R(x, u, grad u, v, grad v) for a batch of quadrature points: x [N, d], u [N, out],
        /// grad u [N, out, d], v [N] and grad v [N, d] to [N], with engine operations.
        /// <see cref="VariationalWeakForm{T}"/> has the standard ones.
        /// </param>
        /// <param name="options">Quadrature and test-function settings.</param>
        /// <remarks>
        /// For Beginners:
        /// The weak form residual should encode the variational formulation of your PDE.
        ///
        /// Example - Poisson Equation (-laplacian u = f):
        /// Weak form: int grad u . grad v dx - int f v dx = 0
        /// Implementation: VariationalWeakForm&lt;T&gt;.Poisson(f)
        ///
        /// As in Kharazmi et al. (2019), the integral is taken with tensor-product Gauss-Legendre quadrature on the
        /// unit cube, against the Legendre test functions v_k = P_{k+1} - P_{k-1}, which vanish on the boundary.
        /// The boundary condition comes from the data passed to <c>Train(boundaryPoints, boundaryValues)</c>.
        /// </remarks>
        public VariationalPINN(NeuralNetworkArchitecture<T> architecture,
            Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> weakFormResidual,
            VariationalPINNOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            options ??= new VariationalPINNOptions();
            options.Validate();
            _options = options;
            Options = _options;

            Guard.NotNull(weakFormResidual);
            _weakFormResidual = weakFormResidual;
            _numTestFunctions = options.NumTestFunctions;
            _optimizer = PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
                ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = true;

            InitializeLayers();
            int dimension = architecture.InputSize;
            int perAxis = Math.Max(2, (int)Math.Floor(Math.Pow(options.NumQuadraturePoints, 1.0 / dimension) + 1e-9));
            (_quadraturePoints, _quadratureWeights) = BuildQuadrature(perAxis, dimension);
            _numQuadraturePoints = _quadraturePoints.Shape[0];
            (_testValues, _testGradients) = BuildTestFunctions(_quadraturePoints, _numTestFunctions);
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultVariationalPINNLayers(Architecture));
            }
        }

        /// <summary>
        /// Gauss-Legendre nodes (ascending) and weights of <paramref name="count"/> points on [-1, 1], in T.
        /// </summary>
        /// <remarks>
        /// The rule is computed once in double by <see cref="GaussLegendreRule"/> and rounded to T once. Computed in T,
        /// P'_n at the outer nodes is near n (n + 1) / 2, so P'^2 overflows half precision from n = 16 or so and the
        /// outer weights became zero; the recurrence also held every P_0 .. P_n, count tensors of count values.
        /// </remarks>
        internal static Tensor<T> GaussLegendre(int count, out Tensor<T> weights)
        {
            var (nodes, rule) = GaussLegendreRule(count);
            weights = new Tensor<double>(rule, new[] { count }).Cast<T>();
            return new Tensor<double>(nodes, new[] { count }).Cast<T>();
        }

        /// <summary>
        /// Gauss-Legendre nodes (ascending) and weights of <paramref name="count"/> points on [-1, 1], in double.
        /// </summary>
        /// <remarks>
        /// Newton's method on P_count from the guesses -cos(pi (i + 3/4) / (count + 1/2)), with P_count and P_count-1 from
        /// Bonnet's recurrence (n + 1) P_{n+1} = (2n + 1) x P_n - n P_{n-1}, keeping only the last two terms. The weights
        /// are 2 / ((1 - x^2) P'_count(x)^2), with P'_n(x) = n (x P_n - P_{n-1}) / (x^2 - 1).
        /// </remarks>
        internal static (double[] Nodes, double[] Weights) GaussLegendreRule(int count)
        {
            if (count < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(count), "A Gauss-Legendre rule needs at least one point.");
            }

            var nodes = new double[count];
            var weights = new double[count];
            for (int i = 0; i < count; i++)
            {
                double x = -Math.Cos(Math.PI * (i + 0.75) / (count + 0.5));
                double derivative = 1.0;
                for (int iteration = 0; iteration < 20; iteration++)
                {
                    double value = LegendrePair(x, count, out double previous);
                    derivative = count * (x * value - previous) / (x * x - 1.0);
                    double delta = value / derivative;
                    x -= delta;
                    if (Math.Abs(delta) <= 4e-16 * Math.Max(1.0, Math.Abs(x)))
                    {
                        break;
                    }
                }

                double finalValue = LegendrePair(x, count, out double finalPrevious);
                derivative = count * (x * finalValue - finalPrevious) / (x * x - 1.0);
                nodes[i] = x;
                weights[i] = 2.0 / ((1.0 - x * x) * derivative * derivative);
            }

            return (nodes, weights);
        }

        /// <summary>P_n(x), with P_{n-1}(x) in <paramref name="previous"/>, by the recurrence with two terms kept.</summary>
        private static double LegendrePair(double x, int n, out double previous)
        {
            double lower = 1.0;
            double current = x;
            if (n == 0)
            {
                previous = 0.0;
                return 1.0;
            }

            for (int k = 1; k < n; k++)
            {
                double next = ((2 * k + 1) * x * current - k * lower) / (k + 1);
                lower = current;
                current = next;
            }

            previous = lower;
            return current;
        }

        /// <summary>
        /// P_0 .. P_max at <paramref name="x"/> by Bonnet's recurrence (n + 1) P_{n+1} = (2n + 1) x P_n - n P_{n-1}.
        /// </summary>
        private Tensor<T>[] Legendre(Tensor<T> x, int max)
        {
            var polynomials = new Tensor<T>[max + 1];
            polynomials[0] = Tensor<T>.CreateDefault(x.Shape.ToArray(), NumOps.One);
            if (max >= 1)
            {
                polynomials[1] = x;
            }

            if (max < 2)
            {
                return polynomials;
            }

            // (2n + 1) / (n + 1) = 2 - 1 / (n + 1) and n / (n + 1) = 1 - 1 / (n + 1) for n = 1 .. max - 1.
            T one = NumOps.One;
            var reciprocal = Engine.TensorReciprocal(Engine.TensorAddScalar(Engine.TensorArange<T>(1, max, 1), one));
            var negated = Engine.TensorNegate(reciprocal);
            var leadCoefficients = Engine.TensorAddScalar(negated, NumOps.FromDouble(2.0)).ToArray();
            var lagCoefficients = Engine.TensorAddScalar(negated, one).ToArray();
            for (int n = 1; n < max; n++)
            {
                var lead = Engine.TensorMultiplyScalar(Engine.TensorMultiply(x, polynomials[n]), leadCoefficients[n - 1]);
                polynomials[n + 1] = Engine.TensorSubtract(lead, Engine.TensorMultiplyScalar(polynomials[n - 1], lagCoefficients[n - 1]));
            }

            return polynomials;
        }

        /// <summary>
        /// The tensor-product Gauss-Legendre rule on the unit cube [0, 1]^d: points [count^d, d] and weights
        /// [count^d] that sum to 1.
        /// </summary>
        private (Tensor<T> Points, Tensor<T> Weights) BuildQuadrature(int count, int dimension)
        {
            var nodes = GaussLegendre(count, out var weights);
            var unitNodes = Engine.TensorMultiplyScalar(Engine.TensorAddScalar(nodes, NumOps.One), NumOps.FromDouble(0.5));
            var unitWeights = Engine.TensorMultiplyScalar(weights, NumOps.FromDouble(0.5));

            var gridShape = Enumerable.Repeat(count, dimension).ToArray();
            int total = gridShape.Aggregate(1, (a, b) => a * b);
            var ones = Tensor<T>.CreateDefault(gridShape, NumOps.One);
            var columns = new Tensor<T>[dimension];
            Tensor<T> gridWeights = ones;
            for (int axis = 0; axis < dimension; axis++)
            {
                var axisShape = Enumerable.Repeat(1, dimension).ToArray();
                axisShape[axis] = count;
                columns[axis] = Engine.Reshape(Engine.TensorMultiply(ones, Engine.Reshape(unitNodes, axisShape)), new[] { total });
                gridWeights = Engine.TensorMultiply(gridWeights, Engine.Reshape(unitWeights, axisShape));
            }

            return (Engine.TensorStack(columns, 1), Engine.Reshape(gridWeights, new[] { total }));
        }

        /// <summary>
        /// The test functions of Kharazmi et al. (2019), Eq. 13, at the quadrature points: the tensor products
        /// v_k(x) = prod_j (P_{k_j+1}(xi_j) - P_{k_j-1}(xi_j)) with xi = 2x - 1 and k_j &gt;= 1, so every v_k
        /// vanishes on the boundary of the unit cube, and their gradients, with
        /// d/dx (P_{k+1} - P_{k-1})(2x - 1) = 2 (2k + 1) P_k(2x - 1).
        /// </summary>
        private (Tensor<T>[] Values, Tensor<T>[] Gradients) BuildTestFunctions(Tensor<T> points, int count)
        {
            int dimension = points.Shape[1];
            int perAxis = 1;
            while (Math.Pow(perAxis, dimension) < count)
            {
                perAxis++;
            }

            // 2 (2k + 1) for k = 1 .. perAxis, at index k - 1.
            T one = NumOps.One;
            T two = NumOps.FromDouble(2.0);
            var slopeScales = Engine.TensorAddScalar(
                Engine.TensorMultiplyScalar(Engine.TensorArange<T>(1, perAxis + 1, 1), NumOps.FromDouble(4.0)), two).ToArray();
            var factors = new Tensor<T>[dimension][];
            var slopes = new Tensor<T>[dimension][];
            for (int axis = 0; axis < dimension; axis++)
            {
                var xi = Engine.TensorSubtractScalar(
                    Engine.TensorMultiplyScalar(PinnAutodiff<T>.Column(Engine, points, axis), two), one);
                var polynomials = Legendre(xi, perAxis + 1);
                factors[axis] = new Tensor<T>[perAxis + 1];
                slopes[axis] = new Tensor<T>[perAxis + 1];
                for (int k = 1; k <= perAxis; k++)
                {
                    factors[axis][k] = Engine.TensorSubtract(polynomials[k + 1], polynomials[k - 1]);
                    slopes[axis][k] = Engine.TensorMultiplyScalar(polynomials[k], slopeScales[k - 1]);
                }
            }

            var values = new Tensor<T>[count];
            var gradients = new Tensor<T>[count];
            for (int index = 0; index < count; index++)
            {
                var orders = new int[dimension];
                int remaining = index;
                for (int axis = 0; axis < dimension; axis++)
                {
                    orders[axis] = 1 + remaining % perAxis;
                    remaining /= perAxis;
                }

                var value = factors[0][orders[0]];
                for (int axis = 1; axis < dimension; axis++)
                {
                    value = Engine.TensorMultiply(value, factors[axis][orders[axis]]);
                }

                var partials = new Tensor<T>[dimension];
                for (int axis = 0; axis < dimension; axis++)
                {
                    var partial = slopes[axis][orders[axis]];
                    for (int other = 0; other < dimension; other++)
                    {
                        if (other != axis)
                        {
                            partial = Engine.TensorMultiply(partial, factors[other][orders[other]]);
                        }
                    }

                    partials[axis] = partial;
                }

                values[index] = value;
                gradients[index] = Engine.TensorStack(partials, 1);
            }

            return (values, gradients);
        }

        /// <summary>
        /// Computes the weak form residual R_k = int R(x, u, grad u, v_k, grad v_k) dx by quadrature.
        /// </summary>
        /// <param name="testFunctionIndex">Index of the test function to use.</param>
        /// <returns>The weak residual (should be zero for a perfect solution).</returns>
        /// <remarks>
        /// For Beginners:
        /// For a true solution, this integral is zero for ALL test functions. We approximate "all" by a finite set of
        /// Legendre test functions that vanish on the boundary, so integration by parts leaves no boundary term.
        /// </remarks>
        public T ComputeWeakResidual(int testFunctionIndex)
        {
            if (testFunctionIndex < 0 || testFunctionIndex >= _numTestFunctions)
            {
                throw new ArgumentOutOfRangeException(nameof(testFunctionIndex));
            }

            var residuals = ComputeWeakResiduals(_quadraturePoints);
            return residuals.AsSpan()[testFunctionIndex];
        }

        /// <summary>
        /// All weak residuals [K] at the quadrature points, with grad u through the gradient tape.
        /// </summary>
        private Tensor<T> ComputeWeakResiduals(Tensor<T> points)
        {
            var outputs = PinnAutodiff<T>.Evaluate(Forward, points, 1, out var derivatives);
            var gradients = derivatives.FirstDerivatives
                ?? throw new InvalidOperationException("The tape produced no input gradient.");

            var residuals = new Tensor<T>[_numTestFunctions];
            for (int k = 0; k < _numTestFunctions; k++)
            {
                var integrand = _weakFormResidual(points, outputs, gradients, _testValues[k], _testGradients[k]);
                residuals[k] = Engine.ReduceSum(
                    Engine.TensorMultiply(Engine.Reshape(integrand, new[] { _numQuadraturePoints }), _quadratureWeights),
                    new[] { 0 }, keepDims: true);
            }

            return Engine.TensorConcatenate(residuals, 0);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The VPINN loss of Kharazmi et al. (2019), Eq. 11: (1/K) sum_k R_k^2 over the test functions plus the mean
        /// squared error on the boundary data <paramref name="input"/>, <paramref name="target"/>.
        /// </remarks>
        protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> target)
        {
            return new[]
            {
                new Training.TrainingObjective<T>(
                    new[] { input, target, _quadraturePoints },
                    data =>
                    {
                        var diff = Engine.TensorSubtract(Forward(data[0]), data[1]);
                        var squared = Engine.TensorMultiply(diff, diff);
                        var boundary = Engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), keepDims: false);
                        var residuals = ComputeWeakResiduals(data[2]);
                        var variational = Engine.ReduceMean(Engine.TensorMultiply(residuals, residuals), new[] { 0 }, keepDims: false);
                        return Engine.TensorAdd(variational, boundary);
                    })
            };
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The weak residual differentiates the network with respect to its inputs inside the step, a nested backward
        /// the compile-once plan cannot replay; the eager tape re-runs it every step.
        /// </remarks>
        protected override bool SupportsFusedCompiledTraining => false;

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
        /// Gets the solution at a specific point.
        /// </summary>
        public Vector<T> GetSolution(Vector<T> point)
        {
            Guard.NotNull(point);
            return Forward(Engine.Reshape(Tensor<T>.FromVector(point), new[] { 1, point.Length })).ToVector();
        }

        /// <summary>
        /// Makes a prediction using the VPINN.
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
        /// Gets metadata about the VPINN model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "InputSize", Architecture.InputSize },
                    { "OutputSize", Architecture.OutputSize },
                    { "QuadraturePoints", _numQuadraturePoints },
                    { "TestFunctions", _numTestFunctions },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes VPINN-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes VPINN-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;
    }
}


