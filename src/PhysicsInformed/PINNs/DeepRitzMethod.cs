#pragma warning disable CS0649, CS0414, CS0169
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
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.Regression,
    ///     inputSize: 2, outputSize: 1);
    /// var deepRitz = new DeepRitzMethod&lt;float&gt;(architecture, DeepRitzEnergy&lt;float&gt;.Poisson());
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("The Deep Ritz Method: A Deep Learning-Based Numerical Method for Solving Variational Problems", "https://doi.org/10.1007/s40304-018-0127-z", Year = 2018, Authors = "Weinan E, Bing Yu")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    public partial class DeepRitzMethod<T> : NeuralNetworkBase<T>
    {
        private readonly DeepRitzMethodOptions _options;

        private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> _energyFunctional;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;
        private readonly int _numQuadraturePoints;
        private readonly Random _sampler;
        private Tensor<T> _quadraturePoints;

        /// <summary>
        /// Initializes a new instance of the Deep Ritz Method.
        /// </summary>
        /// <param name="architecture">The neural network architecture.</param>
        /// <param name="energyFunctional">
        /// The energy density E(x, u, grad u) for a batch of points: x [N, d], u [N, out] and grad u [N, out, d] to
        /// the density [N], with engine operations. <see cref="DeepRitzEnergy{T}"/> has the standard ones.
        /// </param>
        /// <param name="options">Quadrature and boundary-penalty settings.</param>
        /// <remarks>
        /// For Beginners:
        /// The energy functional should encode the physics of your problem.
        ///
        /// Example - Poisson Equation (-laplacian u = f):
        /// Energy: E(u) = 1/2 int |grad u|^2 dx - int f u dx
        /// Implementation: DeepRitzEnergy&lt;T&gt;.Poisson(f)
        ///
        /// The energy is integrated over the unit cube [0, 1]^d by Monte Carlo, with a fresh batch of points every
        /// training step, as in E and Yu (2018). The boundary condition is the penalty
        /// beta * mean |u - g|^2 on the boundary data passed to <c>Train(boundaryPoints, boundaryValues)</c>.
        /// </remarks>
        public DeepRitzMethod(NeuralNetworkArchitecture<T> architecture,
            Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> energyFunctional,
            DeepRitzMethodOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            options ??= new DeepRitzMethodOptions();
            options.Validate();
            _options = options;
            Options = _options;

            Guard.NotNull(energyFunctional);
            _energyFunctional = energyFunctional;
            _numQuadraturePoints = options.NumQuadraturePoints;
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = true;
            _sampler = RandomHelper.CreateSeededRandom(42);

            InitializeLayers();
            _quadraturePoints = SampleQuadraturePoints();
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
        /// Draws a fresh Monte Carlo batch of quadrature points in the unit cube [0, 1]^d.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Numerical integration approximates int f(x) dx ~ (1/N) sum f(x_i) for uniform x_i in the unit cube,
        /// simple and effective in high dimensions. E and Yu (2018) redraw the points at every step, which makes
        /// training stochastic gradient descent on the true energy rather than on one fixed sample.
        /// </remarks>
        private Tensor<T> SampleQuadraturePoints()
        {
            return Engine.TensorRandomUniformRange<T>(
                new[] { _numQuadraturePoints, Architecture.InputSize }, NumOps.Zero, NumOps.One, _sampler.Next());
        }

        /// <summary>
        /// The Monte Carlo estimate of the energy integral (1/N) sum_i E(x_i, u(x_i), grad u(x_i)) at the current
        /// quadrature points, with grad u taken through the gradient tape.
        /// </summary>
        /// <returns>The total energy value.</returns>
        /// <remarks>
        /// For Beginners:
        /// This is the key quantity the method minimizes: the gradient of this energy with respect to the network
        /// parameters tells us how to update the network.
        /// </remarks>
        public T ComputeTotalEnergy()
        {
            var energy = ComputeEnergy(_quadraturePoints);
            return energy.AsSpan()[0];
        }

        private Tensor<T> ComputeEnergy(Tensor<T> points)
        {
            var outputs = PinnAutodiff<T>.Evaluate(Forward, points, 1, out var derivatives);
            var gradients = derivatives.FirstDerivatives
                ?? throw new InvalidOperationException("The tape produced no input gradient.");
            var density = _energyFunctional(points, outputs, gradients);
            return Engine.ReduceMean(density, Enumerable.Range(0, density.Rank).ToArray(), keepDims: false);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The Deep Ritz loss of E and Yu (2018), Eq. 7: the energy integral at a freshly drawn batch of quadrature
        /// points plus beta * mean |u - g|^2 on the boundary data <paramref name="input"/>, <paramref name="target"/>.
        /// beta is <see cref="DeepRitzMethodOptions.BoundaryPenalty"/> (500, the paper's Poisson value).
        /// </remarks>
        protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> target)
        {
            _quadraturePoints = SampleQuadraturePoints();
            return new[]
            {
                new Training.TrainingObjective<T>(
                    new[] { input, target, _quadraturePoints },
                    data =>
                    {
                        var diff = Engine.TensorSubtract(Forward(data[0]), data[1]);
                        var squared = Engine.TensorMultiply(diff, diff);
                        var boundary = Engine.ReduceMean(squared, Enumerable.Range(0, squared.Rank).ToArray(), keepDims: false);
                        return Engine.TensorAdd(
                            ComputeEnergy(data[2]),
                            Engine.TensorMultiplyScalar(boundary, NumOps.FromDouble(_options.BoundaryPenalty)));
                    })
            };
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The energy differentiates the network with respect to its inputs inside the step, a nested backward the
        /// compile-once plan cannot replay; the eager tape re-runs it every step.
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
        /// Makes a prediction using the Deep Ritz network.
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
        /// Gets metadata about the Deep Ritz model.
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
                    { "BoundaryPenalty", _options.BoundaryPenalty },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes Deep Ritz-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes Deep Ritz-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;
    }
}


