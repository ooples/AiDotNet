using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Policy using Beta distribution for bounded continuous action spaces.
    /// Network outputs alpha and beta parameters for each action dimension.
    /// Actions are naturally bounded to [0, 1] and can be scaled to any [min, max] range.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create a Beta policy for bounded continuous actions in [0, 1]
    /// var network = new NeuralNetwork&lt;double&gt;();
    /// var policy = new BetaPolicy&lt;double&gt;(network, actionSize: 2, new EpsilonGreedyExploration&lt;double&gt;());
    ///
    /// // Sample an action bounded between 0 and 1
    /// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
    /// var action = policy.SelectAction(state, training: true);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    // Citation URL corrected. arXiv 1707.04544 is "Flavor-singlet charm pentaquark" — an unrelated
    // particle-physics paper. This work appeared at ICML 2017 (PMLR v70, pp. 834-843) and is not on
    // arXiv, so the canonical proceedings URL replaces the invented preprint id.
    [ResearchPaper("Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution",
        "https://proceedings.mlr.press/v70/chou17a.html",
        Year = 2017,
        Authors = "Pei-Wen Chou, Daniel Maturana, Sebastian Scherer")]
    public partial class BetaPolicy<T> : PolicyBase<T>
    {
        private readonly INeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly double _actionMin;
        private readonly double _actionMax;

        /// <summary>
        /// Initializes a new instance of the BetaPolicy class.
        /// </summary>
        /// <param name="policyNetwork">Network that outputs alpha and beta parameters (2 * actionSize outputs).</param>
        /// <summary>
        /// Initializes a new instance with default settings.
        /// </summary>
        public BetaPolicy()
            : this(
                new NeuralNetwork<T>(),
                2,
                new EpsilonGreedyExploration<T>())
        {
        }

        /// <summary>
        /// Builds the policy from <paramref name="options"/>: a dense ReLU network of the options' hidden widths from
        /// StateSize inputs to α and β for each action dimension (2 × ActionSize outputs).
        /// </summary>
        /// <param name="options">The policy's configuration.</param>
        public BetaPolicy(BetaPolicyOptions<T> options)
            : this(
                PolicyNetworks.Build((options ?? throw new ArgumentNullException(nameof(options))).StateSize,
                    options.HiddenLayers, 2 * options.ActionSize, options.LossFunction, nameof(BetaPolicyOptions<T>)),
                options.ActionSize, options.ExplorationStrategy, options.ActionMin, options.ActionMax,
                PolicyNetworks.RandomFor(options.Seed))
        {
        }

        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="explorationStrategy">The exploration strategy.</param>
        /// <param name="actionMin">Minimum action value (default: 0.0).</param>
        /// <param name="actionMax">Maximum action value (default: 1.0).</param>
        /// <param name="random">Optional random number generator.</param>
        public BetaPolicy(
            INeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            double actionMin = 0.0,
            double actionMax = 1.0,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            if (actionSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(actionSize), actionSize, "The action size must be positive.");
            // The range must be finite too: actions are actionMin + sample * (actionMax - actionMin), and two finite
            // bounds such as -MaxValue and MaxValue overflow that difference to infinity.
            if (double.IsNaN(actionMin) || double.IsInfinity(actionMin) || double.IsNaN(actionMax) || double.IsInfinity(actionMax)
                || actionMax <= actionMin || double.IsInfinity(actionMax - actionMin))
            {
                throw new ArgumentException(
                    $"The action bounds must be finite with actionMax > actionMin and a finite range (got [{actionMin}, {actionMax}]).",
                    nameof(actionMax));
            }
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
            _actionMin = actionMin;
            _actionMax = actionMax;
        }

        /// <summary>
        /// Initializes a new instance of the BetaPolicy class from its options: action size, exploration
        /// strategy, action bounds and <see cref="AiDotNet.Models.Options.ModelOptions.Seed"/>.
        /// </summary>
        /// <param name="policyNetwork">Network producing alpha and beta for each action dimension.</param>
        /// <param name="options">The policy options.</param>
        public BetaPolicy(INeuralNetwork<T> policyNetwork, BetaPolicyOptions<T> options)
            : this(
                policyNetwork,
                (options ?? throw new ArgumentNullException(nameof(options))).ActionSize,
                options.ExplorationStrategy,
                options.ActionMin,
                options.ActionMax,
                RandomFromSeed(options.Seed))
        {
        }

        /// <summary>
        /// Selects an action by sampling from Beta distributions.
        /// </summary>
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            // Get alpha and beta parameters from network
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            // Network should output 2 * actionSize values (alpha and beta for each dimension)
            if (output.Length != 2 * _actionSize)
            {
                throw new InvalidOperationException(
                    string.Format("Network output size {0} does not match expected size {1} (2 * actionSize)",
                        output.Length, 2 * _actionSize));
            }

            var action = new Vector<T>(_actionSize);

            for (int i = 0; i < _actionSize; i++)
            {
                // Get alpha and beta (ensure positive via softplus: log(1 + exp(x)) + 1)
                double alphaRaw = NumOps.ToDouble(output[i]);
                double betaRaw = NumOps.ToDouble(output[_actionSize + i]);

                double alpha = Softplus(alphaRaw) + 1.0;  // Add 1 to ensure alpha > 1 for well-behaved Beta
                double beta = Softplus(betaRaw) + 1.0;

                // Training samples the Beta distribution; inference takes its mode, (alpha - 1) / (alpha + beta - 2).
                // A softplus that underflows leaves both parameters at 1, the uniform distribution, whose every point
                // is a mode: the midpoint is taken rather than 0 / 0.
                double sample = training ? SampleBeta(alpha, beta) : BetaMode(alpha, beta);

                // Scale from [0, 1] to [actionMin, actionMax]
                double scaledAction = _actionMin + sample * (_actionMax - _actionMin);
                action[i] = NumOps.FromDouble(scaledAction);
            }

            if (training)
            {
                return _explorationStrategy.GetExplorationAction(state, action, _actionSize, _random);
            }

            return action;
        }

        // The mode of Beta(alpha, beta) for alpha, beta >= 1; 0.5 when both are 1 (uniform).
        internal static double BetaMode(double alpha, double beta)
        {
            double denominator = alpha + beta - 2.0;
            return denominator > 0.0 ? (alpha - 1.0) / denominator : 0.5;
        }

        /// <summary>
        /// Computes the log probability of an action under the Beta distribution policy.
        /// </summary>
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            ValidateState(state, nameof(state));
            ValidateActionSize(_actionSize, action.Length, nameof(action));

            // Get alpha and beta parameters
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            return BetaLogDensity(output, action, _actionSize, _actionMin, _actionMax);
        }

        /// <summary>
        /// Log-density of <paramref name="action"/> under the Beta heads in <paramref name="output"/>
        /// ([raw alpha | raw beta], alpha = softplus(raw) + 1), rescaled from [actionMin, actionMax] to [0, 1].
        /// </summary>
        /// <remarks>The unit action is clipped to [1e-7, 1 - 1e-7]; the upper bound is kept strictly below one in T
        /// (1 - 1e-7 rounds to 1 in Half, where log(1 - x) would be -infinity at the action bound).</remarks>
        internal static T BetaLogDensity(Vector<T> output, Vector<T> action, int actionSize, double actionMin, double actionMax)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var engine = AiDotNetEngine.Current;
            T one = numOps.One;

            // alpha, beta = softplus(raw) + 1 (> 1 for a well-behaved Beta)
            var alpha = engine.TensorAddScalar(engine.Softplus(output.GetSubVector(0, actionSize).AsTensor()), one);
            var beta = engine.TensorAddScalar(engine.Softplus(output.GetSubVector(actionSize, actionSize).AsTensor()), one);

            // Rescale action from [actionMin, actionMax] to [0, 1], clipped to avoid log(0)
            T range = numOps.FromDouble(actionMax - actionMin);
            var unit = engine.TensorDivideScalar(
                engine.TensorSubtractScalar(RLTensorOps.Head(action, actionSize).AsTensor(), numOps.FromDouble(actionMin)), range);
            var x = engine.TensorClamp(unit, numOps.FromDouble(1e-7), RLTensorOps.BelowOne<T>(1e-7));
            var oneMinusX = engine.TensorSubtract(engine.TensorFullLike(x, one), x);

            // Beta distribution log probability
            // log p(x) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
            // where B(α,β) = Γ(α)Γ(β)/Γ(α+β)
            var logBeta = engine.TensorSubtract(
                engine.TensorAdd(engine.TensorLgamma(alpha), engine.TensorLgamma(beta)),
                engine.TensorLgamma(engine.TensorAdd(alpha, beta)));
            var betaLogProb = engine.TensorSubtract(
                engine.TensorAdd(
                    engine.TensorMultiply(engine.TensorSubtractScalar(alpha, one), engine.TensorLog(x)),
                    engine.TensorMultiply(engine.TensorSubtractScalar(beta, one), engine.TensorLog(oneMinusX))),
                logBeta);

            // Account for rescaling Jacobian: log(actionMax - actionMin) per dimension
            T rescaleLogProb = numOps.Multiply(numOps.FromDouble(actionSize), numOps.Log(range));
            return numOps.Subtract(engine.TensorSum(betaLogProb), rescaleLogProb);
        }

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        /// <summary>
        /// Resets the exploration strategy.
        /// </summary>
        public override void Reset()
        {
            _explorationStrategy.Reset();
        }

        // Helper methods

        private double Softplus(double x)
        {
            // Numerically stable softplus: log(1 + exp(x))
            if (x > 20.0)
            {
                return x;  // For large x, softplus ≈ x
            }
            return Math.Log(1.0 + Math.Exp(x));
        }

        private double SampleBeta(double alpha, double beta)
        {
            // Sample from Beta using Gamma samples: if X~Gamma(α) and Y~Gamma(β), then X/(X+Y)~Beta(α,β)
            double x = SampleGamma(alpha);
            double y = SampleGamma(beta);
            return x / (x + y);
        }

        private double SampleGamma(double shape)
        {
            // Marsaglia and Tsang's method for Gamma sampling
            if (shape < 1.0)
            {
                return SampleGamma(shape + 1.0) * Math.Pow(_random.NextDouble(), 1.0 / shape);
            }

            double d = shape - 1.0 / 3.0;
            double c = 1.0 / Math.Sqrt(9.0 * d);

            while (true)
            {
                double x = 0.0;
                double v = 0.0;

                do
                {
                    // Sample from standard normal
                    x = _random.NextGaussian();
                    v = 1.0 + c * x;
                } while (v <= 0.0);

                v = v * v * v;
                double u = _random.NextDouble();

                if (u < 1.0 - 0.0331 * x * x * x * x)
                {
                    return d * v;
                }

                if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v;
                }
            }
        }
    }
}
