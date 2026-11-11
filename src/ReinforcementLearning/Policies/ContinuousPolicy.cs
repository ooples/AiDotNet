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
    /// Policy for continuous action spaces using a neural network to output Gaussian parameters.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create a Gaussian continuous policy for unbounded continuous actions
    /// var network = new NeuralNetwork&lt;double&gt;();
    /// var policy = new ContinuousPolicy&lt;double&gt;(network, actionSize: 2, new EpsilonGreedyExploration&lt;double&gt;());
    ///
    /// // Sample an action from the Gaussian distribution
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
    [ResearchPaper("Reinforcement Learning: An Introduction",
        "https://incompleteideas.net/book/the-book-2nd.html",
        Year = 2018,
        Authors = "Sutton, R. S. & Barto, A. G.")]
    public partial class ContinuousPolicy<T> : PolicyBase<T>
    {
        private readonly INeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly bool _useTanhSquashing;

        /// <summary>
        /// Initializes a new instance with default settings.
        /// </summary>
        public ContinuousPolicy()
            : this(
                new NeuralNetwork<T>(new NeuralNetworkArchitecture<T>(
                    inputType: Enums.InputType.OneDimensional,
                    taskType: Enums.NeuralNetworkTaskType.Regression,
                    inputSize: 4,
                    outputSize: 4)),
                2,
                new EpsilonGreedyExploration<T>())
        {
        }

        /// <summary>
        /// Builds the policy from <paramref name="options"/>: a dense ReLU network of the options' hidden widths from
        /// StateSize inputs to a mean and a log standard deviation for each action dimension (2 × ActionSize outputs).
        /// </summary>
        /// <param name="options">The policy's configuration.</param>
        public ContinuousPolicy(ContinuousPolicyOptions<T> options)
            : this(
                PolicyNetworks.Build((options ?? throw new ArgumentNullException(nameof(options))).StateSize,
                    options.HiddenLayers, 2 * options.ActionSize, options.LossFunction, nameof(ContinuousPolicyOptions<T>)),
                options.ActionSize, options.ExplorationStrategy, options.UseTanhSquashing,
                PolicyNetworks.RandomFor(options.Seed))
        {
        }

        public ContinuousPolicy(
            INeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            bool useTanhSquashing = false,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
            _useTanhSquashing = useTanhSquashing;
        }

        /// <summary>
        /// Initializes a new instance of the ContinuousPolicy class from its options: action size, exploration
        /// strategy, tanh squashing and <see cref="AiDotNet.Models.Options.ModelOptions.Seed"/>.
        /// </summary>
        /// <param name="policyNetwork">Network producing the mean and log standard deviation of each action.</param>
        /// <param name="options">The policy options.</param>
        public ContinuousPolicy(INeuralNetwork<T> policyNetwork, ContinuousPolicyOptions<T> options)
            : this(
                policyNetwork,
                (options ?? throw new ArgumentNullException(nameof(options))).ActionSize,
                options.ExplorationStrategy,
                options.UseTanhSquashing,
                RandomFromSeed(options.Seed))
        {
        }

        /// <inheritdoc />
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            // Get mean and log_std from network
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            // Split output into mean and log_std
            var mean = output.GetSubVector(0, _actionSize);
            var logStd = output.GetSubVector(_actionSize, _actionSize);

            // Inference takes the distribution's mode (its mean), as DiscretePolicy takes the argmax and
            // Stable-Baselines3 evaluates with deterministic=True; training samples mean + sigma * N(0, 1).
            var action = training ? RLTensorOps.SampleGaussian(mean, logStd, _random) : mean;

            if (_useTanhSquashing)
            {
                action = AiDotNetEngine.Current.Tanh(action);
            }

            if (training)
            {
                return _explorationStrategy.GetExplorationAction(state, action, _actionSize, _random);
            }

            return action;
        }

        /// <inheritdoc />
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            // Get mean and log_std from network
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            var engine = AiDotNetEngine.Current;
            var mean = output.GetSubVector(0, _actionSize);
            var logStd = output.GetSubVector(_actionSize, _actionSize);
            var taken = RLTensorOps.Head(action, _actionSize);

            if (!_useTanhSquashing)
            {
                // Gaussian log probability: sum_i -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
                return RLTensorOps.GaussianLogProb(mean, logStd, taken);
            }

            // Tanh-squashed Gaussian (Haarnoja et al. 2018, SAC, Appendix C): the action is a = tanh(u) with
            // u ~ N(mu, sigma), so log pi(a) = log N(u) - sum_i log(1 - a_i^2). The pre-squash sample is recovered
            // as u = atanh(a) (a clipped just inside (-1, 1)), as Stable-Baselines3's TanhBijector does; the
            // former code evaluated the Gaussian at the squashed action and squashed it a second time.
            T epsilon = NumOps.FromDouble(1e-6);
            var clipped = engine.Clamp(taken, NumOps.FromDouble(-1.0 + 1e-6), NumOps.FromDouble(1.0 - 1e-6));
            T gaussianLogProb = RLTensorOps.GaussianLogProb(mean, logStd, engine.Atanh(clipped));
            var oneMinusSquare = engine.Subtract(
                Vector<T>.CreateDefault(_actionSize, NumOps.Add(NumOps.One, epsilon)),
                engine.Multiply(taken, taken));
            T correction = engine.Sum(engine.TensorLog(oneMinusSquare.AsTensor()).ToVector());
            return NumOps.Subtract(gaussianLogProb, correction);
        }

        /// <inheritdoc />
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        /// <inheritdoc />
        public override void Reset()
        {
            _explorationStrategy.Reset();
        }
    }
}
