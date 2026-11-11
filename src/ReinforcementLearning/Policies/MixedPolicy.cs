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
    /// Policy for environments with both discrete and continuous action spaces.
    /// Outputs both categorical distribution for discrete actions and Gaussian for continuous actions.
    /// Common in robotics where you have discrete mode selection and continuous parameter control.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create a mixed policy for environments with both discrete and continuous actions
    /// var discreteNet = new NeuralNetwork&lt;double&gt;();
    /// var continuousNet = new NeuralNetwork&lt;double&gt;();
    /// var policy = new MixedPolicy&lt;double&gt;(discreteNet, continuousNet,
    ///     discreteActionSize: 3, continuousActionSize: 2,
    ///     new EpsilonGreedyExploration&lt;double&gt;(), new GaussianNoiseExploration&lt;double&gt;());
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelTask(ModelTask.Classification)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Reinforcement Learning: An Introduction",
        "https://incompleteideas.net/book/the-book-2nd.html",
        Year = 2018,
        Authors = "Sutton, R. S. & Barto, A. G.")]
    public partial class MixedPolicy<T> : PolicyBase<T>
    {
        private readonly INeuralNetwork<T> _discreteNetwork;
        private readonly INeuralNetwork<T> _continuousNetwork;
        private readonly IExplorationStrategy<T> _discreteExploration;
        private readonly IExplorationStrategy<T> _continuousExploration;
        private readonly int _discreteActionSize;
        private readonly int _continuousActionSize;
        private readonly bool _sharedFeatures;

        /// <summary>
        /// Initializes a new instance of the MixedPolicy class.
        /// </summary>
        /// <summary>
        /// Initializes a new instance with default settings.
        /// </summary>
        public MixedPolicy()
            : this(
                new NeuralNetwork<T>(),
                new NeuralNetwork<T>(),
                2,
                2,
                new EpsilonGreedyExploration<T>(),
                new EpsilonGreedyExploration<T>())
        {
        }

        /// <summary>
        /// Builds the policy from <paramref name="options"/>: dense ReLU networks of the options' hidden widths from
        /// StateSize inputs, a DiscreteActionSize-logit head and a 2 × ContinuousActionSize head (means, then log standard
        /// deviations); with <see cref="MixedPolicyOptions{T}.SharedFeatures"/>, one network carries both heads.
        /// </summary>
        /// <param name="options">The policy's configuration.</param>
        public MixedPolicy(MixedPolicyOptions<T> options)
            : this(BuildNetworks(options ?? throw new ArgumentNullException(nameof(options))), options)
        {
        }

        private MixedPolicy((INeuralNetwork<T> Discrete, INeuralNetwork<T> Continuous) networks, MixedPolicyOptions<T> options)
            : this(networks.Discrete, networks.Continuous, options.DiscreteActionSize, options.ContinuousActionSize,
                options.DiscreteExplorationStrategy, options.ContinuousExplorationStrategy, options.SharedFeatures,
                PolicyNetworks.RandomFor(options.Seed))
        {
        }

        private static (INeuralNetwork<T> Discrete, INeuralNetwork<T> Continuous) BuildNetworks(MixedPolicyOptions<T> options)
        {
            const string name = nameof(MixedPolicyOptions<T>);
            if (options.DiscreteActionSize <= 0 || options.ContinuousActionSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), name + ": DiscreteActionSize and ContinuousActionSize must be positive.");
            if (options.SharedFeatures)
            {
                var trunk = PolicyNetworks.Build(options.StateSize, options.HiddenLayers,
                    options.DiscreteActionSize + 2 * options.ContinuousActionSize, options.LossFunction, name);
                return (trunk, trunk);
            }
            return (PolicyNetworks.Build(options.StateSize, options.HiddenLayers, options.DiscreteActionSize, options.LossFunction, name),
                PolicyNetworks.Build(options.StateSize, options.HiddenLayers, 2 * options.ContinuousActionSize, options.LossFunction, name));
        }

        /// <param name="discreteNetwork">Network for discrete action logits.</param>
        /// <param name="continuousNetwork">Network for continuous action parameters (mean and log_std).</param>
        /// <param name="discreteActionSize">Number of discrete actions.</param>
        /// <param name="continuousActionSize">Number of continuous action dimensions.</param>
        /// <param name="discreteExploration">Exploration strategy for discrete actions.</param>
        /// <param name="continuousExploration">Exploration strategy for continuous actions.</param>
        /// <param name="sharedFeatures">Whether networks share feature extraction layers.</param>
        /// <param name="random">Optional random number generator.</param>

        public MixedPolicy(
            INeuralNetwork<T> discreteNetwork,
            INeuralNetwork<T> continuousNetwork,
            int discreteActionSize,
            int continuousActionSize,
            IExplorationStrategy<T> discreteExploration,
            IExplorationStrategy<T> continuousExploration,
            bool sharedFeatures = false,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(discreteNetwork);
            _discreteNetwork = discreteNetwork;
            Guard.NotNull(continuousNetwork);
            _continuousNetwork = continuousNetwork;
            Guard.NotNull(discreteExploration);
            _discreteExploration = discreteExploration;
            Guard.NotNull(continuousExploration);
            _continuousExploration = continuousExploration;
            Guard.Positive(discreteActionSize);
            _discreteActionSize = discreteActionSize;
            Guard.Positive(continuousActionSize);
            _continuousActionSize = continuousActionSize;
            // One trunk serves both heads: the same network, read as [logits, means, log standard deviations].
            if (sharedFeatures && !ReferenceEquals(discreteNetwork, continuousNetwork))
                throw new ArgumentException(
                    "With sharedFeatures the discrete and continuous networks must be the same network, whose output is " +
                    "[discreteActionSize logits, continuousActionSize means, continuousActionSize log standard deviations].",
                    nameof(continuousNetwork));
            _sharedFeatures = sharedFeatures;
        }

        /// <summary>
        /// Initializes a new instance of the MixedPolicy class from its options: both action sizes, both
        /// exploration strategies, feature sharing and <see cref="AiDotNet.Models.Options.ModelOptions.Seed"/>.
        /// </summary>
        /// <param name="discreteNetwork">Network producing the discrete action logits.</param>
        /// <param name="continuousNetwork">Network producing the continuous action distribution.</param>
        /// <param name="options">The policy options.</param>
        public MixedPolicy(INeuralNetwork<T> discreteNetwork, INeuralNetwork<T> continuousNetwork, MixedPolicyOptions<T> options)
            : this(
                discreteNetwork,
                continuousNetwork,
                (options ?? throw new ArgumentNullException(nameof(options))).DiscreteActionSize,
                options.ContinuousActionSize,
                options.DiscreteExplorationStrategy,
                options.ContinuousExplorationStrategy,
                options.SharedFeatures,
                RandomFromSeed(options.Seed))
        {
        }

        // The one-hot of the most probable discrete action.
        private Vector<T> OneHotArgMax(Vector<T> logits)
        {
            // The argmax of the logits: softmax preserves their order, and a rounded probability can tie two
            // logits that differ.
            return RLTensorOps.OneHot<T>(logits.Length, RLTensorOps.ArgMax(logits));
        }

        /// <summary>
        /// Selects mixed action: [discrete_action, continuous_actions]
        /// </summary>
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            var stateTensor = Tensor<T>.FromVector(state);

            // Get discrete action (one-hot)
            var (discreteLogits, continuousOutput) = ReadHeads(stateTensor);
            var discreteProbabilities = Softmax(discreteLogits);
            // Inference takes each part's mode (the most probable discrete action, the Gaussian mean); training samples.
            var discreteAction = training ? SampleCategorical(discreteProbabilities) : OneHotArgMax(discreteLogits);

            if (training)
            {
                discreteAction = _discreteExploration.GetExplorationAction(state, discreteAction, _discreteActionSize, _random);
            }

            // Get continuous action (Gaussian)

            // Training samples mean + sigma * N(0, 1); inference takes the mean.
            var continuousMean = continuousOutput.GetSubVector(0, _continuousActionSize);
            var continuousAction = training
                ? RLTensorOps.SampleGaussian(continuousMean, continuousOutput.GetSubVector(_continuousActionSize, _continuousActionSize), _random)
                : continuousMean;

            if (training)
            {
                continuousAction = _continuousExploration.GetExplorationAction(state, continuousAction, _continuousActionSize, _random);
            }

            // Concatenate: [discrete, continuous]
            var mixedAction = RLTensorOps.Concat(
                RLTensorOps.Head(discreteAction, _discreteActionSize),
                RLTensorOps.Head(continuousAction, _continuousActionSize));

            return mixedAction;
        }

        /// <summary>
        /// Computes log probability of mixed action.
        /// </summary>
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            ValidateState(state, nameof(state));
            ValidateActionSize(_discreteActionSize + _continuousActionSize, action.Length, nameof(action));

            var stateTensor = Tensor<T>.FromVector(state);

            // Split action into discrete and continuous parts
            var discreteAction = action.GetSubVector(0, _discreteActionSize);
            var continuousAction = action.GetSubVector(_discreteActionSize, _continuousActionSize);

            // Discrete log prob
            var (discreteLogits, continuousOutput) = ReadHeads(stateTensor);

            int discreteActionIndex = RLTensorOps.FirstPositiveIndex(discreteAction, NumOps.FromDouble(0.5));

            // Exact log-softmax from the logits, log p(a) = z_a - logsumexp(z), as DiscretePolicy computes it: the
            // former log(p + 1e-8) of the rounded softmax biased every log-probability.
            T discreteLogProb = RLTensorOps.LogSoftmaxAt(discreteLogits, discreteActionIndex);

            // Continuous log prob: the diagonal Gaussian over [means | log standard deviations]
            T continuousLogProb = RLTensorOps.GaussianLogProb(
                continuousOutput.GetSubVector(0, _continuousActionSize),
                continuousOutput.GetSubVector(_continuousActionSize, _continuousActionSize),
                continuousAction);

            // Total log prob is sum (since actions are independent)
            return NumOps.Add(discreteLogProb, continuousLogProb);
        }

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return _sharedFeatures
                ? new List<INeuralNetwork<T>> { _discreteNetwork }
                : new List<INeuralNetwork<T>> { _discreteNetwork, _continuousNetwork };
        }

        /// <summary>
        /// Resets both exploration strategies.
        /// </summary>
        public override void Reset()
        {
            _discreteExploration.Reset();
            _continuousExploration.Reset();
        }

        // The discrete logits and the continuous [means, log standard deviations]: two networks' outputs, or the one
        // shared trunk's output split at discreteActionSize.
        private (Vector<T> Logits, Vector<T> Continuous) ReadHeads(Tensor<T> stateTensor)
        {
            if (!_sharedFeatures)
                return (_discreteNetwork.Predict(stateTensor).ToVector(), _continuousNetwork.Predict(stateTensor).ToVector());
            var output = _discreteNetwork.Predict(stateTensor).ToVector();
            int expected = _discreteActionSize + 2 * _continuousActionSize;
            if (output.Length != expected)
                throw new InvalidOperationException(string.Format(
                    "The shared network outputs {0} values; a mixed policy reads {1}.", output.Length, expected));
            return (output.GetSubVector(0, _discreteActionSize), output.GetSubVector(_discreteActionSize, 2 * _continuousActionSize));
        }

        // Helper methods
        private Vector<T> Softmax(Vector<T> logits)
        {
            // exp(z - max z) / sum exp(z - max z)
            return RLTensorOps.StableSoftmax(logits);
        }

        private Vector<T> SampleCategorical(Vector<T> probabilities)
        {
            // First index whose running probability reaches the uniform draw; the last action when rounding
            // leaves the total below it.
            double randomValue = _random.NextDouble();
            int index = RLTensorOps.SampleFromCdf(probabilities, randomValue, probabilities.Length - 1);
            return RLTensorOps.OneHot<T>(probabilities.Length, index);
        }
    }
}
