using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Policy for discrete action spaces using a neural network to output action logits.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create a discrete policy that outputs action probabilities via softmax
    /// var network = new NeuralNetwork&lt;double&gt;();
    /// var policy = new DiscretePolicy&lt;double&gt;(network, actionSize: 4, new EpsilonGreedyExploration&lt;double&gt;());
    ///
    /// // Select a discrete action from the policy
    /// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
    /// var action = policy.SelectAction(state, training: true);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelTask(ModelTask.Classification)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Reinforcement Learning: An Introduction",
        "https://incompleteideas.net/book/the-book-2nd.html",
        Year = 2018,
        Authors = "Sutton, R. S. & Barto, A. G.")]
    public partial class DiscretePolicy<T> : PolicyBase<T>
    {
        private readonly INeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;

        /// <summary>
        /// Initializes a new instance with default settings.
        /// </summary>
        public DiscretePolicy()
            : this(
                new NeuralNetwork<T>(),
                2,
                new EpsilonGreedyExploration<T>())
        {
        }

        /// <summary>
        /// Builds the policy from <paramref name="options"/>: a dense ReLU network of the options' hidden widths from
        /// StateSize inputs to one logit per action (ActionSize outputs).
        /// </summary>
        /// <param name="options">The policy's configuration.</param>
        public DiscretePolicy(DiscretePolicyOptions<T> options)
            : this(
                PolicyNetworks.Build((options ?? throw new ArgumentNullException(nameof(options))).StateSize,
                    options.HiddenLayers, options.ActionSize, options.LossFunction, nameof(DiscretePolicyOptions<T>)),
                options.ActionSize, options.ExplorationStrategy, PolicyNetworks.RandomFor(options.Seed))
        {
        }

        public DiscretePolicy(
            INeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
        }

        /// <summary>
        /// Initializes a new instance of the DiscretePolicy class from its options: action count, exploration
        /// strategy and <see cref="AiDotNet.Models.Options.ModelOptions.Seed"/>.
        /// </summary>
        /// <param name="policyNetwork">Network producing one logit per action.</param>
        /// <param name="options">The policy options.</param>
        public DiscretePolicy(INeuralNetwork<T> policyNetwork, DiscretePolicyOptions<T> options)
            : this(
                policyNetwork,
                (options ?? throw new ArgumentNullException(nameof(options))).ActionSize,
                options.ExplorationStrategy,
                RandomFromSeed(options.Seed))
        {
        }

        /// <inheritdoc />
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            // Get action probabilities from network
            var stateTensor = Tensor<T>.FromVector(state);
            var logitsTensor = _policyNetwork.Predict(stateTensor);
            var logits = logitsTensor.ToVector();

            // Apply softmax to get probabilities
            var probabilities = Softmax(logits);

            // Training samples the categorical distribution; inference takes its mode (the most probable action).
            var policyAction = training ? SampleCategorical(probabilities) : OneHotArgMax(logits);

            if (training)
            {
                // Apply exploration strategy
                return _explorationStrategy.GetExplorationAction(state, policyAction, _actionSize, _random);
            }

            return policyAction;
        }

        /// <inheritdoc />
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            // Get logits from network
            var stateTensor = Tensor<T>.FromVector(state);
            var logitsTensor = _policyNetwork.Predict(stateTensor);
            var logits = logitsTensor.ToVector();

            // Find which action was taken (one-hot encoded)
            int actionIndex = RLTensorOps.FirstPositiveIndex(action, NumOps.FromDouble(0.5));

            // Exact log-softmax, log p(a) = z_a - logsumexp(z), as torch's Categorical.log_prob. The former
            // log(p + 1e-8) biased every log-probability and left the probabilities summing past 1.
            return RLTensorOps.LogSoftmaxAt(logits, actionIndex);
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

        // Helper methods
        private Vector<T> OneHotArgMax(Vector<T> logits)
        {
            // The argmax of the logits: softmax preserves their order, and a rounded probability can tie two
            // logits that differ.
            return RLTensorOps.OneHot<T>(logits.Length, RLTensorOps.ArgMax(logits));
        }

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
