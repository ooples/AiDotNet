using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Epsilon-greedy exploration: with probability epsilon, select random action.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create epsilon-greedy exploration with annealing from 1.0 to 0.01
    /// var exploration = new EpsilonGreedyExploration&lt;double&gt;(epsilonStart: 1.0, epsilonEnd: 0.01, epsilonDecay: 0.995);
    ///
    /// // With probability epsilon, select a random action instead of the policy action
    /// var policyAction = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0 });
    /// var action = exploration.GetExplorationAction(state, policyAction, actionSpaceSize: 3, random);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelTask(ModelTask.Classification)]
    [ModelComplexity(ModelComplexity.Low)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ModelPaper("Reinforcement Learning: An Introduction",
        "https://incompleteideas.net/book/the-book-2nd.html",
        Year = 2018,
        Authors = "Sutton, R. S. & Barto, A. G.")]
    public class EpsilonGreedyExploration<T> : ExplorationStrategyBase<T>
    {
        private double _epsilon;
        private readonly double _epsilonStart;
        private readonly double _epsilonEnd;
        private readonly double _epsilonDecay;

        public EpsilonGreedyExploration(double epsilonStart = 1.0, double epsilonEnd = 0.01, double epsilonDecay = 0.995)
        {
            _epsilonStart = epsilonStart;
            _epsilonEnd = epsilonEnd;
            _epsilonDecay = epsilonDecay;
            _epsilon = epsilonStart;
        }

        public override Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
        {
            if (random.NextDouble() < _epsilon)
            {
                // Random action
                int randomActionIndex = random.Next(actionSpaceSize);
                var randomAction = new Vector<T>(actionSpaceSize);
                randomAction[randomActionIndex] = NumOps.One;
                return randomAction;
            }

            // Greedy action from policy
            return policyAction;
        }

        public override void Update()
        {
            _epsilon = Math.Max(_epsilonEnd, _epsilon * _epsilonDecay);
        }

        public override void Reset()
        {
            _epsilon = _epsilonStart;
        }

        public double CurrentEpsilon => _epsilon;
    }
}
