using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Epsilon-greedy exploration: with probability epsilon, select random action.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
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
