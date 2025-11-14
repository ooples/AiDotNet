using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Epsilon-greedy exploration: with probability epsilon, select random action.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class EpsilonGreedyExploration<T> : IExplorationStrategy<T>
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

        public Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
        {
            if (random.NextDouble() < _epsilon)
            {
                // Random action
                int randomActionIndex = random.Next(actionSpaceSize);
                var randomAction = new Vector<T>(actionSpaceSize);
                randomAction[randomActionIndex] = NumOps<T>.One;
                return randomAction;
            }

            // Greedy action from policy
            return policyAction;
        }

        public void Update()
        {
            _epsilon = Math.Max(_epsilonEnd, _epsilon * _epsilonDecay);
        }

        public void Reset()
        {
            _epsilon = _epsilonStart;
        }

        public double CurrentEpsilon => _epsilon;
    }
}
