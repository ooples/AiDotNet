using AiDotNet.LinearAlgebra;
using System;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// No exploration - always use the policy's action directly (greedy).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class NoExploration<T> : IExplorationStrategy<T>
    {
        public Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
        {
            return policyAction;
        }

        public void Update()
        {
            // Nothing to update
        }

        public void Reset()
        {
            // Nothing to reset
        }
    }
}
