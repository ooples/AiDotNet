using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// No exploration - always use the policy's action directly (greedy).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class NoExploration<T> : ExplorationStrategyBase<T>
    {
        public override Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
        {
            return policyAction;
        }

        public override void Update()
        {
            // Nothing to update
        }

        public override void Reset()
        {
            // Nothing to reset
        }
    }
}
