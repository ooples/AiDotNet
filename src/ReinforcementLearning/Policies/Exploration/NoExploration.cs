using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// No exploration - always use the policy's action directly (greedy).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelTask(ModelTask.Classification)]
    [ModelComplexity(ModelComplexity.Low)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ModelPaper("Reinforcement Learning: An Introduction",
        "https://incompleteideas.net/book/the-book-2nd.html",
        Year = 2018,
        Authors = "Sutton, R. S. & Barto, A. G.")]
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
