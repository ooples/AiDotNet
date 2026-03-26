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
    /// <example>
    /// <code>
    /// // Create a no-exploration strategy that always uses the greedy action
    /// var exploration = new NoExploration&lt;double&gt;();
    ///
    /// // Returns the policy action unchanged (pure exploitation)
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
