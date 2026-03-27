using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Upper Confidence Bound (UCB) exploration for discrete action spaces.
    /// Balances exploration and exploitation using confidence intervals: UCB(a) = Q(a) + c * √(ln(t) / N(a))
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create UCB exploration that favors under-explored actions
    /// var exploration = new UpperConfidenceBoundExploration&lt;double&gt;(explorationConstant: 2.0);
    ///
    /// // Select action using upper confidence bound: Q(a) + c * sqrt(ln(t) / N(a))
    /// var policyAction = new Vector&lt;double&gt;(new double[] { 1.5, 0.8, 2.3 });
    /// var action = exploration.GetExplorationAction(state, policyAction, actionSpaceSize: 3, random);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelTask(ModelTask.Classification)]
    [ModelComplexity(ModelComplexity.Low)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ModelPaper("Finite-time Analysis of the Multiarmed Bandit Problem",
        "https://link.springer.com/article/10.1023/A:1013689704352",
        Year = 2002,
        Authors = "Auer, P., Cesa-Bianchi, N., & Fischer, P.")]
    public class UpperConfidenceBoundExploration<T> : ExplorationStrategyBase<T>
    {
        private readonly Dictionary<int, int> _actionCounts;
        private int _totalSteps;
        private readonly double _explorationConstant;

        /// <summary>
        /// Initializes a new instance of the Upper Confidence Bound exploration strategy.
        /// </summary>
        /// <param name="explorationConstant">Exploration constant 'c' that controls exploration level (default: 2.0).</param>
        public UpperConfidenceBoundExploration(double explorationConstant = 2.0)
        {
            _actionCounts = new Dictionary<int, int>();
            _totalSteps = 0;
            _explorationConstant = explorationConstant;
        }

        /// <summary>
        /// Selects action using UCB: action with highest Q(a) + c * √(ln(t) / N(a))
        /// </summary>
        public override Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random)
        {
            _totalSteps++;

            // Interpret policyAction as Q-values for each action
            double maxUcbValue = double.NegativeInfinity;
            int bestAction = 0;

            for (int i = 0; i < actionSpaceSize; i++)
            {
                double qValue = NumOps.ToDouble(policyAction[i]);

                // Get action count, default to 0 if not seen
                int actionCount = 0;
                if (_actionCounts.ContainsKey(i))
                {
                    actionCount = _actionCounts[i];
                }

                // UCB bonus: c * √(ln(t) / N(a))
                // If action never taken, give it maximum priority
                double ucbBonus = 0.0;
                if (actionCount == 0)
                {
                    ucbBonus = double.PositiveInfinity;
                }
                else
                {
                    ucbBonus = _explorationConstant * Math.Sqrt(Math.Log(_totalSteps) / actionCount);
                }

                double ucbValue = qValue + ucbBonus;

                if (ucbValue > maxUcbValue)
                {
                    maxUcbValue = ucbValue;
                    bestAction = i;
                }
            }

            // Update action count
            if (_actionCounts.ContainsKey(bestAction))
            {
                _actionCounts[bestAction]++;
            }
            else
            {
                _actionCounts[bestAction] = 1;
            }

            // Return one-hot encoded action
            var action = new Vector<T>(actionSpaceSize);
            action[bestAction] = NumOps.One;
            return action;
        }

        /// <summary>
        /// Updates internal parameters (UCB is count-based, no explicit decay).
        /// </summary>
        public override void Update()
        {
            // UCB is self-regulating through action counts
        }

        /// <summary>
        /// Resets action counts and total steps.
        /// </summary>
        public override void Reset()
        {
            _actionCounts.Clear();
            _totalSteps = 0;
        }

        /// <summary>
        /// Gets the current exploration constant.
        /// </summary>
        public double ExplorationConstant => _explorationConstant;

        /// <summary>
        /// Gets the total number of steps taken.
        /// </summary>
        public int TotalSteps => _totalSteps;
    }
}
