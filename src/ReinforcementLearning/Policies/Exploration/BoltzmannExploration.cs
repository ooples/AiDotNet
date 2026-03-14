using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Boltzmann (softmax) exploration with temperature-based action selection.
    /// Uses temperature parameter to control exploration: higher temperature = more random.
    /// Action probability: P(a) = exp(Q(a)/τ) / Σ exp(Q(a')/τ)
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create Boltzmann exploration with temperature decay
    /// var exploration = new BoltzmannExploration&lt;double&gt;(temperatureStart: 1.0, temperatureEnd: 0.01, temperatureDecay: 0.995);
    ///
    /// // Get an exploration action weighted by Q-value estimates
    /// var policyAction = new Vector&lt;double&gt;(new double[] { 1.5, 0.8, 2.3 });
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
    public class BoltzmannExploration<T> : ExplorationStrategyBase<T>
    {
        private double _temperature;
        private readonly double _temperatureStart;
        private readonly double _temperatureEnd;
        private readonly double _temperatureDecay;

        /// <summary>
        /// Initializes a new instance of the Boltzmann exploration strategy.
        /// </summary>
        /// <param name="temperatureStart">Initial temperature (default: 1.0).</param>
        /// <param name="temperatureEnd">Minimum temperature (default: 0.01).</param>
        /// <param name="temperatureDecay">Temperature decay rate per update (default: 0.995).</param>
        public BoltzmannExploration(
            double temperatureStart = 1.0,
            double temperatureEnd = 0.01,
            double temperatureDecay = 0.995)
        {
            _temperatureStart = temperatureStart;
            _temperatureEnd = temperatureEnd;
            _temperatureDecay = temperatureDecay;
            _temperature = temperatureStart;
        }

        /// <summary>
        /// Applies Boltzmann (softmax) exploration to select an action.
        /// </summary>
        public override Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random)
        {
            // For discrete actions, apply softmax with temperature to the policy action (assumed to be logits or Q-values)
            // For continuous actions, treat policyAction as mean and sample with temperature-scaled noise

            // Check if this is discrete (one-hot) or continuous
            bool isDiscrete = IsOneHot(policyAction);

            if (isDiscrete)
            {
                // Apply softmax with temperature
                var probabilities = SoftmaxWithTemperature(policyAction);
                return SampleCategorical(probabilities, random);
            }
            else
            {
                // For continuous: add temperature-scaled Gaussian noise
                var noisyAction = new Vector<T>(actionSpaceSize);
                for (int i = 0; i < actionSpaceSize; i++)
                {
                    double noise = NumOps.ToDouble(BoxMullerSample(random)) * _temperature;
                    double actionValue = NumOps.ToDouble(policyAction[i]) + noise;
                    noisyAction[i] = NumOps.FromDouble(actionValue);
                }
                return ClampAction(noisyAction);
            }
        }

        /// <summary>
        /// Updates the temperature using exponential decay.
        /// </summary>
        public override void Update()
        {
            _temperature = Math.Max(_temperatureEnd, _temperature * _temperatureDecay);
        }

        /// <summary>
        /// Resets the temperature to its initial value.
        /// </summary>
        public override void Reset()
        {
            _temperature = _temperatureStart;
        }

        /// <summary>
        /// Gets the current temperature value.
        /// </summary>
        public double CurrentTemperature => _temperature;

        // Helper methods

        private Vector<T> SoftmaxWithTemperature(Vector<T> logits)
        {
            var probabilities = new Vector<T>(logits.Length);
            T maxLogit = logits[0];

            // Find max for numerical stability
            for (int i = 1; i < logits.Length; i++)
            {
                if (NumOps.GreaterThan(logits[i], maxLogit))
                {
                    maxLogit = logits[i];
                }
            }

            // Compute exp((logit - max) / temperature) and sum
            T sumExp = NumOps.Zero;
            for (int i = 0; i < logits.Length; i++)
            {
                double scaledLogit = (NumOps.ToDouble(logits[i]) - NumOps.ToDouble(maxLogit)) / _temperature;
                var expValue = NumOps.FromDouble(Math.Exp(scaledLogit));
                probabilities[i] = expValue;
                sumExp = NumOps.Add(sumExp, expValue);
            }

            // Normalize
            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = NumOps.Divide(probabilities[i], sumExp);
            }

            return probabilities;
        }

        private Vector<T> SampleCategorical(Vector<T> probabilities, Random random)
        {
            double randomValue = random.NextDouble();
            double cumulativeProbability = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulativeProbability += NumOps.ToDouble(probabilities[i]);
                if (randomValue <= cumulativeProbability)
                {
                    var action = new Vector<T>(probabilities.Length);
                    action[i] = NumOps.One;
                    return action;
                }
            }

            // Fallback: select last action
            var fallbackAction = new Vector<T>(probabilities.Length);
            fallbackAction[probabilities.Length - 1] = NumOps.One;
            return fallbackAction;
        }

        private bool IsOneHot(Vector<T> action)
        {
            int onesCount = 0;
            for (int i = 0; i < action.Length; i++)
            {
                double val = NumOps.ToDouble(action[i]);
                if (Math.Abs(val - 1.0) < 1e-6)
                {
                    onesCount++;
                }
                else if (Math.Abs(val) > 1e-6)
                {
                    // Non-zero, non-one value found
                    return false;
                }
            }
            return onesCount == 1;
        }
    }
}
