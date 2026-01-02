using System;
using System.Collections.Generic;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Thompson Sampling (Bayesian) exploration for discrete action spaces.
    /// Maintains Beta distributions for each action and samples from posteriors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ThompsonSamplingExploration<T> : ExplorationStrategyBase<T>
    {
        private readonly Dictionary<int, BetaDistribution> _actionDistributions;
        private readonly double _priorAlpha;
        private readonly double _priorBeta;

        /// <summary>
        /// Initializes a new instance of the Thompson Sampling exploration strategy.
        /// </summary>
        /// <param name="priorAlpha">Prior alpha parameter for Beta distribution (default: 1.0).</param>
        /// <param name="priorBeta">Prior beta parameter for Beta distribution (default: 1.0).</param>
        public ThompsonSamplingExploration(double priorAlpha = 1.0, double priorBeta = 1.0)
        {
            _actionDistributions = new Dictionary<int, BetaDistribution>();
            _priorAlpha = priorAlpha;
            _priorBeta = priorBeta;
        }

        /// <summary>
        /// Selects action by sampling from Beta posteriors for each action.
        /// </summary>
        public override Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random)
        {
            // Initialize distributions for actions we haven't seen
            for (int i = 0; i < actionSpaceSize; i++)
            {
                if (!_actionDistributions.ContainsKey(i))
                {
                    _actionDistributions[i] = new BetaDistribution(_priorAlpha, _priorBeta);
                }
            }

            // Sample from each action's Beta distribution
            double maxSample = double.NegativeInfinity;
            int bestAction = 0;

            for (int i = 0; i < actionSpaceSize; i++)
            {
                double sample = _actionDistributions[i].Sample(random);
                if (sample > maxSample)
                {
                    maxSample = sample;
                    bestAction = i;
                }
            }

            // Return one-hot encoded action
            var action = new Vector<T>(actionSpaceSize);
            action[bestAction] = NumOps.One;
            return action;
        }

        /// <summary>
        /// Updates the Beta distribution for a specific action based on reward.
        /// </summary>
        /// <param name="actionIndex">The action that was taken.</param>
        /// <param name="reward">The reward received (should be in [0, 1]).</param>
        public void UpdateDistribution(int actionIndex, double reward)
        {
            if (!_actionDistributions.ContainsKey(actionIndex))
            {
                _actionDistributions[actionIndex] = new BetaDistribution(_priorAlpha, _priorBeta);
            }

            // Update based on reward (Bernoulli feedback)
            // If reward is positive/high, increment alpha (success)
            // If reward is negative/low, increment beta (failure)
            if (reward > 0.5)
            {
                _actionDistributions[actionIndex].Alpha += 1.0;
            }
            else
            {
                _actionDistributions[actionIndex].Beta += 1.0;
            }
        }

        /// <summary>
        /// Updates internal parameters (call UpdateDistribution separately for each action).
        /// </summary>
        public override void Update()
        {
            // Updates happen via UpdateDistribution method
        }

        /// <summary>
        /// Resets all action distributions to prior.
        /// </summary>
        public override void Reset()
        {
            _actionDistributions.Clear();
        }

        /// <summary>
        /// Simple Beta distribution implementation for Thompson Sampling.
        /// </summary>
        private class BetaDistribution
        {
            public double Alpha { get; set; }
            public double Beta { get; set; }

            public BetaDistribution(double alpha, double beta)
            {
                Alpha = alpha;
                Beta = beta;
            }

            public double Sample(Random random)
            {
                // Sample from Beta using Gamma samples: if X~Gamma(α) and Y~Gamma(β), then X/(X+Y)~Beta(α,β)
                double x = SampleGamma(Alpha, random);
                double y = SampleGamma(Beta, random);
                return x / (x + y);
            }

            private double SampleGamma(double shape, Random random)
            {
                // Marsaglia and Tsang's method for Gamma sampling
                if (shape < 1.0)
                {
                    return SampleGamma(shape + 1.0, random) * Math.Pow(random.NextDouble(), 1.0 / shape);
                }

                double d = shape - 1.0 / 3.0;
                double c = 1.0 / Math.Sqrt(9.0 * d);

                while (true)
                {
                    double x = 0.0;
                    double v = 0.0;

                    do
                    {
                        // Standard normal
                        x = random.NextGaussian();
                        v = 1.0 + c * x;
                    } while (v <= 0.0);

                    v = v * v * v;
                    double u = random.NextDouble();

                    if (u < 1.0 - 0.0331 * x * x * x * x)
                    {
                        return d * v;
                    }

                    if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                    {
                        return d * v;
                    }
                }
            }
        }
    }
}
