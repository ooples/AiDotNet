using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Ornstein-Uhlenbeck process noise for temporally correlated exploration.
    /// Commonly used in DDPG and other continuous control algorithms.
    /// Process equation: dx = θ(μ - x)dt + σdW
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <example>
    /// <code>
    /// // Create Ornstein-Uhlenbeck noise for temporally correlated exploration
    /// var ouNoise = new OrnsteinUhlenbeckNoise&lt;double&gt;(actionSize: 2, theta: 0.15, sigma: 0.2);
    ///
    /// // Add temporally correlated noise to continuous actions
    /// var policyAction = new Vector&lt;double&gt;(new double[] { 0.5, -0.3 });
    /// var noisyAction = ouNoise.GetExplorationAction(state, policyAction, actionSpaceSize: 2, random);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.ReinforcementLearningAgent)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.Medium)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ModelPaper("Continuous Control with Deep Reinforcement Learning",
        "https://arxiv.org/abs/1509.02971",
        Year = 2016,
        Authors = "Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D.")]
    public class OrnsteinUhlenbeckNoise<T> : ExplorationStrategyBase<T>
    {
        private readonly double _theta;      // Mean reversion rate
        private readonly double _sigma;      // Volatility/noise scale
        private readonly double _mu;         // Long-term mean
        private readonly double _dt;         // Time step
        private Vector<T> _state;            // Current noise state

        /// <summary>
        /// Initializes a new instance of the Ornstein-Uhlenbeck noise exploration.
        /// </summary>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="theta">Mean reversion rate (default: 0.15).</param>
        /// <param name="sigma">Volatility/noise scale (default: 0.2).</param>
        /// <param name="mu">Long-term mean (default: 0.0).</param>
        /// <param name="dt">Time step (default: 0.01).</param>
        /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public OrnsteinUhlenbeckNoise()
        : this(4)
    {
    }

    public OrnsteinUhlenbeckNoise(
            int actionSize,
            double theta = 0.15,
            double sigma = 0.2,
            double mu = 0.0,
            double dt = 0.01)
        {
            _theta = theta;
            _sigma = sigma;
            _mu = mu;
            _dt = dt;
            _state = new Vector<T>(actionSize);

            // Initialize state to zeros
            for (int i = 0; i < actionSize; i++)
            {
                _state[i] = NumOps.Zero;
            }
        }

        /// <summary>
        /// Applies Ornstein-Uhlenbeck noise to the policy action.
        /// </summary>
        public override Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random)
        {
            ValidateActionSize(_state.Length, actionSpaceSize, nameof(actionSpaceSize));

            var noisyAction = new Vector<T>(actionSpaceSize);

            for (int i = 0; i < actionSpaceSize; i++)
            {
                // Ornstein-Uhlenbeck process: dx = θ(μ - x)dt + σ√dt * dW
                double x = NumOps.ToDouble(_state[i]);
                double dW = NumOps.ToDouble(BoxMullerSample(random));
                double dx = _theta * (_mu - x) * _dt + _sigma * Math.Sqrt(_dt) * dW;

                // Update noise state
                double newX = x + dx;
                _state[i] = NumOps.FromDouble(newX);

                // Add noise to action
                double actionValue = NumOps.ToDouble(policyAction[i]) + newX;
                noisyAction[i] = NumOps.FromDouble(actionValue);
            }

            // Clamp to valid action range
            return ClampAction(noisyAction);
        }

        /// <summary>
        /// Updates internal parameters (no-op for OU noise as it self-regulates).
        /// </summary>
        public override void Update()
        {
            // OU noise is self-regulating through mean reversion
            // No explicit decay needed
        }

        /// <summary>
        /// Resets the noise state to zero.
        /// </summary>
        public override void Reset()
        {
            for (int i = 0; i < _state.Length; i++)
            {
                _state[i] = NumOps.Zero;
            }
        }
    }
}
