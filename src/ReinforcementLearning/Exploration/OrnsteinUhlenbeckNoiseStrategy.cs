using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Exploration
{
    /// <summary>
    /// Implements Ornstein-Uhlenbeck noise process for exploration in continuous action spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// The Ornstein-Uhlenbeck process generates temporally correlated noise, which is useful
    /// for exploring continuous action spaces in physical environments where actions should
    /// change smoothly over time. This is commonly used in algorithms like DDPG.
    /// </para>
    /// </remarks>
    public class OrnsteinUhlenbeckNoiseStrategy<T> : IExplorationStrategy<Vector<T>, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        private readonly T _theta = default!;
        private readonly T _sigma = default!;
        private readonly T _decayRate = default!;
        private readonly long _decaySteps;
        private readonly Random _random = default!;
        private Vector<T> _state = default!;
        private Vector<T> _mu = default!;
        private readonly int _actionDim;

        /// <summary>
        /// Gets the current exploration rate (scale of the noise).
        /// </summary>
        public T ExplorationRate { get; private set; }

        /// <summary>
        /// Gets a value indicating whether the strategy is suitable for continuous action spaces.
        /// </summary>
        public bool IsContinuous => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="OrnsteinUhlenbeckNoiseStrategy{T}"/> class.
        /// </summary>
        /// <param name="actionDim">The dimensionality of the action space.</param>
        /// <param name="theta">The rate at which the process reverts to the mean (higher: faster reversion).</param>
        /// <param name="sigma">The scale of the noise.</param>
        /// <param name="initialScale">The initial scale factor for the noise.</param>
        /// <param name="finalScale">The final scale factor for the noise after decay.</param>
        /// <param name="decayRate">The rate at which the noise scale decays.</param>
        /// <param name="decaySteps">The number of steps over which to decay the noise scale.</param>
        /// <param name="mu">The mean of the process (default: zeros).</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        public OrnsteinUhlenbeckNoiseStrategy(
            int actionDim,
            double theta = 0.15,
            double sigma = 0.2,
            double initialScale = 1.0,
            double finalScale = 0.1,
            double decayRate = 0.9999,
            long decaySteps = 100000,
            Vector<T>? mu = null,
            int? seed = null)
        {
            _actionDim = actionDim;
            _theta = NumOps.FromDouble(theta);
            _sigma = NumOps.FromDouble(sigma);
            ExplorationRate = NumOps.FromDouble(initialScale);
            _decayRate = NumOps.FromDouble(decayRate);
            _decaySteps = decaySteps;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Initialize mean
            _mu = mu ?? Vector<T>.CreateDefault(actionDim, NumOps.Zero);
            
            // Initialize state to mean
            _state = new Vector<T>(_mu);
        }

        /// <summary>
        /// Applies the exploration strategy to potentially modify an action.
        /// </summary>
        /// <param name="action">The original action selected by the policy.</param>
        /// <param name="step">The current training step, used to adjust exploration parameters over time.</param>
        /// <returns>The potentially modified action after applying exploration.</returns>
        public Vector<T> ApplyExploration(Vector<T> action, long step)
        {
            // Decay noise scale
            Decay(step);

            // Update the noise state
            UpdateNoiseState();

            // Add scaled noise to the action
            var noisyAction = new Vector<T>(action.Length);
            for (int i = 0; i < action.Length; i++)
            {
                noisyAction[i] = NumOps.Add(action[i], NumOps.Multiply(_state[i], ExplorationRate));
            }

            return noisyAction;
        }

        /// <summary>
        /// Updates the internal noise state according to the Ornstein-Uhlenbeck process.
        /// </summary>
        private void UpdateNoiseState()
        {
            var dx = new Vector<T>(_actionDim);
            
            // Calculate the drift term (theta * (mu - x))
            for (int i = 0; i < _actionDim; i++)
            {
                // Drift term
                T drift = NumOps.Multiply(_theta, NumOps.Subtract(_mu[i], _state[i]));
                
                // Diffusion term (Gaussian noise scaled by sigma)
                T diffusion = NumOps.Multiply(_sigma, NumOps.FromDouble(GaussianSample()));
                
                // Combine terms
                dx[i] = NumOps.Add(drift, diffusion);
                
                // Update state
                _state[i] = NumOps.Add(_state[i], dx[i]);
            }
        }

        /// <summary>
        /// Generates a sample from a standard Gaussian distribution.
        /// </summary>
        /// <returns>A random sample from N(0,1).</returns>
        private double GaussianSample()
        {
            // Box-Muller transform to generate Gaussian samples
            double u1 = 1.0 - _random.NextDouble(); // Uniform(0,1) sample
            double u2 = 1.0 - _random.NextDouble(); // Uniform(0,1) sample
            
            // Convert uniform to standard normal
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Decays the exploration rate according to the strategy's schedule.
        /// </summary>
        /// <param name="step">The current training step.</param>
        public void Decay(long step)
        {
            if (step < _decaySteps)
            {
                // Exponential decay
                ExplorationRate = NumOps.Multiply(ExplorationRate, _decayRate);
            }
        }

        /// <summary>
        /// Resets the exploration parameters to their initial values.
        /// </summary>
        public void Reset()
        {
            _state = new Vector<T>(_mu);
        }

        /// <summary>
        /// Gets a value indicating whether the exploration is active.
        /// </summary>
        /// <param name="step">The current training step.</param>
        /// <returns>True if exploration is still active at the current step, otherwise false.</returns>
        public bool IsActive(long step)
        {
            Decay(step);
            return NumOps.GreaterThan(ExplorationRate, NumOps.FromDouble(0.001));
        }
    }
}