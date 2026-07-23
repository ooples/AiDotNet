using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for continuous action space policies in reinforcement learning.
    /// Continuous policies output actions as real-valued vectors using Gaussian (normal) distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Continuous policies are essential for reinforcement learning in environments where actions are
    /// real-valued rather than discrete choices. Common applications include robotic control (joint angles,
    /// velocities, torques), autonomous driving (steering angle, acceleration), and financial trading
    /// (position sizes, portfolio weights). The policy network typically outputs both the mean (μ) and
    /// standard deviation (σ) of a Gaussian distribution for each action dimension, enabling the agent
    /// to express uncertainty and explore through stochastic sampling.
    /// </para>
    /// <para>
    /// This configuration provides defaults optimized for continuous control tasks, based on best practices
    /// from algorithms like SAC (Soft Actor-Critic), PPO (Proximal Policy Optimization), and TD3 (Twin Delayed
    /// DDPG). The larger default network size [256, 256] compared to discrete policies reflects the higher
    /// complexity typically required for smooth continuous control.
    /// </para>
    /// <para><b>For Beginners:</b> Continuous policies are for when your actions are numbers on a scale
    /// rather than discrete choices.
    ///
    /// Think of the difference:
    /// - Discrete: "Turn left, right, or go straight" (3 choices)
    /// - Continuous: "Turn the wheel 17.3 degrees" (infinite precision)
    ///
    /// Real-world examples:
    /// - Robot arm: How much to rotate each joint (0° to 180°)
    /// - Self-driving car: Steering angle (-30° to +30°), acceleration (-5 to +5 m/s²)
    /// - Temperature control: Set thermostat (60°F to 80°F)
    ///
    /// The policy learns a "range of good actions" for each situation:
    /// - Mean: The average/best action to take
    /// - Standard deviation: How much to vary around that (exploration)
    ///
    /// During training: Sample actions from this range (adds randomness for exploration)
    /// During evaluation: Use the mean action (most confident choice)
    ///
    /// This options class lets you configure the network that learns these action ranges.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    public class ContinuousPolicyOptions<T> : ModelOptions
    {
        /// <summary>
        /// Gets or sets the size of the observation/state space.
        /// </summary>
        /// <value>The number of input features describing the environment state. Must be greater than 0.</value>
        /// <remarks>
        /// <para>
        /// For continuous control tasks, state representations often include positions, velocities, accelerations,
        /// and other physical quantities. For example, a quadrotor might have 12-dimensional state (3D position,
        /// 3D orientation, 3D linear velocity, 3D angular velocity). The state size directly impacts the network's
        /// input layer size and should match the environment's observation space exactly.
        /// </para>
        /// <para><b>For Beginners:</b> How many numbers describe the current situation?
        ///
        /// Examples for continuous control:
        /// - Pendulum: 2 numbers (angle, angular velocity)
        /// - Car: 4 numbers (X position, Y position, heading angle, speed)
        /// - Humanoid robot: 376 numbers (joint angles, velocities, body positions)
        ///
        /// Continuous tasks often have larger state spaces than discrete ones because they track
        /// precise physical quantities rather than simplified representations.
        /// </para>
        /// </remarks>
        public int StateSize { get; set; } = 0;

        /// <summary>
        /// Gets or sets the dimensionality of the continuous action space.
        /// </summary>
        /// <value>The number of continuous action dimensions. Must be greater than 0.</value>
        /// <remarks>
        /// <para>
        /// Each action dimension represents an independent continuous control variable. The policy network
        /// outputs 2 × ActionSize values: mean and log-standard-deviation for each dimension's Gaussian
        /// distribution. Common dimensionalities range from 1 (simple control like temperature) to 20+
        /// (complex robots with many joints). Higher dimensionality makes learning harder due to the
        /// exponential growth of the action space volume.
        /// </para>
        /// <para><b>For Beginners:</b> How many different continuous values does your agent control?
        ///
        /// Examples:
        /// - Thermostat: 1 dimension (temperature setpoint)
        /// - 2D navigation: 2 dimensions (forward/backward speed, turning rate)
        /// - Robot arm: 6 dimensions (one for each joint)
        /// - Quadrotor: 4 dimensions (thrust for each rotor)
        ///
        /// Each dimension is independent, so a 4-dimensional action space means the agent outputs
        /// 4 separate numbers each step. More dimensions = harder to learn, but necessary for
        /// complex control tasks.
        /// </para>
        /// </remarks>
        public int ActionSize { get; set; } = 0;

        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new GaussianNoiseExploration<T>();
        public bool UseTanhSquashing { get; set; } = false;
        public new int? Seed { get; set; } = null;
    }
}
