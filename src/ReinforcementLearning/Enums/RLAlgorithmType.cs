namespace AiDotNet.ReinforcementLearning.Enums;

/// <summary>
/// Defines the different types of reinforcement learning algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Reinforcement learning algorithms can be categorized into different families based on their approach
/// to learning. This enum provides a comprehensive taxonomy of RL algorithm types.
/// </para>
/// <para><b>For Beginners:</b> This lists different approaches to teaching AI agents.
///
/// Think of these as different teaching methods:
/// - Some methods teach by showing examples (supervised)
/// - Some methods teach by trial and error (reinforcement learning)
/// - Within RL, there are different strategies for learning
///
/// The main categories:
/// - Value-Based: Learn which situations are valuable
/// - Policy-Based: Learn which actions to take
/// - Actor-Critic: Combine both approaches
/// - Model-Based: Learn how the world works, then plan
///
/// Choose based on your problem:
/// - Discrete actions? Try value-based (DQN)
/// - Continuous actions? Try policy-based (PPO) or actor-critic (SAC)
/// - Need sample efficiency? Try model-based
/// </para>
/// </remarks>
public enum RLAlgorithmType
{
    /// <summary>
    /// Deep Q-Network: Value-based algorithm for discrete action spaces.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DQN learns to estimate "how good is each action in this situation?"
    ///
    /// Best for:
    /// - Discrete actions (button presses, menu selections)
    /// - Simple to moderate complexity
    /// - When you have lots of data
    ///
    /// Examples: Atari games, grid worlds, simple robot control
    /// </para>
    /// </remarks>
    DQN,

    /// <summary>
    /// Double DQN: Reduces overestimation bias in DQN.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> An improved version of DQN that makes more accurate predictions.
    ///
    /// Fixes: DQN tends to be too optimistic about action values
    /// Result: More stable and reliable learning
    ///
    /// Use this instead of regular DQN for better results with minimal extra cost.
    /// </para>
    /// </remarks>
    DoubleDQN,

    /// <summary>
    /// Dueling DQN: Separates state value and action advantages.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A DQN variant that learns "how good is this situation?" separately from "how good is each action?"
    ///
    /// Better for: Environments where some states are generally good/bad regardless of action
    /// Example: In a racing game, being near the track is generally good, specific steering is the advantage
    ///
    /// Often performs better than regular DQN, especially in complex environments.
    /// </para>
    /// </remarks>
    DuelingDQN,

    /// <summary>
    /// Rainbow DQN: Combines multiple DQN improvements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The "kitchen sink" DQN with all the best improvements combined.
    ///
    /// Includes:
    /// - Double Q-learning
    /// - Prioritized replay
    /// - Dueling networks
    /// - Multi-step learning
    /// - Distributional RL
    /// - Noisy networks
    ///
    /// State-of-the-art for value-based methods, but more complex to implement and tune.
    /// </para>
    /// </remarks>
    RainbowDQN,

    /// <summary>
    /// REINFORCE: Basic policy gradient algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Directly learns a policy (action strategy) using gradient ascent.
    ///
    /// Best for:
    /// - Continuous action spaces
    /// - Stochastic policies needed
    /// - Simple problems
    ///
    /// Pros: Simple, works with continuous actions
    /// Cons: High variance, slow learning, needs lots of data
    ///
    /// Usually a stepping stone to better algorithms like PPO.
    /// </para>
    /// </remarks>
    REINFORCE,

    /// <summary>
    /// Actor-Critic: Combines policy gradient (actor) with value function (critic).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Two neural networks working together: one decides actions, one evaluates them.
    ///
    /// The actor: "I think we should do this"
    /// The critic: "That's a good/bad idea because..."
    ///
    /// Better than pure policy gradient because the critic reduces variance.
    /// Foundation for advanced algorithms like A2C, A3C, SAC, TD3.
    /// </para>
    /// </remarks>
    ActorCritic,

    /// <summary>
    /// Advantage Actor-Critic (A2C): Synchronous version of A3C.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> An improved actor-critic that focuses on "advantages" (how much better is this action than average?).
    ///
    /// Best for:
    /// - Continuous or discrete actions
    /// - Faster learning than REINFORCE
    /// - Moderate complexity problems
    ///
    /// Popular for: Many continuous control tasks, robotics
    /// </para>
    /// </remarks>
    A2C,

    /// <summary>
    /// Asynchronous Advantage Actor-Critic (A3C): Parallel training version of A2C.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A2C but with multiple agents learning in parallel.
    ///
    /// Pros: Faster training, more stable, explores more diverse strategies
    /// Cons: Requires more computational resources
    ///
    /// Good for: When you have multiple CPU cores and want faster training
    /// </para>
    /// </remarks>
    A3C,

    /// <summary>
    /// Proximal Policy Optimization: State-of-the-art policy gradient method.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Currently one of the best all-around RL algorithms.
    ///
    /// Why it's popular:
    /// - Good sample efficiency
    /// - Stable training
    /// - Works well out-of-the-box
    /// - Handles both continuous and discrete actions
    ///
    /// Used by: OpenAI, DeepMind, many production systems
    /// Great for: Most problems, especially continuous control, robotics
    ///
    /// Often the first algorithm to try for new problems.
    /// </para>
    /// </remarks>
    PPO,

    /// <summary>
    /// Trust Region Policy Optimization: Theoretical foundation for PPO.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like PPO but more complex, with stronger theoretical guarantees.
    ///
    /// Ensures: Policy updates are not too large (stays in "trust region")
    /// Result: Very stable learning
    ///
    /// Pros: Theoretically sound, very stable
    /// Cons: More complex, computationally expensive
    ///
    /// Most people use PPO instead (simpler, almost as good).
    /// </para>
    /// </remarks>
    TRPO,

    /// <summary>
    /// Deep Deterministic Policy Gradient: Actor-critic for continuous actions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Made for continuous action spaces (like robot joint angles).
    ///
    /// Deterministic: Always outputs same action for same state (no randomness)
    /// Exploration: Uses noise added to actions
    ///
    /// Best for: Continuous control, robotics, control systems
    /// Examples: Robot manipulation, autonomous vehicles
    ///
    /// Foundation for more advanced algorithms like TD3 and SAC.
    /// </para>
    /// </remarks>
    DDPG,

    /// <summary>
    /// Twin Delayed DDPG: Improved version of DDPG.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DDPG with three key improvements to make it more reliable.
    ///
    /// Improvements:
    /// 1. Two critics (twin): More accurate value estimates
    /// 2. Delayed updates: Update policy less often than critic
    /// 3. Target policy smoothing: Add noise to reduce overfitting
    ///
    /// Result: Much more stable than DDPG
    /// Use this instead of DDPG for most continuous control tasks.
    /// </para>
    /// </remarks>
    TD3,

    /// <summary>
    /// Soft Actor-Critic: Maximum entropy actor-critic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Currently one of the best algorithms for continuous control.
    ///
    /// Special feature: Encourages exploration by maximizing "entropy" (randomness)
    /// Result: More robust policies that explore well
    ///
    /// Best for:
    /// - Continuous control
    /// - Robotics
    /// - Tasks requiring robust policies
    ///
    /// Often outperforms TD3 and PPO on continuous control tasks.
    /// </para>
    /// </remarks>
    SAC,

    /// <summary>
    /// Model-Based RL: Learns a model of the environment dynamics.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Learns how the world works, then uses that knowledge to plan.
    ///
    /// Two steps:
    /// 1. Learn a model: Predict what happens when you take actions
    /// 2. Use the model: Plan or simulate to choose actions
    ///
    /// Pros: Very sample efficient (learns faster with less data)
    /// Cons: More complex, model errors can compound
    ///
    /// Best for: Expensive real-world interactions (robots, control systems)
    /// </para>
    /// </remarks>
    ModelBased,

    /// <summary>
    /// Multi-Agent RL: Multiple agents learning simultaneously.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multiple AI agents learning together (cooperative or competitive).
    ///
    /// Challenges:
    /// - Non-stationary environment (other agents are changing)
    /// - Credit assignment (who contributed to the outcome?)
    /// - Coordination (how do agents work together?)
    ///
    /// Examples: Team games, traffic control, negotiation, swarm robotics
    ///
    /// Algorithms: MADDPG, QMIX, CommNet
    /// </para>
    /// </remarks>
    MultiAgent,

    /// <summary>
    /// Offline RL: Learn from fixed dataset without environment interaction.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Learn from a recorded dataset, can't try new things.
    ///
    /// Use when:
    /// - Environment interaction is expensive or dangerous
    /// - You have lots of historical data
    /// - Real-time interaction isn't possible
    ///
    /// Examples:
    /// - Healthcare: Learn from patient records
    /// - Autonomous driving: Learn from recorded trips
    /// - Recommendations: Learn from user history
    ///
    /// Algorithms: CQL, IQL, Decision Transformer
    ///
    /// Harder than regular RL because can't explore - must extract value from existing data.
    /// </para>
    /// </remarks>
    OfflineRL,

    /// <summary>
    /// Inverse RL: Learn reward function from expert demonstrations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Learn what the goal is by watching an expert.
    ///
    /// Regular RL: Given a goal (reward), learn how to achieve it
    /// Inverse RL: Given expert behavior, figure out what their goal was
    ///
    /// Use when:
    /// - Hard to specify reward function
    /// - Have expert demonstrations
    /// - Want agent to imitate human values/goals
    ///
    /// Examples: Learn driving preferences, imitate expert pilot, discover human objectives
    ///
    /// Often combined with imitation learning and apprenticeship learning.
    /// </para>
    /// </remarks>
    InverseRL
}
