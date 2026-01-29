namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Represents a single experience tuple (s, a, r, s', done) for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TState">The type representing the state observation (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TAction">The type representing the action (e.g., Vector&lt;T&gt; for continuous, int for discrete).</typeparam>
/// <remarks>
/// <para>
/// An Experience is a fundamental data structure in reinforcement learning that captures a single
/// interaction between an agent and its environment. It consists of five components: the current state,
/// the action taken, the reward received, the resulting next state, and a flag indicating whether
/// the episode has ended. This tuple is used to train reinforcement learning agents in algorithms
/// like Q-learning, Deep Q-Networks (DQN), PPO, and many others.
/// </para>
/// <para><b>For Beginners:</b> An experience is one step of interaction with the environment.
/// It contains everything the agent needs to learn from that step:
///
/// - **State**: What the situation looked like before the agent acted (like a snapshot)
/// - **Action**: What the agent decided to do
/// - **Reward**: The feedback received (positive = good, negative = bad, zero = neutral)
/// - **NextState**: What the situation looks like after the action
/// - **Done**: Whether this action ended the episode (game over, goal reached, etc.)
///
/// For example, in a maze-solving robot:
/// - State: Robot's current position and sensor readings
/// - Action: "move forward" or "turn left"
/// - Reward: +10 for reaching the exit, -1 for hitting a wall, 0 otherwise
/// - NextState: Robot's new position after the action
/// - Done: True if robot reached the exit or got stuck
///
/// **Common Type Combinations:**
/// - `Experience&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;` - For continuous actions (e.g., robotic control)
/// - `Experience&lt;double, Vector&lt;double&gt;, int&gt;` - For discrete actions (e.g., game playing)
/// - `Experience&lt;float, Tensor&lt;float&gt;, int&gt;` - For image-based states (e.g., Atari games)
/// </para>
/// </remarks>
public record Experience<T, TState, TAction>(
    TState State,
    TAction Action,
    T Reward,
    TState NextState,
    bool Done)
{
    /// <summary>
    /// Gets or sets the priority for prioritized experience replay.
    /// </summary>
    /// <value>A double representing the experience's sampling priority. Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// In prioritized experience replay, experiences with higher priority are sampled more frequently.
    /// The priority is typically based on the TD-error (temporal difference error), meaning experiences
    /// that surprise the agent (large prediction errors) are replayed more often.
    /// </para>
    /// <para><b>For Beginners:</b> Priority determines how often this experience gets picked for learning.
    ///
    /// Think of it like highlighting important notes in a textbook:
    /// - Higher priority = more important = reviewed more often
    /// - Experiences where the agent made big mistakes get higher priority
    /// - This helps the agent learn from its most surprising or educational moments
    ///
    /// Default is 1.0 (all experiences equal). Values greater than 1.0 mean "sample this more often."
    /// </para>
    /// </remarks>
    public double Priority { get; set; } = 1.0;
}

/// <summary>
/// Simplified Experience record for Vector-based states and actions.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public record Experience<T>(
    Vector<T> State,
    Vector<T> Action,
    T Reward,
    Vector<T> NextState,
    bool Done) : Experience<T, Vector<T>, Vector<T>>(State, Action, Reward, NextState, Done);
