using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Represents a single experience tuple (s, a, r, s', done) for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An experience is one step of interaction with the environment.
/// It contains everything the agent needs to learn from that step:
/// - What the situation was (State)
/// - What the agent did (Action)
/// - What reward it got (Reward)
/// - What happened next (NextState)
/// - Whether the episode ended (Done)
/// </remarks>
public record Experience<T>(
    Vector<T> State,
    Vector<T> Action,
    T Reward,
    Vector<T> NextState,
    bool Done)
{
    /// <summary>
    /// Optional priority for prioritized experience replay.
    /// </summary>
    public double Priority { get; set; } = 1.0;
}
