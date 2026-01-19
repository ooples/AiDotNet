using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Marker interface for reinforcement learning agents that integrate with AiModelBuilder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This interface extends IFullModel to ensure RL agents integrate seamlessly with AiDotNet's
/// existing architecture. RL agents are models where:
/// - TInput = Tensor&lt;T&gt; (state observations, though often flattened to Vector in practice)
/// - TOutput = Vector&lt;T&gt; (actions)
/// </para>
/// <para><b>For Beginners:</b>
/// An RL agent is just a special kind of model that learns through interaction with an environment.
/// By implementing IFullModel, RL agents work with all of AiDotNet's existing infrastructure:
/// - They can be saved and loaded
/// - They work with the AiModelBuilder pattern
/// - They support serialization, cloning, etc.
///
/// The key difference is how they're trained:
/// - Regular models: trained on fixed datasets (x, y)
/// - RL agents: trained by interacting with environments and getting rewards
/// </para>
/// </remarks>
public interface IRLAgent<T> : IFullModel<T, Vector<T>, Vector<T>>
{
    /// <summary>
    /// Selects an action given the current state observation.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="explore">Whether to use exploration (epsilon-greedy, etc.).</param>
    /// <returns>Action as a Vector (one-hot for discrete, continuous values for continuous action spaces).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is how the agent decides what to do in a given situation.
    /// During training, it might explore (try random things), but during evaluation it uses its learned policy.
    /// </remarks>
    Vector<T> SelectAction(Vector<T> state, bool explore = true);

    /// <summary>
    /// Stores an experience tuple for later learning.
    /// </summary>
    /// <param name="state">The state before taking action.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The state after taking action.</param>
    /// <param name="done">Whether the episode terminated.</param>
    /// <remarks>
    /// <b>For Beginners:</b> RL agents learn from experiences. This stores one experience
    /// (state, action, reward, next state) for the agent to learn from later.
    /// </remarks>
    void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done);

    /// <summary>
    /// Performs one training step using stored experiences.
    /// </summary>
    /// <returns>Training loss for monitoring.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is where the agent actually learns from its experiences.
    /// It looks at what happened (stored experiences) and updates its strategy to get better rewards.
    /// </remarks>
    T Train();

    /// <summary>
    /// Gets current training metrics.
    /// </summary>
    /// <returns>Dictionary of metric names to values.</returns>
    Dictionary<string, T> GetMetrics();

    /// <summary>
    /// Resets episode-specific state (if any).
    /// </summary>
    void ResetEpisode();
}
