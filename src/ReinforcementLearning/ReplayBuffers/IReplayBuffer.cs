namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Interface for experience replay buffers used in reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TState">The type representing the state observation (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TAction">The type representing the action (e.g., Vector&lt;T&gt; for continuous, int for discrete).</typeparam>
/// <remarks>
/// <para>
/// Experience replay is a technique where the agent stores past experiences and learns from them
/// multiple times. This breaks temporal correlations and improves sample efficiency. The replay
/// buffer stores experience tuples (state, action, reward, next_state, done) and provides random
/// sampling for training.
/// </para>
/// <para><b>For Beginners:</b>
/// A replay buffer is like a memory bank for the agent. Instead of learning only from the most
/// recent experience, the agent stores experiences and learns from random samples of past experiences.
/// This makes learning more stable and efficient.
///
/// Think of it like studying for an exam:
/// - You don't just study the most recent lesson
/// - You review random material from throughout the course
/// - This helps you learn connections between different topics
/// - And prevents forgetting older material
///
/// **Common Buffer Types:**
/// - **Uniform**: All experiences sampled with equal probability
/// - **Prioritized**: Important experiences (big errors) sampled more often
/// </para>
/// </remarks>
public interface IReplayBuffer<T, TState, TAction>
{
    /// <summary>
    /// Gets the maximum capacity of the buffer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the maximum number of experiences the buffer can hold.
    /// Once full, old experiences are typically replaced with new ones (FIFO).
    /// Common values: 10,000 to 1,000,000 depending on available memory.
    /// </remarks>
    int Capacity { get; }

    /// <summary>
    /// Gets the current number of experiences in the buffer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How many experiences are currently stored.
    /// Training usually starts only when the buffer has enough experiences (e.g., Count >= BatchSize).
    /// </remarks>
    int Count { get; }

    /// <summary>
    /// Adds an experience to the buffer.
    /// </summary>
    /// <param name="experience">The experience to add.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Call this after each step in the environment to store
    /// what happened for later learning. If the buffer is full, the oldest experience
    /// is typically removed to make room.
    /// </remarks>
    void Add(Experience<T, TState, TAction> experience);

    /// <summary>
    /// Samples a batch of experiences from the buffer.
    /// </summary>
    /// <param name="batchSize">Number of experiences to sample.</param>
    /// <returns>List of sampled experiences.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Randomly selects experiences for training.
    /// Random sampling breaks temporal correlations (nearby experiences being similar)
    /// which helps the neural network learn more stable patterns.
    /// </remarks>
    List<Experience<T, TState, TAction>> Sample(int batchSize);

    /// <summary>
    /// Checks if the buffer has enough experiences to sample a batch.
    /// </summary>
    /// <param name="batchSize">The desired batch size.</param>
    /// <returns>True if buffer contains at least batchSize experiences.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Check this before sampling to avoid errors.
    /// Training should wait until CanSample(batchSize) returns true.
    /// </remarks>
    bool CanSample(int batchSize);

    /// <summary>
    /// Clears all experiences from the buffer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Removes all stored experiences. Use this when starting
    /// fresh training or when the environment changes significantly.
    /// </remarks>
    void Clear();
}
