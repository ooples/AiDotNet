namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Interface for experience replay buffers used in reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Experience replay is a technique where the agent stores past experiences and learns from them
/// multiple times. This breaks temporal correlations and improves sample efficiency.
/// </para>
/// <para><b>For Beginners:</b>
/// A replay buffer is like a memory bank for the agent. Instead of learning only from the most
/// recent experience, the agent stores experiences and learns from random samples of past experiences.
/// This makes learning more stable and efficient.
///
/// Think of it like studying for an exam - you don't just study the most recent lesson,
/// you review random material from throughout the course to learn better.
/// </para>
/// </remarks>
public interface IReplayBuffer<T>
{
    /// <summary>
    /// Gets the maximum capacity of the buffer.
    /// </summary>
    int Capacity { get; }

    /// <summary>
    /// Gets the current number of experiences in the buffer.
    /// </summary>
    int Count { get; }

    /// <summary>
    /// Adds an experience to the buffer.
    /// </summary>
    /// <param name="experience">The experience to add.</param>
    void Add(Experience<T> experience);

    /// <summary>
    /// Samples a batch of experiences from the buffer.
    /// </summary>
    /// <param name="batchSize">Number of experiences to sample.</param>
    /// <returns>List of sampled experiences.</returns>
    List<Experience<T>> Sample(int batchSize);

    /// <summary>
    /// Checks if the buffer has enough experiences to sample a batch.
    /// </summary>
    /// <param name="batchSize">The desired batch size.</param>
    /// <returns>True if buffer contains at least batchSize experiences.</returns>
    bool CanSample(int batchSize);

    /// <summary>
    /// Clears all experiences from the buffer.
    /// </summary>
    void Clear();
}
