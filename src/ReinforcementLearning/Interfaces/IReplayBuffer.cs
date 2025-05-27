using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Interface for a replay buffer that stores and samples experiences for reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IReplayBuffer<TState, T>
    {
        /// <summary>
        /// Gets the current number of experiences in the buffer.
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Gets the maximum capacity of the buffer.
        /// </summary>
        int Capacity { get; }

        /// <summary>
        /// Adds a new experience to the buffer.
        /// </summary>
        void Add(TState state, int action, T reward, TState nextState, bool done);

        /// <summary>
        /// Samples a batch of experiences from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A batch of experiences.</returns>
        ReplayBatch<TState, T> Sample(int batchSize);

        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        void Clear();
    }
}