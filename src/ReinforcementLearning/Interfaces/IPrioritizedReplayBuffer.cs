using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Interface for a prioritized replay buffer that supports experience prioritization.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IPrioritizedReplayBuffer<TState, T> : IReplayBuffer<TState, T>
    {
        /// <summary>
        /// Updates priorities for specific experiences in the buffer.
        /// </summary>
        void UpdatePriority(int index, T priority);

        /// <summary>
        /// Samples a batch of experiences with importance sampling weights.
        /// </summary>
        ReplayBatch<TState, T> SamplePrioritized(int batchSize, T beta);
    }
}