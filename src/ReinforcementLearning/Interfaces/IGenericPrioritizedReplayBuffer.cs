using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Interface for a prioritized replay buffer that supports experience prioritization with generic action types.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IPrioritizedReplayBuffer<TState, TAction, T> : IReplayBuffer<TState, TAction, T>
    {
        /// <summary>
        /// Updates priorities for specific experiences in the buffer.
        /// </summary>
        void UpdatePriority(int index, T priority);

        /// <summary>
        /// Samples a batch of experiences with importance sampling weights.
        /// </summary>
        ReplayBatch<TState, TAction, T> SamplePrioritizedBatch(int batchSize, T beta);
    }
}