using AiDotNet.ReinforcementLearning.Memory;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Interface for a sequential replay buffer that stores trajectories of experiences.
    /// </summary>
    /// <typeparam name="TState">The type of state observation.</typeparam>
    /// <typeparam name="TAction">The type of action.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface ISequentialReplayBuffer<TState, TAction, T>
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
        /// Adds an experience to the buffer.
        /// </summary>
        /// <param name="state">The state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next state observation.</param>
        /// <param name="done">Whether the episode is done.</param>
        void Add(TState state, TAction action, T reward, TState nextState, bool done);
        
        /// <summary>
        /// Samples a batch of trajectories from the buffer.
        /// </summary>
        /// <param name="batchSize">The number of trajectories to sample.</param>
        /// <returns>A batch of sampled trajectories.</returns>
        TrajectoryBatch<TState, TAction, T> SampleBatch(int batchSize);
        
        /// <summary>
        /// Clears all experiences from the buffer.
        /// </summary>
        void Clear();
    }
}