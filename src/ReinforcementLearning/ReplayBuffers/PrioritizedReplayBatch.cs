namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// Represents a batch of experiences sampled from a prioritized replay buffer.
    /// </summary>
    /// <typeparam name="TState">Type of state representation.</typeparam>
    /// <typeparam name="TAction">Type of action representation.</typeparam>
    /// <typeparam name="T">The numeric type used for computations.</typeparam>
    public class PrioritizedReplayBatch<TState, TAction, T> : ReplayBatch<TState, TAction, T>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PrioritizedReplayBatch{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="states">Array of states.</param>
        /// <param name="actions">Array of actions.</param>
        /// <param name="rewards">Array of rewards.</param>
        /// <param name="nextStates">Array of next states.</param>
        /// <param name="dones">Array of done flags.</param>
        /// <param name="indices">Array of indices in the replay buffer.</param>
        /// <param name="weights">Array of importance sampling weights.</param>
        public PrioritizedReplayBatch(TState[] states, TAction[] actions, T[] rewards, 
            TState[] nextStates, bool[] dones, int[] indices, T[] weights)
            : base(states, actions, rewards, nextStates, dones, indices, weights)
        {
        }
    }
}