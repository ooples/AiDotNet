namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// Represents a batch of experiences sampled from a replay buffer.
    /// </summary>
    /// <typeparam name="TState">The type of the state representation.</typeparam>
    /// <typeparam name="TAction">The type of the action representation.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ReplayBatch<TState, TAction, T>
       
    {
        /// <summary>
        /// Gets the states before actions were taken.
        /// </summary>
        public TState[] States { get; }

        /// <summary>
        /// Gets the actions that were taken.
        /// </summary>
        public TAction[] Actions { get; }

        /// <summary>
        /// Gets the rewards that were received.
        /// </summary>
        public T[] Rewards { get; }

        /// <summary>
        /// Gets the states after actions were taken.
        /// </summary>
        public TState[] NextStates { get; }

        /// <summary>
        /// Gets the flags indicating whether episodes ended.
        /// </summary>
        public bool[] Dones { get; }

        /// <summary>
        /// Gets the indices of the experiences in the replay buffer.
        /// </summary>
        public int[] Indices { get; }

        /// <summary>
        /// Gets the importance sampling weights for prioritized experience replay.
        /// </summary>
        public T[] Weights { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ReplayBatch{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="states">The states before actions were taken.</param>
        /// <param name="actions">The actions that were taken.</param>
        /// <param name="rewards">The rewards that were received.</param>
        /// <param name="nextStates">The states after actions were taken.</param>
        /// <param name="dones">The flags indicating whether episodes ended.</param>
        /// <param name="indices">The indices of the experiences in the replay buffer.</param>
        /// <param name="weights">The importance sampling weights for prioritized experience replay.</param>
        public ReplayBatch(
            TState[] states, 
            TAction[] actions, 
            T[] rewards, 
            TState[] nextStates, 
            bool[] dones, 
            int[]? indices = null, 
            T[]? weights = null)
        {
            States = states;
            Actions = actions;
            Rewards = rewards;
            NextStates = nextStates;
            Dones = dones;
            Indices = indices ?? new int[states.Length];
            Weights = weights ?? Array.Empty<T>();
        }
        
        /// <summary>
        /// Gets the batch size (number of experiences in the batch).
        /// </summary>
        public int BatchSize => States.Length;
    }
}