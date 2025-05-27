namespace AiDotNet.ReinforcementLearning.Memory
{
    /// <summary>
    /// Represents a single reinforcement learning experience tuple.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class Experience<TState, TAction, T>
    {
        /// <summary>
        /// Gets or sets the state before the action was taken.
        /// </summary>
        public TState State { get; set; }

        /// <summary>
        /// Gets or sets the action that was taken.
        /// </summary>
        public TAction Action { get; set; }

        /// <summary>
        /// Gets or sets the reward received after taking the action.
        /// </summary>
        public T Reward { get; set; }

        /// <summary>
        /// Gets or sets the state after the action was taken.
        /// </summary>
        public TState NextState { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the episode ended after this action.
        /// </summary>
        public bool Done { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Experience{TState, TAction, T}"/> class.
        /// </summary>
        public Experience()
        {
            State = default!;
            Action = default!;
            Reward = default!;
            NextState = default!;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Experience{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public Experience(TState state, TAction action, T reward, TState nextState, bool done)
        {
            State = state;
            Action = action;
            Reward = reward;
            NextState = nextState;
            Done = done;
        }
    }
}