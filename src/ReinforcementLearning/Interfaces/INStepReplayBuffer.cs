namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Interface for a replay buffer that supports n-step returns.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface INStepReplayBuffer<TState, T> : IReplayBuffer<TState, T>
    {
        /// <summary>
        /// Gets the number of steps used for n-step returns.
        /// </summary>
        int NSteps { get; }

        /// <summary>
        /// Gets the discount factor used for n-step returns.
        /// </summary>
        T Gamma { get; }
    }
}