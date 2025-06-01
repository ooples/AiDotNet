namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Defines the interface for a reinforcement learning exploration strategy.
    /// </summary>
    /// <typeparam name="TAction">The type used to represent actions, typically int for discrete actions or Vector<double>&lt;T&gt; for continuous.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
    /// <remarks>
    /// <para>
    /// An exploration strategy determines how an agent balances exploration (trying new actions to discover the environment)
    /// and exploitation (using known good actions to maximize reward). This interface defines methods for selecting actions
    /// with exploration and updating exploration parameters over time.
    /// </para>
    /// </remarks>
    public interface IExplorationStrategy<TAction, T>
    {
        /// <summary>
        /// Applies the exploration strategy to potentially modify an action.
        /// </summary>
        /// <param name="action">The original action selected by the policy.</param>
        /// <param name="step">The current training step, used to adjust exploration parameters over time.</param>
        /// <returns>The potentially modified action after applying exploration.</returns>
        TAction ApplyExploration(TAction action, long step);

        /// <summary>
        /// Gets the current exploration rate or parameter.
        /// </summary>
        T ExplorationRate { get; }

        /// <summary>
        /// Decays the exploration rate according to the strategy's schedule.
        /// </summary>
        /// <param name="step">The current training step.</param>
        void Decay(long step);

        /// <summary>
        /// Resets the exploration parameters to their initial values.
        /// </summary>
        void Reset();

        /// <summary>
        /// Gets a value indicating whether the strategy is suitable for continuous action spaces.
        /// </summary>
        bool IsContinuous { get; }

        /// <summary>
        /// Gets a value indicating whether the exploration is active.
        /// </summary>
        /// <param name="step">The current training step.</param>
        /// <returns>True if exploration is still active at the current step, otherwise false.</returns>
        bool IsActive(long step);
    }
}