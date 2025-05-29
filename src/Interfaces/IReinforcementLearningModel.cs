namespace AiDotNet.Interfaces;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Defines the common interface for all reinforcement learning algorithms in the AiDotNet library.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface specifies the contract that all reinforcement learning models must fulfill,
/// including methods for selecting actions, updating from experience, and accessing model parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> Reinforcement learning is a type of machine learning where an agent learns to make
/// decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.
/// 
/// For example, reinforcement learning can be used to:
/// - Train a robot to navigate through complex environments
/// - Develop AI agents that play games (like chess, Go, or video games)
/// - Optimize resource allocation in systems like data centers
/// - Create trading algorithms that adapt to market conditions
/// 
/// Unlike supervised learning (which learns from labeled examples), reinforcement learning
/// learns through trial and error based on feedback from the environment.
/// </para>
/// </remarks>
public interface IReinforcementLearningModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets whether the model uses a continuous action space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates whether the agent operates in a continuous action space (true)
    /// or a discrete action space (false). This affects how actions are represented and processed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you whether the agent's actions are:
    /// - Continuous: Actions represented as floating-point values that can vary smoothly 
    ///   (like steering wheel angles, motor speeds, or joint torques)
    /// - Discrete: Actions represented as distinct choices from a fixed set 
    ///   (like "move left", "move right", or selecting from numbered options)
    /// </para>
    /// </remarks>
    bool IsContinuous { get; }

    /// <summary>
    /// Selects an action based on the current state.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether to use exploration during action selection.</param>
    /// <returns>The selected action.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the agent's action in a given state. During training,
    /// it may incorporate exploration to try new actions. During evaluation, it typically
    /// selects the best action according to the learned policy.
    /// </para>
    /// </remarks>
    Vector<T> SelectAction(Tensor<T> state, bool isTraining = false);

    /// <summary>
    /// Updates the model based on experience with the environment.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    /// <returns>The loss value from the update.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the agent's knowledge based on a single step of interaction
    /// with the environment (a transition). It stores the experience and may perform
    /// a learning update if enough experiences have been collected.
    /// </para>
    /// </remarks>
    T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done);


    
    /// <summary>
    /// Gets the last computed loss value.
    /// </summary>
    /// <returns>The last computed loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the loss value from the most recent training update.
    /// It can be used to monitor the agent's learning progress.
    /// </para>
    /// </remarks>
    T GetLoss();
    
    /// <summary>
    /// Sets the model to training mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In training mode, the agent may use exploration strategies and update its parameters
    /// based on experiences.
    /// </para>
    /// </remarks>
    void SetTrainingMode();
    
    /// <summary>
    /// Sets the model to evaluation mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In evaluation mode, the agent typically uses a deterministic policy without exploration
    /// and does not update its parameters.
    /// </para>
    /// </remarks>
    void SetEvaluationMode();
}