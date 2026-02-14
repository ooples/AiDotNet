namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Interface for learning rate schedulers that adjust the learning rate during training.
/// </summary>
/// <remarks>
/// <para>
/// Learning rate schedulers are essential for training neural networks effectively. They adjust
/// the learning rate according to various strategies, enabling better convergence and final performance.
/// </para>
/// <para><b>For Beginners:</b> The learning rate controls how big each step is when the model is learning.
/// A scheduler automatically adjusts this step size during training - typically starting with larger steps
/// to make fast progress, then smaller steps to fine-tune the solution. Think of it like driving:
/// you go faster on the highway (early training) and slow down as you approach your destination (later training).
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LearningRateScheduler")]
public interface ILearningRateScheduler
{
    /// <summary>
    /// Gets the current learning rate.
    /// </summary>
    double CurrentLearningRate { get; }

    /// <summary>
    /// Gets the base (initial) learning rate.
    /// </summary>
    double BaseLearningRate { get; }

    /// <summary>
    /// Gets the current step (iteration or epoch count depending on scheduler type).
    /// </summary>
    int CurrentStep { get; }

    /// <summary>
    /// Advances the scheduler by one step and returns the new learning rate.
    /// </summary>
    /// <returns>The updated learning rate for the next step.</returns>
    double Step();

    /// <summary>
    /// Gets the learning rate for a specific step without advancing the scheduler.
    /// </summary>
    /// <param name="step">The step number to get the learning rate for.</param>
    /// <returns>The learning rate at the specified step.</returns>
    double GetLearningRateAtStep(int step);

    /// <summary>
    /// Resets the scheduler to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the scheduler state for serialization/checkpointing.
    /// </summary>
    /// <returns>A dictionary containing the scheduler state.</returns>
    Dictionary<string, object> GetState();

    /// <summary>
    /// Loads the scheduler state from a checkpoint.
    /// </summary>
    /// <param name="state">The state dictionary to load from.</param>
    void LoadState(Dictionary<string, object> state);
}
