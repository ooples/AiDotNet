using AiDotNet.Training;
using AiDotNet.Training.Configuration;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for training machine learning models from configuration-driven recipes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A trainer takes a training recipe (configuration) and runs the full
/// training process: loading data, creating the model, running epochs, and returning results.
/// This is the "run my experiment" interface.
/// </para>
/// </remarks>
public interface ITrainer<T>
{
    /// <summary>
    /// Gets the training recipe configuration used by this trainer.
    /// </summary>
    TrainingRecipeConfig Config { get; }

    /// <summary>
    /// Runs the full training loop and returns the result.
    /// </summary>
    /// <returns>A <see cref="TrainingResult{T}"/> containing the trained model, loss history, and metadata.</returns>
    TrainingResult<T> Run();

    /// <summary>
    /// Runs the full training loop asynchronously and returns the result.
    /// </summary>
    /// <returns>A task that resolves to a <see cref="TrainingResult{T}"/> containing the trained model, loss history, and metadata.</returns>
    Task<TrainingResult<T>> RunAsync();
}
