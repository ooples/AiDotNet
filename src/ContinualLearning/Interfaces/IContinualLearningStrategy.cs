using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Represents a strategy for preventing catastrophic forgetting in continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A continual learning strategy defines HOW to prevent forgetting
/// when learning new tasks. Different strategies use different techniques:</para>
///
/// <para>
/// - <b>EWC (Elastic Weight Consolidation):</b> Protects important parameters from changing
/// - <b>LwF (Learning without Forgetting):</b> Uses knowledge distillation
/// - <b>GEM (Gradient Episodic Memory):</b> Ensures gradients don't hurt previous tasks
/// - <b>Experience Replay:</b> Intermixes old and new data during training
/// </para>
/// </remarks>
public interface IContinualLearningStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes the strategy for a new task.
    /// </summary>
    /// <param name="model">The model to be trained.</param>
    /// <param name="taskData">Data for the new task.</param>
    void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData);

    /// <summary>
    /// Computes the regularization loss to prevent forgetting.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <returns>The regularization loss value.</returns>
    T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Adjusts the gradients to prevent forgetting previous tasks.
    /// </summary>
    /// <param name="gradients">The computed gradients for the current batch.</param>
    /// <returns>The adjusted gradients.</returns>
    Vector<T> AdjustGradients(Vector<T> gradients);

    /// <summary>
    /// Finalizes learning after completing a task (e.g., storing important parameters).
    /// </summary>
    /// <param name="model">The trained model.</param>
    void FinalizeTask(IFullModel<T, TInput, TOutput> model);
}
