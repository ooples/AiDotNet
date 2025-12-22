using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;

namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Strategy interface for continual learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Different continual learning methods use different strategies
/// to prevent forgetting. This interface allows the trainer to work with any strategy.</para>
///
/// <para><b>Strategy Types:</b>
/// <list type="bullet">
/// <item><description><b>Regularization-based:</b> Add penalty terms to protect important weights (EWC, SI, MAS)</description></item>
/// <item><description><b>Replay-based:</b> Store and replay old examples (Experience Replay, GEM)</description></item>
/// <item><description><b>Architecture-based:</b> Use separate parameters for different tasks (Progressive Networks, PackNet)</description></item>
/// <item><description><b>Distillation-based:</b> Use teacher model to preserve old knowledge (LwF)</description></item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> De Lange et al. "A Continual Learning Survey: Defying Forgetting" (2021)</para>
/// </remarks>
public interface IContinualLearningStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether this strategy requires storing examples from previous tasks.
    /// </summary>
    bool RequiresMemoryBuffer { get; }

    /// <summary>
    /// Gets whether this strategy modifies the model architecture.
    /// </summary>
    bool ModifiesArchitecture { get; }

    /// <summary>
    /// Gets the current memory usage of the strategy in bytes.
    /// </summary>
    long MemoryUsageBytes { get; }

    /// <summary>
    /// Prepares the strategy for learning a new task.
    /// </summary>
    /// <param name="model">The model being trained.</param>
    /// <param name="taskData">The data for the new task.</param>
    /// <remarks>
    /// <para>This is called at the start of training on a new task. Strategies use this to:
    /// - EWC: Store current parameters as optimal for previous tasks
    /// - LwF: Create a copy of the model as the teacher
    /// - GEM: Compute reference gradients from stored examples
    /// </para>
    /// </remarks>
    void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData);

    /// <summary>
    /// Computes the regularization loss to prevent forgetting.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <returns>The regularization loss value.</returns>
    /// <remarks>
    /// <para>This is added to the task loss during training:
    /// - EWC: Sum of (current params - optimal params)^2 * Fisher Information
    /// - SI: Sum of (current params - optimal params)^2 * importance
    /// - MAS: Sum of (current params - optimal params)^2 * gradient magnitude
    /// </para>
    /// </remarks>
    T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Adjusts gradients to prevent forgetting.
    /// </summary>
    /// <param name="gradients">The gradients from the current task loss.</param>
    /// <returns>Adjusted gradients that respect previous task constraints.</returns>
    /// <remarks>
    /// <para>This modifies gradients before they're applied:
    /// - GEM: Project gradients to not increase loss on previous tasks
    /// - A-GEM: Project onto average gradient of reference samples
    /// - OWM: Project gradients into orthogonal subspace
    /// </para>
    /// </remarks>
    Vector<T> AdjustGradients(Vector<T> gradients);

    /// <summary>
    /// Finalizes the task after training is complete.
    /// </summary>
    /// <param name="model">The trained model.</param>
    /// <remarks>
    /// <para>This is called after training on a task completes. Strategies use this to:
    /// - EWC: Compute Fisher Information Matrix
    /// - SI: Compute path integral importance
    /// - PackNet: Prune and freeze weights
    /// </para>
    /// </remarks>
    void FinalizeTask(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Resets the strategy to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Saves the strategy state to a file.
    /// </summary>
    /// <param name="path">Path to save the state.</param>
    void Save(string path);

    /// <summary>
    /// Loads the strategy state from a file.
    /// </summary>
    /// <param name="path">Path to load the state from.</param>
    void Load(string path);

    /// <summary>
    /// Gets strategy-specific metrics for monitoring.
    /// </summary>
    /// <returns>Dictionary of metric name to value.</returns>
    IReadOnlyDictionary<string, object> GetMetrics();

    /// <summary>
    /// Validates that the strategy is compatible with the given model.
    /// </summary>
    /// <param name="model">The model to validate against.</param>
    /// <returns>True if compatible, false otherwise.</returns>
    bool IsCompatibleWith(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Gets a description of why the strategy is incompatible with a model.
    /// </summary>
    /// <param name="model">The model to check.</param>
    /// <returns>Description of incompatibility, or null if compatible.</returns>
    string? GetIncompatibilityReason(IFullModel<T, TInput, TOutput> model);
}

/// <summary>
/// Extended strategy interface for strategies that store task examples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IMemoryBasedStrategy<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of examples stored in memory.
    /// </summary>
    int StoredExampleCount { get; }

    /// <summary>
    /// Gets the maximum number of examples that can be stored.
    /// </summary>
    int MaxExamples { get; }

    /// <summary>
    /// Stores examples from a completed task.
    /// </summary>
    /// <param name="taskData">The task data to sample from.</param>
    void StoreTaskExamples(IDataset<T, TInput, TOutput> taskData);

    /// <summary>
    /// Samples a batch of stored examples for replay.
    /// </summary>
    /// <param name="batchSize">Number of examples to sample.</param>
    /// <returns>List of input-output pairs.</returns>
    IReadOnlyList<(TInput Input, TOutput Output, int TaskId)> SampleExamples(int batchSize);

    /// <summary>
    /// Clears all stored examples.
    /// </summary>
    void ClearMemory();
}

/// <summary>
/// Extended strategy interface for knowledge distillation-based strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IDistillationStrategy<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the teacher model (frozen copy from before current task).
    /// </summary>
    IFullModel<T, TInput, TOutput>? TeacherModel { get; }

    /// <summary>
    /// Gets the distillation temperature.
    /// </summary>
    T Temperature { get; }

    /// <summary>
    /// Gets the weight for distillation loss vs task loss.
    /// </summary>
    T DistillationWeight { get; }

    /// <summary>
    /// Computes the distillation loss between teacher and student outputs.
    /// </summary>
    /// <param name="teacherOutput">Output from the teacher model.</param>
    /// <param name="studentOutput">Output from the current student model.</param>
    /// <returns>The distillation loss.</returns>
    T ComputeDistillationLoss(TOutput teacherOutput, TOutput studentOutput);
}

/// <summary>
/// Extended strategy interface for gradient-based constraint strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IGradientConstraintStrategy<T, TInput, TOutput> : IMemoryBasedStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of tasks with stored gradients.
    /// </summary>
    int StoredGradientCount { get; }

    /// <summary>
    /// Stores the gradient for a completed task.
    /// </summary>
    /// <param name="taskGradient">The average gradient for the task.</param>
    void StoreTaskGradient(Vector<T> taskGradient);

    /// <summary>
    /// Projects a gradient to satisfy all task constraints.
    /// </summary>
    /// <param name="gradient">The gradient to project.</param>
    /// <returns>The projected gradient.</returns>
    Vector<T> ProjectGradient(Vector<T> gradient);

    /// <summary>
    /// Checks if a gradient violates any task constraint.
    /// </summary>
    /// <param name="gradient">The gradient to check.</param>
    /// <returns>True if any constraint is violated.</returns>
    bool ViolatesConstraint(Vector<T> gradient);
}
