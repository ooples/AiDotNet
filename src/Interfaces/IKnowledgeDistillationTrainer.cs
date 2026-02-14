namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for knowledge distillation trainers that train student models
/// using knowledge transferred from teacher models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Matrix&lt;T&gt; for batch outputs, Tensor&lt;T&gt; for structured outputs).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A knowledge distillation trainer orchestrates the process of
/// transferring knowledge from a large, accurate teacher model to a smaller, faster student model.</para>
///
/// <para><b>Why an interface?</b>
/// - **Flexibility**: Multiple trainer implementations (standard, self-distillation, multi-teacher, etc.)
/// - **Testability**: Easy to mock for unit testing
/// - **Extensibility**: New training strategies can be added without breaking existing code
/// - **Dependency Injection**: Can be injected into other components</para>
///
/// <para><b>Common Implementations:</b>
/// - **Standard Trainer**: Single teacher → single student
/// - **Self-Distillation Trainer**: Model teaches itself (improves calibration)
/// - **Multi-Teacher Trainer**: Multiple teachers → one student (ensemble distillation)
/// - **Online Trainer**: Teacher updates during student training
/// - **Mutual Learning Trainer**: Multiple students learn from each other</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("KnowledgeDistillationTrainer")]
public interface IKnowledgeDistillationTrainer<T, TInput, TOutput>
{
    /// <summary>
    /// Trains the student model on a single batch of data.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass on student model.</param>
    /// <param name="studentBackward">Function to perform backward pass on student model (takes gradient).</param>
    /// <param name="inputs">Input batch.</param>
    /// <param name="trueLabels">Ground truth labels (optional). If null, uses only teacher supervision.</param>
    /// <returns>Average loss for the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method processes one batch of training data:
    /// 1. Get predictions from both teacher and student
    /// 2. Compute distillation loss
    /// 3. Update student weights via backpropagation</para>
    /// </remarks>
    T TrainBatch(
        Func<TInput, TOutput> studentForward,
        Action<TOutput> studentBackward,
        Vector<TInput> inputs,
        Vector<TOutput>? trueLabels = null);

    /// <summary>
    /// Trains the student model for multiple epochs.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass on student model.</param>
    /// <param name="studentBackward">Function to perform backward pass on student model.</param>
    /// <param name="trainInputs">Training input data.</param>
    /// <param name="trainLabels">Training labels (optional). If null, uses only teacher supervision.</param>
    /// <param name="epochs">Number of epochs to train.</param>
    /// <param name="batchSize">Batch size for training.</param>
    /// <param name="onEpochComplete">Optional callback invoked after each epoch with (epoch, avgLoss).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> An epoch is one complete pass through the training data.
    /// This method runs the complete training process for the specified number of epochs.</para>
    /// </remarks>
    void Train(
        Func<TInput, TOutput> studentForward,
        Action<TOutput> studentBackward,
        Vector<TInput> trainInputs,
        Vector<TOutput>? trainLabels,
        int epochs,
        int batchSize = 32,
        Action<int, T>? onEpochComplete = null);

    /// <summary>
    /// Evaluates the student model's accuracy on test data.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass on student model.</param>
    /// <param name="testInputs">Test input data.</param>
    /// <param name="testLabels">Test labels (one-hot encoded).</param>
    /// <returns>Classification accuracy as a percentage (0-100).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well the student has learned.
    /// Returns a percentage between 0 and 100, where 100.0 means perfect accuracy.</para>
    /// </remarks>
    double Evaluate(
        Func<TInput, TOutput> studentForward,
        Vector<TInput> testInputs,
        Vector<TOutput> testLabels);

    /// <summary>
    /// Gets the teacher model used for distillation.
    /// </summary>
    ITeacherModel<TInput, TOutput> Teacher { get; }

    /// <summary>
    /// Gets the distillation strategy used for computing loss and gradients.
    /// </summary>
    IDistillationStrategy<T> DistillationStrategy { get; }
}
