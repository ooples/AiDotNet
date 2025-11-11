using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Abstract base class for all knowledge distillation trainers.
/// Provides common functionality for training loops, data shuffling, validation, and evaluation.
/// </summary>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class implements the common training workflow shared by all
/// distillation trainers. Specific trainer types (standard, self-distillation, online, etc.) inherit
/// from this and customize only the parts that differ.</para>
///
/// <para><b>Design Pattern:</b> Template Method Pattern - the base class defines the training algorithm
/// structure, and derived classes fill in specific steps.</para>
///
/// <para><b>Common Functionality Provided:</b>
/// - Data shuffling using Fisher-Yates algorithm (O(n) efficiency)
/// - Epoch and batch management
/// - Validation after each epoch
/// - Progress callbacks
/// - Evaluation metrics (accuracy, loss)
/// - Teacher and strategy property management</para>
///
/// <para><b>Derived Classes Override:</b>
/// - GetTeacherPredictions(): How to obtain teacher outputs for training
/// - OnEpochStart(): Custom logic before each epoch
/// - OnEpochEnd(): Custom logic after each epoch
/// - OnTrainingStart(): Custom logic before training begins
/// - OnTrainingEnd(): Custom logic after training completes</para>
/// </remarks>
public abstract class KnowledgeDistillationTrainerBase<TInput, TOutput, T> : IKnowledgeDistillationTrainer<TInput, TOutput, T>
{
    /// <summary>
    /// Gets the numeric operations helper for the type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the random number generator for data shuffling.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Gets the teacher model used for distillation.
    /// </summary>
    public ITeacherModel<TInput, TOutput> Teacher { get; protected set; }

    /// <summary>
    /// Gets the distillation strategy for computing loss and gradients.
    /// </summary>
    public IDistillationStrategy<TOutput, T> DistillationStrategy { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the KnowledgeDistillationTrainerBase class.
    /// </summary>
    /// <param name="teacher">The teacher model.</param>
    /// <param name="distillationStrategy">The distillation strategy.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The teacher and strategy are the core components:
    /// - Teacher: Provides the "expert" knowledge to transfer
    /// - Strategy: Defines how to measure and optimize the knowledge transfer</para>
    /// </remarks>
    protected KnowledgeDistillationTrainerBase(
        ITeacherModel<TInput, TOutput> teacher,
        IDistillationStrategy<TOutput, T> distillationStrategy,
        int? seed = null)
    {
        Teacher = teacher ?? throw new ArgumentNullException(nameof(teacher));
        DistillationStrategy = distillationStrategy ?? throw new ArgumentNullException(nameof(distillationStrategy));
        NumOps = MathHelper.GetNumericOperations<T>();
        Random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Trains the student model on a single batch using knowledge distillation.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass through student model.</param>
    /// <param name="studentBackward">Function to perform backward pass and update weights.</param>
    /// <param name="inputs">Batch of input data.</param>
    /// <param name="trueLabels">Optional true labels for the batch.</param>
    /// <returns>Average loss for the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training a batch involves:
    /// 1. Get student predictions (forward pass)
    /// 2. Get teacher predictions (from teacher model or cache)
    /// 3. Compute distillation loss (how different student is from teacher)
    /// 4. Compute gradients (how to improve student)
    /// 5. Update student weights (backward pass)</para>
    /// </remarks>
    public virtual T TrainBatch(
        Func<TInput, TOutput> studentForward,
        Action<TOutput> studentBackward,
        TInput[] inputs,
        TOutput[]? trueLabels = null)
    {
        ArgumentNullException.ThrowIfNull(studentForward);
        ArgumentNullException.ThrowIfNull(studentBackward);
        ArgumentNullException.ThrowIfNull(inputs);

        T totalLoss = NumOps.Zero;

        for (int i = 0; i < inputs.Length; i++)
        {
            var input = inputs[i];
            var label = trueLabels?[i];

            // Student forward pass
            var studentOutput = studentForward(input);

            // Get teacher predictions (may be cached or computed on-demand)
            var teacherOutput = GetTeacherPredictions(input, i);

            // Compute loss and gradient
            var loss = DistillationStrategy.ComputeLoss(studentOutput, teacherOutput, label);
            var gradient = DistillationStrategy.ComputeGradient(studentOutput, teacherOutput, label);

            totalLoss = NumOps.Add(totalLoss, loss);

            // Student backward pass
            studentBackward(gradient);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(inputs.Length));
    }

    /// <summary>
    /// Trains the student model for multiple epochs using knowledge distillation.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass through student model.</param>
    /// <param name="studentBackward">Function to perform backward pass and update weights.</param>
    /// <param name="trainInputs">Training input data.</param>
    /// <param name="trainLabels">Training labels (optional for pure soft distillation).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="batchSize">Batch size for mini-batch training.</param>
    /// <param name="validationInputs">Optional validation inputs for monitoring.</param>
    /// <param name="validationLabels">Optional validation labels.</param>
    /// <param name="onEpochComplete">Optional callback invoked after each epoch with (epoch, avgLoss).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method orchestrates the complete training process:
    /// 1. Initialize (OnTrainingStart)
    /// 2. For each epoch:
    ///    a. Shuffle training data
    ///    b. Process data in batches
    ///    c. Validate if requested
    ///    d. Invoke callbacks
    /// 3. Cleanup (OnTrainingEnd)</para>
    ///
    /// <para><b>Training Tips:</b>
    /// - Use batch sizes that fit in memory (32-128 typical)
    /// - Monitor validation loss to detect overfitting
    /// - Invoke callbacks to log progress or save checkpoints</para>
    /// </remarks>
    public virtual void Train(
        Func<TInput, TOutput> studentForward,
        Action<TOutput> studentBackward,
        TInput[] trainInputs,
        TOutput[]? trainLabels = null,
        int epochs = 20,
        int batchSize = 32,
        TInput[]? validationInputs = null,
        TOutput[]? validationLabels = null,
        Action<int, T>? onEpochComplete = null)
    {
        ArgumentNullException.ThrowIfNull(studentForward);
        ArgumentNullException.ThrowIfNull(studentBackward);
        ArgumentNullException.ThrowIfNull(trainInputs);

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive", nameof(epochs));
        if (batchSize <= 0)
            throw new ArgumentException("BatchSize must be positive", nameof(batchSize));
        if (trainLabels != null && trainInputs.Length != trainLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");
        if (validationInputs != null && validationLabels != null && validationInputs.Length != validationLabels.Length)
            throw new ArgumentException("Validation inputs and labels must have the same length");

        // Prepare for training
        OnTrainingStart(trainInputs, trainLabels);

        int numBatches = (trainInputs.Length + batchSize - 1) / batchSize;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Custom logic before epoch
            OnEpochStart(epoch, trainInputs, trainLabels);

            T epochLoss = NumOps.Zero;

            // Shuffle data
            var (shuffledInputs, shuffledLabels) = ShuffleData(trainInputs, trainLabels);

            // Train on batches
            for (int b = 0; b < numBatches; b++)
            {
                int start = b * batchSize;
                int end = Math.Min(start + batchSize, trainInputs.Length);
                int currentBatchSize = end - start;

                var batchInputs = new TInput[currentBatchSize];
                Array.Copy(shuffledInputs, start, batchInputs, 0, currentBatchSize);

                TOutput[]? batchLabels = null;
                if (shuffledLabels != null)
                {
                    batchLabels = new TOutput[currentBatchSize];
                    Array.Copy(shuffledLabels, start, batchLabels, 0, currentBatchSize);
                }

                var batchLoss = TrainBatch(studentForward, studentBackward, batchInputs, batchLabels);
                epochLoss = NumOps.Add(epochLoss, batchLoss);
            }

            var avgEpochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(numBatches));

            // Validate if requested
            if (validationInputs != null && validationLabels != null)
            {
                var valAccuracy = Evaluate(studentForward, validationInputs, validationLabels);
                OnValidationComplete(epoch, valAccuracy);
            }

            // Custom logic after epoch
            OnEpochEnd(epoch, avgEpochLoss);

            // Invoke callback
            onEpochComplete?.Invoke(epoch, avgEpochLoss);
        }

        // Cleanup after training
        OnTrainingEnd(trainInputs, trainLabels);
    }

    /// <summary>
    /// Evaluates the student model's accuracy on a dataset.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass through student model.</param>
    /// <param name="inputs">Evaluation input data.</param>
    /// <param name="trueLabels">True labels for evaluation.</param>
    /// <returns>Accuracy as a percentage (0-100).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Evaluation measures how well the student performs:
    /// - For each input, get student's prediction
    /// - Compare with true label (argmax for classification)
    /// - Calculate percentage of correct predictions</para>
    ///
    /// <para>This is used to monitor training progress and detect overfitting.</para>
    /// </remarks>
    public virtual double Evaluate(
        Func<TInput, TOutput> studentForward,
        TInput[] inputs,
        TOutput[] trueLabels)
    {
        ArgumentNullException.ThrowIfNull(studentForward);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(trueLabels);

        if (inputs.Length != trueLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");

        if (inputs.Length == 0)
            return 0.0;

        int correct = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            var prediction = studentForward(inputs[i]);
            if (IsCorrectPrediction(prediction, trueLabels[i]))
                correct++;
        }

        return (double)correct / inputs.Length * 100.0;
    }

    /// <summary>
    /// Shuffles training data using Fisher-Yates algorithm.
    /// </summary>
    /// <param name="inputs">Input data to shuffle.</param>
    /// <param name="labels">Labels to shuffle (maintains alignment with inputs).</param>
    /// <returns>Tuple of shuffled inputs and labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Data shuffling is important because:
    /// - Prevents model from learning order-dependent patterns
    /// - Improves gradient descent convergence
    /// - Reduces overfitting to batch ordering</para>
    ///
    /// <para>Fisher-Yates is O(n) compared to O(n log n) for sort-based shuffling.</para>
    /// </remarks>
    protected virtual (TInput[] shuffledInputs, TOutput[]? shuffledLabels) ShuffleData(
        TInput[] inputs,
        TOutput[]? labels)
    {
        var indices = FisherYatesShuffle(inputs.Length);
        var shuffledInputs = indices.Select(i => inputs[i]).ToArray();
        TOutput[]? shuffledLabels = null;

        if (labels != null)
        {
            shuffledLabels = indices.Select(i => labels[i]).ToArray();
        }

        return (shuffledInputs, shuffledLabels);
    }

    /// <summary>
    /// Generates a random permutation of indices using Fisher-Yates shuffle.
    /// </summary>
    /// <param name="length">The length of the array to shuffle.</param>
    /// <returns>An array of shuffled indices.</returns>
    protected int[] FisherYatesShuffle(int length)
    {
        var indices = Enumerable.Range(0, length).ToArray();
        for (int i = length - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        return indices;
    }

    /// <summary>
    /// Gets teacher predictions for a given input. Abstract method to be implemented by derived classes.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="index">The index of this input in the training batch (useful for caching).</param>
    /// <returns>Teacher's output predictions.</returns>
    /// <remarks>
    /// <para><b>For Derived Classes:</b>
    /// - Standard trainers: Call teacher.GetLogits(input)
    /// - Self-distillation: Return cached predictions from previous generation
    /// - Online distillation: Get predictions from dynamically updated teacher
    /// - Ensemble: Combine predictions from multiple teachers</para>
    /// </remarks>
    protected abstract TOutput GetTeacherPredictions(TInput input, int index);

    /// <summary>
    /// Determines if a prediction matches the true label. Default implementation for Vector outputs.
    /// </summary>
    /// <param name="prediction">Student's prediction.</param>
    /// <param name="trueLabel">True label.</param>
    /// <returns>True if prediction is correct.</returns>
    /// <remarks>
    /// <para><b>For Classification:</b> Uses argmax to find predicted class and compares with true class.</para>
    /// <para><b>Override This:</b> If you have different output types or evaluation criteria.</para>
    /// </remarks>
    protected virtual bool IsCorrectPrediction(TOutput prediction, TOutput trueLabel)
    {
        // Default implementation for Vector<T> - finds argmax
        if (prediction is Vector<T> predVector && trueLabel is Vector<T> labelVector)
        {
            return ArgMax(predVector) == ArgMax(labelVector);
        }

        throw new NotImplementedException(
            $"IsCorrectPrediction must be overridden for output type {typeof(TOutput).Name}");
    }

    /// <summary>
    /// Finds the index of the maximum value in a vector (argmax).
    /// </summary>
    /// <param name="vector">The vector to search.</param>
    /// <returns>Index of the maximum value.</returns>
    protected int ArgMax(Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        int maxIndex = 0;
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Called before training starts. Override for custom initialization logic.
    /// </summary>
    /// <param name="trainInputs">Training inputs.</param>
    /// <param name="trainLabels">Training labels.</param>
    /// <remarks>
    /// <para><b>Use Cases:</b>
    /// - Cache teacher predictions (for efficiency)
    /// - Initialize EMA buffers (for self-distillation)
    /// - Setup curriculum schedules
    /// - Allocate temporary buffers</para>
    /// </remarks>
    protected virtual void OnTrainingStart(TInput[] trainInputs, TOutput[]? trainLabels)
    {
        // Default: no-op, derived classes can override
    }

    /// <summary>
    /// Called after training completes. Override for custom cleanup logic.
    /// </summary>
    /// <param name="trainInputs">Training inputs.</param>
    /// <param name="trainLabels">Training labels.</param>
    /// <remarks>
    /// <para><b>Use Cases:</b>
    /// - Clear caches
    /// - Save final checkpoints
    /// - Log final metrics
    /// - Free temporary resources</para>
    /// </remarks>
    protected virtual void OnTrainingEnd(TInput[] trainInputs, TOutput[]? trainLabels)
    {
        // Default: no-op, derived classes can override
    }

    /// <summary>
    /// Called before each epoch starts. Override for custom per-epoch initialization.
    /// </summary>
    /// <param name="epoch">Current epoch number (0-indexed).</param>
    /// <param name="trainInputs">Training inputs.</param>
    /// <param name="trainLabels">Training labels.</param>
    /// <remarks>
    /// <para><b>Use Cases:</b>
    /// - Update learning rate schedules
    /// - Adjust temperature schedules
    /// - Update curriculum difficulty
    /// - Refresh teacher in online distillation</para>
    /// </remarks>
    protected virtual void OnEpochStart(int epoch, TInput[] trainInputs, TOutput[]? trainLabels)
    {
        // Default: no-op, derived classes can override
    }

    /// <summary>
    /// Called after each epoch completes. Override for custom per-epoch cleanup/logging.
    /// </summary>
    /// <param name="epoch">Current epoch number (0-indexed).</param>
    /// <param name="avgLoss">Average loss for this epoch.</param>
    /// <remarks>
    /// <para><b>Use Cases:</b>
    /// - Log epoch metrics
    /// - Save checkpoints
    /// - Update adaptive parameters
    /// - Implement early stopping logic</para>
    /// </remarks>
    protected virtual void OnEpochEnd(int epoch, T avgLoss)
    {
        // Default: no-op, derived classes can override
    }

    /// <summary>
    /// Called after validation completes for an epoch. Override for custom validation handling.
    /// </summary>
    /// <param name="epoch">Current epoch number (0-indexed).</param>
    /// <param name="accuracy">Validation accuracy (0-100).</param>
    /// <remarks>
    /// <para><b>Use Cases:</b>
    /// - Log validation metrics
    /// - Implement early stopping
    /// - Track best model
    /// - Adjust hyperparameters based on validation performance</para>
    /// </remarks>
    protected virtual void OnValidationComplete(int epoch, double accuracy)
    {
        // Default: no-op, derived classes can override
    }
}
