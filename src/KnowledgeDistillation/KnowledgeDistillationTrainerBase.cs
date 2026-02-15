using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Abstract base class for all knowledge distillation trainers.
/// Provides common functionality for training loops, data shuffling, validation, and evaluation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
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
public abstract class KnowledgeDistillationTrainerBase<T, TInput, TOutput> : IKnowledgeDistillationTrainer<T, TInput, TOutput>
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
    public IDistillationStrategy<T> DistillationStrategy { get; protected set; }

    /// <summary>
    /// Checkpoint configuration for automatic model saving during training (internal).
    /// </summary>
    private readonly DistillationCheckpointConfig? _checkpointConfig;

    /// <summary>
    /// Checkpoint manager for handling checkpoint operations (internal).
    /// </summary>
    private DistillationCheckpointManager<T>? _checkpointManager;

    /// <summary>
    /// Student model reference for checkpointing (internal).
    /// </summary>
    private ICheckpointableModel? _student;

    private double _lastValidationMetric;
    private double _lastValidationLoss;
    private T _lastTrainingLoss;
    private int _currentEpoch;
    private double _bestMonitoredMetric;
    private int _patienceCounter;
    private readonly bool _useEarlyStopping;
    private readonly double _earlyStoppingMinDelta;
    private readonly int _earlyStoppingPatience;

    /// <summary>
    /// Initializes a new instance of the KnowledgeDistillationTrainerBase class.
    /// </summary>
    /// <param name="teacher">The teacher model.</param>
    /// <param name="distillationStrategy">The distillation strategy.</param>
    /// <param name="checkpointConfig">Optional checkpoint configuration for automatic model saving during training.</param>
    /// <param name="useEarlyStopping">Enable early stopping based on validation loss.</param>
    /// <param name="earlyStoppingMinDelta">Minimum improvement required to count as progress.</param>
    /// <param name="earlyStoppingPatience">Number of epochs without improvement before stopping.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The teacher and strategy are the core components:
    /// - Teacher: Provides the "expert" knowledge to transfer
    /// - Strategy: Defines how to measure and optimize the knowledge transfer</para>
    ///
    /// <para><b>Automatic Checkpointing:</b> To enable automatic checkpointing, pass a
    /// <see cref="DistillationCheckpointConfig"/> instance. If null (default), no automatic checkpointing occurs.
    /// When enabled, the trainer will automatically:
    /// - Save checkpoints based on your configuration (e.g., every 5 epochs)
    /// - Keep only the best N checkpoints to save disk space
    /// - Load the best checkpoint after training completes</para>
    ///
    /// <para><b>Example with Checkpointing:</b>
    /// <code>
    /// var config = new DistillationCheckpointConfig
    /// {
    ///     SaveEveryEpochs = 5,
    ///     KeepBestN = 3
    /// };
    /// var trainer = new KnowledgeDistillationTrainer(teacher, strategy, checkpointConfig: config);
    /// </code>
    /// </para>
    /// </remarks>
    protected KnowledgeDistillationTrainerBase(
        ITeacherModel<TInput, TOutput> teacher,
        IDistillationStrategy<T> distillationStrategy,
        DistillationCheckpointConfig? checkpointConfig = null,
        bool useEarlyStopping = false,
        double earlyStoppingMinDelta = 0.001,
        int earlyStoppingPatience = 10,
        int? seed = null)
    {
        Guard.NotNull(teacher);
        Teacher = teacher;
        Guard.NotNull(distillationStrategy);
        DistillationStrategy = distillationStrategy;
        _checkpointConfig = checkpointConfig;
        _useEarlyStopping = useEarlyStopping;
        _earlyStoppingMinDelta = earlyStoppingMinDelta;
        _earlyStoppingPatience = earlyStoppingPatience;
        _bestMonitoredMetric = double.MaxValue;
        _patienceCounter = 0;
        NumOps = MathHelper.GetNumericOperations<T>();
        Random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _lastTrainingLoss = NumOps.Zero;
        _lastValidationLoss = double.MaxValue;
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
        Vector<TInput> inputs,
        Vector<TOutput>? trueLabels = null)
    {
        if (studentForward == null) throw new ArgumentNullException(nameof(studentForward));
        if (studentBackward == null) throw new ArgumentNullException(nameof(studentBackward));
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0) throw new ArgumentException("Input batch cannot be empty.", nameof(inputs));

        int batchSize = inputs.Length;

        // Collect all outputs into matrices using ConversionsHelper for generic TOutput support
        var studentOutputsList = new List<Vector<T>>(batchSize);
        var teacherOutputsList = new List<Vector<T>>(batchSize);

        for (int i = 0; i < batchSize; i++)
        {
            var input = inputs[i];

            // Student forward pass - use ConversionsHelper to convert any TOutput to Vector<T>
            var studentOutput = studentForward(input);
            if (studentOutput == null)
            {
                throw new InvalidOperationException(
                    $"Student forward pass returned null at index {i}");
            }
            studentOutputsList.Add(ConversionsHelper.ConvertToVector<T, TOutput>(studentOutput));

            // Get teacher predictions - use ConversionsHelper to convert any TOutput to Vector<T>
            var teacherOutput = GetTeacherPredictions(input, i);
            if (teacherOutput == null)
            {
                throw new InvalidOperationException(
                    $"Teacher forward pass returned null at index {i}");
            }
            teacherOutputsList.Add(ConversionsHelper.ConvertToVector<T, TOutput>(teacherOutput));
        }

        // Convert lists to matrices
        int outputDim = studentOutputsList[0].Length;
        var studentBatchMatrix = new Matrix<T>(batchSize, outputDim);
        var teacherBatchMatrix = new Matrix<T>(batchSize, outputDim);

        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < outputDim; c++)
            {
                studentBatchMatrix[r, c] = studentOutputsList[r][c];
                teacherBatchMatrix[r, c] = teacherOutputsList[r][c];
            }
        }

        // Convert labels to matrix if provided - use ConversionsHelper for generic TOutput support
        Matrix<T>? labelsBatchMatrix = null;
        if (trueLabels != null && trueLabels.Length > 0)
        {
            labelsBatchMatrix = new Matrix<T>(batchSize, outputDim);
            for (int r = 0; r < batchSize; r++)
            {
                if (trueLabels[r] == null)
                {
                    throw new InvalidOperationException(
                        $"Label at index {r} is null");
                }
                var labelVec = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabels[r]);
                for (int c = 0; c < outputDim; c++)
                {
                    labelsBatchMatrix[r, c] = labelVec[c];
                }
            }
        }

        // Compute loss and gradient for entire batch
        var batchLoss = DistillationStrategy.ComputeLoss(studentBatchMatrix, teacherBatchMatrix, labelsBatchMatrix);
        var batchGradient = DistillationStrategy.ComputeGradient(studentBatchMatrix, teacherBatchMatrix, labelsBatchMatrix);

        // Apply gradients to each sample - use ConversionsHelper to convert Vector<T> back to TOutput
        for (int i = 0; i < batchSize; i++)
        {
            var sampleGradient = batchGradient.GetRow(i);
            studentBackward(ConversionsHelper.ConvertVectorToInputWithoutReference<T, TOutput>(sampleGradient));
        }

        return batchLoss;
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
    /// <param name="student">Optional student model for automatic checkpointing (must implement ICheckpointableModel).</param>
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
    ///
    /// <para><b>Automatic Checkpointing:</b> If a checkpoint configuration was provided to the constructor
    /// and you pass the student model parameter, automatic checkpointing will be enabled.</para>
    /// </remarks>
    public virtual void Train(
        Func<TInput, TOutput> studentForward,
        Action<TOutput> studentBackward,
        Vector<TInput> trainInputs,
        Vector<TOutput>? trainLabels = null,
        int epochs = 20,
        int batchSize = 32,
        Vector<TInput>? validationInputs = null,
        Vector<TOutput>? validationLabels = null,
        ICheckpointableModel? student = null,
        Action<int, T>? onEpochComplete = null)
    {
        if (studentForward == null) throw new ArgumentNullException(nameof(studentForward));
        if (studentBackward == null) throw new ArgumentNullException(nameof(studentBackward));
        if (trainInputs == null) throw new ArgumentNullException(nameof(trainInputs));

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive", nameof(epochs));
        if (batchSize <= 0)
            throw new ArgumentException("BatchSize must be positive", nameof(batchSize));
        if (trainLabels != null && trainInputs.Length != trainLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");
        if (validationInputs != null && validationLabels != null && validationInputs.Length != validationLabels.Length)
            throw new ArgumentException("Validation inputs and labels must have the same length");
        if (trainInputs.Length == 0)
            throw new ArgumentException("Training dataset cannot be empty.", nameof(trainInputs));

        // Store student reference for checkpointing
        _student = student;

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

                var batchInputs = new Vector<TInput>(shuffledInputs.Skip(start).Take(currentBatchSize));

                Vector<TOutput>? batchLabels = null;
                if (shuffledLabels != null)
                {
                    batchLabels = new Vector<TOutput>(shuffledLabels.Skip(start).Take(currentBatchSize));
                }

                var batchLoss = TrainBatch(studentForward, studentBackward, batchInputs, batchLabels);
                epochLoss = NumOps.Add(epochLoss, batchLoss);
            }

            var avgEpochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(numBatches));

            // Validate if requested
            if (validationInputs != null && validationLabels != null)
            {
                var valAccuracy = Evaluate(studentForward, validationInputs, validationLabels);
                _lastValidationMetric = valAccuracy;

                // Compute validation loss for early stopping
                if (_useEarlyStopping)
                {
                    // Collect validation outputs into matrices using ConversionsHelper
                    int valSize = validationInputs.Length;
                    var valStudentOutputs = new List<Vector<T>>(valSize);
                    var valTeacherOutputs = new List<Vector<T>>(valSize);

                    for (int i = 0; i < valSize; i++)
                    {
                        var studentOutput = studentForward(validationInputs[i]);
                        if (studentOutput == null)
                        {
                            throw new InvalidOperationException(
                                $"Validation student output at index {i} is null");
                        }
                        valStudentOutputs.Add(ConversionsHelper.ConvertToVector<T, TOutput>(studentOutput));

                        var teacherOutput = GetTeacherPredictions(validationInputs[i], i);
                        if (teacherOutput == null)
                        {
                            throw new InvalidOperationException(
                                $"Validation teacher output at index {i} is null");
                        }
                        valTeacherOutputs.Add(ConversionsHelper.ConvertToVector<T, TOutput>(teacherOutput));
                    }

                    // Convert to matrices
                    int valOutputDim = valStudentOutputs[0].Length;
                    var valStudentMatrix = new Matrix<T>(valSize, valOutputDim);
                    var valTeacherMatrix = new Matrix<T>(valSize, valOutputDim);

                    for (int r = 0; r < valSize; r++)
                    {
                        for (int c = 0; c < valOutputDim; c++)
                        {
                            valStudentMatrix[r, c] = valStudentOutputs[r][c];
                            valTeacherMatrix[r, c] = valTeacherOutputs[r][c];
                        }
                    }

                    // Convert labels to matrix using ConversionsHelper
                    Matrix<T>? valLabelsMatrix = null;
                    if (validationLabels != null && validationLabels.Length > 0)
                    {
                        valLabelsMatrix = new Matrix<T>(valSize, valOutputDim);
                        for (int r = 0; r < valSize; r++)
                        {
                            if (validationLabels[r] == null)
                            {
                                throw new InvalidOperationException(
                                    $"Validation label at index {r} is null");
                            }
                            var labelVec = ConversionsHelper.ConvertToVector<T, TOutput>(validationLabels[r]);
                            for (int c = 0; c < valOutputDim; c++)
                            {
                                valLabelsMatrix[r, c] = labelVec[c];
                            }
                        }
                    }

                    // Compute validation loss on entire batch
                    var valLoss = DistillationStrategy.ComputeLoss(valStudentMatrix, valTeacherMatrix, valLabelsMatrix);
                    _lastValidationLoss = Convert.ToDouble(valLoss);

                    // Check early stopping
                    if (_bestMonitoredMetric - _lastValidationLoss > _earlyStoppingMinDelta)
                    {
                        _bestMonitoredMetric = _lastValidationLoss;
                        _patienceCounter = 0;
                    }
                    else
                    {
                        _patienceCounter++;
                        if (_patienceCounter >= _earlyStoppingPatience)
                        {
                            Console.WriteLine($"Early stopping triggered at epoch {epoch + 1}. Validation loss has not improved for {_earlyStoppingPatience} epochs.");
                            OnValidationComplete(epoch, valAccuracy);
                            OnEpochEnd(epoch, avgEpochLoss);
                            onEpochComplete?.Invoke(epoch, avgEpochLoss);
                            break;
                        }
                    }
                }

                OnValidationComplete(epoch, valAccuracy);

                // Console output for epoch progress
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs}: Train Loss = {Convert.ToDouble(avgEpochLoss):F4}, Val Acc = {valAccuracy:F2}");
            }
            else
            {
                // Console output for epoch progress (no validation)
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs}: Train Loss = {Convert.ToDouble(avgEpochLoss):F4}");
            }

            // Custom logic after epoch
            OnEpochEnd(epoch, avgEpochLoss);

            // Invoke callback
            onEpochComplete?.Invoke(epoch, avgEpochLoss);
        }

        // Cleanup after training
        OnTrainingEnd(trainInputs, trainLabels);

        Console.WriteLine();
        Console.WriteLine("Knowledge Distillation training completed successfully!");
    }

    /// <summary>
    /// Trains the student model for multiple epochs (interface-compliant overload).
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass on student model.</param>
    /// <param name="studentBackward">Function to perform backward pass on student model.</param>
    /// <param name="trainInputs">Training input data.</param>
    /// <param name="trainLabels">Training labels (optional). If null, uses only teacher supervision.</param>
    /// <param name="epochs">Number of epochs to train.</param>
    /// <param name="batchSize">Batch size for training.</param>
    /// <param name="onEpochComplete">Optional callback invoked after each epoch with (epoch, avgLoss).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This overload matches the interface contract and delegates to the
    /// extended version with no validation data.</para>
    /// </remarks>
    public virtual void Train(
        Func<TInput, TOutput> studentForward,
        Action<TOutput> studentBackward,
        Vector<TInput> trainInputs,
        Vector<TOutput>? trainLabels,
        int epochs,
        int batchSize = 32,
        Action<int, T>? onEpochComplete = null)
    {
        Train(studentForward, studentBackward, trainInputs, trainLabels, epochs, batchSize, null, null, null, onEpochComplete);
    }

    /// <summary>
    /// Evaluates the student model's accuracy on a dataset.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass through student model.</param>
    /// <param name="inputs">Evaluation input data.</param>
    /// <param name="trueLabels">True labels for evaluation.</param>
    /// <returns>Accuracy as a fraction (0-1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Evaluation measures how well the student performs:
    /// - For each input, get student's prediction
    /// - Compare with true label (argmax for classification)
    /// - Calculate fraction of correct predictions</para>
    ///
    /// <para>This is used to monitor training progress and detect overfitting.</para>
    /// </remarks>
    public virtual double Evaluate(
        Func<TInput, TOutput> studentForward,
        Vector<TInput> inputs,
        Vector<TOutput> trueLabels)
    {
        if (studentForward == null) throw new ArgumentNullException(nameof(studentForward));
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (trueLabels == null) throw new ArgumentNullException(nameof(trueLabels));

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

        return (double)correct / inputs.Length;
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
    protected virtual (Vector<TInput> shuffledInputs, Vector<TOutput>? shuffledLabels) ShuffleData(
        Vector<TInput> inputs,
        Vector<TOutput>? labels)
    {
        var indices = FisherYatesShuffle(inputs.Length);
        var shuffledInputs = new Vector<TInput>(indices.Select(i => inputs[i]));
        Vector<TOutput>? shuffledLabels = null;

        if (labels != null)
        {
            shuffledLabels = new Vector<TOutput>(indices.Select(i => labels[i]));
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
    /// Determines if a prediction matches the true label.
    /// </summary>
    /// <param name="prediction">Student's prediction.</param>
    /// <param name="trueLabel">True label.</param>
    /// <returns>True if prediction is correct.</returns>
    /// <remarks>
    /// <para><b>For Classification:</b> Uses argmax to find predicted class and compares with true class.
    /// The conversion to Vector&lt;T&gt; is handled by ConversionsHelper, so this works for any TOutput type
    /// (Vector&lt;T&gt;, Tensor&lt;T&gt;, T[], or scalar T).</para>
    /// <para><b>Override This:</b> If you need different evaluation criteria (e.g., regression).</para>
    /// </remarks>
    protected virtual bool IsCorrectPrediction(TOutput prediction, TOutput trueLabel)
    {
        // Use ConversionsHelper to convert any TOutput to Vector<T>
        var predVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
        var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);

        return ArgMax(predVector) == ArgMax(labelVector);
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
    ///
    /// <para><b>Automatic Checkpointing:</b> If checkpoint configuration was provided to the constructor,
    /// this method automatically initializes the checkpoint manager.</para>
    /// </remarks>
    protected virtual void OnTrainingStart(Vector<TInput> trainInputs, Vector<TOutput>? trainLabels)
    {
        // Initialize checkpoint manager if config is provided
        if (_checkpointConfig != null)
        {
            _checkpointManager = new DistillationCheckpointManager<T>(_checkpointConfig);
        }
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
    ///
    /// <para><b>Automatic Checkpointing:</b> If checkpoint configuration was provided to the constructor,
    /// this method automatically loads the best checkpoint (based on validation metrics) after training completes.</para>
    /// </remarks>
    protected virtual void OnTrainingEnd(Vector<TInput> trainInputs, Vector<TOutput>? trainLabels)
    {
        // Load best checkpoint if checkpointing was enabled
        if (_checkpointManager != null && _student != null)
        {
            var bestCheckpoint = _checkpointManager.LoadBestCheckpoint(
                student: _student,
                teacher: Teacher as ICheckpointableModel
            );

            if (bestCheckpoint != null)
            {
                Console.WriteLine($"[Checkpointing] Loaded best checkpoint from epoch {bestCheckpoint.Epoch}");
                if (bestCheckpoint.Metrics.ContainsKey(_checkpointConfig!.BestMetric))
                {
                    Console.WriteLine($"[Checkpointing] Best {_checkpointConfig.BestMetric}: {bestCheckpoint.Metrics[_checkpointConfig.BestMetric]:F4}");
                }
            }
        }
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
    protected virtual void OnEpochStart(int epoch, Vector<TInput> trainInputs, Vector<TOutput>? trainLabels)
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
    ///
    /// <para><b>IMPORTANT:</b> This base implementation calls Reset() on RelationalDistillationStrategy
    /// to flush partial batches and prevent buffer leakage between epochs. Derived classes should
    /// call base.OnEpochEnd() if they override this method.</para>
    ///
    /// <para><b>Automatic Checkpointing:</b> If checkpoint configuration was provided to the constructor,
    /// this method automatically saves checkpoints based on your configuration.</para>
    /// </remarks>
    protected virtual void OnEpochEnd(int epoch, T avgLoss)
    {
        // Reset RelationalDistillationStrategy buffers at epoch boundaries to prevent leakage
        // This ensures partial batches are flushed and amortization uses correct counts
        if (DistillationStrategy is Strategies.RelationalDistillationStrategy<T> relationalStrategy)
        {
            relationalStrategy.Reset();
        }

        // Track current state for checkpointing
        _currentEpoch = epoch;
        _lastTrainingLoss = avgLoss;

        // Automatic checkpoint saving
        if (_checkpointManager != null)
        {
            var metrics = new Dictionary<string, double>
            {
                { "training_loss", Convert.ToDouble(_lastTrainingLoss) }
            };

            // Include validation metric if available
            if (_lastValidationMetric > 0)
            {
                // Use "validation_accuracy" since _lastValidationMetric stores accuracy, not loss
                metrics["validation_accuracy"] = _lastValidationMetric;
            }

            _checkpointManager.SaveCheckpointIfNeeded(
                epoch: epoch,
                student: _student,
                teacher: Teacher as ICheckpointableModel,
                strategy: DistillationStrategy,
                metrics: metrics
            );
        }
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
    ///
    /// <para><b>Automatic Checkpointing:</b> If <see cref="CheckpointConfig"/> is set, this method
    /// automatically tracks validation metrics for best checkpoint selection.</para>
    /// </remarks>
    protected virtual void OnValidationComplete(int epoch, double accuracy)
    {
        // Track validation metric for checkpointing
        _lastValidationMetric = accuracy;
    }

    /// <summary>
    /// Collects intermediate activations from a layered model by performing a forward pass
    /// and recording the output of each layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method runs input data through the model layer by layer,
    /// recording what each layer outputs. These intermediate outputs (activations) can then be
    /// compared between a teacher and student model for hint-based distillation (FitNets-style).</para>
    ///
    /// <para><b>How it works:</b></para>
    /// <list type="number">
    /// <item><description>The model must implement <see cref="ILayeredModel{T}"/></description></item>
    /// <item><description>For each layer, it performs a forward pass and records the output as a matrix</description></item>
    /// <item><description>Activations are keyed by layer name for matching between teacher and student</description></item>
    /// </list>
    ///
    /// <para><b>Research References:</b></para>
    /// <list type="bullet">
    /// <item><description>FitNets (Romero et al., 2015): Hint-based distillation matching intermediate features</description></item>
    /// <item><description>Attention Transfer (Zagoruyko &amp; Komodakis, 2017): Transfer attention maps between layers</description></item>
    /// </list>
    /// </remarks>
    /// <param name="model">The neural network model (must implement <see cref="ILayeredModel{T}"/>).</param>
    /// <param name="input">The input tensor to forward through the model.</param>
    /// <param name="targetCategories">Optional set of layer categories to collect activations from.
    /// If null, collects from all trainable layers.</param>
    /// <returns>An <see cref="IntermediateActivations{T}"/> containing per-layer activations,
    /// or an empty instance if the model is not layer-aware.</returns>
    protected IntermediateActivations<T> CollectIntermediateActivations(
        IFullModel<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        HashSet<LayerCategory>? targetCategories = null)
    {
        var activations = new IntermediateActivations<T>();

        if (model is not ILayeredModel<T> layeredModel)
        {
            return activations;
        }

        var allInfo = layeredModel.GetAllLayerInfo();
        var current = input;

        for (int i = 0; i < allInfo.Count; i++)
        {
            var info = allInfo[i];

            // Forward through this layer
            current = info.Layer.Forward(current);

            // Only collect from targeted categories (or all trainable if no filter)
            bool shouldCollect = targetCategories is not null
                ? targetCategories.Contains(info.Category)
                : info.IsTrainable;

            if (shouldCollect)
            {
                // Convert tensor output to matrix: [batchSize, features]
                var outputShape = current.Shape;
                if (outputShape.Length == 0 || outputShape[0] <= 0 || current.Length == 0)
                    continue;

                int batchSize = outputShape.Length >= 2 ? outputShape[0] : 1;

                if (current.Length % batchSize != 0)
                {
                    throw new InvalidOperationException(
                        $"Tensor length ({current.Length}) is not evenly divisible by batch size ({batchSize}) " +
                        $"at layer '{info.Name}' (index {i}). The output shape {string.Join("x", outputShape)} " +
                        "is not compatible with batch-major flattening.");
                }

                int features = current.Length / batchSize;

                var matrix = new Matrix<T>(batchSize, features);
                var flatData = current.ToVector();

                for (int b = 0; b < batchSize; b++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        matrix[b, f] = flatData[b * features + f];
                    }
                }

                activations.Add(info.Name, matrix);
            }
        }

        return activations;
    }

    /// <summary>
    /// Computes intermediate activation loss between teacher and student for hint-based distillation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After collecting activations from both teacher and student, this method
    /// computes how different their internal representations are. The student is then trained to minimize
    /// this difference, which helps it learn not just the final outputs but also the internal reasoning
    /// patterns of the teacher.</para>
    /// </remarks>
    /// <param name="studentActivations">The student's intermediate activations.</param>
    /// <param name="teacherActivations">The teacher's intermediate activations.</param>
    /// <returns>The total intermediate loss across all matched layers.</returns>
    protected T ComputeIntermediateActivationLoss(
        IntermediateActivations<T> studentActivations,
        IntermediateActivations<T> teacherActivations)
    {
        // If the distillation strategy supports intermediate activations, delegate to it
        if (DistillationStrategy is IIntermediateActivationStrategy<T> intermediateStrategy)
        {
            return intermediateStrategy.ComputeIntermediateLoss(studentActivations, teacherActivations);
        }

        // Default: compute MSE loss between matched layers
        T totalLoss = NumOps.Zero;
        int matchedLayers = 0;

        foreach (var layerName in studentActivations.LayerNames)
        {
            var studentMatrix = studentActivations.Get(layerName);
            var teacherMatrix = teacherActivations.Get(layerName);

            if (studentMatrix is null || teacherMatrix is null)
            {
                continue;
            }

            // Require matching dimensions - mismatched shapes indicate a configuration error
            if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[KnowledgeDistillation] Skipping layer '{layerName}': shape mismatch between " +
                    $"student ({studentMatrix.Rows}x{studentMatrix.Columns}) and " +
                    $"teacher ({teacherMatrix.Rows}x{teacherMatrix.Columns}). " +
                    "Ensure teacher and student layer mappings produce matching dimensions.");
                continue;
            }

            // Compute per-element MSE between student and teacher activations
            int rows = studentMatrix.Rows;
            int cols = studentMatrix.Columns;

            T layerLoss = NumOps.Zero;
            int elementCount = rows * cols;

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    T diff = NumOps.Subtract(studentMatrix[r, c], teacherMatrix[r, c]);
                    layerLoss = NumOps.Add(layerLoss, NumOps.Multiply(diff, diff));
                }
            }

            if (elementCount > 0)
            {
                layerLoss = NumOps.Divide(layerLoss, NumOps.FromDouble(elementCount));
            }

            totalLoss = NumOps.Add(totalLoss, layerLoss);
            matchedLayers++;
        }

        // Average across matched layers
        if (matchedLayers > 0)
        {
            totalLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(matchedLayers));
        }

        return totalLoss;
    }
}
