using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Trains a student model using knowledge distillation from a teacher model.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class orchestrates the knowledge distillation training process.
/// It takes a large, accurate teacher model and uses it to train a smaller, faster student model.</para>
///
/// <para>The training process works as follows:
/// 1. For each input, get predictions from both teacher and student
/// 2. Compute distillation loss (how different are their predictions?)
/// 3. Update student parameters to minimize this loss
/// 4. Repeat until student learns to mimic teacher</para>
///
/// <para><b>Real-world Analogy:</b>
/// Think of this as an apprenticeship program. The master (teacher) demonstrates how to solve
/// problems, and the apprentice (student) learns by trying to replicate the master's approach.
/// The apprentice doesn't just learn the final answers, but also the reasoning process.</para>
///
/// <para><b>Benefits of Knowledge Distillation:</b>
/// - **Model Compression**: Deploy a 10x smaller model with &gt;90% of original accuracy
/// - **Faster Inference**: Smaller models run much faster on edge devices
/// - **Ensemble Distillation**: Combine knowledge from multiple teachers into one student
/// - **Transfer Learning**: Transfer knowledge across different architectures</para>
///
/// <para><b>Success Stories:</b>
/// - DistilBERT: 40% smaller than BERT, 97% of performance, 60% faster
/// - MobileNet: Distilled from ResNet, 10x fewer parameters, deployable on phones
/// - TinyBERT: 7.5x smaller than BERT, suitable for edge deployment</para>
/// </remarks>
public class KnowledgeDistillationTrainer<T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _teacher;
    private readonly IDistillationStrategy<Vector<T>, T> _distillationStrategy;
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the KnowledgeDistillationTrainer class.
    /// </summary>
    /// <param name="teacher">The teacher model to learn from.</param>
    /// <param name="distillationStrategy">The strategy for computing distillation loss.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create a trainer by providing:
    /// 1. A trained teacher model (already performing well on your task)
    /// 2. A distillation strategy (defines how to transfer knowledge)</para>
    ///
    /// <para>Example:
    /// <code>
    /// var teacher = new TeacherModelWrapper&lt;double&gt;(...);
    /// var distillationLoss = new DistillationLoss&lt;double&gt;(temperature: 3.0, alpha: 0.3);
    /// var trainer = new KnowledgeDistillationTrainer&lt;double&gt;(teacher, distillationLoss);
    /// </code>
    /// </para>
    /// </remarks>
    public KnowledgeDistillationTrainer(
        ITeacherModel<Vector<T>, Vector<T>> teacher,
        IDistillationStrategy<Vector<T>, T> distillationStrategy,
        int? seed = null)
    {
        _teacher = teacher ?? throw new ArgumentNullException(nameof(teacher));
        _distillationStrategy = distillationStrategy ?? throw new ArgumentNullException(nameof(distillationStrategy));
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Trains the student model on a single batch of data.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass on student model.</param>
    /// <param name="studentBackward">Function to perform backward pass on student model (takes gradient).</param>
    /// <param name="inputs">Input batch.</param>
    /// <param name="trueLabels">Ground truth labels (optional). If null, uses only teacher's soft targets.</param>
    /// <returns>Average loss for the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method processes one batch of training data:
    /// 1. For each sample, get predictions from teacher and student
    /// 2. Compute how different they are (distillation loss)
    /// 3. Calculate gradients and update student weights</para>
    ///
    /// <para>The studentForward function should return logits (raw outputs before softmax).
    /// The studentBackward function should accept gradients and update model parameters.</para>
    /// </remarks>
    public T TrainBatch(
        Func<Vector<T>, Vector<T>> studentForward,
        Action<Vector<T>> studentBackward,
        Vector<T>[] inputs,
        Vector<T>[]? trueLabels = null)
    {
        ArgumentNullException.ThrowIfNull(studentForward);
        ArgumentNullException.ThrowIfNull(studentBackward);
        ArgumentNullException.ThrowIfNull(inputs);

        if (inputs.Length == 0)
            throw new ArgumentException("Input batch cannot be empty", nameof(inputs));
        if (trueLabels != null && inputs.Length != trueLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");

        T totalLoss = _numOps.Zero;

        for (int i = 0; i < inputs.Length; i++)
        {
            var input = inputs[i];
            var trueLabel = trueLabels?[i];

            // Get teacher predictions (soft targets)
            var teacherLogits = _teacher.GetLogits(input);

            // Get student predictions
            var studentLogits = studentForward(input);

            // Compute distillation loss
            var loss = _distillationStrategy.ComputeLoss(studentLogits, teacherLogits, trueLabel);
            totalLoss = _numOps.Add(totalLoss, loss);

            // Compute gradient
            var gradient = _distillationStrategy.ComputeGradient(studentLogits, teacherLogits, trueLabel);

            // Backpropagate through student
            studentBackward(gradient);
        }

        // Return average loss
        return _numOps.Divide(totalLoss, _numOps.FromDouble(inputs.Length));
    }

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
    /// <para><b>For Beginners:</b> This method runs the complete training process:
    /// - Splits data into batches
    /// - Trains for the specified number of epochs
    /// - Reports progress after each epoch</para>
    ///
    /// <para>Training tips:
    /// - Start with 10-20 epochs
    /// - Use batch size 32-128 (depending on memory)
    /// - Monitor loss: should decrease steadily
    /// - If loss doesn't decrease, try:
    ///   * Lower learning rate
    ///   * Different temperature (2-5)
    ///   * Adjust alpha (0.2-0.5)</para>
    /// </remarks>
    public void Train(
        Func<Vector<T>, Vector<T>> studentForward,
        Action<Vector<T>> studentBackward,
        Vector<T>[] trainInputs,
        Vector<T>[]? trainLabels,
        int epochs,
        int batchSize = 32,
        Action<int, T>? onEpochComplete = null)
    {
        ArgumentNullException.ThrowIfNull(studentForward);
        ArgumentNullException.ThrowIfNull(studentBackward);
        ArgumentNullException.ThrowIfNull(trainInputs);

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive", nameof(epochs));
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));
        if (trainInputs.Length == 0)
            throw new ArgumentException("Training data cannot be empty", nameof(trainInputs));
        if (trainLabels != null && trainInputs.Length != trainLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");

        int numBatches = (trainInputs.Length + batchSize - 1) / batchSize;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            T epochLoss = _numOps.Zero;

            // Shuffle data (simple random permutation)
            var (shuffledInputs, shuffledLabels) = ShuffleData(trainInputs, trainLabels);

            // Train on batches
            for (int b = 0; b < numBatches; b++)
            {
                int start = b * batchSize;
                int end = Math.Min(start + batchSize, trainInputs.Length);
                int currentBatchSize = end - start;

                var batchInputs = new Vector<T>[currentBatchSize];
                Array.Copy(shuffledInputs, start, batchInputs, 0, currentBatchSize);

                Vector<T>[]? batchLabels = null;
                if (shuffledLabels != null)
                {
                    batchLabels = new Vector<T>[currentBatchSize];
                    Array.Copy(shuffledLabels, start, batchLabels, 0, currentBatchSize);
                }

                var batchLoss = TrainBatch(studentForward, studentBackward, batchInputs, batchLabels);
                epochLoss = _numOps.Add(epochLoss, batchLoss);
            }

            var avgEpochLoss = _numOps.Divide(epochLoss, _numOps.FromDouble(numBatches));

            // Invoke callback if provided
            onEpochComplete?.Invoke(epoch, avgEpochLoss);
        }
    }

    /// <summary>
    /// Evaluates the student model's accuracy on test data.
    /// </summary>
    /// <param name="studentForward">Function to perform forward pass on student model.</param>
    /// <param name="testInputs">Test input data.</param>
    /// <param name="testLabels">Test labels (one-hot encoded).</param>
    /// <returns>Classification accuracy (fraction of correct predictions).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well the student has learned:
    /// - Returns a value between 0 and 1 (higher is better)
    /// - Compares student's predicted class to true class
    /// - Use this to compare student performance to teacher</para>
    ///
    /// <para>Expected results:
    /// - Good distillation: Student achieves 90-97% of teacher's accuracy
    /// - Poor distillation: Student much worse than teacher (check hyperparameters)
    /// - Perfect distillation: Unlikely unless student has similar capacity to teacher</para>
    /// </remarks>
    public double Evaluate(
        Func<Vector<T>, Vector<T>> studentForward,
        Vector<T>[] testInputs,
        Vector<T>[] testLabels)
    {
        ArgumentNullException.ThrowIfNull(studentForward);
        ArgumentNullException.ThrowIfNull(testInputs);
        ArgumentNullException.ThrowIfNull(testLabels);

        if (testInputs.Length == 0)
            throw new ArgumentException("Test data cannot be empty", nameof(testInputs));
        if (testInputs.Length != testLabels.Length)
            throw new ArgumentException("Test inputs and labels must have the same length");

        int correct = 0;

        for (int i = 0; i < testInputs.Length; i++)
        {
            var studentLogits = studentForward(testInputs[i]);

            // Check if prediction is correct (argmax comparison)
            if (IsPredictionCorrect(studentLogits, testLabels[i]))
                correct++;
        }

        return (double)correct / testInputs.Length;
    }

    /// <summary>
    /// Shuffles training data using Fisher-Yates algorithm.
    /// </summary>
    /// <param name="inputs">Input data to shuffle.</param>
    /// <param name="labels">Labels to shuffle (optional, shuffled in sync with inputs).</param>
    /// <returns>Shuffled inputs and labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shuffling prevents the model from learning spurious patterns
    /// based on the order of training examples. It's a standard practice in deep learning.</para>
    /// </remarks>
    private (Vector<T>[] inputs, Vector<T>[]? labels) ShuffleData(Vector<T>[] inputs, Vector<T>[]? labels)
    {
        var indices = Enumerable.Range(0, inputs.Length).ToArray();

        // Fisher-Yates shuffle
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var shuffledInputs = indices.Select(i => inputs[i]).ToArray();
        var shuffledLabels = labels != null ? indices.Select(i => labels[i]).ToArray() : null;

        return (shuffledInputs, shuffledLabels);
    }

    /// <summary>
    /// Checks if the student's prediction matches the true label.
    /// </summary>
    /// <param name="prediction">Student's predicted logits.</param>
    /// <param name="trueLabel">True label (one-hot encoded).</param>
    /// <returns>True if predicted class matches true class, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This compares the predicted class (argmax of logits)
    /// to the true class (argmax of one-hot label). If they match, the prediction is correct.</para>
    /// </remarks>
    private bool IsPredictionCorrect(Vector<T> prediction, Vector<T> trueLabel)
    {
        int predClass = ArgMax(prediction);
        int trueClass = ArgMax(trueLabel);
        return predClass == trueClass;
    }

    /// <summary>
    /// Finds the index of the maximum value in a vector.
    /// </summary>
    /// <param name="vector">Input vector.</param>
    /// <returns>Index of the maximum value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ArgMax finds the position of the largest value.
    /// For classification, this gives the predicted class.</para>
    ///
    /// <para>Example: For logits [0.2, 0.8, 0.5], argmax = 1 (class 1 has highest score).</para>
    /// </remarks>
    private int ArgMax(Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Vector cannot be empty", nameof(vector));

        int maxIndex = 0;
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (_numOps.GreaterThan(vector[i], maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
