using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Implements self-distillation where a model acts as its own teacher to improve calibration and generalization.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Self-distillation is a clever technique where a model learns from itself!
/// Instead of using a separate larger teacher, you train a model normally, then use it as a teacher
/// to train itself again. This often improves:
/// - **Calibration**: Model confidence matches actual accuracy
/// - **Generalization**: Better performance on unseen data
/// - **Robustness**: Less sensitive to noisy labels or adversarial examples</para>
///
/// <para><b>How It Works:</b>
/// 1. Train model normally on hard labels (standard training)
/// 2. Save the trained model's predictions
/// 3. Retrain the model using its own soft predictions as teacher
/// 4. Repeat for multiple generations if desired</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine studying for an exam, then teaching the material to yourself as if you were a student.
/// By explaining concepts in your own words, you deepen your understanding and identify gaps
/// in your knowledge. Self-distillation works similarly for neural networks.</para>
///
/// <para><b>Variants:</b>
/// - **Iterative Self-Distillation**: Multiple rounds of self-teaching
/// - **Born-Again Networks**: Same architecture, trained from scratch with self as teacher
/// - **Online Self-Distillation**: Student learns from earlier checkpoints of itself</para>
///
/// <para><b>Benefits:</b>
/// - No need for a separate teacher model
/// - Improves calibration without model compression
/// - Can be combined with data augmentation for better regularization
/// - Often provides 1-3% accuracy improvement for free</para>
///
/// <para><b>When to Use:</b>
/// - You want better calibrated predictions
/// - You have limited model capacity (can't afford a larger teacher)
/// - You want to improve an existing trained model
/// - You're training on noisy or imperfect labels</para>
///
/// <para><b>References:</b>
/// - Furlanello, T., et al. (2018). Born Again Neural Networks. ICML.
/// - Zhang, L., et al. (2019). Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self-Distillation.</para>
/// </remarks>
public class SelfDistillationTrainer<T>
{
    private readonly IDistillationStrategy<Vector<T>, T> _distillationStrategy;
    private readonly INumericOperations<T> _numOps;
    private readonly int _generations;

    /// <summary>
    /// Gets or sets whether to use exponential moving average for teacher predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> EMA smooths out the teacher's predictions over time,
    /// making them more stable and reliable. This can improve training stability.</para>
    /// </remarks>
    public bool UseEMA { get; set; }

    /// <summary>
    /// Gets or sets the EMA decay rate (default 0.99). Higher values give more weight to history.
    /// </summary>
    public double EMADecay { get; set; }

    /// <summary>
    /// Initializes a new instance of the SelfDistillationTrainer class.
    /// </summary>
    /// <param name="distillationStrategy">The strategy for computing distillation loss.</param>
    /// <param name="generations">Number of self-distillation generations (default 1).
    /// More generations can improve performance but take longer to train.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generations control how many times the model relearns from itself:
    /// - 1 generation: Train normally, then retrain with self as teacher
    /// - 2 generations: Do it twice (teacher → student1 → student2)
    /// - More generations: Diminishing returns, usually not worth it beyond 2-3</para>
    ///
    /// <para>Example:
    /// <code>
    /// var distillationLoss = new DistillationLoss&lt;double&gt;(temperature: 3.0, alpha: 0.5);
    /// var selfTrainer = new SelfDistillationTrainer&lt;double&gt;(distillationLoss, generations: 2);
    /// </code>
    /// </para>
    /// </remarks>
    public SelfDistillationTrainer(
        IDistillationStrategy<Vector<T>, T> distillationStrategy,
        int generations = 1)
    {
        if (generations < 1)
            throw new ArgumentException("Generations must be at least 1", nameof(generations));

        _distillationStrategy = distillationStrategy ?? throw new ArgumentNullException(nameof(distillationStrategy));
        _numOps = MathHelper.GetNumericOperations<T>();
        _generations = generations;
        UseEMA = false;
        EMADecay = 0.99;
    }

    /// <summary>
    /// Performs self-distillation training for the specified number of generations.
    /// </summary>
    /// <param name="modelForward">Function to perform forward pass and get logits.</param>
    /// <param name="modelBackward">Function to perform backward pass with gradients.</param>
    /// <param name="trainInputs">Training input data.</param>
    /// <param name="trainLabels">Training labels.</param>
    /// <param name="epochs">Number of epochs per generation.</param>
    /// <param name="batchSize">Batch size for training.</param>
    /// <param name="onGenerationComplete">Optional callback invoked after each generation with (generation, avgLoss).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method runs the complete self-distillation process:
    /// 1. **Generation 0**: Train model normally (if starting from scratch)
    /// 2. **Generation 1**: Retrain using self as teacher
    /// 3. **Generation 2+**: Continue if requested</para>
    ///
    /// <para>Each generation:
    /// - Saves current model predictions as "teacher"
    /// - Retrains model to match both teacher predictions and true labels
    /// - Typically sees 0.5-2% improvement per generation</para>
    ///
    /// <para><b>Training Tips:</b>
    /// - Use temperature 2-4 (lower than standard distillation)
    /// - Set alpha = 0.5 (equal weight to self and labels)
    /// - Train for fewer epochs in later generations (half of first)
    /// - Watch for overfitting in later generations</para>
    /// </remarks>
    public void TrainMultipleGenerations(
        Func<Vector<T>, Vector<T>> modelForward,
        Action<Vector<T>> modelBackward,
        Vector<T>[] trainInputs,
        Vector<T>[] trainLabels,
        int epochs,
        int batchSize = 32,
        Action<int, T>? onGenerationComplete = null)
    {
        ArgumentNullException.ThrowIfNull(modelForward);
        ArgumentNullException.ThrowIfNull(modelBackward);
        ArgumentNullException.ThrowIfNull(trainInputs);
        ArgumentNullException.ThrowIfNull(trainLabels);

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive", nameof(epochs));
        if (trainInputs.Length != trainLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");

        // Store teacher predictions for current generation
        Vector<T>[]? teacherPredictions = null;

        for (int generation = 0; generation < _generations; generation++)
        {
            T generationLoss = _numOps.Zero;
            int numBatches = (trainInputs.Length + batchSize - 1) / batchSize;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T epochLoss = _numOps.Zero;

                // Shuffle data - use same indices for inputs, labels, and teacher predictions
                var indices = Enumerable.Range(0, trainInputs.Length).OrderBy(_ => Guid.NewGuid()).ToArray();
                var shuffledInputs = indices.Select(i => trainInputs[i]).ToArray();
                var shuffledLabels = indices.Select(i => trainLabels[i]).ToArray();
                Vector<T>[]? shuffledTeacher = null;

                if (teacherPredictions != null)
                {
                    // Use same indices to ensure alignment
                    shuffledTeacher = indices.Select(i => teacherPredictions[i]).ToArray();
                }

                // Train on batches
                for (int b = 0; b < numBatches; b++)
                {
                    int start = b * batchSize;
                    int end = Math.Min(start + batchSize, trainInputs.Length);
                    int currentBatchSize = end - start;

                    var batchInputs = new Vector<T>[currentBatchSize];
                    var batchLabels = new Vector<T>[currentBatchSize];
                    Array.Copy(shuffledInputs, start, batchInputs, 0, currentBatchSize);
                    Array.Copy(shuffledLabels, start, batchLabels, 0, currentBatchSize);

                    Vector<T>[]? batchTeacher = null;
                    if (shuffledTeacher != null)
                    {
                        batchTeacher = new Vector<T>[currentBatchSize];
                        Array.Copy(shuffledTeacher, start, batchTeacher, 0, currentBatchSize);
                    }

                    var batchLoss = TrainBatch(
                        modelForward,
                        modelBackward,
                        batchInputs,
                        batchLabels,
                        batchTeacher);
                    epochLoss = _numOps.Add(epochLoss, batchLoss);
                }

                generationLoss = _numOps.Add(generationLoss, epochLoss);
            }

            var avgGenLoss = _numOps.Divide(generationLoss, _numOps.FromDouble(epochs * numBatches));

            // Invoke callback if provided
            onGenerationComplete?.Invoke(generation, avgGenLoss);

            // After training this generation, save predictions as teacher for next generation
            if (generation < _generations - 1)
            {
                teacherPredictions = new Vector<T>[trainInputs.Length];
                for (int i = 0; i < trainInputs.Length; i++)
                {
                    teacherPredictions[i] = modelForward(trainInputs[i]);
                }
            }
        }
    }

    /// <summary>
    /// Trains on a single batch with self-distillation.
    /// </summary>
    private T TrainBatch(
        Func<Vector<T>, Vector<T>> modelForward,
        Action<Vector<T>> modelBackward,
        Vector<T>[] batchInputs,
        Vector<T>[] batchLabels,
        Vector<T>[]? teacherPredictions)
    {
        T totalLoss = _numOps.Zero;

        for (int i = 0; i < batchInputs.Length; i++)
        {
            var input = batchInputs[i];
            var label = batchLabels[i];

            // Get current predictions
            var studentLogits = modelForward(input);

            // Compute loss
            T loss;
            Vector<T> gradient;

            if (teacherPredictions != null)
            {
                // Use self-distillation: learn from both labels and previous self
                loss = _distillationStrategy.ComputeLoss(studentLogits, teacherPredictions[i], label);
                gradient = _distillationStrategy.ComputeGradient(studentLogits, teacherPredictions[i], label);
            }
            else
            {
                // First generation: standard training (just use labels)
                // Compute as if teacher predictions match labels
                loss = _distillationStrategy.ComputeLoss(studentLogits, studentLogits, label);
                gradient = _distillationStrategy.ComputeGradient(studentLogits, studentLogits, label);
            }

            totalLoss = _numOps.Add(totalLoss, loss);
            modelBackward(gradient);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batchInputs.Length));
    }

    /// <summary>
    /// Shuffles data using random permutation.
    /// </summary>
    private (Vector<T>[] inputs, Vector<T>[] labels) ShuffleData(Vector<T>[] inputs, Vector<T>[] labels)
    {
        var indices = Enumerable.Range(0, inputs.Length).OrderBy(_ => Guid.NewGuid()).ToArray();
        var shuffledInputs = indices.Select(i => inputs[i]).ToArray();
        var shuffledLabels = indices.Select(i => labels[i]).ToArray();
        return (shuffledInputs, shuffledLabels);
    }
}
