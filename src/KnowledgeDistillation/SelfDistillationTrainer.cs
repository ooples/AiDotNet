
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
public class SelfDistillationTrainer<T> : KnowledgeDistillationTrainerBase<T, Vector<T>, Vector<T>>
{
    private readonly int _generations;
    private Dictionary<Vector<T>, Vector<T>>? _cachedTeacherPredictions;
    private Func<Vector<T>, Vector<T>>? _studentForward;

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
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not between 0 and 1.</exception>
    public double EMADecay
    {
        get => _emaDecay;
        set
        {
            if (value <= 0 || value >= 1)
                throw new ArgumentOutOfRangeException(nameof(value),
                    "EMADecay must be between 0 and 1 (exclusive). Typical values are 0.9-0.999.");
            _emaDecay = value;
        }
    }
    private double _emaDecay = 0.99;

    /// <summary>
    /// Initializes a new instance of the SelfDistillationTrainer class.
    /// </summary>
    /// <param name="distillationStrategy">The strategy for computing distillation loss.</param>
    /// <param name="generations">Number of self-distillation generations (default 1).
    /// More generations can improve performance but take longer to train.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generations control how many times the model relearns from itself:
    /// - 1 generation: Train normally (standard training, no self-distillation)
    /// - 2 generations: Train, then retrain using self as teacher (first self-distillation)
    /// - 3 generations: Train → self-teach → self-teach again
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
        IDistillationStrategy<T> distillationStrategy,
        int generations = 1,
        int? seed = null)
        : base(new SelfTeacherModelPlaceholder<T>(), distillationStrategy, checkpointConfig: null, seed: seed)
    {
        if (generations < 1)
            throw new ArgumentException("Generations must be at least 1", nameof(generations));

        _generations = generations;
        UseEMA = false;
        EMADecay = 0.99;
    }

    /// <summary>
    /// Gets teacher predictions from the cached predictions dictionary (for self-distillation).
    /// </summary>
    /// <param name="input">The input data to look up cached predictions for.</param>
    /// <param name="index">The index in the training batch (unused - we use input for lookup).</param>
    /// <returns>Cached teacher prediction for this input.</returns>
    /// <remarks>
    /// <para><b>For Self-Distillation:</b> Instead of calling a separate teacher model,
    /// we return predictions that were cached from the previous generation. We use the input
    /// itself as the key (via reference equality) to handle shuffled batches correctly.</para>
    ///
    /// <para><b>Generation 0 Handling:</b> When no cached predictions exist (first generation),
    /// we use the student's own predictions as the teacher. This makes distillation a no-op for
    /// generation 0, effectively training normally. This avoids dimension mismatches since the
    /// placeholder teacher has OutputDimension = 0.</para>
    /// </remarks>
    protected override Vector<T> GetTeacherPredictions(Vector<T> input, int index)
    {
        if (_cachedTeacherPredictions == null)
        {
            // First generation - use student's own predictions as teacher
            // This makes distillation a no-op (teacher same as student) for generation 0
            if (_studentForward == null)
            {
                throw new InvalidOperationException(
                    "Student forward function not set. Call TrainMultipleGenerations to properly initialize self-distillation.");
            }
            return _studentForward(input);
        }

        // Look up by input reference to handle shuffled data correctly
        if (_cachedTeacherPredictions.TryGetValue(input, out var cachedPrediction))
        {
            return cachedPrediction;
        }

        // Fallback for inputs not in cache (shouldn't happen in normal flow)
        // Use student predictions if available
        if (_studentForward != null)
        {
            return _studentForward(input);
        }

        throw new InvalidOperationException(
            $"Input not found in cached teacher predictions and no fallback available. " +
            $"Cache size: {_cachedTeacherPredictions.Count}, Input hash: {input.GetHashCode()}");
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
        Vector<Vector<T>> trainInputs,
        Vector<Vector<T>> trainLabels,
        int epochs,
        int batchSize = 32,
        Action<int, T>? onGenerationComplete = null)
    {
        if (modelForward == null) throw new ArgumentNullException(nameof(modelForward));
        if (modelBackward == null) throw new ArgumentNullException(nameof(modelBackward));
        if (trainInputs == null) throw new ArgumentNullException(nameof(trainInputs));
        if (trainLabels == null) throw new ArgumentNullException(nameof(trainLabels));

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive", nameof(epochs));
        if (trainInputs.Length != trainLabels.Length)
            throw new ArgumentException("Inputs and labels must have the same length");

        // Store student forward function for GetTeacherPredictions to use
        // This is needed for generation 0 where no cached predictions exist yet
        _studentForward = modelForward;

        for (int generation = 0; generation < _generations; generation++)
        {
            // Cache predictions from previous generation (if not first generation)
            if (generation > 0)
            {
                // Use dictionary to map input instances to their predictions
                var newPredictions = new Dictionary<Vector<T>, Vector<T>>();
                for (int i = 0; i < trainInputs.Length; i++)
                {
                    var input = trainInputs[i];
                    newPredictions[input] = modelForward(input);
                }

                // Apply EMA if enabled
                if (UseEMA && _cachedTeacherPredictions != null)
                {
                    var blendedPredictions = new Dictionary<Vector<T>, Vector<T>>();
                    foreach (var kvp in newPredictions)
                    {
                        var input = kvp.Key;
                        var newPred = kvp.Value;

                        if (_cachedTeacherPredictions.TryGetValue(input, out var oldPred))
                        {
                            var blended = new Vector<T>(newPred.Length);
                            for (int j = 0; j < newPred.Length; j++)
                            {
                                var oldValue = NumOps.Multiply(
                                    oldPred[j],
                                    NumOps.FromDouble(EMADecay));
                                var newValue = NumOps.Multiply(
                                    newPred[j],
                                    NumOps.FromDouble(1.0 - EMADecay));
                                blended[j] = NumOps.Add(oldValue, newValue);
                            }
                            blendedPredictions[input] = blended;
                        }
                        else
                        {
                            // No old prediction, use new one as-is
                            blendedPredictions[input] = newPred;
                        }
                    }
                    _cachedTeacherPredictions = blendedPredictions;
                }
                else
                {
                    _cachedTeacherPredictions = newPredictions;
                }
            }

            // Train using base class Train method
            T generationLoss = NumOps.Zero;
            int epochCount = 0;

            Train(
                modelForward,
                modelBackward,
                trainInputs,
                trainLabels,
                epochs,
                batchSize,
                onEpochComplete: (epoch, loss) =>
                {
                    generationLoss = NumOps.Add(generationLoss, loss);
                    epochCount++;
                });

            var avgGenLoss = NumOps.Divide(generationLoss, NumOps.FromDouble(epochCount));
            onGenerationComplete?.Invoke(generation, avgGenLoss);
        }
    }
}

/// <summary>
/// Placeholder teacher model for self-distillation (not actually used for predictions).
/// </summary>
/// <remarks>
/// <para><b>Architecture Note:</b> This placeholder satisfies the ITeacherModel requirement
/// in the base class constructor, but GetLogits is never called in practice because
/// SelfDistillationTrainer overrides GetTeacherPredictions to use cached student predictions instead.</para>
///
/// <para>This design allows SelfDistillationTrainer to inherit from KnowledgeDistillationTrainerBase
/// without requiring a real teacher model, since the student acts as its own teacher.</para>
///
/// <para><b>LSP Compliance:</b> Even though this class isn't used in the normal flow, it provides
/// valid implementations to avoid violating the Liskov Substitution Principle. GetLogits returns
/// an empty vector rather than throwing exceptions.</para>
/// </remarks>
internal class SelfTeacherModelPlaceholder<T> : ITeacherModel<Vector<T>, Vector<T>>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes the placeholder teacher model.
    /// </summary>
    public SelfTeacherModelPlaceholder()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Returns 0 because the actual output dimension comes from the student model.
    /// </summary>
    /// <remarks>
    /// <para>In self-distillation, the student determines the output dimension.
    /// This placeholder doesn't represent a real model with a fixed output size.</para>
    /// </remarks>
    public int OutputDimension => 0;

    /// <summary>
    /// Returns an empty vector. This method is not called in practice.
    /// </summary>
    /// <param name="input">Input data (ignored).</param>
    /// <returns>An empty vector to maintain LSP compliance.</returns>
    /// <remarks>
    /// <para><b>Important:</b> This method is never called in normal self-distillation flow
    /// because SelfDistillationTrainer overrides GetTeacherPredictions. It returns an empty
    /// vector rather than throwing an exception to maintain Liskov Substitution Principle.</para>
    ///
    /// <para>If this method is called, it indicates a programming error - the caller should
    /// be using SelfDistillationTrainer's GetTeacherPredictions override instead.</para>
    /// </remarks>
    public Vector<T> GetLogits(Vector<T> input)
    {
        // Return empty vector for LSP compliance (never called in practice)
        return new Vector<T>(0);
    }
}
