using AiDotNet.NeuralNetworks;
using AiDotNet.LossFunctions;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Base class for audio classification models (genre, event detection, scene classification).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio classification assigns labels to audio clips. This base class provides
/// common functionality for various classification tasks including:
/// - Genre classification (rock, jazz, classical)
/// - Audio event detection (dog bark, car horn)
/// - Scene classification (office, park, street)
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio classification is like teaching a computer to recognize
/// different types of sounds, similar to how you can tell the difference between
/// a dog barking and a car horn.
///
/// This base class provides:
/// - Class label management
/// - Softmax for probability conversion
/// - Common feature extraction
/// </para>
/// </remarks>
public abstract class AudioClassifierBase<T> : AudioNeuralNetworkBase<T>,
    AiDotNet.Interfaces.ITrainingObjectiveProvider<T>
{
    AiDotNet.Enums.TrainingObjectiveKind
        AiDotNet.Interfaces.ITrainingObjectiveProvider<T>.TrainingObjectiveKind
        => AiDotNet.Enums.TrainingObjectiveKind.Supervised;

    AiDotNet.Tensors.LinearAlgebra.Tensor<T>
        AiDotNet.Interfaces.ITrainingObjectiveProvider<T>.ResolveTrainingTarget(
            AiDotNet.Tensors.LinearAlgebra.Tensor<T> input,
            AiDotNet.Tensors.LinearAlgebra.Tensor<T> proposedTarget)
        => proposedTarget;

    T AiDotNet.Interfaces.ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(
        AiDotNet.Tensors.LinearAlgebra.Tensor<T> input,
        AiDotNet.Tensors.LinearAlgebra.Tensor<T> target)
        => base.EvaluateTrainingObjective(input, target);

    /// <summary>
    /// Gets the list of class labels this model can classify.
    /// </summary>
    public IReadOnlyList<string> ClassLabels { get; protected set; } = Array.Empty<string>();

    /// <summary>
    /// Gets the number of classes this model can classify.
    /// </summary>
    public int NumClasses => ClassLabels.Count;

    /// <summary>
    /// Initializes a new instance of the AudioClassifierBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">Optional loss function. When omitted, the audio-network MSE default is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm used for clipping.</param>
    protected AudioClassifierBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm)
    {
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Raw model output (logits).</param>
    /// <returns>Dictionary mapping class labels to probabilities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Softmax converts raw scores into probabilities that sum to 1.
    /// For example, raw scores [2.0, 1.0, 0.5] might become probabilities [0.6, 0.27, 0.13].
    /// </para>
    /// </remarks>
    protected Dictionary<string, T> ApplySoftmax(Vector<T> logits)
    {
        if (logits.Length != NumClasses)
        {
            throw new ArgumentException($"Expected {NumClasses} logits but got {logits.Length}.");
        }

        // Numerically stable softmax (exp(logit - max) / sum) as one engine op.
        T[] probabilities = Engine.Softmax(logits).ToArray();

        var result = new Dictionary<string, T>();
        for (int i = 0; i < probabilities.Length; i++)
        {
            result[ClassLabels[i]] = probabilities[i];
        }

        return result;
    }

    /// <summary>
    /// Applies softmax to convert logits tensor to probabilities.
    /// </summary>
    /// <param name="logits">Raw model output tensor.</param>
    /// <returns>Dictionary mapping class labels to probabilities.</returns>
    protected Dictionary<string, T> ApplySoftmax(Tensor<T> logits)
    {
        return ApplySoftmax(logits.ToVector());
    }

    /// <summary>
    /// Gets the top-K predictions sorted by probability.
    /// </summary>
    /// <param name="probabilities">Class probabilities.</param>
    /// <param name="k">Number of top predictions to return.</param>
    /// <returns>List of top predictions with their probabilities.</returns>
    protected IReadOnlyList<(string Label, T Probability, int Rank)> GetTopK(
        Dictionary<string, T> probabilities,
        int k)
    {
        var entries = probabilities.ToList();
        if (entries.Count == 0) return new List<(string Label, T Probability, int Rank)>();
        var values = new Tensor<T>(entries.Select(p => p.Value).ToArray(), [entries.Count]);
        var sorted = AudioEngineOps.StableDescendingOrder(Engine, values)
            .Take(k)
            .Select((i, index) => (entries[i].Key, entries[i].Value, index + 1))
            .ToList();

        return sorted;
    }

    /// <summary>
    /// Gets the predicted class (highest probability).
    /// </summary>
    /// <param name="probabilities">Class probabilities.</param>
    /// <returns>Tuple of (predicted label, confidence).</returns>
    protected (string Label, T Confidence) GetPrediction(Dictionary<string, T> probabilities)
    {
        if (probabilities.Count == 0) return (string.Empty, NumOps.Zero);

        var entries = probabilities.ToList();
        var values = new Tensor<T>(entries.Select(p => p.Value).ToArray(), [entries.Count]);
        var best = entries[AudioEngineOps.FirstArgMaxSequential(Engine, values)];
        return (best.Key, best.Value);
    }

    /// <summary>
    /// Applies threshold for multi-label classification.
    /// </summary>
    /// <param name="probabilities">Class probabilities.</param>
    /// <param name="threshold">Minimum probability to consider as positive.</param>
    /// <returns>List of labels above the threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In multi-label classification, an audio clip can belong
    /// to multiple classes (e.g., both "speech" and "music"). This method returns
    /// all classes with probability above the threshold.
    /// </para>
    /// </remarks>
    protected IReadOnlyList<(string Label, T Probability)> ApplyThreshold(
        Dictionary<string, T> probabilities,
        T threshold)
    {
        var entries = probabilities.ToList();
        if (entries.Count == 0) return new List<(string Label, T Probability)>();
        var values = new Tensor<T>(entries.Select(p => p.Value).ToArray(), [entries.Count]);
        var mask = Engine.TensorGreaterEqual(values, Tensor<T>.CreateDefault([entries.Count], threshold));
        int[] kept = Engine.TensorNonzero(mask).ToArray();
        if (kept.Length == 0) return new List<(string Label, T Probability)>();
        var keptValues = new Tensor<T>(kept.Select(i => entries[i].Value).ToArray(), [kept.Length]);
        return AudioEngineOps.StableDescendingOrder(Engine, keptValues)
            .Select(j => (entries[kept[j]].Key, entries[kept[j]].Value))
            .ToList();
    }

    /// <summary>
    /// Computes class weights for imbalanced datasets.
    /// </summary>
    /// <param name="classCounts">Dictionary of class labels to sample counts.</param>
    /// <returns>Dictionary of class labels to weights (inverse frequency).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If some classes have many more examples than others,
    /// the model might become biased. Class weights help balance training by
    /// giving more importance to rare classes.
    /// </para>
    /// </remarks>
    protected Dictionary<string, T> ComputeClassWeights(Dictionary<string, int> classCounts)
    {
        // Filter out classes with zero counts to prevent division by zero
        var nonZeroCounts = classCounts.Where(kvp => kvp.Value > 0).ToList();

        if (nonZeroCounts.Count == 0)
        {
            // Return equal weights of 1.0 for all classes if all counts are zero
            return classCounts.ToDictionary(kvp => kvp.Key, _ => NumOps.FromDouble(1.0));
        }

        int totalSamples = nonZeroCounts.Sum(kvp => kvp.Value);
        int numClasses = nonZeroCounts.Count;

        // Inverse frequency weighting: total / (numClasses * count), for every class at once.
        var entries = classCounts.ToList();
        var counts = new Tensor<int>(entries.Select(kvp => kvp.Value).ToArray(), [entries.Count]).Cast<T>();
        var positive = Engine.TensorGreaterThan(counts, NumOps.Zero);
        // Non-positive counts divide by one instead (their weight is replaced below), so a type whose
        // division by zero throws (decimal) never sees a zero denominator.
        var safeCounts = Engine.TensorWhere(positive, counts, Tensor<T>.CreateDefault([entries.Count], NumOps.One));
        var inverse = Engine.TensorDivide(
            Tensor<T>.CreateDefault([entries.Count], NumOps.FromDouble(totalSamples)),
            Engine.TensorMultiplyScalar(safeCounts, NumOps.FromDouble(numClasses)));
        // Classes with zero samples get the maximum weight (they need the most attention): the weight
        // of the rarest non-empty class, total / (numClasses * minCount).
        T maxWeight = NumOps.FromDouble((double)totalSamples / ((double)numClasses * nonZeroCounts.Min(kvp => kvp.Value)));
        T[] weightValues = Engine.TensorWhere(
            positive,
            inverse,
            Tensor<T>.CreateDefault([entries.Count], maxWeight)).ToArray();

        var weights = new Dictionary<string, T>();
        for (int i = 0; i < entries.Count; i++)
        {
            weights[entries[i].Key] = weightValues[i];
        }

        return weights;
    }
}
