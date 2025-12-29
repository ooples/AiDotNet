using AiDotNet.NeuralNetworks;

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
public abstract class AudioClassifierBase<T> : AudioNeuralNetworkBase<T>
{
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
    protected AudioClassifierBase(NeuralNetworkArchitecture<T> architecture)
        : base(architecture)
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

        // Find max for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.GreaterThan(logits[i], maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        // Compute exp(logit - max) and sum
        T[] expValues = new T[logits.Length];
        T expSum = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            expValues[i] = NumOps.Exp(NumOps.Subtract(logits[i], maxLogit));
            expSum = NumOps.Add(expSum, expValues[i]);
        }

        // Normalize and create result dictionary
        var result = new Dictionary<string, T>();
        for (int i = 0; i < logits.Length; i++)
        {
            T probability = NumOps.Divide(expValues[i], expSum);
            result[ClassLabels[i]] = probability;
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
        var sorted = probabilities
            .OrderByDescending(p => NumOps.ToDouble(p.Value))
            .Take(k)
            .Select((p, index) => (p.Key, p.Value, index + 1))
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
        string bestLabel = string.Empty;
        T bestProb = NumOps.Zero;

        foreach (var (label, prob) in probabilities)
        {
            if (string.IsNullOrEmpty(bestLabel) || NumOps.GreaterThan(prob, bestProb))
            {
                bestLabel = label;
                bestProb = prob;
            }
        }

        return (bestLabel, bestProb);
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
        var result = new List<(string Label, T Probability)>();

        foreach (var (label, prob) in probabilities)
        {
            if (NumOps.GreaterThanOrEquals(prob, threshold))
            {
                result.Add((label, prob));
            }
        }

        return result.OrderByDescending(p => NumOps.ToDouble(p.Probability)).ToList();
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
        int totalSamples = classCounts.Values.Sum();
        int numClasses = classCounts.Count;

        var weights = new Dictionary<string, T>();
        foreach (var (label, count) in classCounts)
        {
            // Inverse frequency weighting
            double weight = (double)totalSamples / (numClasses * count);
            weights[label] = NumOps.FromDouble(weight);
        }

        return weights;
    }
}
