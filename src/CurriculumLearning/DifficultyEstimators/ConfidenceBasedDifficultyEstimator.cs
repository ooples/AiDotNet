using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning.DifficultyEstimators;

/// <summary>
/// Difficulty estimator based on model prediction confidence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This estimator uses the model's confidence in its
/// predictions as a measure of difficulty. Low confidence predictions indicate
/// the model is uncertain, suggesting the sample is harder.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>The model predicts probabilities for each class</description></item>
/// <item><description>Confidence is measured (highest probability, margin, or entropy)</description></item>
/// <item><description>Lower confidence = harder sample</description></item>
/// </list>
///
/// <para><b>Confidence Metrics:</b></para>
/// <list type="bullet">
/// <item><description><b>MaxProbability:</b> Highest class probability</description></item>
/// <item><description><b>Margin:</b> Difference between top two probabilities</description></item>
/// <item><description><b>Entropy:</b> Information-theoretic uncertainty</description></item>
/// </list>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Kumar et al. "Self-Paced Learning for Latent Variable Models" (NIPS 2010)</description></item>
/// </list>
/// </remarks>
public class ConfidenceBasedDifficultyEstimator<T, TInput, TOutput>
    : DifficultyEstimatorBase<T, TInput, TOutput>, IConfidentDifficultyEstimator<T, TInput, TOutput>
{
    private readonly ConfidenceMetricType _metricType;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => $"ConfidenceBased_{_metricType}";

    /// <summary>
    /// Gets whether this estimator requires the model.
    /// </summary>
    public override bool RequiresModel => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConfidenceBasedDifficultyEstimator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="metricType">The type of confidence metric to use.</param>
    /// <param name="normalize">Whether to normalize difficulties to [0, 1].</param>
    public ConfidenceBasedDifficultyEstimator(
        ConfidenceMetricType metricType = ConfidenceMetricType.Entropy,
        bool normalize = true)
    {
        _metricType = metricType;
        _normalize = normalize;
    }

    /// <summary>
    /// Estimates the difficulty of a single sample based on prediction confidence.
    /// </summary>
    public override T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model),
                "ConfidenceBasedDifficultyEstimator requires a model to compute confidence.");
        }

        // Get prediction probabilities
        var probabilities = GetPredictionProbabilities(model, input);

        // Calculate confidence based on metric type
        var confidence = CalculateConfidence(probabilities);

        // Convert confidence to difficulty (higher confidence = easier)
        // Difficulty = 1 - confidence
        return NumOps.Subtract(NumOps.One, confidence);
    }

    /// <summary>
    /// Gets both difficulty estimate and confidence for a sample.
    /// </summary>
    public (T Difficulty, T Confidence) EstimateDifficultyWithConfidence(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        var probabilities = GetPredictionProbabilities(model, input);
        var confidence = CalculateConfidence(probabilities);
        var difficulty = NumOps.Subtract(NumOps.One, confidence);

        return (difficulty, confidence);
    }

    /// <summary>
    /// Estimates difficulty scores for all samples.
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        var difficulties = base.EstimateDifficulties(dataset, model);

        if (_normalize)
        {
            difficulties = NormalizeDifficulties(difficulties);
        }

        return difficulties;
    }

    /// <summary>
    /// Gets prediction probabilities from the model.
    /// </summary>
    private Vector<T> GetPredictionProbabilities(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        // Get raw prediction
        var prediction = model.Predict(input);

        // Try to get probabilities if model supports it
        if (model is IProbabilisticModel<T, TInput, TOutput> probModel)
        {
            return probModel.PredictProbabilities(input);
        }

        // Fallback: convert prediction to probability-like vector
        return ConvertToProbabilities(prediction);
    }

    /// <summary>
    /// Converts a prediction output to a probability vector.
    /// </summary>
    private Vector<T> ConvertToProbabilities(TOutput prediction)
    {
        // Handle common cases
        if (prediction is Vector<T> vector)
        {
            return ApplySoftmax(vector);
        }

        if (prediction is T[] array)
        {
            return ApplySoftmax(new Vector<T>(array));
        }

        if (prediction is T scalar)
        {
            // Binary classification: convert to two probabilities
            var p = ClampProbability(scalar);
            return new Vector<T>([
                NumOps.Subtract(NumOps.One, p),
                p
            ]);
        }

        // Default: treat as single confident prediction
        return new Vector<T>([NumOps.One]);
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    private Vector<T> ApplySoftmax(Vector<T> logits)
    {
        if (logits.Length == 0)
        {
            return logits;
        }

        // Find max for numerical stability
        var max = logits.Max();

        // Compute exp(x - max) for each element
        var expValues = new Vector<T>(logits.Length);
        var sum = NumOps.Zero;

        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], max);
            expValues[i] = NumOps.Exp(shifted);
            sum = NumOps.Add(sum, expValues[i]);
        }

        // Normalize
        for (int i = 0; i < expValues.Length; i++)
        {
            expValues[i] = NumOps.Divide(expValues[i], sum);
        }

        return expValues;
    }

    /// <summary>
    /// Clamps a value to valid probability range [0, 1].
    /// </summary>
    private T ClampProbability(T value)
    {
        if (NumOps.Compare(value, NumOps.Zero) < 0)
            return NumOps.Zero;
        if (NumOps.Compare(value, NumOps.One) > 0)
            return NumOps.One;
        return value;
    }

    /// <summary>
    /// Calculates confidence from probabilities based on the selected metric.
    /// </summary>
    private T CalculateConfidence(Vector<T> probabilities)
    {
        return _metricType switch
        {
            ConfidenceMetricType.MaxProbability => CalculateMaxProbabilityConfidence(probabilities),
            ConfidenceMetricType.Margin => CalculateMarginConfidence(probabilities),
            ConfidenceMetricType.Entropy => CalculateEntropyConfidence(probabilities),
            _ => CalculateEntropyConfidence(probabilities)
        };
    }

    /// <summary>
    /// Calculates confidence as maximum probability.
    /// </summary>
    private T CalculateMaxProbabilityConfidence(Vector<T> probabilities)
    {
        if (probabilities.Length == 0)
            return NumOps.Zero;

        return probabilities.Max();
    }

    /// <summary>
    /// Calculates confidence as margin between top two probabilities.
    /// </summary>
    private T CalculateMarginConfidence(Vector<T> probabilities)
    {
        if (probabilities.Length < 2)
            return NumOps.One;

        // Find top two probabilities
        var sorted = probabilities.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.Compare(b, a)); // Descending

        return NumOps.Subtract(sorted[0], sorted[1]);
    }

    /// <summary>
    /// Calculates confidence from entropy (low entropy = high confidence).
    /// </summary>
    private T CalculateEntropyConfidence(Vector<T> probabilities)
    {
        if (probabilities.Length == 0)
            return NumOps.Zero;

        // Calculate entropy: -sum(p * log(p))
        var entropy = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-10); // Avoid log(0)

        foreach (var p in probabilities)
        {
            if (NumOps.Compare(p, epsilon) > 0)
            {
                var logP = NumOps.Log(p);
                entropy = NumOps.Add(entropy, NumOps.Multiply(p, logP));
            }
        }

        entropy = NumOps.Negate(entropy);

        // Normalize by maximum possible entropy (uniform distribution)
        var maxEntropy = NumOps.Log(NumOps.FromDouble(probabilities.Length));
        if (NumOps.Compare(maxEntropy, NumOps.Zero) > 0)
        {
            var normalizedEntropy = NumOps.Divide(entropy, maxEntropy);
            // Confidence = 1 - normalized entropy
            return NumOps.Subtract(NumOps.One, normalizedEntropy);
        }

        return NumOps.One;
    }
}

/// <summary>
/// Type of confidence metric used for difficulty estimation.
/// </summary>
public enum ConfidenceMetricType
{
    /// <summary>
    /// Uses the maximum predicted probability.
    /// Higher max probability = higher confidence.
    /// </summary>
    MaxProbability,

    /// <summary>
    /// Uses the margin between top two probabilities.
    /// Larger margin = higher confidence.
    /// </summary>
    Margin,

    /// <summary>
    /// Uses prediction entropy.
    /// Lower entropy = higher confidence.
    /// </summary>
    Entropy
}

/// <summary>
/// Interface for models that provide probability predictions.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public interface IProbabilisticModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Predicts class probabilities for the given input.
    /// </summary>
    Vector<T> PredictProbabilities(TInput input);
}
