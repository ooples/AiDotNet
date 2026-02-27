namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements FedRoD (Representation on Demand) personalization with dual classifiers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedRoD trains two classification heads on top of a shared
/// feature extractor: a generic head (aggregated globally, works well on average) and a
/// personalized head (kept locally, works well on this client's data). At inference time,
/// the client can choose to use either head or combine their predictions depending on
/// whether the input is more like global data or local data.</para>
///
/// <para>Architecture:</para>
/// <code>
/// Input → SharedBody → ┌ GenericHead (aggregated)     → prediction_g
///                       └ PersonalizedHead (local only) → prediction_p
/// Final = alpha * prediction_g + (1 - alpha) * prediction_p
/// </code>
///
/// <para>Reference: Chen, H.-Y. &amp; Chao, W.-L. (2023). "On Bridging Generic and Personalized
/// Federated Learning for Image Classification." ICLR 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedRoDPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _headFraction;
    private readonly double _mixingAlpha;

    /// <summary>
    /// Creates a new FedRoD personalization strategy.
    /// </summary>
    /// <param name="headFraction">Fraction of model for each head. Default: 0.1.</param>
    /// <param name="mixingAlpha">Weight for generic head in combined prediction. Default: 0.5.</param>
    public FedRoDPersonalization(double headFraction = 0.1, double mixingAlpha = 0.5)
    {
        if (headFraction <= 0 || headFraction >= 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(headFraction), "Head fraction must be in (0, 0.5).");
        }

        if (mixingAlpha < 0 || mixingAlpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(mixingAlpha), "Mixing alpha must be in [0, 1].");
        }

        _headFraction = headFraction;
        _mixingAlpha = mixingAlpha;
    }

    /// <summary>
    /// Extracts the shared body + generic head (to be aggregated).
    /// </summary>
    public Dictionary<string, T[]> ExtractSharedParameters(Dictionary<string, T[]> fullParameters)
    {
        var layerNames = fullParameters.Keys.ToArray();
        int personalizedHeadCount = (int)(layerNames.Length * _headFraction);
        int sharedCount = layerNames.Length - personalizedHeadCount;

        var shared = new Dictionary<string, T[]>(sharedCount);
        for (int i = 0; i < sharedCount; i++)
        {
            shared[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return shared;
    }

    /// <summary>
    /// Extracts the personalized head parameters (kept local).
    /// </summary>
    public Dictionary<string, T[]> ExtractPersonalizedHead(Dictionary<string, T[]> fullParameters)
    {
        var layerNames = fullParameters.Keys.ToArray();
        int personalizedHeadCount = (int)(layerNames.Length * _headFraction);
        int sharedCount = layerNames.Length - personalizedHeadCount;

        var personalized = new Dictionary<string, T[]>(personalizedHeadCount);
        for (int i = sharedCount; i < layerNames.Length; i++)
        {
            personalized[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return personalized;
    }

    /// <summary>
    /// Combines generic and personalized predictions.
    /// </summary>
    /// <param name="genericPrediction">Output from generic head.</param>
    /// <param name="personalizedPrediction">Output from personalized head.</param>
    /// <returns>Mixed prediction.</returns>
    public T[] CombinePredictions(T[] genericPrediction, T[] personalizedPrediction)
    {
        var result = new T[genericPrediction.Length];
        var alpha = NumOps.FromDouble(_mixingAlpha);
        var oneMinusAlpha = NumOps.FromDouble(1.0 - _mixingAlpha);

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.Add(
                NumOps.Multiply(genericPrediction[i], alpha),
                NumOps.Multiply(personalizedPrediction[i], oneMinusAlpha));
        }

        return result;
    }

    /// <summary>
    /// Computes the balanced softmax loss, which adjusts for class frequency imbalance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When clients have very different class distributions (e.g., one
    /// client has mostly cats, another mostly dogs), standard softmax biases towards the majority
    /// class. Balanced softmax adds a correction term log(n_c / N) based on class frequencies,
    /// so rare classes get higher effective logits. This is applied to the generic head to
    /// compensate for the class imbalance across all clients.</para>
    /// </remarks>
    /// <param name="logits">Raw logits from the model.</param>
    /// <param name="classFrequencies">Per-class sample counts on the client.</param>
    /// <returns>Adjusted logits with class-frequency correction.</returns>
    public T[] ComputeBalancedSoftmaxLogits(T[] logits, int[] classFrequencies)
    {
        int numClasses = logits.Length;
        long totalSamples = 0;
        for (int c = 0; c < classFrequencies.Length; c++)
        {
            totalSamples += classFrequencies[c];
        }

        var adjusted = new T[numClasses];
        for (int c = 0; c < numClasses; c++)
        {
            double freq = c < classFrequencies.Length && totalSamples > 0
                ? (double)classFrequencies[c] / totalSamples
                : 1.0 / numClasses;

            // Balanced softmax: adjusted_logit = logit + log(freq + epsilon)
            double correction = Math.Log(Math.Max(freq, 1e-10));
            adjusted[c] = NumOps.Add(logits[c], NumOps.FromDouble(correction));
        }

        return adjusted;
    }

    /// <summary>
    /// Extracts the generic head parameters (aggregated with the body).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The generic head sits between the body and the personalized head
    /// in parameter ordering. It is aggregated globally so it captures the "average" classification
    /// function across all clients.</para>
    /// </remarks>
    /// <param name="fullParameters">Full model parameter dictionary.</param>
    /// <returns>Generic head parameters.</returns>
    public Dictionary<string, T[]> ExtractGenericHead(Dictionary<string, T[]> fullParameters)
    {
        var layerNames = fullParameters.Keys.ToArray();
        int personalizedCount = (int)(layerNames.Length * _headFraction);
        int genericStart = layerNames.Length - 2 * personalizedCount;
        int genericEnd = layerNames.Length - personalizedCount;

        var genericHead = new Dictionary<string, T[]>(personalizedCount);
        for (int i = genericStart; i < genericEnd; i++)
        {
            genericHead[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return genericHead;
    }

    /// <summary>Gets the mixing weight for generic head.</summary>
    public double MixingAlpha => _mixingAlpha;

    /// <summary>Gets the head fraction.</summary>
    public double HeadFraction => _headFraction;
}
