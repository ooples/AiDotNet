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

    /// <summary>Gets the mixing weight for generic head.</summary>
    public double MixingAlpha => _mixingAlpha;
}
