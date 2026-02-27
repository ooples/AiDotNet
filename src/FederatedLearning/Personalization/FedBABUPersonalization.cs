namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements FedBABU (Body And Bottom Update) personalization strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedBABU takes a deliberately simple approach to personalization:
/// during federated training, only the model body (feature extractor) is trained and aggregated,
/// while the classification head is frozen at random initialization. After FL converges, each
/// client locally fine-tunes just the head on their own data. This works surprisingly well
/// because a good feature extractor transfers across clients, and a few local epochs on the
/// head are enough for personalization.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Initialize model with random head, random body</item>
/// <item>During FL: freeze head, train body, aggregate body via FedAvg</item>
/// <item>After FL converges: freeze body, fine-tune head locally</item>
/// </list>
///
/// <para>Reference: Oh, J., et al. (2022). "FedBABU: Toward Enhanced Representation for
/// Federated Image Classification." ICLR 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedBABUPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _headFraction;
    private readonly int _localFineTuneEpochs;

    /// <summary>
    /// Creates a new FedBABU personalization strategy.
    /// </summary>
    /// <param name="headFraction">Fraction of parameters that form the head (frozen during FL). Default: 0.1.</param>
    /// <param name="localFineTuneEpochs">Epochs for local head fine-tuning after FL. Default: 5.</param>
    public FedBABUPersonalization(double headFraction = 0.1, int localFineTuneEpochs = 5)
    {
        if (headFraction <= 0 || headFraction >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(headFraction), "Head fraction must be in (0, 1).");
        }

        if (localFineTuneEpochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(localFineTuneEpochs), "Fine-tune epochs must be at least 1.");
        }

        _headFraction = headFraction;
        _localFineTuneEpochs = localFineTuneEpochs;
    }

    /// <summary>
    /// Extracts the body parameters (shared/aggregated) from the full model.
    /// </summary>
    /// <param name="fullParameters">Full model parameter dictionary.</param>
    /// <returns>Body-only parameter dictionary.</returns>
    public Dictionary<string, T[]> ExtractBody(Dictionary<string, T[]> fullParameters)
    {
        Guard.NotNull(fullParameters);
        var layerNames = fullParameters.Keys.ToArray();
        int headStart = (int)(layerNames.Length * (1.0 - _headFraction));

        var body = new Dictionary<string, T[]>(headStart);
        for (int i = 0; i < headStart; i++)
        {
            body[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return body;
    }

    /// <summary>
    /// Merges aggregated body with local head parameters.
    /// </summary>
    /// <param name="aggregatedBody">Aggregated body parameters from server.</param>
    /// <param name="localHead">Client's local head parameters.</param>
    /// <returns>Full model with updated body and preserved head.</returns>
    public Dictionary<string, T[]> MergeBodyAndHead(
        Dictionary<string, T[]> aggregatedBody,
        Dictionary<string, T[]> localHead)
    {
        Guard.NotNull(aggregatedBody);
        Guard.NotNull(localHead);
        var merged = new Dictionary<string, T[]>(aggregatedBody.Count + localHead.Count);
        foreach (var kvp in aggregatedBody)
        {
            merged[kvp.Key] = kvp.Value;
        }

        foreach (var kvp in localHead)
        {
            merged[kvp.Key] = kvp.Value;
        }

        return merged;
    }

    /// <summary>
    /// Extracts the head parameters (frozen during FL, fine-tuned locally after convergence).
    /// </summary>
    /// <param name="fullParameters">Full model parameter dictionary.</param>
    /// <returns>Head-only parameter dictionary.</returns>
    public Dictionary<string, T[]> ExtractHead(Dictionary<string, T[]> fullParameters)
    {
        var layerNames = fullParameters.Keys.ToArray();
        int headStart = (int)(layerNames.Length * (1.0 - _headFraction));

        var head = new Dictionary<string, T[]>(layerNames.Length - headStart);
        for (int i = headStart; i < layerNames.Length; i++)
        {
            head[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return head;
    }

    /// <summary>
    /// Initializes the classification head with random values (Kaiming uniform initialization).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> FedBABU starts with a randomly initialized head and keeps it frozen
    /// during federated training. The head is only fine-tuned locally after the shared body converges.
    /// Kaiming initialization scales random values based on the layer size to maintain gradient flow.</para>
    /// </remarks>
    /// <param name="headParams">Head parameter dictionary to reinitialize.</param>
    /// <param name="seed">Random seed for reproducibility. Default: 42.</param>
    /// <returns>Reinitialized head parameters.</returns>
    public Dictionary<string, T[]> InitializeRandomHead(Dictionary<string, T[]> headParams, int seed = 42)
    {
        var rng = new Random(seed);
        var initialized = new Dictionary<string, T[]>(headParams.Count);

        foreach (var (layerName, values) in headParams)
        {
            var result = new T[values.Length];
            // Kaiming uniform: limit = sqrt(6 / fan_in), assume fan_in ≈ param count
            double limit = Math.Sqrt(6.0 / Math.Max(1, values.Length));
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * limit);
            }

            initialized[layerName] = result;
        }

        return initialized;
    }

    /// <summary>
    /// Applies a gradient mask that zeros out head gradients during FL body training.
    /// </summary>
    /// <param name="gradients">Full model gradient dictionary.</param>
    /// <returns>Masked gradients with head gradients set to zero.</returns>
    public Dictionary<string, T[]> MaskHeadGradients(Dictionary<string, T[]> gradients)
    {
        var layerNames = gradients.Keys.ToArray();
        int headStart = (int)(layerNames.Length * (1.0 - _headFraction));

        var masked = new Dictionary<string, T[]>(gradients.Count);
        for (int i = 0; i < layerNames.Length; i++)
        {
            if (i < headStart)
            {
                masked[layerNames[i]] = gradients[layerNames[i]]; // Body: keep gradients.
            }
            else
            {
                // Head: zero gradients (frozen during FL).
                var zeros = new T[gradients[layerNames[i]].Length];
                masked[layerNames[i]] = zeros;
            }
        }

        return masked;
    }

    /// <summary>Gets the head fraction.</summary>
    public double HeadFraction => _headFraction;

    /// <summary>Gets the local fine-tuning epochs.</summary>
    public int LocalFineTuneEpochs => _localFineTuneEpochs;
}
