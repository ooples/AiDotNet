namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements FedSelect — learned sparse binary masks for personalization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of deciding up-front which layers to personalize,
/// FedSelect learns a binary mask for each client that determines, for each parameter,
/// whether it should be shared (aggregated globally) or personalized (kept local). The mask
/// itself is learned during training using straight-through estimator gradients. This gives
/// each client a different "personalization pattern" that best fits their data.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// mask_k = sigmoid(logits_k) > threshold   // per-client binary mask
/// shared_params = params * (1 - mask_k)     // aggregated globally
/// personal_params = params * mask_k          // kept local
/// </code>
///
/// <para>Reference: FedSelect: Personalizing FL with Learned Parameter Selection (2023).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedSelectPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _maskThreshold;
    private readonly double _maskRegularization;
    private Dictionary<string, double[]>? _maskLogits;

    /// <summary>
    /// Creates a new FedSelect personalization strategy.
    /// </summary>
    /// <param name="maskThreshold">Threshold for binarizing mask (after sigmoid). Default: 0.5.</param>
    /// <param name="maskRegularization">L1 regularization on mask to encourage sparsity. Default: 0.01.</param>
    public FedSelectPersonalization(double maskThreshold = 0.5, double maskRegularization = 0.01)
    {
        if (maskThreshold <= 0 || maskThreshold >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maskThreshold), "Mask threshold must be in (0, 1).");
        }

        if (maskRegularization < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maskRegularization), "Regularization must be non-negative.");
        }

        _maskThreshold = maskThreshold;
        _maskRegularization = maskRegularization;
    }

    /// <summary>
    /// Initializes mask logits for a model structure.
    /// </summary>
    /// <param name="modelStructure">Model parameter dictionary to create masks for.</param>
    /// <param name="initialBias">Initial bias for mask logits (negative = mostly shared). Default: -2.0.</param>
    public void InitializeMasks(Dictionary<string, T[]> modelStructure, double initialBias = -2.0)
    {
        _maskLogits = new Dictionary<string, double[]>(modelStructure.Count);
        foreach (var kvp in modelStructure)
        {
            var logits = new double[kvp.Value.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] = initialBias;
            }

            _maskLogits[kvp.Key] = logits;
        }
    }

    /// <summary>
    /// Extracts shared parameters (where mask is 0) for aggregation.
    /// </summary>
    public Dictionary<string, T[]> ExtractSharedParameters(Dictionary<string, T[]> fullParameters)
    {
        if (_maskLogits == null)
        {
            InitializeMasks(fullParameters);
        }

        var shared = new Dictionary<string, T[]>(fullParameters.Count);
        foreach (var kvp in fullParameters)
        {
            var logits = _maskLogits![kvp.Key];
            var result = new T[kvp.Value.Length];
            for (int i = 0; i < result.Length; i++)
            {
                double mask = 1.0 / (1.0 + Math.Exp(-logits[i])); // sigmoid
                if (mask < _maskThreshold)
                {
                    result[i] = kvp.Value[i]; // shared
                }
                else
                {
                    result[i] = NumOps.Zero; // personalized, zero out for aggregation
                }
            }

            shared[kvp.Key] = result;
        }

        return shared;
    }

    /// <summary>
    /// Computes the mask sparsity (fraction of parameters that are personalized).
    /// </summary>
    public double GetPersonalizationRatio()
    {
        if (_maskLogits == null)
        {
            return 0;
        }

        int total = 0, personalized = 0;
        foreach (var logits in _maskLogits.Values)
        {
            foreach (double logit in logits)
            {
                total++;
                double mask = 1.0 / (1.0 + Math.Exp(-logit));
                if (mask >= _maskThreshold)
                {
                    personalized++;
                }
            }
        }

        return total > 0 ? (double)personalized / total : 0;
    }

    /// <summary>Gets the mask binarization threshold.</summary>
    public double MaskThreshold => _maskThreshold;

    /// <summary>Gets the mask regularization weight.</summary>
    public double MaskRegularization => _maskRegularization;
}
