using AiDotNet.Enums;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Defines precision policies for different layer types during mixed-precision training.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Not all layers behave well in low precision. This class helps you
/// configure which layers should stay in higher precision (FP32 or FP16) even when using
/// mixed-precision training.
/// </para>
/// <para>
/// <b>Why Some Layers Need Higher Precision:</b>
/// </para>
/// <list type="bullet">
/// <item><description><b>Normalization layers (BatchNorm, LayerNorm):</b> Compute mean and variance which can be unstable in low precision</description></item>
/// <item><description><b>Softmax:</b> Involves exponentials that can overflow/underflow easily</description></item>
/// <item><description><b>Attention:</b> Contains softmax and can have large value ranges</description></item>
/// <item><description><b>Embeddings:</b> Often have sparse gradients that benefit from higher precision</description></item>
/// <item><description><b>Output layers:</b> Final predictions should be precise</description></item>
/// </list>
/// </remarks>
public class LayerPrecisionPolicy
{
    private readonly Dictionary<string, MixedPrecisionType> _exactMatches = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<(string Pattern, MixedPrecisionType Precision)> _patterns = new();
    private readonly MixedPrecisionType _defaultPrecision;

    /// <summary>
    /// Gets the default precision for layers not matching any rule.
    /// </summary>
    public MixedPrecisionType DefaultPrecision => _defaultPrecision;

    /// <summary>
    /// Creates a new layer precision policy with the specified default precision.
    /// </summary>
    /// <param name="defaultPrecision">Default precision for layers not matching any rule.</param>
    public LayerPrecisionPolicy(MixedPrecisionType defaultPrecision = MixedPrecisionType.FP16)
    {
        _defaultPrecision = defaultPrecision;
    }

    /// <summary>
    /// Sets the precision for a specific layer by exact name.
    /// </summary>
    /// <param name="layerName">Exact layer name.</param>
    /// <param name="precision">Precision to use for this layer.</param>
    /// <returns>This policy for chaining.</returns>
    public LayerPrecisionPolicy SetPrecision(string layerName, MixedPrecisionType precision)
    {
        _exactMatches[layerName] = precision;
        return this;
    }

    /// <summary>
    /// Adds a pattern-based precision rule. Layers whose names contain the pattern will use the specified precision.
    /// </summary>
    /// <param name="pattern">Substring to match in layer names (case-insensitive).</param>
    /// <param name="precision">Precision to use for matching layers.</param>
    /// <returns>This policy for chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Patterns are matched as substrings. For example, "Norm" will match
    /// "LayerNorm", "BatchNorm", "RMSNorm", etc.</para>
    /// </remarks>
    public LayerPrecisionPolicy AddPattern(string pattern, MixedPrecisionType precision)
    {
        _patterns.Add((pattern, precision));
        return this;
    }

    /// <summary>
    /// Keeps layers matching the pattern in full precision (FP32).
    /// </summary>
    /// <param name="pattern">Substring to match in layer names.</param>
    /// <returns>This policy for chaining.</returns>
    public LayerPrecisionPolicy KeepInFP32(string pattern)
    {
        return AddPattern(pattern, MixedPrecisionType.None);
    }

    /// <summary>
    /// Keeps layers matching the pattern in FP16 (useful when default is FP8).
    /// </summary>
    /// <param name="pattern">Substring to match in layer names.</param>
    /// <returns>This policy for chaining.</returns>
    public LayerPrecisionPolicy KeepInFP16(string pattern)
    {
        return AddPattern(pattern, MixedPrecisionType.FP16);
    }

    /// <summary>
    /// Gets the precision to use for a layer with the given name.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>The precision type to use.</returns>
    public MixedPrecisionType GetPrecision(string layerName)
    {
        // Check exact matches first
        if (_exactMatches.TryGetValue(layerName, out var precision))
        {
            return precision;
        }

        // Check patterns (first match wins)
        foreach (var (pattern, patternPrecision) in _patterns)
        {
            if (layerName.Contains(pattern, StringComparison.OrdinalIgnoreCase))
            {
                return patternPrecision;
            }
        }

        return _defaultPrecision;
    }

    /// <summary>
    /// Determines if a layer should be kept in higher precision (FP32 or FP16).
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>True if the layer should use higher precision than the default.</returns>
    public bool ShouldUseHigherPrecision(string layerName)
    {
        var precision = GetPrecision(layerName);
        return precision == MixedPrecisionType.None || // FP32
               (precision == MixedPrecisionType.FP16 && _defaultPrecision >= MixedPrecisionType.FP8_E4M3);
    }

    /// <summary>
    /// Determines if a layer should skip mixed precision entirely (stay in FP32).
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>True if the layer should remain in FP32.</returns>
    public bool ShouldSkipMixedPrecision(string layerName)
    {
        return GetPrecision(layerName) == MixedPrecisionType.None;
    }

    #region Factory Methods

    /// <summary>
    /// Creates the default policy for FP16 mixed-precision training.
    /// </summary>
    /// <remarks>
    /// <para>Keeps normalization layers in FP32 as recommended by NVIDIA.</para>
    /// </remarks>
    public static LayerPrecisionPolicy ForFP16()
    {
        return new LayerPrecisionPolicy(MixedPrecisionType.FP16)
            .KeepInFP32("BatchNorm")
            .KeepInFP32("LayerNorm")
            .KeepInFP32("RMSNorm")
            .KeepInFP32("GroupNorm");
    }

    /// <summary>
    /// Creates the default policy for BF16 mixed-precision training.
    /// </summary>
    /// <remarks>
    /// <para>BF16 is more stable, so fewer layers need FP32.</para>
    /// </remarks>
    public static LayerPrecisionPolicy ForBF16()
    {
        return new LayerPrecisionPolicy(MixedPrecisionType.BF16)
            .KeepInFP32("Loss"); // Only loss computation in FP32
    }

    /// <summary>
    /// Creates the default policy for FP8 mixed-precision training.
    /// </summary>
    /// <remarks>
    /// <para>FP8 has reduced precision, so more layers need higher precision fallback.</para>
    /// </remarks>
    public static LayerPrecisionPolicy ForFP8()
    {
        return new LayerPrecisionPolicy(MixedPrecisionType.FP8_Hybrid)
            .KeepInFP32("BatchNorm")
            .KeepInFP32("LayerNorm")
            .KeepInFP32("RMSNorm")
            .KeepInFP32("GroupNorm")
            .KeepInFP16("Softmax")
            .KeepInFP16("Attention")
            .KeepInFP16("Embedding")
            .KeepInFP16("LMHead")
            .KeepInFP16("Output");
    }

    /// <summary>
    /// Creates a policy for transformer models with FP8.
    /// </summary>
    /// <remarks>
    /// <para>Optimized for large language models and vision transformers.</para>
    /// </remarks>
    public static LayerPrecisionPolicy ForFP8Transformers()
    {
        return new LayerPrecisionPolicy(MixedPrecisionType.FP8_Hybrid)
            // Normalization in FP32
            .KeepInFP32("LayerNorm")
            .KeepInFP32("RMSNorm")
            // Attention components in FP16 (due to softmax)
            .KeepInFP16("attention")
            .KeepInFP16("self_attn")
            .KeepInFP16("cross_attn")
            .KeepInFP16("MultiHead")
            // Input/Output in FP16
            .KeepInFP16("Embedding")
            .KeepInFP16("embed_tokens")
            .KeepInFP16("lm_head")
            .KeepInFP16("final_proj")
            // Q/K/V projections can stay in FP8
            // MLP/FFN can stay in FP8
            ;
    }

    /// <summary>
    /// Creates a policy for convolutional networks with FP8.
    /// </summary>
    /// <remarks>
    /// <para>Optimized for image classification and detection models.</para>
    /// </remarks>
    public static LayerPrecisionPolicy ForFP8ConvNets()
    {
        return new LayerPrecisionPolicy(MixedPrecisionType.FP8_Hybrid)
            .KeepInFP32("BatchNorm")
            .KeepInFP32("GroupNorm")
            .KeepInFP16("stem")      // First conv layer
            .KeepInFP16("classifier") // Final FC layer
            .KeepInFP16("head")
            .KeepInFP16("fc");
    }

    /// <summary>
    /// Creates a fully FP32 policy (no mixed precision).
    /// </summary>
    public static LayerPrecisionPolicy FullPrecision()
    {
        return new LayerPrecisionPolicy(MixedPrecisionType.None);
    }

    #endregion

    /// <summary>
    /// Gets all layer patterns that should be excluded from a given precision level.
    /// </summary>
    /// <param name="minimumPrecision">The minimum precision level. Patterns requiring higher precision will be returned.</param>
    /// <returns>List of layer patterns that need higher precision.</returns>
    public IReadOnlyList<string> GetExcludedPatterns(MixedPrecisionType minimumPrecision)
    {
        var excluded = new List<string>();

        // Add patterns that need higher precision than the minimum
        foreach (var (pattern, precision) in _patterns)
        {
            // None = FP32 (highest), then FP16, then BF16, then FP8 variants (lowest)
            // If a pattern requires higher precision, it should be excluded
            if (ShouldExcludeForPrecision(precision, minimumPrecision))
            {
                excluded.Add(pattern);
            }
        }

        // Add exact matches that need higher precision
        foreach (var (layerName, precision) in _exactMatches)
        {
            if (ShouldExcludeForPrecision(precision, minimumPrecision))
            {
                excluded.Add(layerName);
            }
        }

        return excluded;
    }

    /// <summary>
    /// Determines if a layer's required precision should exclude it from a target precision.
    /// </summary>
    private static bool ShouldExcludeForPrecision(MixedPrecisionType requiredPrecision, MixedPrecisionType targetPrecision)
    {
        // FP32 (None) is highest precision, followed by FP16/BF16, then FP8 variants
        // Exclude if required precision is higher than target
        if (requiredPrecision == MixedPrecisionType.None)
        {
            // FP32 required - exclude from anything lower
            return targetPrecision != MixedPrecisionType.None;
        }

        if (requiredPrecision == MixedPrecisionType.FP16 || requiredPrecision == MixedPrecisionType.BF16)
        {
            // FP16/BF16 required - exclude from FP8 variants
            return targetPrecision >= MixedPrecisionType.FP8_E4M3;
        }

        // FP8 or lower - no need to exclude
        return false;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"LayerPrecisionPolicy: Default={_defaultPrecision}, " +
               $"ExactRules={_exactMatches.Count}, Patterns={_patterns.Count}";
    }
}

/// <summary>
/// Extension methods for applying layer precision policies.
/// </summary>
public static class LayerPrecisionPolicyExtensions
{
    /// <summary>
    /// Applies a precision policy to a MixedPrecisionConfig.
    /// </summary>
    /// <param name="config">The config to modify.</param>
    /// <param name="policy">The policy to apply.</param>
    /// <returns>The modified config.</returns>
    public static MixedPrecisionConfig WithLayerPolicy(this MixedPrecisionConfig config, LayerPrecisionPolicy policy)
    {
        // Get excluded patterns from the policy based on the config's precision type
        var excludedPatterns = policy.GetExcludedPatterns(config.PrecisionType);

        // Always set the exclusions (even if empty) to clear any prior configuration
        config.FP8ExcludedLayers = new List<string>(excludedPatterns);

        return config;
    }
}
