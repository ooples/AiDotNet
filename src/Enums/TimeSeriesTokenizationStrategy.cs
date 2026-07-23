namespace AiDotNet.Enums;

/// <summary>
/// Tokenization strategies for time series foundation models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Different foundation models use different ways to convert raw time
/// series data into "tokens" that the model can process:
/// <list type="bullet">
/// <item><b>Patching:</b> Splits the series into fixed-size chunks (most common)</item>
/// <item><b>Quantization:</b> Converts continuous values to discrete vocabulary tokens</item>
/// <item><b>AdaptivePatching:</b> Variable-size patches based on local complexity</item>
/// <item><b>LagFeatures:</b> Uses historical lags as features instead of raw values</item>
/// <item><b>RawSequence:</b> No tokenization — feeds raw values directly</item>
/// </list>
/// </para>
/// <para>
/// <b>Reference:</b> See Chronos (ICML 2024) for quantization, PatchTST (ICLR 2023) for
/// patching, Kairos (2025) for adaptive patching, and Lag-Llama (2023) for lag features.
/// </para>
/// </remarks>
public enum TimeSeriesTokenizationStrategy
{
    /// <summary>
    /// Non-overlapping or overlapping fixed-size patches.
    /// Used by: PatchTST, Chronos-2, Moirai, MOMENT, TTM, Sundial.
    /// </summary>
    Patching = 0,

    /// <summary>
    /// Discrete vocabulary quantization via uniform or learned binning.
    /// Used by: Chronos v1.
    /// </summary>
    Quantization = 1,

    /// <summary>
    /// Variable-size patches based on local information density.
    /// Used by: Kairos (Mixture-of-Size encoder).
    /// </summary>
    AdaptivePatching = 2,

    /// <summary>
    /// Historical lag values as input features.
    /// Used by: Lag-Llama.
    /// </summary>
    LagFeatures = 3,

    /// <summary>
    /// Raw numerical sequence without tokenization.
    /// Used by: TimesFM, DeepAR.
    /// </summary>
    RawSequence = 4
}
