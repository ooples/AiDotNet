using System.Collections.Generic;

namespace AiDotNet.ModelCompression;

/// <summary>
/// One operating point on the compression-versus-accuracy frontier: the configured strategy applied to a
/// magnitude-ranked fraction of the model's weights, with the resulting model re-evaluated on real data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Compressing more of a model makes it smaller but usually less accurate. Each
/// point here answers "if I compress this fraction of the weights, how small does the model get and how much
/// accuracy do I keep?" — so you can pick the trade-off you want instead of guessing.</para>
/// </remarks>
public sealed class CompressionFrontierPoint<T>
{
    /// <summary>The fraction of weights (smallest-magnitude first) compressed at this operating point, 0..1.</summary>
    public double Fraction { get; init; }

    /// <summary>Original bytes divided by compressed bytes at this point (higher means smaller).</summary>
    public double CompressionRatio { get; init; }

    /// <summary>The compressed size in bytes at this point (compressed fraction plus the exact remainder).</summary>
    public long CompressedSizeBytes { get; init; }

    /// <summary>
    /// The share of the original model's fit retained after compressing and rebuilding, 0..1 — measured by
    /// re-running the decompressed model on the prepared data, not by a proxy.
    /// </summary>
    public double AccuracyRetained { get; init; }

    /// <summary>Root-mean-square error between the original weights and the decompressed weights at this point.</summary>
    public double ReconstructionError { get; init; }
}

/// <summary>
/// The model-compression audit produced by a configured compression strategy: the true size-versus-accuracy
/// trade-off of the actual decompressed model, a sensitivity-aware Pareto frontier, and weight reconstruction
/// error — everything needed to choose a compression level rather than trust a single headline ratio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Most toolkits report a compression ratio in isolation. This audit compresses the trained weights, decompresses
/// them, <b>rebuilds the model from the decompressed weights, and re-evaluates it on the prepared data</b>, so the
/// reported accuracy retention is the model's real predictive fit — not a weight-error proxy. The frontier sweeps
/// magnitude-ranked partial compression (compress the least important weights first) to trace how accuracy falls
/// as size shrinks, and the knee marks the most compression that still clears the retention tolerance.
/// </para>
/// </remarks>
public sealed class ModelCompressionReport<T>
{
    /// <summary>The configured compression strategy's type name.</summary>
    public string StrategyName { get; init; } = string.Empty;

    /// <summary>The number of trainable parameters that were compressed.</summary>
    public long ParameterCount { get; init; }

    /// <summary>The uncompressed size of the parameters in bytes.</summary>
    public long OriginalSizeBytes { get; init; }

    /// <summary>The compressed size in bytes at full compression (every weight through the strategy).</summary>
    public long CompressedSizeBytes { get; init; }

    /// <summary>Original bytes divided by compressed bytes at full compression (higher means smaller).</summary>
    public double CompressionRatio { get; init; }

    /// <summary>The share of the original model's fit retained at full compression, 0..1 (real re-evaluation).</summary>
    public double AccuracyRetained { get; init; }

    /// <summary>Root-mean-square error between the original and decompressed weights at full compression.</summary>
    public double ReconstructionError { get; init; }

    /// <summary>The prediction loss (RMSE on the prepared data) of the original, uncompressed model.</summary>
    public double BaselineLoss { get; init; }

    /// <summary>The prediction loss (RMSE on the prepared data) of the fully compressed, rebuilt model.</summary>
    public double CompressedLoss { get; init; }

    /// <summary>
    /// The compression-versus-accuracy operating points, from lightest to full compression — the Pareto frontier
    /// a user reads to choose a level.
    /// </summary>
    public IReadOnlyList<CompressionFrontierPoint<T>> Frontier { get; init; } = new List<CompressionFrontierPoint<T>>();

    /// <summary>The compressed fraction at the knee: the most compression that still meets the retention tolerance.</summary>
    public double KneeFraction { get; init; }

    /// <summary>The compression ratio achieved at the knee point.</summary>
    public double KneeCompressionRatio { get; init; }

    /// <summary>The accuracy retained at the knee point.</summary>
    public double KneeAccuracyRetained { get; init; }

    /// <summary>The retention tolerance the knee was selected against (a point qualifies when it clears this).</summary>
    public double RetentionTolerance { get; init; }

    /// <summary>Whether more than one frontier point was produced (a real sweep, not a single operating point).</summary>
    public bool SweepAvailable { get; init; }
}
