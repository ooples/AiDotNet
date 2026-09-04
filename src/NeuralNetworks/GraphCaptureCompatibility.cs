using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Selects graph-capture-compatible inference paths while a forward is being traced or validated.
/// </summary>
/// <remarks>
/// <para>
/// Some inference fast paths intentionally operate on raw arrays and therefore cannot participate in
/// <see cref="GraphMode"/>. Validation must execute the same canonical forward as tracing; comparing a
/// captured graph with a separate optimized kernel confuses legitimate platform-specific rounding with
/// stale input binding.
/// </para>
/// <para>
/// The explicit scope is used only for the validation forward, where no graph is active. Ordinary eager,
/// compiled, CPU, and GPU inference remain unaffected.
/// </para>
/// </remarks>
internal static class GraphCaptureCompatibility
{
    [ThreadStatic]
    private static bool _isExplicitlyActive;

    internal static bool IsActive => _isExplicitlyActive || GraphMode.IsActive;

    internal static Scope Enter()
    {
        bool previous = _isExplicitlyActive;
        _isExplicitlyActive = true;
        return new Scope(previous);
    }

    /// <summary>
    /// Copies the caller's complete optimization policy while disabling only nested compilation.
    /// </summary>
    internal static TensorCodecOptions CreateOptionsWithoutCompilation(TensorCodecOptions source)
    {
        return new TensorCodecOptions
        {
            EnableCompilation = false,
            EnableDataflowFusion = source.EnableDataflowFusion,
            EnableAlgebraicBackward = source.EnableAlgebraicBackward,
            EnableSpectralDecomposition = source.EnableSpectralDecomposition,
            EnableFftConv = source.EnableFftConv,
            FftConvKernelThreshold = source.FftConvKernelThreshold,
            SpectralErrorTolerance = source.SpectralErrorTolerance,
            DataflowFusionMaxHidden = source.DataflowFusionMaxHidden,
            EnableConvBnFusion = source.EnableConvBnFusion,
            EnableAttentionFusion = source.EnableAttentionFusion,
            EnablePointwiseFusion = source.EnablePointwiseFusion,
            EnableConstantFolding = source.EnableConstantFolding,
            EnableForwardCSE = source.EnableForwardCSE,
            EnableBlasBatch = source.EnableBlasBatch,
            EnableBackwardGradientPooling = source.EnableBackwardGradientPooling,
            EnableMixedPrecision = source.EnableMixedPrecision,
            MixedPrecisionPolicy = source.MixedPrecisionPolicy,
            Deterministic = source.Deterministic,
            UseCudnn = source.UseCudnn,
            UseCudnnBatchNorm = source.UseCudnnBatchNorm,
            UseCublas = source.UseCublas
        };
    }

    internal readonly struct Scope : IDisposable
    {
        private readonly bool _previous;

        internal Scope(bool previous)
        {
            _previous = previous;
        }

        public void Dispose()
        {
            _isExplicitlyActive = _previous;
        }
    }
}
