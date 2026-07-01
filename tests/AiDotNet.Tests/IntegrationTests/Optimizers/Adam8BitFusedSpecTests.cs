using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// #1745: the Adam8BitOptimizer's BF16 moment-storage mode must advertise a fused
/// config so it keeps the compiled fast path (with bf16 m/v) instead of dropping
/// to the eager tape. The deterministic true 8-bit block-quant mode must request
/// fused int8 moments, while unsupported quantization variants must still fall
/// back instead of silently running as plain fp32-moment Adam. These guard the
/// optimizer→fused-kernel mapping.
/// </summary>
public class Adam8BitFusedSpecTests
{
    private static Adam8BitOptimizer<float, Tensor<float>, Tensor<float>> Make(
        bool bf16,
        bool amsGrad = false,
        bool adaptiveLr = false,
        bool useDynamicQuantization = true,
        bool compressBothMoments = true,
        double quantizationPercentile = 99.9,
        bool stochasticRounding = false,
        int fullPrecisionUpdateFrequency = 0,
        int blockSize = 2048)
        => new(
            null,
            new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                UseBFloat16MomentStorage = bf16,
                UseAMSGrad = amsGrad,
                UseAdaptiveLearningRate = adaptiveLr,
                UseDynamicQuantization = useDynamicQuantization,
                CompressBothMoments = compressBothMoments,
                QuantizationPercentile = quantizationPercentile,
                UseStochasticRounding = stochasticRounding,
                FullPrecisionUpdateFrequency = fullPrecisionUpdateFrequency,
                BlockSize = blockSize,
            });

    [Fact]
    public void Bf16Mode_MapsToFusedAdam_WithBf16Moments()
    {
        var opt = Make(bf16: true);
        Assert.True(((IFusedOptimizerSpec)opt).TryGetFusedOptimizerConfig(out var cfg),
            "BF16-mode Adam8Bit should map to a fused config so it keeps the fused fast path.");
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam, cfg.Type);
        Assert.True(cfg.UseBf16Moments, "Fused config must request bf16 moment storage.");
        Assert.False(cfg.UseInt8Moments, "BF16 mode must not also request int8 moment storage.");
    }

    [Fact]
    public void BlockQuantMode_MapsToFusedAdam_WithInt8Moments()
    {
        var opt = Make(bf16: false, quantizationPercentile: 100.0, blockSize: 1024);

        Assert.True(((IFusedOptimizerSpec)opt).TryGetFusedOptimizerConfig(out var cfg),
            "Exact deterministic 8-bit block-quant Adam8Bit should map to fused int8 moments.");
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam, cfg.Type);
        Assert.False(cfg.UseBf16Moments, "Int8 mode must not also request bf16 moment storage.");
        Assert.True(cfg.UseInt8Moments, "Fused config must request int8 moment storage.");
        Assert.Equal(1024, cfg.Int8MomentBlockSize);
    }

    [Fact]
    public void Bf16Mode_WithAmsGradOrAdaptiveLr_DoesNotMap()
    {
        // The bf16 Adam/AdamW kernels don't model AMSGrad's max-second-moment or
        // an adaptive (per-step-mutated) learning rate, so these fall back.
        Assert.False(((IFusedOptimizerSpec)Make(bf16: true, amsGrad: true))
            .TryGetFusedOptimizerConfig(out _), "AMSGrad must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: true, adaptiveLr: true))
            .TryGetFusedOptimizerConfig(out _), "Adaptive LR must fall back to eager.");
    }

    [Fact]
    public void BlockQuantMode_WithUnsupportedInt8Options_DoesNotMap()
    {
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false))
            .TryGetFusedOptimizerConfig(out _), "Percentile clipping must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, compressBothMoments: false))
            .TryGetFusedOptimizerConfig(out _), "Partial moment compression must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, stochasticRounding: true))
            .TryGetFusedOptimizerConfig(out _), "Stochastic rounding must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, fullPrecisionUpdateFrequency: 256))
            .TryGetFusedOptimizerConfig(out _), "Full-precision refreshes must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, useDynamicQuantization: false))
            .TryGetFusedOptimizerConfig(out _), "Static quantization must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, blockSize: 0))
            .TryGetFusedOptimizerConfig(out _), "Invalid block size must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, amsGrad: true))
            .TryGetFusedOptimizerConfig(out _), "AMSGrad must fall back to eager.");
        Assert.False(((IFusedOptimizerSpec)Make(bf16: false, quantizationPercentile: 100.0, adaptiveLr: true))
            .TryGetFusedOptimizerConfig(out _), "Adaptive LR must fall back to eager.");
    }
}
