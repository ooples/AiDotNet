using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// #1745: the Adam8BitOptimizer's BF16 moment-storage mode must advertise a fused
/// config so it keeps the compiled fast path (with bf16 m/v) instead of dropping
/// to the eager tape. The 8-bit block-quant mode maps to the plan's int8 block-quantized
/// Adam with the same block size and minimum quantized length, and declines for the
/// configurations that kernel does not implement. These guard the optimizer→fused-kernel mapping.
/// </summary>
public class Adam8BitFusedSpecTests
{
    private static Adam8BitOptimizer<float, Tensor<float>, Tensor<float>> Make(
        bool bf16, bool amsGrad = false, bool adaptiveLr = false)
        => new(
            null,
            new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                UseBFloat16MomentStorage = bf16,
                UseAMSGrad = amsGrad,
                UseAdaptiveLearningRate = adaptiveLr,
            });

    [Fact]
    public void Bf16Mode_MapsToFusedAdam_WithBf16Moments()
    {
        var opt = Make(bf16: true);
        Assert.True(((IFusedOptimizerSpec)opt).TryGetFusedOptimizerConfig(out var cfg),
            "BF16-mode Adam8Bit should map to a fused config so it keeps the fused fast path.");
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam, cfg.Type);
        Assert.True(cfg.UseBf16Moments, "Fused config must request bf16 moment storage.");
    }

    [Fact]
    public void BlockQuantMode_MapsToFusedInt8Adam_WithItsBlockSizeAndMinimum()
    {
        var opt = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(
            null, new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>> { BlockSize = 256, Min8BitSize = 1024 });
        Assert.True(((IFusedOptimizerSpec)opt).TryGetFusedOptimizerConfig(out var cfg),
            "8-bit block-quant Adam8Bit should map to the plan's int8 block-quantized Adam.");
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam, cfg.Type);
        Assert.False(cfg.UseBf16Moments);
        Assert.Equal(256, cfg.Int8MomentBlockSize);
        Assert.Equal(1024, cfg.Int8MinQuantizedLength);
        Assert.Equal(0f, cfg.WeightDecay);
    }

    [Theory]
    [InlineData("percentile")]
    [InlineData("stochastic")]
    [InlineData("v-only")]
    [InlineData("amsgrad")]
    public void BlockQuantMode_TheKernelDoesNotImplement_DoesNotMap(string scenario)
    {
        var opt = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(
            null, new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                QuantizationPercentile = scenario == "percentile" ? 99.9 : 100.0,
                UseStochasticRounding = scenario == "stochastic",
                CompressBothMoments = scenario != "v-only",
                UseAMSGrad = scenario == "amsgrad",
            });
        Assert.False(((IFusedOptimizerSpec)opt).TryGetFusedOptimizerConfig(out _),
            $"Adam8Bit mapped to the int8 kernel under '{scenario}', which that kernel does not implement.");
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
}
