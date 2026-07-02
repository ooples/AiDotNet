using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// #1745: the Adam8BitOptimizer's BF16 moment-storage mode must advertise a fused
/// config so it keeps the compiled fast path (with bf16 m/v) instead of dropping
/// to the eager tape. The true 8-bit block-quant mode has no fused kernel yet and
/// must NOT map (it would otherwise run as plain fp32-moment Adam, silently losing
/// the block quantization). These guard the optimizer→fused-kernel mapping.
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
    public void BlockQuantMode_DoesNotMapToFused()
    {
        // UseBFloat16MomentStorage == false ⇒ true int8 block-quant moments,
        // which has no fused kernel yet — must fall back to the eager tape.
        var opt = Make(bf16: false);
        Assert.False(((IFusedOptimizerSpec)opt).TryGetFusedOptimizerConfig(out _),
            "8-bit block-quant Adam8Bit has no fused kernel and must not map.");
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
