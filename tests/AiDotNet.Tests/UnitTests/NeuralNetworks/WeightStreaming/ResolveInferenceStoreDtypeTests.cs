using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.WeightStreaming;

/// <summary>
/// Tier-1 quant-resident inference store selection (AiDotNet#1622): for a foundation-scale model
/// in inference, <see cref="NeuralNetworkBase{T}.ResolveInferenceStoreDtype"/> keeps the weight set
/// resident at the loosest precision that fits the budget (bf16 → int8 → … → bf16-streaming
/// fallback), so multi-forward Predict pays no per-forward paging I/O.
/// </summary>
public class ResolveInferenceStoreDtypeTests
{
    private const long GiB = 1024L * 1024 * 1024;

    [Fact]
    public void Bf16Fits_KeepsBf16_Unchanged()
    {
        // 1B params → bf16 footprint 2 GB, well under an 11 GB budget → stay bf16 (least lossy,
        // the existing Auto-inference behaviour for models that already fit resident).
        Assert.Equal(StreamingStoreDtype.Bf16,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(1_000_000_000L, 11 * GiB));
    }

    [Fact]
    public void Bf16TooBig_Int8Fits_StepsDownToInt8Resident()
    {
        // 7B params → bf16 = 14 GB (> 11 GB budget) but int8 = 7 GB (< 11 GB) → int8-resident.
        Assert.Equal(StreamingStoreDtype.Int8,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(7_000_000_000L, 11 * GiB));
    }

    [Fact]
    public void NeitherFits_FallsBackToBf16Streaming()
    {
        // 30B params → bf16 = 60 GB, int8 = 30 GB, both over an 11 GB budget → bf16 (pool pages).
        // (int4-resident — 15 GB — is the next rung once the int4 Tensors release is consumed.)
        Assert.Equal(StreamingStoreDtype.Bf16,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(30_000_000_000L, 11 * GiB));
    }

    [Fact]
    public void BoundaryAtExactBudget_PrefersThatTier()
    {
        // bf16 == budget exactly → bf16 (the <= comparison keeps the looser tier when it just fits).
        Assert.Equal(StreamingStoreDtype.Bf16,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(5 * GiB / 2, 5 * GiB));
        // int8 == budget exactly (bf16 = 2x over) → int8.
        Assert.Equal(StreamingStoreDtype.Int8,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(5 * GiB, 5 * GiB));
    }

    [Theory]
    [InlineData(0L, 11)]
    [InlineData(1_000_000_000L, 0)]
    public void InvalidInputs_ReturnAuto(long paramCount, int budgetGiB)
    {
        Assert.Equal(StreamingStoreDtype.Auto,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(paramCount, budgetGiB * GiB));
    }
}
