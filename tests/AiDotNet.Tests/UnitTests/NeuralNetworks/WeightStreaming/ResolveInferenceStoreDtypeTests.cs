using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.WeightStreaming;

/// <summary>
/// Tier-1 quant-resident inference store selection (AiDotNet#1622): for a foundation-scale model
/// in inference, <see cref="NeuralNetworkBase{T}.ResolveInferenceStoreDtype"/> keeps the weight set
/// resident at the loosest precision whose execution footprint fits the budget (bf16 → int8 → int4
/// fallback), so multi-forward Predict pays no per-forward paging I/O.
/// </summary>
public class ResolveInferenceStoreDtypeTests
{
    private const long GiB = 1024L * 1024 * 1024;

    [Fact]
    public void Bf16Fits_KeepsBf16_Unchanged()
    {
        // 1B params → fp32 execution footprint 4 GB, well under an 11 GB budget → stay bf16
        // (least lossy; its decoded GEMM owner also fits the execution budget).
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
    public void NeitherInt8NorInt4Fits_UsesInt4Streaming()
    {
        // 30B params → int8 = 30 GB and packed int4 = 15 GB, both over an 11 GB budget. Keep
        // int4 and page the smallest/no-upcast representation instead of regressing to bf16 I/O.
        Assert.Equal(StreamingStoreDtype.Int4,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(30_000_000_000L, 11 * GiB));
    }

    [Fact]
    public void BoundaryAtExactBudget_PrefersThatTier()
    {
        // Native fp32 execution == budget exactly → bf16 store (the <= comparison keeps the
        // highest-fidelity tier when its executable owner just fits).
        Assert.Equal(StreamingStoreDtype.Bf16,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(5 * GiB / 4, 5 * GiB));
        // int8 == budget exactly (native execution = 4x over) → int8.
        Assert.Equal(StreamingStoreDtype.Int8,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(5 * GiB, 5 * GiB));
        // packed int4 == budget exactly → int4.
        Assert.Equal(StreamingStoreDtype.Int4,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(10 * GiB, 5 * GiB));
    }

    [Fact]
    public void Bf16StoreFitsButDecodedExecutionDoesNot_StepsDownToInt8()
    {
        // This is the Transfusion timeout mechanism: 6.5B params have a ~13 GB bf16 store that
        // appears to fit a 16 GB cap, but GEMM needs decoded fp32 owners (~26 GB). The executable
        // footprint does not fit, while the no-upcast int8 representation does.
        Assert.Equal(StreamingStoreDtype.Int8,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(6_500_000_000L, 16 * GiB));
    }

    [Fact]
    public void OddParameterCount_UsesCeilingForPackedInt4Boundary()
    {
        Assert.Equal(StreamingStoreDtype.Int4,
            NeuralNetworkBase<float>.ResolveInferenceStoreDtype(11, 6));
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
