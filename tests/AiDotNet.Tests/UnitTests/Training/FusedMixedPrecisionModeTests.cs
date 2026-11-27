using AiDotNet.Enums;
using AiDotNet.MixedPrecision;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The matmul autocast mode on the mixed-precision configuration reaches the fused compiled step: PyTorch's
/// cast-the-operands semantics by default, the fast compute type only when asked for, and a change of mode is a change
/// of the configuration's signature (so a committed plan refuses it rather than half-applying it).
/// </summary>
public class FusedMixedPrecisionModeTests
{
    [Fact]
    public void Default_is_cast_operands()
    {
        Assert.Equal(MatMulAutocastMode.CastOperands, new MixedPrecisionConfig().MatMulAutocastMode);
        Assert.Equal(MatMulAutocastMode.CastOperands, MixedPrecisionConfig.ForBF16().MatMulAutocastMode);
    }

    [Theory]
    [InlineData(MixedPrecisionType.FP16, MatMulAutocastMode.CastOperands)]
    [InlineData(MixedPrecisionType.FP16, MatMulAutocastMode.FastComputeType)]
    [InlineData(MixedPrecisionType.BF16, MatMulAutocastMode.FastComputeType)]
    public void The_fused_form_carries_the_mode_and_its_signature_names_it(MixedPrecisionType type, MatMulAutocastMode mode)
    {
        var config = new MixedPrecisionConfig { PrecisionType = type, MatMulAutocastMode = mode };
        using var context = new MixedPrecisionContext(config);
        var fused = FusedMixedPrecision.From(context, out var reason);
        Assert.Null(reason);
        Assert.NotNull(fused);
        Assert.Equal(mode, fused!.MatMulAutocastMode);

        var other = mode == MatMulAutocastMode.CastOperands ? MatMulAutocastMode.FastComputeType : MatMulAutocastMode.CastOperands;
        using var otherContext = new MixedPrecisionContext(new MixedPrecisionConfig { PrecisionType = type, MatMulAutocastMode = other });
        var otherFused = FusedMixedPrecision.From(otherContext, out _);
        Assert.NotEqual(fused.Signature, otherFused!.Signature);
    }
}
