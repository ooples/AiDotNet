using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The persistence probe asks whether any gradient reached the parameters. A host-resident gradient is scanned in
/// place: squaring each one on the engine allocated a gradient-sized temporary per parameter on every probed step,
/// and a small float gradient squared to zero and read as "no gradient".
/// </summary>
public class FusedTrainingSessionProbeTests
{
    private static bool AnyGradientNonZero(params Tensor<float>[] gradients)
    {
        var method = typeof(FusedTrainingSession<float>).GetMethod(
            "AnyGradientNonZero", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        var map = new Dictionary<Tensor<float>, Tensor<float>>(TensorReferenceComparer<Tensor<float>>.Instance);
        foreach (var gradient in gradients) map[new Tensor<float>(new[] { 1 })] = gradient;
        var result = method?.Invoke(null, new object[] { map });
        return result is bool reached && reached;
    }

    private static Tensor<float> Filled(int length, int at, float value)
    {
        var tensor = new Tensor<float>(new[] { length });
        if (at >= 0) tensor[at] = value;
        return tensor;
    }

    [Fact]
    public void ZeroGradients_DidNotReach()
        => Assert.False(AnyGradientNonZero(Filled(64, -1, 0f), Filled(32, 5, -0f)));

    [Fact]
    public void ASmallGradient_Reached()
        // 1e-30 squares to 0 in float: the engine sum of squares called this gradient zero.
        => Assert.True(AnyGradientNonZero(Filled(64, -1, 0f), Filled(1024, 1000, 1e-30f)));

    [Fact]
    public void ANonFiniteGradient_Reached()
        => Assert.True(AnyGradientNonZero(Filled(16, 3, float.NaN)));

    [Fact]
    public void TheProbe_DoesNotCopyTheGradients()
    {
        var large = Filled(1 << 20, (1 << 20) - 1, 0.5f);
        AnyGradientNonZero(large); // warm the reflection and JIT
        long before = GC.GetAllocatedBytesForCurrentThread();
        Assert.True(AnyGradientNonZero(large));
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 64 * 1024,
            $"probing a 4 MB gradient allocated {allocated} bytes; squaring it on the engine copied it.");
    }
}