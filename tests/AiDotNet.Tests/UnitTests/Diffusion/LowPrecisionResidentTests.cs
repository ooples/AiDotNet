using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Fast correctness coverage for the fp16-resident weight path (issue #1672). A foundation-scale
/// DiT exceeds 16 GB in fp32; <see cref="LayerBase{T}.LowPrecisionResident"/> stores the large weight
/// matrices as fp16 and upcasts them transiently per matmul (via the SIMD <c>FromHalfSpan</c>/<c>ToHalfSpan</c>
/// bulk casts), halving resident weight memory while computing in full precision. This test exercises the
/// mechanism on a small layer (no foundation-scale memory/time needed): the fp16-resident forward must
/// match the fp32 forward within fp16 rounding, be deterministic across repeated forwards (the fp16 master
/// is reused), and leave the reported parameter count unchanged after the fp32 master is freed.
/// </summary>
public sealed class LowPrecisionResidentTests
{
    [Fact]
    public void LowPrecisionResident_MatchesFp32Forward_AndIsDeterministic()
    {
        var layer = new DenseLayer<float>(64, (AiDotNet.Interfaces.IActivationFunction<float>)new IdentityActivation<float>());

        var rng = new Random(1);
        var input = new Tensor<float>(new[] { 4, 64 });
        var inSpan = input.AsWritableSpan();
        for (int i = 0; i < inSpan.Length; i++) inSpan[i] = (float)(rng.NextDouble() - 0.5);

        // Baseline: fp32 weights (first forward also lazily initializes them).
        var fp32 = layer.Forward(input);
        long paramsBefore = layer.ParameterCount;

        // Enable fp16-resident: the next forward downcasts the SAME weights to the fp16 master, frees the
        // fp32 copy, and upcasts transiently for the matmul. A second forward must reuse the fp16 master.
        layer.LowPrecisionResident = true;
        var resident1 = layer.Forward(input);
        var resident2 = layer.Forward(input);

        Assert.Equal(fp32.Length, resident1.Length);
        var a = fp32.AsSpan();
        var b = resident1.AsSpan();
        var c = resident2.AsSpan();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.False(float.IsNaN(b[i]) || float.IsInfinity(b[i]), $"non-finite fp16-resident output at {i}");
            // fp16 mantissa ~11 bits; allow rounding error scaled by magnitude.
            double tol = 1e-2 * (Math.Abs(a[i]) + 1.0);
            Assert.True(Math.Abs(a[i] - b[i]) <= tol, $"fp16-resident mismatch at {i}: fp32={a[i]} fp16={b[i]}");
            Assert.Equal(b[i], c[i]); // deterministic: same fp16 master reused across forwards
        }

        // Parameter count must be unchanged even though the fp32 master was freed.
        Assert.Equal(paramsBefore, layer.ParameterCount);
    }
}
