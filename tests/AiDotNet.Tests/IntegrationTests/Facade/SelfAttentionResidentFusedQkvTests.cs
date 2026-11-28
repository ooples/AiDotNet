using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Facade;

/// <summary>
/// The fused QKV projection on the fp16-resident path must build from the authoritative weights. The separate
/// projection path offloads each projection to an fp16 master and leaves a <c>[0,0]</c> fp32 placeholder, so a fused
/// build that later read the fp32 fields failed (or used empty weights).
/// </summary>
/// <remarks>
/// Serial: the fused path is selected by <see cref="ForwardScratchGate.Override"/>, a process-wide static that would
/// switch every scratch sub-path on for tests running beside this one.
/// </remarks>
[Collection("NonParallelIntegration")]
public sealed class SelfAttentionResidentFusedQkvTests
{
    [Fact]
    public void FusedQkv_AfterSeparateResidentForward_BuildsFromTheFp16Masters()
    {
        var attn = new SelfAttentionLayer<float>(
            sequenceLength: 4,
            embeddingDimension: 16,
            headCount: 4,
            activationFunction: (IActivationFunction<float>?)null);
        attn.SetTrainingMode(false);

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2101);
        var input = new Tensor<float>(new[] { 2, 4, 16 });
        var span = input.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() - 0.5);

        var fp32 = attn.Forward(input);

        bool? previousGate = ForwardScratchGate.Override;
        try
        {
            ForwardScratchGate.Override = true;
            attn.LowPrecisionResident = true;

            // Derived caching off: the separate projection path runs and offloads Q/K/V to their fp16 masters.
            DerivedTensorCacheSettings.SetThreadOverride(false);
            var separate = attn.Forward(input);

            // Derived caching on: the fused path now builds its [E, 3E] weight and must use those masters.
            DerivedTensorCacheSettings.SetThreadOverride(true);
            var fused = attn.Forward(input);

            Assert.Equal(fp32.Length, fused.Length);
            var a = fp32.AsSpan();
            var b = separate.AsSpan();
            var c = fused.AsSpan();
            for (int i = 0; i < a.Length; i++)
            {
                Assert.False(float.IsNaN(c[i]) || float.IsInfinity(c[i]), $"non-finite fused resident output at {i}");
                double tol = 1e-2 * (Math.Abs(a[i]) + 1.0);
                Assert.True(Math.Abs(a[i] - c[i]) <= tol, $"fused resident output diverges from fp32 at {i}: {a[i]} vs {c[i]}");
                Assert.True(Math.Abs(b[i] - c[i]) <= 1e-4 * (Math.Abs(b[i]) + 1.0),
                    $"fused and separate resident paths disagree at {i}: {b[i]} vs {c[i]}");
            }
        }
        finally
        {
            DerivedTensorCacheSettings.SetThreadOverride(null);
            ForwardScratchGate.Override = previousGate;
        }
    }
}
