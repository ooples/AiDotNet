using System;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Controlled-scale validation of transparent weight streaming for DiT noise
/// predictors (Tensors #602 / #603, issue #430). Foundation-scale DiT models
/// (Sora / Flux2 / WanVideo) can't be forwarded on a 16 GB host, so these tests
/// engage the exact same code path on a tiny DiT via the test-only threshold +
/// resident-cap overrides, and assert the two guarantees that matter:
///
///   1. Faithful paging — the CORE streaming contract: forward-1 (weights
///      resident) and forward-2 (weights registered, dropped to the disk-backed
///      pool, and auto-rehydrated on access) must be BIT-IDENTICAL. That is the
///      "transparent" guarantee — a model computes the same result it would with
///      all weights resident, just within a bounded memory budget.
///   2. Bounded residency — after the weights are registered, a forward keeps
///      the pool's resident set under the cap (EvictionCount &gt; 0,
///      ResidentBytes ≤ cap); the symmetric owner-drop frees the tensors'
///      resident copies.
///
/// <para>The streaming forward reaches each weight through <c>Tensor.Memory</c>
/// (SelfAttention etc.), which only auto-rehydrates a paged-out weight on the
/// Tensors build that carries the #603 <c>.Memory</c> gate. These tests probe
/// for that gate at runtime and <see cref="Skip"/> cleanly on an older Tensors
/// (e.g. 0.95.0) — they run and pass once the consuming project bumps to the
/// #603 release. This is a cross-repo version gate, not a masked failure.</para>
///
/// <para>The tests mutate the process-wide <see cref="WeightRegistry"/> and the
/// static engagement overrides, so each resets all global state in a finally
/// block. Tests in one class are one xUnit collection (run sequentially), and no
/// other diffusion test forwards a DiT model, so the global override window
/// doesn't race a neighbour.</para>
/// </summary>
[Collection("WeightStreaming-Singleton")]
public class DiTWeightStreamingTests
{
    private const int InputChannels = 4;
    private const int HiddenSize = 64;
    private const int NumLayers = 4;
    private const int NumHeads = 4;
    private const int PatchSize = 2;
    private const int ContextDim = 32;
    private const int LatentSpatial = 8;
    private const int Seed = 42;

    // Cap above the largest single weight (~128 KiB here, so AllocateStreaming
    // never goes over-budget on one tensor) but below the model's total
    // (~525 KiB), so the pool pages cold weights out ACROSS the forward. The
    // production floor is 512 MiB (>> any single weight); this mirrors that
    // relationship at the tiny test scale.
    private const long ResidentCap = 256 * 1024;

    private static DiTNoisePredictor<double> NewPredictor() => new(
        inputChannels: InputChannels,
        hiddenSize: HiddenSize,
        numLayers: NumLayers,
        numHeads: NumHeads,
        patchSize: PatchSize,
        contextDim: ContextDim,
        latentSpatialSize: LatentSpatial,
        seed: Seed);

    private static Tensor<double> NewInput() =>
        new(new[] { 1, InputChannels, LatentSpatial, LatentSpatial });

    /// <summary>
    /// True when the referenced Tensors build auto-rehydrates a paged-out
    /// streaming weight read through <c>Tensor.Memory</c> (PR #603). Detected by
    /// running a tiny streamed forward — SelfAttention reaches weights via
    /// <c>.Memory</c>, which on a pre-#603 build slices empty dropped storage and
    /// throws <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    private static bool MemoryGatePresent()
    {
        WeightRegistry.Reset();
        NoisePredictorBase<double>.StreamingThresholdOverride = 1;
        NoisePredictorBase<double>.StreamingResidentCapOverride = ResidentCap;
        try
        {
            var p = NewPredictor();
            _ = p.PredictNoise(NewInput(), timestep: 250); // engage + register + drop
            _ = p.PredictNoise(NewInput(), timestep: 250); // rehydrate via .Memory
            return true;
        }
        catch (ArgumentOutOfRangeException)
        {
            return false;
        }
        finally
        {
            NoisePredictorBase<double>.StreamingThresholdOverride = null;
            NoisePredictorBase<double>.StreamingResidentCapOverride = null;
            WeightRegistry.Reset();
        }
    }

    [SkippableFact]
    public void Streaming_FaithfulPaging_AndBoundedResidentSet()
    {
        Skip.IfNot(MemoryGatePresent(),
            "Requires the Tensors .Memory auto-rehydrate gate (PR #603); the referenced " +
            "build lacks it. Bump AiDotNet.Tensors to the #603 release to run this test.");

        try
        {
            WeightRegistry.Reset();
            // threshold 1 forces engage on the tiny model; the cap forces the
            // pool to page weights out across the forward.
            NoisePredictorBase<double>.StreamingThresholdOverride = 1;
            NoisePredictorBase<double>.StreamingResidentCapOverride = ResidentCap;

            var streamed = NewPredictor();

            // First forward: engages streaming, resolves + registers weights.
            var out1 = streamed.PredictNoise(NewInput(), timestep: 250);
            var report1 = WeightRegistry.GetStreamingReport();
            Assert.True(report1.RegisteredEntryCount > 0,
                "Streaming should have engaged and registered the predictor's weights.");

            // Second forward: runs against the bounded, paged-out resident set.
            var out2 = streamed.PredictNoise(NewInput(), timestep: 250);
            var report2 = WeightRegistry.GetStreamingReport();

            // Bounded residency: the pool paged cold weights to disk under the
            // cap and never held more than the cap resident.
            Assert.True(report2.EvictionCount > 0,
                "The cap (below the model's total weight bytes) must force at least one eviction.");
            Assert.True(report2.ResidentBytes <= ResidentCap,
                $"Resident bytes ({report2.ResidentBytes}) must stay within the {ResidentCap}-byte cap.");

            var firstOut = out1.ToArray();
            var streamedOut = out2.ToArray();
            Assert.True(firstOut.Length > 0);
            Assert.Equal(firstOut.Length, streamedOut.Length);

            // Output must be finite and non-trivial — an all-zero or NaN/Inf
            // forward would make the faithfulness check meaningless.
            bool anyNonZero = false;
            foreach (var v in streamedOut)
            {
                Assert.True(!double.IsNaN(v) && !double.IsInfinity(v), "Streamed output must be finite.");
                if (Math.Abs(v) > 1e-12) anyNonZero = true;
            }
            Assert.True(anyNonZero, "Streamed forward produced an all-zero output (weights never resolved).");

            // THE core streaming guarantee: paging weights to disk and back must
            // not change the result. forward-1 (resident) vs forward-2 (paged out
            // then auto-rehydrated on access) must be BIT-IDENTICAL.
            for (int i = 0; i < firstOut.Length; i++)
                Assert.True(Math.Abs(firstOut[i] - streamedOut[i]) < 1e-12,
                    $"Rehydrated forward-2[{i}]={streamedOut[i]} diverged from resident forward-1 {firstOut[i]} — disk paging is not faithful.");
        }
        finally
        {
            NoisePredictorBase<double>.StreamingThresholdOverride = null;
            NoisePredictorBase<double>.StreamingResidentCapOverride = null;
            WeightRegistry.Reset();
        }
    }

    [Fact]
    public void Streaming_NotEngaged_BelowThreshold()
    {
        // With no override, a tiny predictor is far below the 500M default and
        // must NOT engage streaming — leaving the registry untouched. (No
        // .Memory-gate dependency: nothing pages out, so this always runs.)
        WeightRegistry.Reset();
        NoisePredictorBase<double>.StreamingThresholdOverride = null;
        NoisePredictorBase<double>.StreamingResidentCapOverride = null;
        try
        {
            var predictor = NewPredictor();
            _ = predictor.PredictNoise(NewInput(), timestep: 100);

            var report = WeightRegistry.GetStreamingReport();
            Assert.Equal(0, report.RegisteredEntryCount);
        }
        finally
        {
            WeightRegistry.Reset();
        }
    }
}
