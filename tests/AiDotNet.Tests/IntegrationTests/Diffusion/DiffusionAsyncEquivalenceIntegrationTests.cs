using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Numerical-equivalence tests for the new async <see cref="DiffusionModelBase{T}.GenerateAsync"/>
/// surface added in #1273. The async path runs the same denoising loop as
/// <see cref="DiffusionModelBase{T}.Generate"/> but routes per-step noise
/// prediction through <c>PredictNoiseAsync</c> (the compile-host's
/// <c>ExecuteAsync</c> path); given the same seed and shape, both surfaces
/// must produce bit-equivalent output.
/// </summary>
/// <remarks>
/// Acceptance criterion for #1273 Workstream A: <em>Numerical equivalence
/// test passes within <c>1e-4</c> relative tolerance.</em> A trained DDPM
/// predicts noise that's bounded — but for these tests the underlying
/// <see cref="DDPMModel{T}"/> uses its placeholder zero-prediction and
/// the scheduler's deterministic Step under <c>NumOps.Zero</c> eta, so
/// both paths take the same numerical trajectory. We assert exact equality
/// (no floating-point drift) because the only difference between paths is
/// the awaitable wrapper around the same engine ops.
/// </remarks>
public class DiffusionAsyncEquivalenceIntegrationTests
{
    [Fact]
    public async Task GenerateAsync_MatchesGenerate_OnSameSeedAndShape()
    {
        // Same seed → same initial noise; same scheduler → same per-step
        // alpha/beta values; same shape → same per-step plan key.
        const int seed = 42;
        const int steps = 10;
        var shape = new[] { 1, 4, 8, 8 };

        var sync = new DDPMModel<float>(seed);
        var async_ = new DDPMModel<float>(seed);

        var syncOutput = sync.Generate(shape, numInferenceSteps: steps, seed: seed);
        var asyncOutput = await async_.GenerateAsync(shape, numInferenceSteps: steps, seed: seed)
            .ConfigureAwait(false);

        Assert.Equal(syncOutput.Shape, asyncOutput.Shape);
        Assert.Equal(syncOutput.Length, asyncOutput.Length);

        for (int i = 0; i < syncOutput.Length; i++)
        {
            // Bit-equivalent: same op sequence, same numeric inputs.
            Assert.Equal(syncOutput[i], asyncOutput[i]);
        }
    }

    [Fact]
    public async Task GenerateAsync_BehavesIdenticallyAcrossMultipleAwaits()
    {
        // Replay determinism — calling GenerateAsync twice with the same
        // seed must produce the same output. Catches state bleed between
        // calls (compile-cache contamination, scheduler-step mutation
        // leaking into the next generation).
        const int seed = 7;
        var shape = new[] { 1, 4, 4, 4 };
        var model = new DDPMModel<float>(seed);

        var first = await model.GenerateAsync(shape, numInferenceSteps: 5, seed: seed);
        var second = await model.GenerateAsync(shape, numInferenceSteps: 5, seed: seed);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i]);
    }

    [Fact]
    public async Task GenerateAsync_IsCancellable()
    {
        // Cancellation observed at the per-step boundary in
        // GenerateAsyncCore — should propagate as OperationCanceledException
        // rather than completing the full denoising loop.
        var model = new DDPMModel<float>(seed: 1);
        var shape = new[] { 1, 4, 16, 16 };
        using var cts = new System.Threading.CancellationTokenSource();
        cts.Cancel(); // Pre-cancelled.

        await Assert.ThrowsAnyAsync<System.OperationCanceledException>(async () =>
        {
            await model.GenerateAsync(shape, numInferenceSteps: 50, seed: 1, cts.Token)
                .ConfigureAwait(false);
        });
    }
}
