using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// #1643: the Neural Turing Machine and Differentiable Neural Computer have a <b>dynamic,
/// stateful</b> forward — an external memory matrix mutated across an internal recurrence with
/// content/location-based addressing. That is not a static op graph, so they must train on the
/// eager autograd tape, never on the compile-once/replay-many fused compiled training path.
/// </summary>
/// <remarks>
/// The fused path caches a compiled plan per thread in a <c>[ThreadStatic]</c> field keyed by
/// tensor shape. The ModelFamily test base awaits <c>Task.Yield()</c>, so each model's
/// <c>Train</c> resumes on a pooled worker thread; on the CI runners' small thread pool a sibling
/// model's same-shape plan lingered on the thread NTM landed on and replayed against the wrong
/// tensors, freezing NTM's loss with exactly-zero gradients. That surfaced as the intermittent
/// M-N CI-shard failures (GradientFlow_ShouldBeNonZeroAndFinite, Training_ShouldChangeParameters,
/// LossStrictlyDecreasesOnMemorizationTask, TrainingError_ShouldNotExceedTestError). The fix makes
/// these models opt out of the fused path (<c>SupportsFusedCompiledTraining =&gt; false</c>); the
/// eager tape re-runs the true dynamic forward every step, so it is both correct and immune to the
/// cross-test plan-cache hazard. These tests run synchronously (no <c>Task.Yield</c>) so the
/// thread-static fused-step counter reflects exactly the training under test.
/// </remarks>
public class StatefulMemoryModelEagerTrainingTests
{
    private static Tensor<float> Rand(int[] shape, int seed, float scale = 0.5f)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static bool AnyChanged(float[] before, Vector<float> after)
    {
        int n = System.Math.Min(before.Length, after.Length);
        for (int i = 0; i < n; i++)
            if (System.Math.Abs(after[i] - before[i]) > 1e-9f) return true;
        return false;
    }

    [Fact]
    public void Ntm_TrainsViaEagerTape_NeverFusedCompiledPath_AndUpdatesParameters()
    {
        var network = new NeuralTuringMachine<float>();          // [128] -> [1] regression
        var input = Rand(new[] { 128 }, 7);
        var target = Rand(new[] { 1 }, 8);

        var before = network.GetParameters().ToArray();

        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        for (int step = 0; step < 20; step++)
            network.Train(input, target);

        // The dynamic-forward model must never engage the fused compiled path — a single fused
        // step would mean the [ThreadStatic] plan-cache hazard is back in play for NTM.
        Assert.Equal(0L, AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount());

        // ...and the eager tape must still move parameters (gradients flow every step).
        Assert.True(AnyChanged(before, network.GetParameters()),
            "NTM parameters did not change after 20 eager training steps — gradients are zero.");
    }

    [Fact]
    public void Dnc_TrainsViaEagerTape_NeverFusedCompiledPath_AndUpdatesParameters()
    {
        var network = new DifferentiableNeuralComputer<float>();   // [128] -> [1] regression
        var input = Rand(new[] { 128 }, 21);
        var target = Rand(new[] { 1 }, 22);
        var before = network.GetParameters().ToArray();

        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        for (int step = 0; step < 20; step++)
            network.Train(input, target);

        Assert.Equal(0L, AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount());
        Assert.True(AnyChanged(before, network.GetParameters()),
            "DNC parameters did not change after 20 eager training steps — gradients are zero.");
    }
}
