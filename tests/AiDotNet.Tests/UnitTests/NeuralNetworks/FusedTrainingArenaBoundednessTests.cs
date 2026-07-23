// Regression test for the REAL #1624 OOM. The DEFAULT training route for a
// float model with a plain Adam/SGD optimizer is the fused-optimizer path
// (NeuralNetworkBase.TryTrainWithFusedOptimizer) — it returns before the eager
// tape's per-step arena and, before the fix, opened NO arena of its own. So when
// a caller wraps the whole training loop in ONE outer TensorArena and never
// Reset()s it between steps (the shared ModelFamilyTests base does exactly this,
// and so does any user pooling buffers across steps), every step's forward/
// backward activations (~one step's working set) accumulated in the outer arena's
// ring -> linear growth -> OOM over a long run.
//
// The fix opens a per-step arena inside the fused path, gated on
// CompiledTapeTrainingStep.GetFusedStepCount() > 0 so the CONFIGURING step (which
// creates the persistent compiled plan holding Adam's m/v moments) runs arena-free
// and those long-lived buffers are never recycled. Every subsequent step's pure
// transients are scoped to a recycling per-step arena.
//
// Contract asserted here: under the un-Reset outer arena the live heap PLATEAUS
// after warmup instead of climbing with the step count. Pre-fix this grew ~290 MB
// per step (measured on SimCSE); post-fix it is flat from ~step 10 on.

using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class FusedTrainingArenaBoundednessTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = (float)(((i * 7 + seed * 13) % 17) - 8) * 0.1f;
        return new Tensor<float>(data, shape);
    }

    [Fact]
    public void FusedTraining_UnderUnResetOuterArena_HeapPlateausInsteadOfClimbing()
    {
        const int dim = 256;
        // Sized so ONE step's activation working set is large enough that a per-step
        // leak of the pre-fix kind is unmistakably above GC/measurement noise, while
        // still training in a couple of seconds for CI. (A tiny model leaks only a
        // few MB/step, which the slack below would swallow.)
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: dim,
            outputSize: dim);
        var model = new SimCSE<float>(
            arch,
            embeddingDimension: dim,
            maxSequenceLength: 256,
            numLayers: 8,
            numHeads: 8,
            feedForwardDim: 1024);

        // Default route: let the autotuner pick the eager/fused path (NOT streaming).
        // ForceOff guarantees we exercise the fused-optimizer path the fix touches.
        model.StreamingTraining = StreamingTrainingMode.ForceOff;

        var input = Rand(new[] { 1, dim }, seed: 1);
        var target = Rand(new[] { 1, dim }, seed: 2);

        // The exact leak shape: ONE outer arena around the whole loop, never Reset().
        using var outerArena = TensorArena.Create();

        // Warm up (configure the compiled plan / fill the steady-state pools), then
        // sample the live heap at increasing step counts.
        model.Train(input, target);
        long heapWarm = FullGcLiveBytes();
        float lossAfterWarmup = System.Convert.ToSingle(model.GetLastLoss());

        for (int s = 0; s < 10; s++) model.Train(input, target);
        long heapAt11 = FullGcLiveBytes();

        for (int s = 0; s < 15; s++) model.Train(input, target);
        long heapAt26 = FullGcLiveBytes();
        float lossFinal = System.Convert.ToSingle(model.GetLastLoss());

        // CONVERGENCE contract: the per-step arena must reclaim only TRANSIENTS — the
        // Adam m/v moments live on the GC heap (held by the compiled plan) and must
        // survive every step's arena scope. If the scope wrongly recycled them, Adam
        // would corrupt and the memorization loss would stop falling. So a memory fix
        // that bounded the heap by silently breaking training fails HERE, not just the
        // plateau check below. (Mirrors PyTorch: optimizer.state persists across the
        // per-iteration activation churn.)
        Assert.True(
            !float.IsNaN(lossFinal) && !float.IsInfinity(lossFinal) && lossFinal < lossAfterWarmup,
            $"Fused training did not converge under the per-step arena: loss {lossAfterWarmup:F6} " +
            $"(after warmup) -> {lossFinal:F6} (after 26 steps). The arena scope must recycle only " +
            $"transient activations and leave the persistent Adam moments intact.");

        // PLATEAU contract: once the pools are warm, additional steps must not grow
        // the live heap. The window from step 11 -> 26 (15 more steps) is the
        // sensitive one: a per-step leak of the pre-fix magnitude (~290 MB/step on
        // full SimCSE; proportionally smaller here) would add hundreds of MB across
        // it. We allow a small slack for measurement noise / GC settling but fail
        // hard on any sustained climb.
        double growthPerStep = (heapAt26 - heapAt11) / 15.0;
        const double maxSlackBytesPerStep = 4 * 1024 * 1024; // 4 MB/step ceiling (noise only)

        Assert.True(
            growthPerStep < maxSlackBytesPerStep,
            $"Fused training accumulated in the un-Reset outer arena: live heap " +
            $"{heapWarm / (1024.0 * 1024.0):F1} MB (warm) -> {heapAt11 / (1024.0 * 1024.0):F1} MB (11 steps) -> " +
            $"{heapAt26 / (1024.0 * 1024.0):F1} MB (26 steps) = {growthPerStep / (1024.0 * 1024.0):F2} MB/step. " +
            $"The per-step arena in TryTrainWithFusedOptimizer must bound this to a post-warmup plateau.");
    }

    private static long FullGcLiveBytes()
    {
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect();
        return System.GC.GetTotalMemory(forceFullCollection: true);
    }
}
