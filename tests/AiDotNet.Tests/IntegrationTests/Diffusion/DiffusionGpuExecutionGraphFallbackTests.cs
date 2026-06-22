using System.Threading.Tasks;
using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Fallback-contract tests for the opt-in GPU deferred-execution-graph denoising step
/// (<see cref="DiffusionModelOptions{T}.UseGpuExecutionGraph"/>, #642).
/// </summary>
/// <remarks>
/// <para>
/// <see cref="DiffusionModelBase{T}.PredictNoiseStep"/> only routes through the deferred GPU
/// execution graph when the active engine is a CUDA <c>DirectGpuTensorEngine</c>; on any other
/// engine (CPU, or no GPU) it must fall back to the eager <see cref="DiffusionModelBase{T}.PredictNoise"/>
/// with NO change in output. These tests run on the CPU engine, so they validate exactly that
/// transparent-fallback safety contract: flipping the flag on must not perturb generation.
/// </para>
/// <para>
/// They deliberately do NOT validate the GPU graph path itself — that produces output only on
/// CUDA hardware and must be verified on a CUDA box before the option is enabled for any model
/// (the per-model op-coverage audit). What is asserted here is the invariant that matters for
/// every non-CUDA consumer: the option is inert unless a CUDA engine is active.
/// </para>
/// </remarks>
public class DiffusionGpuExecutionGraphFallbackTests
{
    private static DiffusionModelOptions<float> Options(bool useGpuExecutionGraph) => new()
    {
        TrainTimesteps = 1000,
        BetaStart = 0.0001,
        BetaEnd = 0.02,
        BetaSchedule = BetaSchedule.Linear,
        DefaultInferenceSteps = 50,
        UseGpuExecutionGraph = useGpuExecutionGraph,
    };

    [Fact(Timeout = 120000)]
    public async Task UseGpuExecutionGraph_OnCpuEngine_FallsBackToEager_BitEquivalentOutput()
    {
        await Task.Yield(); // keep the body genuinely async so [Fact(Timeout)] is enforced (xUnit v2)

        const int seed = 42;
        const int steps = 10;
        var shape = new[] { 1, 4, 8, 8 };

        // Two models identical except for the flag; same seed -> same UNet weights + same
        // initial noise + same scheduler trajectory. On the CPU engine the flag is inert
        // (PredictNoiseStep sees no CUDA engine and calls PredictNoise directly), so the
        // outputs must be element-wise identical — proving the fallback is transparent.
        var eager = new DDPMModel<float>(architecture: null, options: Options(useGpuExecutionGraph: false), seed: seed);
        var withFlag = new DDPMModel<float>(architecture: null, options: Options(useGpuExecutionGraph: true), seed: seed);

        var eagerOutput = eager.Generate(shape, numInferenceSteps: steps, seed: seed);
        var flagOutput = withFlag.Generate(shape, numInferenceSteps: steps, seed: seed);

        Assert.Equal(eagerOutput.Shape, flagOutput.Shape);
        Assert.Equal(eagerOutput.Length, flagOutput.Length);
        for (int i = 0; i < eagerOutput.Length; i++)
        {
            // Bit-equivalent: the only difference is a flag that is a no-op without a CUDA engine.
            Assert.Equal(eagerOutput[i], flagOutput[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task UseGpuExecutionGraph_DefaultsToOff()
    {
        await Task.Yield();
        // Opt-in: the option must be OFF by default so no model silently engages the
        // not-yet-audited GPU graph path on a CUDA box.
        Assert.False(new DiffusionModelOptions<float>().UseGpuExecutionGraph);
    }
}
