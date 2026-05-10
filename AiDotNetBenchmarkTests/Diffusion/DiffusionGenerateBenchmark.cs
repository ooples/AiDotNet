#if NET10_0_OR_GREATER
using System;
using System.Threading.Tasks;
using AiDotNet.Diffusion;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.Diffusion;

/// <summary>
/// Measures <see cref="DiffusionModelBase{T}.Generate"/> (sync) vs
/// <see cref="DiffusionModelBase{T}.GenerateAsync"/> (async, compile-host
/// per step) for #1273 Workstream A. Steady-state replay cost is reported
/// (WarmupCount=2 amortises the first-call trace + JIT warmup).
/// </summary>
/// <remarks>
/// <para>
/// Acceptance criterion for #1273 W-A: <em>≥ 15% improvement on 50-step DDIM
/// at SDXL-class shapes ([1, 4, 128, 128] latent), ≥ 40% on 4-step distilled.</em>
/// On a CPU engine the per-step ExecuteAsync completes inline, so the async
/// column's per-step host-side cost is identical to sync — the win shows up
/// when the engine is GPU and the CUDA stream's tail kernels overlap with
/// host-side scheduler.Step / RNG advance work between awaits.
/// </para>
/// <para>
/// The benchmark uses <see cref="DDPMModel{T}"/>'s placeholder zero-prediction
/// path so we measure the framework's per-step plumbing cost in isolation,
/// not the noise predictor's forward — the noise-predictor cost is
/// benchmarked separately under each predictor's own benchmark file. With
/// the placeholder predictor, the only difference between sync and async
/// paths is the await-vs-block plumbing cost, which is the signal we want
/// to measure.
/// </para>
/// </remarks>
[Config(typeof(DiffusionGenerateBenchmarkConfig))]
public class DiffusionGenerateBenchmark
{
    [Params(10, 50)]
    public int NumInferenceSteps { get; set; }

    private DDPMModel<float>? _model;
    private int[]? _shape;

    [GlobalSetup]
    public void Setup()
    {
        _model = new DDPMModel<float>(seed: 42);
        // SDXL-class latent shape per the issue's acceptance criterion.
        _shape = new[] { 1, 4, 128, 128 };
    }

    [Benchmark(Baseline = true, Description = "DiffusionModelBase.Generate (sync, eager)")]
    public Tensor<float> GenerateSync()
        => _model!.Generate(_shape!, numInferenceSteps: NumInferenceSteps, seed: 42);

    [Benchmark(Description = "DiffusionModelBase.GenerateAsync (#1273 W-A, true-async PredictNoiseAsync per step)")]
    public Tensor<float> GenerateAsync()
        => _model!.GenerateAsync(_shape!, numInferenceSteps: NumInferenceSteps, seed: 42)
            .GetAwaiter().GetResult();
}

/// <summary>
/// Short-job config: 2 warmup iterations to amortise the per-step
/// compile-cache trace (first call traces the noise-predictor plan;
/// subsequent calls replay), 5 measurement iterations.
/// <see cref="MemoryDiagnoser.Default"/> tracks alloc count for compile-
/// cache correctness verification — a successful replay should allocate
/// orders of magnitude less than the initial trace.
/// </summary>
public sealed class DiffusionGenerateBenchmarkConfig : ManualConfig
{
    public DiffusionGenerateBenchmarkConfig()
    {
        AddJob(Job.ShortRun
            .WithWarmupCount(2)
            .WithIterationCount(5)
            .WithLaunchCount(1));
        AddDiagnoser(MemoryDiagnoser.Default);
    }
}
#endif
