#if NET10_0_OR_GREATER
using System;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.Diffusion;

/// <summary>
/// VAE encode + decode round-trip benchmark for #1272 acceptance criterion #2.
/// Measures the steady-state replay cost of <see cref="StandardVAE{T}"/> with
/// the compile-cache wrapper added in #1272 W1, on the canonical SD-VAE
/// 512×512 RGB → 64×64×4 latent shape.
/// </summary>
/// <remarks>
/// <para>
/// First call traces and compiles the encoder backbone + decoder body; every
/// subsequent call replays the cached plan. The benchmark's <c>WarmupCount=2</c>
/// amortises the trace, so the measured iterations report replay cost only.
/// </para>
/// <para>
/// PyTorch head-to-head: a TorchSharp-side comparison module is a
/// follow-up commit on this branch. Hand-porting the full SD-VAE topology
/// (4 down/up ResNet stages + mid attention + group-norm + SiLU) to
/// TorchSharp primitives takes ~300 lines and needs API-stable Conv2d /
/// GroupNorm parameter names — a benchmark commit on its own. For now the
/// AiDotNet absolute timing is reported; the ratio computation is wired
/// up in a sibling benchmark file once the TorchSharp module lands.
/// </para>
/// </remarks>
[Config(typeof(VAEEncodeDecodeBenchmarkConfig))]
public class VAEEncodeDecodeBenchmark : IDisposable
{
    private const int InputSpatialSize = 512;
    private const int LatentChannels = 4;
    private const int BaseChannels = 128;

    private StandardVAE<float>? _aidotnetVae;
    private Tensor<float>? _aidotnetInput;

    [GlobalSetup]
    public void Setup()
    {
        _aidotnetVae = new StandardVAE<float>(
            inputChannels: 3,
            latentChannels: LatentChannels,
            baseChannels: BaseChannels);

        _aidotnetInput = new Tensor<float>(new[] { 1, 3, InputSpatialSize, InputSpatialSize });
        var rng = new Random(42);
        for (int i = 0; i < _aidotnetInput.Length; i++)
            _aidotnetInput[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }

    [GlobalCleanup]
    public void Cleanup() => Dispose();

    public void Dispose()
    {
        _aidotnetVae = null;
        _aidotnetInput = null;
        GC.SuppressFinalize(this);
    }

    [Benchmark(Description = "AiDotNet StandardVAE encode + decode (compile-cache wrapped, #1272 W1)")]
    public Tensor<float> AidotnetEncodeDecode()
    {
        var (mean, _) = _aidotnetVae!.EncodeWithDistribution(_aidotnetInput!);
        return _aidotnetVae.Decode(mean);
    }
}

/// <summary>
/// Short-job config: 2 warmup iterations to amortise compile-cache trace,
/// 5 measurement iterations. <see cref="MemoryDiagnoser.Default"/> tracks
/// allocations — a key correctness signal for the compile cache, since a
/// successful replay should allocate orders of magnitude less than the
/// initial trace.
/// </summary>
public sealed class VAEEncodeDecodeBenchmarkConfig : ManualConfig
{
    public VAEEncodeDecodeBenchmarkConfig()
    {
        AddJob(Job.ShortRun
            .WithWarmupCount(2)
            .WithIterationCount(5)
            .WithLaunchCount(1));
        AddDiagnoser(MemoryDiagnoser.Default);
    }
}
#endif
