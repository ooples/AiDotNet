#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using TorchSharp;

namespace AiDotNetBenchmarkTests.Diffusion;

/// <summary>
/// Head-to-head VAE encode + decode benchmark for #1272 acceptance criterion #2.
/// AiDotNet's <see cref="StandardVAE{T}"/> with the compile-cache wrapper from
/// #1272 W1 vs an equivalent TorchSharp implementation of the same SD-style
/// VAE topology, on identical fp32 input shapes.
/// </summary>
/// <remarks>
/// <para>
/// Acceptance criterion #2: AiDotNet's VAE encode + decode round-trip must
/// measure ≤ 1.05× PyTorch (TorchSharp) on a 512×512 RGB input at fp32, CPU.
/// </para>
/// <para>
/// Methodology:
/// </para>
/// <list type="bullet">
///   <item>Both implementations use the same channel-multiplier topology
///         <c>[1, 2, 4, 4]</c>, same baseChannels=128, same latentChannels=4,
///         so per-op cost differences come from engine implementations not
///         architectural choices.</item>
///   <item>Each iteration runs <c>Encode(image) → Decode(latent)</c>.</item>
///   <item><c>WarmupCount=2</c> amortises the first-call compile (AiDotNet)
///         and JIT/runtime warmup (TorchSharp) before measurement.</item>
///   <item>Measured iterations report steady-state replay cost.</item>
/// </list>
/// </remarks>
[Config(typeof(VAEEncodeDecodeBenchmarkConfig))]
public class VAEEncodeDecodeBenchmark : IDisposable
{
    // Canonical SD-VAE: 512×512 RGB → 64×64×4 latent.
    private const int InputSpatialSize = 512;
    private const int LatentChannels = 4;
    private const int BaseChannels = 128;

    private StandardVAE<float>? _aidotnetVae;
    private Tensor<float>? _aidotnetInput;

    private torch.nn.Module<torch.Tensor, torch.Tensor>? _torchsharpEncoder;
    private torch.nn.Module<torch.Tensor, torch.Tensor>? _torchsharpDecoder;
    private torch.Tensor? _torchsharpInput;

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

        _torchsharpEncoder = BuildTorchsharpVaeEncoder();
        _torchsharpDecoder = BuildTorchsharpVaeDecoder();
        _torchsharpInput = torch.randn(new long[] { 1, 3, InputSpatialSize, InputSpatialSize });
    }

    [GlobalCleanup]
    public void Cleanup() => Dispose();

    public void Dispose()
    {
        _torchsharpEncoder?.Dispose();
        _torchsharpDecoder?.Dispose();
        _torchsharpInput?.Dispose();
        _aidotnetVae = null;
        _aidotnetInput = null;
        GC.SuppressFinalize(this);
    }

    [Benchmark(Baseline = true, Description = "TorchSharp SD-VAE encode + decode (PyTorch-on-.NET reference)")]
    public torch.Tensor TorchsharpEncodeDecode()
    {
        // Encoder emits [1, 2*latentChannels, H/8, W/8] (mean + logvar
        // concatenated along channel dim). Take the mean half — same as
        // diffusers' AutoencoderKL.encode(...).latent_dist.mean.
        using var encoded = _torchsharpEncoder!.forward(_torchsharpInput!);
        using var meanHalf = encoded.split(LatentChannels, dim: 1)[0];
        return _torchsharpDecoder!.forward(meanHalf);
    }

    [Benchmark(Description = "AiDotNet StandardVAE encode + decode (compile-cache wrapped, #1272 W1)")]
    public Tensor<float> AidotnetEncodeDecode()
    {
        var (mean, _) = _aidotnetVae!.EncodeWithDistribution(_aidotnetInput!);
        return _aidotnetVae.Decode(mean);
    }

    /// <summary>
    /// Builds an SD-VAE encoder using TorchSharp primitives. Same topology
    /// AiDotNet's <see cref="StandardVAE{T}"/> uses: input conv → 4 down
    /// stages (each: GroupNorm + SiLU + Conv ×2; stride-2 conv between
    /// stages on all but the last) → 2 mid blocks → norm + SiLU → output
    /// projection emitting <c>[N, 2*latentChannels, H/8, W/8]</c>.
    /// </summary>
    /// <remarks>
    /// ResNet blocks abbreviated to a single Conv-Norm-SiLU-Conv stack (no
    /// skip connection) — the benchmark cares about FLOP-equivalence on the
    /// hot path, not exact weight loading from a pretrained checkpoint.
    /// </remarks>
    private static torch.nn.Module<torch.Tensor, torch.Tensor> BuildTorchsharpVaeEncoder()
    {
        var ch = new[] { BaseChannels, BaseChannels * 2, BaseChannels * 4, BaseChannels * 4 };
        var modules = new List<torch.nn.Module<torch.Tensor, torch.Tensor>>();

        // Input convolution: 3 → baseChannels at full resolution.
        modules.Add(torch.nn.Conv2d(3, ch[0], kernel_size: 3, padding: 1));

        for (int i = 0; i < ch.Length; i++)
        {
            long inCh = i == 0 ? ch[0] : ch[i - 1];
            long outCh = ch[i];

            // ResNet block (abbreviated): GroupNorm + SiLU + Conv 3x3 ×2.
            modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: inCh));
            modules.Add(torch.nn.SiLU());
            modules.Add(torch.nn.Conv2d(inCh, outCh, kernel_size: 3, padding: 1));
            modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: outCh));
            modules.Add(torch.nn.SiLU());
            modules.Add(torch.nn.Conv2d(outCh, outCh, kernel_size: 3, padding: 1));

            // Stride-2 downsample on all but last stage.
            if (i < ch.Length - 1)
                modules.Add(torch.nn.Conv2d(outCh, outCh, kernel_size: 3, stride: 2, padding: 1));
        }

        // Mid blocks at the bottleneck.
        modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: ch[^1]));
        modules.Add(torch.nn.SiLU());
        modules.Add(torch.nn.Conv2d(ch[^1], ch[^1], kernel_size: 3, padding: 1));

        // Output projection to 2*latentChannels (mean + logvar concatenated).
        modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: ch[^1]));
        modules.Add(torch.nn.SiLU());
        modules.Add(torch.nn.Conv2d(ch[^1], LatentChannels * 2, kernel_size: 3, padding: 1));

        return torch.nn.Sequential(modules.ToArray());
    }

    /// <summary>
    /// Builds an SD-VAE decoder — inverse of the encoder, with nearest-neighbor
    /// upsampling + 3×3 conv (matches diffusers' Upsample2D).
    /// </summary>
    private static torch.nn.Module<torch.Tensor, torch.Tensor> BuildTorchsharpVaeDecoder()
    {
        // Reverse channel progression for decoder.
        var ch = new[] { BaseChannels * 4, BaseChannels * 4, BaseChannels * 2, BaseChannels };
        var modules = new List<torch.nn.Module<torch.Tensor, torch.Tensor>>();

        // Post-quant conv: latentChannels → first stage channels.
        modules.Add(torch.nn.Conv2d(LatentChannels, ch[0], kernel_size: 3, padding: 1));

        for (int i = 0; i < ch.Length; i++)
        {
            long inCh = i == 0 ? ch[0] : ch[i - 1];
            long outCh = ch[i];

            modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: inCh));
            modules.Add(torch.nn.SiLU());
            modules.Add(torch.nn.Conv2d(inCh, outCh, kernel_size: 3, padding: 1));
            modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: outCh));
            modules.Add(torch.nn.SiLU());
            modules.Add(torch.nn.Conv2d(outCh, outCh, kernel_size: 3, padding: 1));

            // Nearest-neighbor 2× upsample + 3×3 conv on all but last stage.
            if (i < ch.Length - 1)
            {
                modules.Add(torch.nn.Upsample(scale_factor: new double[] { 2.0, 2.0 }, mode: torch.UpsampleMode.Nearest));
                modules.Add(torch.nn.Conv2d(outCh, outCh, kernel_size: 3, padding: 1));
            }
        }

        // Output convolution: project back to 3 RGB channels.
        modules.Add(torch.nn.GroupNorm(num_groups: 32, num_channels: ch[^1]));
        modules.Add(torch.nn.SiLU());
        modules.Add(torch.nn.Conv2d(ch[^1], 3, kernel_size: 3, padding: 1));

        return torch.nn.Sequential(modules.ToArray());
    }
}

/// <summary>
/// Short-job config: 2 warmup iterations to amortise compile-cache trace,
/// 5 measurement iterations. <see cref="MemoryDiagnoser.Default"/> tracks
/// allocations — a successful replay should allocate orders of magnitude
/// less than the initial trace.
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
