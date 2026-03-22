using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Diffusion.NoisePredictors;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Benchmarks the DiffusionResBlock forward pass at various channel counts.
/// Measures: GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 + skip
/// </summary>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[RankColumn]
public class DiffusionResBlockBenchmarks
{
    // Small block (early UNet level)
    private DiffusionResBlock<double> _block64 = null!;
    private Tensor<double> _input64 = null!;

    // Medium block (mid level)
    private DiffusionResBlock<double> _block256 = null!;
    private Tensor<double> _input256 = null!;

    // Large block (bottleneck — SD15 deepest level)
    private DiffusionResBlock<double> _block1280 = null!;
    private Tensor<double> _input1280 = null!;

    // Channel-changing block (with skip conv)
    private DiffusionResBlock<double> _block64to128 = null!;
    private Tensor<double> _input64for128 = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _block64 = new DiffusionResBlock<double>(64, 64, 32);
        _input64 = CreateRandom(1, 64, 32, 32, rng);

        _block256 = new DiffusionResBlock<double>(256, 256, 16);
        _input256 = CreateRandom(1, 256, 16, 16, rng);

        _block1280 = new DiffusionResBlock<double>(1280, 1280, 8);
        _input1280 = CreateRandom(1, 1280, 8, 8, rng);

        _block64to128 = new DiffusionResBlock<double>(64, 128, 32);
        _input64for128 = CreateRandom(1, 64, 32, 32, rng);
    }

    [Benchmark(Description = "ResBlock 64ch 32x32 (same channels)")]
    public Tensor<double> ResBlock_64ch_32x32()
        => _block64.Forward(_input64);

    [Benchmark(Description = "ResBlock 256ch 16x16 (same channels)")]
    public Tensor<double> ResBlock_256ch_16x16()
        => _block256.Forward(_input256);

    [Benchmark(Description = "ResBlock 1280ch 8x8 (SD15 bottleneck)")]
    public Tensor<double> ResBlock_1280ch_8x8()
        => _block1280.Forward(_input1280);

    [Benchmark(Description = "ResBlock 64->128ch 32x32 (with skip conv)")]
    public Tensor<double> ResBlock_64to128_32x32()
        => _block64to128.Forward(_input64for128);

    private static Tensor<double> CreateRandom(int b, int c, int h, int w, Random rng)
    {
        var tensor = new Tensor<double>([b, c, h, w]);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() - 0.5;
        return tensor;
    }
}
