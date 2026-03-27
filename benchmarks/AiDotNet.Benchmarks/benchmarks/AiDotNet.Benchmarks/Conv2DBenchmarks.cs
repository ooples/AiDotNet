using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Benchmarks Conv2D at various channel counts and spatial sizes.
/// Compares allocating Conv2D vs zero-allocation Conv2DInto.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
[RankColumn]
public class Conv2DBenchmarks
{
    private readonly IEngine _engine = AiDotNetEngine.GetEngine();

    // Small (early UNet levels)
    private Tensor<double> _input64x32 = null!;
    private Tensor<double> _kernel64 = null!;
    private Tensor<double> _output64x32 = null!;

    // Medium (mid UNet levels)
    private Tensor<double> _input256x16 = null!;
    private Tensor<double> _kernel256 = null!;
    private Tensor<double> _output256x16 = null!;

    // Large (deep UNet levels — SD15 bottleneck)
    private Tensor<double> _input1280x8 = null!;
    private Tensor<double> _kernel1280 = null!;
    private Tensor<double> _output1280x8 = null!;

    // Float versions for comparison
    private Tensor<float> _inputF256x16 = null!;
    private Tensor<float> _kernelF256 = null!;
    private Tensor<float> _outputF256x16 = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        // Double precision
        _input64x32 = CreateRandom<double>(1, 64, 32, 32, rng);
        _kernel64 = CreateRandom<double>(64, 64, 3, 3, rng);
        _output64x32 = new Tensor<double>([1, 64, 32, 32]);

        _input256x16 = CreateRandom<double>(1, 256, 16, 16, rng);
        _kernel256 = CreateRandom<double>(256, 256, 3, 3, rng);
        _output256x16 = new Tensor<double>([1, 256, 16, 16]);

        _input1280x8 = CreateRandom<double>(1, 1280, 8, 8, rng);
        _kernel1280 = CreateRandom<double>(1280, 1280, 3, 3, rng);
        _output1280x8 = new Tensor<double>([1, 1280, 8, 8]);

        // Float precision
        _inputF256x16 = CreateRandom<float>(1, 256, 16, 16, rng);
        _kernelF256 = CreateRandom<float>(256, 256, 3, 3, rng);
        _outputF256x16 = new Tensor<float>([1, 256, 16, 16]);
    }

    [Benchmark(Description = "Conv2D 64ch 32x32 (double, allocating)")]
    public Tensor<double> Conv2D_64ch_32x32_Double()
        => _engine.Conv2D(_input64x32, _kernel64, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 64ch 32x32 (double, zero-alloc)")]
    public void Conv2DInto_64ch_32x32_Double()
        => _engine.Conv2DInto(_output64x32, _input64x32, _kernel64, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 256ch 16x16 (double, allocating)")]
    public Tensor<double> Conv2D_256ch_16x16_Double()
        => _engine.Conv2D(_input256x16, _kernel256, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 256ch 16x16 (double, zero-alloc)")]
    public void Conv2DInto_256ch_16x16_Double()
        => _engine.Conv2DInto(_output256x16, _input256x16, _kernel256, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 1280ch 8x8 (double, allocating)")]
    public Tensor<double> Conv2D_1280ch_8x8_Double()
        => _engine.Conv2D(_input1280x8, _kernel1280, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 1280ch 8x8 (double, zero-alloc)")]
    public void Conv2DInto_1280ch_8x8_Double()
        => _engine.Conv2DInto(_output1280x8, _input1280x8, _kernel1280, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 256ch 16x16 (float, allocating)")]
    public Tensor<float> Conv2D_256ch_16x16_Float()
        => _engine.Conv2D(_inputF256x16, _kernelF256, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 256ch 16x16 (float, zero-alloc)")]
    public void Conv2DInto_256ch_16x16_Float()
        => _engine.Conv2DInto(_outputF256x16, _inputF256x16, _kernelF256, stride: 1, padding: 1);

    private static Tensor<TNum> CreateRandom<TNum>(int b, int c, int h, int w, Random rng)
    {
        var tensor = new Tensor<TNum>([b, c, h, w]);
        var span = tensor.AsWritableSpan();
        if (typeof(TNum) == typeof(double))
        {
            var dSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<TNum, double>(span);
            for (int i = 0; i < dSpan.Length; i++)
                dSpan[i] = rng.NextDouble() - 0.5;
        }
        else if (typeof(TNum) == typeof(float))
        {
            var fSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<TNum, float>(span);
            for (int i = 0; i < fSpan.Length; i++)
                fSpan[i] = (float)(rng.NextDouble() - 0.5);
        }
        return tensor;
    }
}
