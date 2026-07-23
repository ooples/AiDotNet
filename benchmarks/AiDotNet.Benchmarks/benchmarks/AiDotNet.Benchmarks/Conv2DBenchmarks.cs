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
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // Shallow input 64x64 (SD15 VAE input / UNet first-stage)
    private Tensor<double> _input64ch64x64 = null!;
    private Tensor<double> _kernel64 = null!;
    private Tensor<double> _output64ch64x64 = null!;

    // Small (early UNet levels) 64 channels @ 32x32
    private Tensor<double> _input64x32 = null!;
    private Tensor<double> _output64x32 = null!;

    // SD15 level-1 128ch @ 32x32
    private Tensor<double> _input128x32 = null!;
    private Tensor<double> _kernel128 = null!;
    private Tensor<double> _output128x32 = null!;

    // Medium (mid UNet levels) 256ch @ 16x16
    private Tensor<double> _input256x16 = null!;
    private Tensor<double> _kernel256 = null!;
    private Tensor<double> _output256x16 = null!;

    // SD15 level-3 512ch @ 16x16
    private Tensor<double> _input512x16 = null!;
    private Tensor<double> _kernel512 = null!;
    private Tensor<double> _output512x16 = null!;

    // Large (deep UNet levels — SD15 bottleneck) 1280ch @ 8x8
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

        // Double precision — spatial sizes 64x64, 32x32, 16x16, 8x8;
        // channels 64, 128, 256, 512, 1280 (the full SD15 channel ladder).
        _input64ch64x64 = CreateRandomDouble(1, 64, 64, 64, rng);
        _kernel64 = CreateRandomDouble(64, 64, 3, 3, rng);
        _output64ch64x64 = new Tensor<double>([1, 64, 64, 64]);

        _input64x32 = CreateRandomDouble(1, 64, 32, 32, rng);
        _output64x32 = new Tensor<double>([1, 64, 32, 32]);

        _input128x32 = CreateRandomDouble(1, 128, 32, 32, rng);
        _kernel128 = CreateRandomDouble(128, 128, 3, 3, rng);
        _output128x32 = new Tensor<double>([1, 128, 32, 32]);

        _input256x16 = CreateRandomDouble(1, 256, 16, 16, rng);
        _kernel256 = CreateRandomDouble(256, 256, 3, 3, rng);
        _output256x16 = new Tensor<double>([1, 256, 16, 16]);

        _input512x16 = CreateRandomDouble(1, 512, 16, 16, rng);
        _kernel512 = CreateRandomDouble(512, 512, 3, 3, rng);
        _output512x16 = new Tensor<double>([1, 512, 16, 16]);

        _input1280x8 = CreateRandomDouble(1, 1280, 8, 8, rng);
        _kernel1280 = CreateRandomDouble(1280, 1280, 3, 3, rng);
        _output1280x8 = new Tensor<double>([1, 1280, 8, 8]);

        // Float precision
        _inputF256x16 = CreateRandomFloat(1, 256, 16, 16, rng);
        _kernelF256 = CreateRandomFloat(256, 256, 3, 3, rng);
        _outputF256x16 = new Tensor<float>([1, 256, 16, 16]);
    }

    [Benchmark(Description = "Conv2D 64ch 64x64 (double, allocating)")]
    public Tensor<double> Conv2D_64ch_64x64_Double()
        => _engine.Conv2D(_input64ch64x64, _kernel64, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 64ch 64x64 (double, zero-alloc)")]
    public void Conv2DInto_64ch_64x64_Double()
        => _engine.Conv2DInto(_output64ch64x64, _input64ch64x64, _kernel64, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 64ch 32x32 (double, allocating)")]
    public Tensor<double> Conv2D_64ch_32x32_Double()
        => _engine.Conv2D(_input64x32, _kernel64, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 64ch 32x32 (double, zero-alloc)")]
    public void Conv2DInto_64ch_32x32_Double()
        => _engine.Conv2DInto(_output64x32, _input64x32, _kernel64, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 128ch 32x32 (double, allocating)")]
    public Tensor<double> Conv2D_128ch_32x32_Double()
        => _engine.Conv2D(_input128x32, _kernel128, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 128ch 32x32 (double, zero-alloc)")]
    public void Conv2DInto_128ch_32x32_Double()
        => _engine.Conv2DInto(_output128x32, _input128x32, _kernel128, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 256ch 16x16 (double, allocating)")]
    public Tensor<double> Conv2D_256ch_16x16_Double()
        => _engine.Conv2D(_input256x16, _kernel256, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 256ch 16x16 (double, zero-alloc)")]
    public void Conv2DInto_256ch_16x16_Double()
        => _engine.Conv2DInto(_output256x16, _input256x16, _kernel256, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2D 512ch 16x16 (double, allocating)")]
    public Tensor<double> Conv2D_512ch_16x16_Double()
        => _engine.Conv2D(_input512x16, _kernel512, stride: 1, padding: 1);

    [Benchmark(Description = "Conv2DInto 512ch 16x16 (double, zero-alloc)")]
    public void Conv2DInto_512ch_16x16_Double()
        => _engine.Conv2DInto(_output512x16, _input512x16, _kernel512, stride: 1, padding: 1);

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

    private static Tensor<double> CreateRandomDouble(int b, int c, int h, int w, Random rng)
    {
        var tensor = new Tensor<double>([b, c, h, w]);
        var data = new double[tensor.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = rng.NextDouble() - 0.5;
        tensor.CopyFromArray(data);
        return tensor;
    }

    private static Tensor<float> CreateRandomFloat(int b, int c, int h, int w, Random rng)
    {
        var tensor = new Tensor<float>([b, c, h, w]);
        var data = new float[tensor.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() - 0.5);
        tensor.CopyFromArray(data);
        return tensor;
    }
}
