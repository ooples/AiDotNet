using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Benchmarks memory allocation patterns: raw allocation vs TensorAllocator vs new Tensor.
/// Measures GC pressure, allocation count, and throughput.
/// </summary>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[RankColumn]
public class MemoryBenchmarks
{
    private readonly IEngine _engine = AiDotNetEngine.GetEngine();

    private Tensor<double> _tensor256 = null!;
    private Tensor<double> _tensor256b = null!;
    private Tensor<double> _dest256 = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _tensor256 = new Tensor<double>([1, 256, 16, 16]);
        _tensor256b = new Tensor<double>([1, 256, 16, 16]);
        _dest256 = new Tensor<double>([1, 256, 16, 16]);
        for (int i = 0; i < _tensor256.Length; i++)
        {
            _tensor256[i] = rng.NextDouble();
            _tensor256b[i] = rng.NextDouble();
        }
    }

    [Benchmark(Description = "TensorAdd (allocating)")]
    public Tensor<double> Add_Allocating()
        => _engine.TensorAdd(_tensor256, _tensor256b);

    [Benchmark(Description = "TensorAddInPlace (zero-alloc)")]
    public void Add_InPlace()
        => _engine.TensorAddInPlace(_tensor256, _tensor256b);

    [Benchmark(Description = "TensorAddInto (zero-alloc, pre-alloc dest)")]
    public void Add_Into()
        => _engine.TensorAddInto(_dest256, _tensor256, _tensor256b);

    [Benchmark(Description = "new Tensor<double>([1,256,16,16])")]
    public Tensor<double> Alloc_NewTensor()
        => new([1, 256, 16, 16]);

    [Benchmark(Description = "TensorAllocator.Rent([1,256,16,16])")]
    public Tensor<double> Alloc_Rent()
        => TensorAllocator.Rent<double>([1, 256, 16, 16]);

    [Benchmark(Description = "TensorAllocator.RentUninitialized([1,256,16,16])")]
    public Tensor<double> Alloc_RentUninitialized()
        => TensorAllocator.RentUninitialized<double>([1, 256, 16, 16]);

    [Benchmark(Description = "Conv2D 256ch 16x16 (allocating)")]
    public Tensor<double> Conv2D_Allocating()
    {
        var kernel = new Tensor<double>([256, 256, 3, 3]);
        return _engine.Conv2D(_tensor256, kernel, stride: 1, padding: 1);
    }

    [Benchmark(Description = "Sigmoid (allocating)")]
    public Tensor<double> Sigmoid_Allocating()
        => _engine.Sigmoid(_tensor256);

    [Benchmark(Description = "SigmoidInPlace (zero-alloc)")]
    public void Sigmoid_InPlace()
        => _engine.SigmoidInPlace(_tensor256);

    [Benchmark(Description = "SigmoidInto (zero-alloc)")]
    public void Sigmoid_Into()
        => _engine.SigmoidInto(_dest256, _tensor256);
}
