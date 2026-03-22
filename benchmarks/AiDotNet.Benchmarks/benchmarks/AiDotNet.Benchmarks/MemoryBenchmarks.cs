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
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
[RankColumn]
public class MemoryBenchmarks
{
    private readonly IEngine _engine = AiDotNetEngine.GetEngine();

    // Immutable inputs — never modified by benchmarks
    private Tensor<double> _inputA = null!;
    private Tensor<double> _inputB = null!;

    // Mutable scratch tensors — reset before each in-place benchmark via IterationSetup
    private Tensor<double> _scratchA = null!;
    private Tensor<double> _dest = null!;

    // Pre-allocated kernel for Conv2D benchmark (not allocated per iteration)
    private Tensor<double> _convKernel = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _inputA = new Tensor<double>([1, 256, 16, 16]);
        _inputB = new Tensor<double>([1, 256, 16, 16]);
        _scratchA = new Tensor<double>([1, 256, 16, 16]);
        _dest = new Tensor<double>([1, 256, 16, 16]);
        _convKernel = new Tensor<double>([256, 256, 3, 3]);
        for (int i = 0; i < _inputA.Length; i++)
        {
            _inputA[i] = rng.NextDouble();
            _inputB[i] = rng.NextDouble();
        }
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Reset scratch tensor before each iteration so in-place ops don't accumulate
        _inputA.Data.Span.CopyTo(_scratchA.Data.Span);
    }

    [Benchmark(Description = "TensorAdd (allocating)")]
    public Tensor<double> Add_Allocating()
        => _engine.TensorAdd(_inputA, _inputB);

    [Benchmark(Description = "TensorAddInPlace (zero-alloc)")]
    public void Add_InPlace()
        => _engine.TensorAddInPlace(_scratchA, _inputB);

    [Benchmark(Description = "TensorAddInto (zero-alloc, pre-alloc dest)")]
    public void Add_Into()
        => _engine.TensorAddInto(_dest, _inputA, _inputB);

    [Benchmark(Description = "new Tensor<double>([1,256,16,16])")]
    public Tensor<double> Alloc_NewTensor()
        => new([1, 256, 16, 16]);

    [Benchmark(Description = "TensorAllocator.Rent+Return cycle ([1,256,16,16])")]
    public void Alloc_Rent()
    {
        var t = TensorAllocator.Rent<double>([1, 256, 16, 16]);
        // Use the tensor (prevent dead code elimination)
        _ = t.Length;
        TensorAllocator.Return(t);
    }

    [Benchmark(Description = "TensorAllocator.RentUninitialized+Return cycle ([1,256,16,16])")]
    public void Alloc_RentUninitialized()
    {
        var t = TensorAllocator.RentUninitialized<double>([1, 256, 16, 16]);
        _ = t.Length;
        TensorAllocator.Return(t);
    }

    [Benchmark(Description = "Conv2D 256ch 16x16 (allocating)")]
    public Tensor<double> Conv2D_Allocating()
        => _engine.Conv2D(_inputA, _convKernel, stride: 1, padding: 1);

    [Benchmark(Description = "Sigmoid (allocating)")]
    public Tensor<double> Sigmoid_Allocating()
        => _engine.Sigmoid(_inputA);

    [Benchmark(Description = "SigmoidInPlace (zero-alloc)")]
    public void Sigmoid_InPlace()
        => _engine.SigmoidInPlace(_scratchA);

    [Benchmark(Description = "SigmoidInto (zero-alloc)")]
    public void Sigmoid_Into()
        => _engine.SigmoidInto(_dest, _inputA);
}
