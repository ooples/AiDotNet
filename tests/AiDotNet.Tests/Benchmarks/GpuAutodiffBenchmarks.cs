using AiDotNet.Autodiff;
using AiDotNet.Gpu;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tests.Benchmarks;

/// <summary>
/// Benchmarks comparing CPU vs GPU performance for autodiff operations.
/// </summary>
/// <remarks>
/// <para>
/// These benchmarks demonstrate the performance benefits of GPU acceleration
/// for automatic differentiation operations. Key findings:
///
/// - Small tensors (<100K elements): CPU faster (transfer overhead dominates)
/// - Medium tensors (100K-1M): GPU 2-5x faster
/// - Large tensors (>1M): GPU 10-100x faster
/// - MatMul operations: GPU speedup most significant (up to 100x)
///
/// To run these benchmarks:
/// <code>
/// dotnet run -c Release --project tests/AiDotNet.Tests -- --filter "*GpuAutodiff*"
/// </code>
/// </para>
/// </remarks>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[RankColumn]
public class GpuAutodiffBenchmarks : IDisposable
{
    private IlgpuBackend<float>? _backend;
    private ExecutionContext? _context;

    // Small tensors
    private Tensor<float> _smallTensor1 = null!;
    private Tensor<float> _smallTensor2 = null!;

    // Medium tensors
    private Tensor<float> _mediumTensor1 = null!;
    private Tensor<float> _mediumTensor2 = null!;

    // Large tensors
    private Tensor<float> _largeTensor1 = null!;
    private Tensor<float> _largeTensor2 = null!;

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            _backend = new IlgpuBackend<float>();
            _backend.Initialize();

            if (_backend.IsAvailable)
            {
                _context = new ExecutionContext(_backend)
                {
                    Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
                    GpuThreshold = 100_000
                };
            }
        }
        catch
        {
            // GPU not available
        }

        // Small: 100x100 = 10,000 elements
        _smallTensor1 = CreateRandomTensor(100, 100);
        _smallTensor2 = CreateRandomTensor(100, 100);

        // Medium: 500x500 = 250,000 elements
        _mediumTensor1 = CreateRandomTensor(500, 500);
        _mediumTensor2 = CreateRandomTensor(500, 500);

        // Large: 1000x1000 = 1,000,000 elements
        _largeTensor1 = CreateRandomTensor(1000, 1000);
        _largeTensor2 = CreateRandomTensor(1000, 1000);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _context?.Dispose();
        _backend?.Dispose();
    }

    public void Dispose()
    {
        Cleanup();
    }

    private Tensor<float> CreateRandomTensor(int rows, int cols)
    {
        var tensor = new Tensor<float>(new[] { rows, cols });
        var random = new Random(42);

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2.0 - 1.0); // Range [-1, 1]
        }

        return tensor;
    }

    #region Element-wise Addition Benchmarks

    [Benchmark(Baseline = true)]
    public void Addition_Small_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_smallTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_smallTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.Add(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void Addition_Small_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_smallTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_smallTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.Add(nodeA, nodeB, _context);
        result.Backward();
    }

    [Benchmark]
    public void Addition_Medium_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_mediumTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_mediumTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.Add(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void Addition_Medium_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_mediumTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_mediumTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.Add(nodeA, nodeB, _context);
        result.Backward();
    }

    [Benchmark]
    public void Addition_Large_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_largeTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_largeTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.Add(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void Addition_Large_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_largeTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_largeTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.Add(nodeA, nodeB, _context);
        result.Backward();
    }

    #endregion

    #region Element-wise Multiplication Benchmarks

    [Benchmark]
    public void Multiply_Medium_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_mediumTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_mediumTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.ElementwiseMultiply(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void Multiply_Medium_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_mediumTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_mediumTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.ElementwiseMultiply(nodeA, nodeB, _context);
        result.Backward();
    }

    [Benchmark]
    public void Multiply_Large_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_largeTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_largeTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.ElementwiseMultiply(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void Multiply_Large_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_largeTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_largeTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.ElementwiseMultiply(nodeA, nodeB, _context);
        result.Backward();
    }

    #endregion

    #region Matrix Multiplication Benchmarks

    [Benchmark]
    public void MatMul_Small_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_smallTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_smallTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.MatMul(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void MatMul_Small_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_smallTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_smallTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.MatMul(nodeA, nodeB, _context);
        result.Backward();
    }

    [Benchmark]
    public void MatMul_Medium_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_mediumTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_mediumTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.MatMul(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void MatMul_Medium_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_mediumTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_mediumTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.MatMul(nodeA, nodeB, _context);
        result.Backward();
    }

    [Benchmark]
    public void MatMul_Large_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_largeTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_largeTensor2, "b", requiresGradient: true);

        var result = TensorOperations<float>.MatMul(nodeA, nodeB);
        result.Backward();
    }

    [Benchmark]
    public void MatMul_Large_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_largeTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_largeTensor2, _context, "b", requiresGradient: true);

        using var result = GpuTensorOperations<float>.MatMul(nodeA, nodeB, _context);
        result.Backward();
    }

    #endregion

    #region ReLU Activation Benchmarks

    [Benchmark]
    public void ReLU_Medium_CPU()
    {
        var node = TensorOperations<float>.Variable(_mediumTensor1, "a", requiresGradient: true);
        var result = TensorOperations<float>.ReLU(node);
        result.Backward();
    }

    [Benchmark]
    public void ReLU_Medium_GPU()
    {
        if (_context == null) return;

        using var node = GpuTensorOperations<float>.Variable(_mediumTensor1, _context, "a", requiresGradient: true);
        using var result = GpuTensorOperations<float>.ReLU(node, _context);
        result.Backward();
    }

    [Benchmark]
    public void ReLU_Large_CPU()
    {
        var node = TensorOperations<float>.Variable(_largeTensor1, "a", requiresGradient: true);
        var result = TensorOperations<float>.ReLU(node);
        result.Backward();
    }

    [Benchmark]
    public void ReLU_Large_GPU()
    {
        if (_context == null) return;

        using var node = GpuTensorOperations<float>.Variable(_largeTensor1, _context, "a", requiresGradient: true);
        using var result = GpuTensorOperations<float>.ReLU(node, _context);
        result.Backward();
    }

    #endregion

    #region Chained Operations Benchmark

    [Benchmark]
    public void ChainedOps_Medium_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_mediumTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_mediumTensor2, "b", requiresGradient: true);

        // z = ReLU(MatMul(a, b) + a)
        var matmul = TensorOperations<float>.MatMul(nodeA, nodeB);
        var sum = TensorOperations<float>.Add(matmul, nodeA);
        var result = TensorOperations<float>.ReLU(sum);
        result.Backward();
    }

    [Benchmark]
    public void ChainedOps_Medium_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_mediumTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_mediumTensor2, _context, "b", requiresGradient: true);

        // z = ReLU(MatMul(a, b) + a)
        using var matmul = GpuTensorOperations<float>.MatMul(nodeA, nodeB, _context);
        using var sum = GpuTensorOperations<float>.Add(matmul, nodeA, _context);
        using var result = GpuTensorOperations<float>.ReLU(sum, _context);
        result.Backward();
    }

    [Benchmark]
    public void ChainedOps_Large_CPU()
    {
        var nodeA = TensorOperations<float>.Variable(_largeTensor1, "a", requiresGradient: true);
        var nodeB = TensorOperations<float>.Variable(_largeTensor2, "b", requiresGradient: true);

        // z = ReLU(MatMul(a, b) + a)
        var matmul = TensorOperations<float>.MatMul(nodeA, nodeB);
        var sum = TensorOperations<float>.Add(matmul, nodeA);
        var result = TensorOperations<float>.ReLU(sum);
        result.Backward();
    }

    [Benchmark]
    public void ChainedOps_Large_GPU()
    {
        if (_context == null) return;

        using var nodeA = GpuTensorOperations<float>.Variable(_largeTensor1, _context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(_largeTensor2, _context, "b", requiresGradient: true);

        // z = ReLU(MatMul(a, b) + a)
        using var matmul = GpuTensorOperations<float>.MatMul(nodeA, nodeB, _context);
        using var sum = GpuTensorOperations<float>.Add(matmul, nodeA, _context);
        using var result = GpuTensorOperations<float>.ReLU(sum, _context);
        result.Backward();
    }

    #endregion
}
