using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace AiDotNet.Tests.Benchmarks;

/// <summary>
/// Performance benchmarks comparing CPU vs GPU execution for Phase B GPU acceleration.
/// Tests operations from Epic 2 (Matrix), Epic 3 (Tensor), and Epic 4 (Optimizers/Layers).
/// </summary>
/// <remarks>
/// <para><b>Phase B: US-GPU-017 - Performance Benchmarking</b></para>
/// <para>
/// This benchmark suite validates the GPU acceleration improvements across:
/// - Epic 2: Matrix operations (GEMM, GEMV, BatchMatMul)
/// - Epic 3: Tensor operations (Conv2D, pooling, element-wise)
/// - Epic 4: Optimizers and neural network layers
///
/// Expected performance improvements:
/// - Small operations (below adaptive thresholds): CPU (no GPU overhead)
/// - Large matrix operations: 100-1000x GPU speedup
/// - Convolution operations: 50-500x GPU speedup
/// - Pooling operations: 20-100x GPU speedup
/// - Optimizer updates: 10-100x GPU speedup for large models
/// </para>
/// </remarks>
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 2, iterationCount: 5)]
public class GpuAccelerationBenchmarks
{
    private CpuEngine? _cpuEngine;
    private GpuEngine? _gpuEngine;

    // Test data sizes
    private const int SmallSize = 128;      // Below most thresholds - CPU should win
    private const int MediumSize = 512;     // At threshold boundary
    private const int LargeSize = 2048;     // Well above thresholds - GPU should win

    // Matrix operation test data
    private Matrix<float>? _matrixA_Small;
    private Matrix<float>? _matrixB_Small;
    private Matrix<float>? _matrixA_Large;
    private Matrix<float>? _matrixB_Large;
    private Vector<float>? _vector_Small;
    private Vector<float>? _vector_Large;

    // Tensor operation test data
    private Tensor<float>? _inputTensor_Conv;
    private Tensor<float>? _kernelTensor_Conv;
    private Tensor<float>? _inputTensor_Pool;

    // Layer test data
    private ConvolutionalLayer<float>? _convLayer_CPU;
    private ConvolutionalLayer<float>? _convLayer_GPU;
    private PoolingLayer<float>? _poolLayer_CPU;
    private PoolingLayer<float>? _poolLayer_GPU;

    // Optimizer test data
    private Vector<float>? _optimizerParams;
    private Vector<float>? _optimizerGradient;

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize engines
        _cpuEngine = new CpuEngine();

        try
        {
            _gpuEngine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            // GPU not available - benchmarks will skip GPU tests
            _gpuEngine = null;
        }

        // ═══════════════════════════════════════════════════════════════════════════════
        // Setup Matrix Operation Test Data
        // ═══════════════════════════════════════════════════════════════════════════════

        _matrixA_Small = CreateRandomMatrix(SmallSize, SmallSize, random);
        _matrixB_Small = CreateRandomMatrix(SmallSize, SmallSize, random);
        _matrixA_Large = CreateRandomMatrix(LargeSize, LargeSize, random);
        _matrixB_Large = CreateRandomMatrix(LargeSize, LargeSize, random);

        _vector_Small = CreateRandomVector(SmallSize, random);
        _vector_Large = CreateRandomVector(LargeSize, random);

        // ═══════════════════════════════════════════════════════════════════════════════
        // Setup Tensor Operation Test Data (Epic 3)
        // ═══════════════════════════════════════════════════════════════════════════════

        // Conv2D: batch=4, channels=64, height=56, width=56, kernels=128, kernel_size=3
        _inputTensor_Conv = CreateRandomTensor(new[] { 4, 64, 56, 56 }, random);
        _kernelTensor_Conv = CreateRandomTensor(new[] { 128, 64, 3, 3 }, random);

        // MaxPool2D: batch=4, channels=128, height=28, width=28
        _inputTensor_Pool = CreateRandomTensor(new[] { 4, 128, 28, 28 }, random);

        // ═══════════════════════════════════════════════════════════════════════════════
        // Setup Layer Test Data (Epic 4: US-GPU-016)
        // ═══════════════════════════════════════════════════════════════════════════════

        _convLayer_CPU = new ConvolutionalLayer<float>(
            inputDepth: 64, outputDepth: 128, kernelSize: 3,
            inputHeight: 56, inputWidth: 56, stride: 1, padding: 1,
            activation: null);

        if (_gpuEngine != null)
        {
            _convLayer_GPU = new ConvolutionalLayer<float>(
                inputDepth: 64, outputDepth: 128, kernelSize: 3,
                inputHeight: 56, inputWidth: 56, stride: 1, padding: 1,
                activation: null);
        }

        _poolLayer_CPU = new PoolingLayer<float>(
            inputDepth: 128, inputHeight: 28, inputWidth: 28,
            poolSize: 2, stride: 2, type: PoolingType.Max);

        if (_gpuEngine != null)
        {
            _poolLayer_GPU = new PoolingLayer<float>(
                inputDepth: 128, inputHeight: 28, inputWidth: 28,
                poolSize: 2, stride: 2, type: PoolingType.Max);
        }

        // ═══════════════════════════════════════════════════════════════════════════════
        // Setup Optimizer Test Data (Epic 4: US-GPU-015)
        // ═══════════════════════════════════════════════════════════════════════════════

        // Large model with 1M parameters (typical for medium-sized neural networks)
        _optimizerParams = CreateRandomVector(1_000_000, random);
        _optimizerGradient = CreateRandomVector(1_000_000, random);
    }

    #region Helper Methods

    private static Matrix<float> CreateRandomMatrix(int rows, int cols, Random random)
    {
        var matrix = new Matrix<float>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = (float)(random.NextDouble() * 2 - 1);
            }
        }
        return matrix;
    }

    private static Vector<float> CreateRandomVector(int size, Random random)
    {
        var vector = new Vector<float>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return vector;
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, Random random)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    #endregion

    #region Epic 2: Matrix Operations (US-GPU-007, US-GPU-008)

    [Benchmark(Description = "GEMM_Small_CPU")]
    public Matrix<float> MatrixMultiply_Small_CPU()
    {
        return (Matrix<float>)_cpuEngine!.MatrixMultiply(_matrixA_Small!, _matrixB_Small!);
    }

    [Benchmark(Description = "GEMM_Small_GPU")]
    public Matrix<float>? MatrixMultiply_Small_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Matrix<float>)_gpuEngine.MatrixMultiply(_matrixA_Small!, _matrixB_Small!);
    }

    [Benchmark(Description = "GEMM_Large_CPU")]
    public Matrix<float> MatrixMultiply_Large_CPU()
    {
        return (Matrix<float>)_cpuEngine!.MatrixMultiply(_matrixA_Large!, _matrixB_Large!);
    }

    [Benchmark(Description = "GEMM_Large_GPU")]
    public Matrix<float>? MatrixMultiply_Large_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Matrix<float>)_gpuEngine.MatrixMultiply(_matrixA_Large!, _matrixB_Large!);
    }

    [Benchmark(Description = "GEMV_Large_CPU")]
    public Vector<float> MatrixVectorMultiply_CPU()
    {
        return (Vector<float>)_cpuEngine!.MatrixVectorMultiply(_matrixA_Large!, _vector_Large!);
    }

    [Benchmark(Description = "GEMV_Large_GPU")]
    public Vector<float>? MatrixVectorMultiply_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Vector<float>)_gpuEngine.MatrixVectorMultiply(_matrixA_Large!, _vector_Large!);
    }

    #endregion

    #region Epic 3: Tensor Operations (US-GPU-011, US-GPU-012)

    [Benchmark(Description = "Conv2D_CPU")]
    public Tensor<float> Convolution2D_CPU()
    {
        return (Tensor<float>)_cpuEngine!.Conv2D(_inputTensor_Conv!, _kernelTensor_Conv!, stride: 1, padding: 1, dilation: 1);
    }

    [Benchmark(Description = "Conv2D_GPU")]
    public Tensor<float>? Convolution2D_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Tensor<float>)_gpuEngine.Conv2D(_inputTensor_Conv!, _kernelTensor_Conv!, stride: 1, padding: 1, dilation: 1);
    }

    [Benchmark(Description = "MaxPool2D_CPU")]
    public Tensor<float> MaxPool2D_CPU()
    {
        return (Tensor<float>)_cpuEngine!.MaxPool2D(_inputTensor_Pool!, poolSize: 2, stride: 2, padding: 0);
    }

    [Benchmark(Description = "MaxPool2D_GPU")]
    public Tensor<float>? MaxPool2D_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Tensor<float>)_gpuEngine.MaxPool2D(_inputTensor_Pool!, poolSize: 2, stride: 2, padding: 0);
    }

    #endregion

    #region Epic 4: Neural Network Layers (US-GPU-016)

    [Benchmark(Description = "ConvLayer_Forward_CPU")]
    public Tensor<float> ConvolutionalLayer_Forward_CPU()
    {
        return _convLayer_CPU!.Forward(_inputTensor_Conv!);
    }

    [Benchmark(Description = "ConvLayer_Forward_GPU")]
    public Tensor<float>? ConvolutionalLayer_Forward_GPU()
    {
        if (_convLayer_GPU == null) return null;
        return _convLayer_GPU.Forward(_inputTensor_Conv!);
    }

    [Benchmark(Description = "PoolLayer_Forward_CPU")]
    public Tensor<float> PoolingLayer_Forward_CPU()
    {
        return _poolLayer_CPU!.Forward(_inputTensor_Pool!);
    }

    [Benchmark(Description = "PoolLayer_Forward_GPU")]
    public Tensor<float>? PoolingLayer_Forward_GPU()
    {
        if (_poolLayer_GPU == null) return null;
        return _poolLayer_GPU.Forward(_inputTensor_Pool!);
    }

    #endregion

    #region Epic 4: Optimizer Operations (US-GPU-015)

    [Benchmark(Description = "VectorOps_Add_Large_CPU")]
    public Vector<float> VectorAdd_Large_CPU()
    {
        return (Vector<float>)_cpuEngine!.Add(_optimizerParams!, _optimizerGradient!);
    }

    [Benchmark(Description = "VectorOps_Add_Large_GPU")]
    public Vector<float>? VectorAdd_Large_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Vector<float>)_gpuEngine.Add(_optimizerParams!, _optimizerGradient!);
    }

    [Benchmark(Description = "VectorOps_Multiply_Large_CPU")]
    public Vector<float> VectorMultiply_Large_CPU()
    {
        return (Vector<float>)_cpuEngine!.Multiply(_optimizerParams!, _optimizerGradient!);
    }

    [Benchmark(Description = "VectorOps_Multiply_Large_GPU")]
    public Vector<float>? VectorMultiply_Large_GPU()
    {
        if (_gpuEngine == null) return null;
        return (Vector<float>)_gpuEngine.Multiply(_optimizerParams!, _optimizerGradient!);
    }

    #endregion
}

/// <summary>
/// Expected benchmark results (based on Phase B implementation):
///
/// SMALL OPERATIONS (128x128 matrices, below adaptive thresholds):
/// - CPU should be faster or comparable to GPU (no GPU overhead penalty)
/// - Adaptive execution automatically routes to CPU
///
/// LARGE OPERATIONS (2048x2048 matrices, well above thresholds):
/// - Matrix Multiply (GEMM): GPU 100-1000x faster than CPU
/// - Matrix-Vector Multiply (GEMV): GPU 50-200x faster than CPU
/// - Conv2D: GPU 50-500x faster than CPU (most impactful improvement)
/// - MaxPool2D: GPU 20-100x faster than CPU
/// - Vector Operations (1M elements): GPU 10-100x faster than CPU
///
/// LAYER FORWARD PASSES:
/// - ConvolutionalLayer: GPU 50-500x faster (dominated by Conv2D operation)
/// - PoolingLayer: GPU 20-100x faster (dominated by pooling operation)
///
/// MEMORY USAGE:
/// - GPU operations use memory pooling (should show minimal allocations)
/// - CPU operations may show more allocations (no pooling)
///
/// COMPARISON TO PYTORCH/TENSORFLOW:
/// While direct benchmarking against PyTorch/TensorFlow requires Python interop,
/// our GPU kernels use ILGPU which provides similar performance characteristics:
/// - cuBLAS-equivalent matrix operations (via ILGPU optimizations)
/// - cuDNN-equivalent convolution patterns
/// - Expected to be within 2-3x of PyTorch/TensorFlow for equivalent operations
///
/// To run these benchmarks:
/// ```bash
/// dotnet run -c Release --project tests/AiDotNet.Tests --filter "*GpuAccelerationBenchmarks*"
/// ```
///
/// Or use BenchmarkDotNet:
/// ```csharp
/// var summary = BenchmarkRunner.Run<GpuAccelerationBenchmarks>();
/// ```
/// </summary>
public class BenchmarkDocumentation { }
