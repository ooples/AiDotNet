using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace AiDotNet.Tests.Benchmarks;

/// <summary>
/// Performance benchmarks comparing manual vs autodiff backward passes.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 3, iterationCount: 10)]
public class AutodiffPerformanceBenchmarks
{
    private DenseLayer<float>? _denseLayer;
    private ActivationLayer<float>? _activationLayer;
    private BatchNormalizationLayer<float>? _batchNormLayer;
    private DropoutLayer<float>? _dropoutLayer;

    private Tensor<float>? _denseInput;
    private Tensor<float>? _denseOutputGradient;

    private Tensor<float>? _activationInput;
    private Tensor<float>? _activationOutputGradient;

    private Tensor<float>? _batchNormInput;
    private Tensor<float>? _batchNormOutputGradient;

    private Tensor<float>? _dropoutInput;
    private Tensor<float>? _dropoutOutputGradient;

    private const int BatchSize = 32;
    private const int InputSize = 512;
    private const int OutputSize = 256;
    private const int FeatureSize = 128;

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Setup DenseLayer
        _denseLayer = new DenseLayer<float>(InputSize, OutputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        _denseInput = CreateRandomTensor(new[] { BatchSize, InputSize }, random);
        _denseOutputGradient = CreateRandomTensor(new[] { BatchSize, OutputSize }, random);

        // Setup ActivationLayer
        _activationLayer = new ActivationLayer<float>(
            new[] { BatchSize, FeatureSize },
            (IActivationFunction<float>)new ReLUActivation<float>());
        _activationInput = CreateRandomTensor(new[] { BatchSize, FeatureSize }, random);
        _activationOutputGradient = CreateRandomTensor(new[] { BatchSize, FeatureSize }, random);

        // Setup BatchNormalizationLayer
        _batchNormLayer = new BatchNormalizationLayer<float>(FeatureSize);
        _batchNormInput = CreateRandomTensor(new[] { BatchSize, FeatureSize }, random);
        _batchNormOutputGradient = CreateRandomTensor(new[] { BatchSize, FeatureSize }, random);

        // Setup DropoutLayer
        _dropoutLayer = new DropoutLayer<float>(0.5);
        _dropoutLayer.SetTrainingMode(true);
        _dropoutInput = CreateRandomTensor(new[] { BatchSize, FeatureSize }, random);
        _dropoutOutputGradient = CreateRandomTensor(new[] { BatchSize, FeatureSize }, random);
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, Random random)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1); // Range: [-1, 1]
        }
        return tensor;
    }

    #region DenseLayer Benchmarks

    [Benchmark(Baseline = true, Description = "DenseLayer_Manual")]
    public Tensor<float> DenseLayer_BackwardManual()
    {
        _denseLayer!.UseAutodiff = false;
        _denseLayer.ResetState();
        _denseLayer.Forward(_denseInput!);
        return _denseLayer.Backward(_denseOutputGradient!);
    }

    [Benchmark(Description = "DenseLayer_Autodiff")]
    public Tensor<float> DenseLayer_BackwardAutodiff()
    {
        _denseLayer!.UseAutodiff = true;
        _denseLayer.ResetState();
        _denseLayer.Forward(_denseInput!);
        return _denseLayer.Backward(_denseOutputGradient!);
    }

    #endregion

    #region ActivationLayer Benchmarks

    [Benchmark(Description = "ActivationLayer_Manual")]
    public Tensor<float> ActivationLayer_BackwardManual()
    {
        _activationLayer!.UseAutodiff = false;
        _activationLayer.ResetState();
        _activationLayer.Forward(_activationInput!);
        return _activationLayer.Backward(_activationOutputGradient!);
    }

    [Benchmark(Description = "ActivationLayer_Autodiff")]
    public Tensor<float> ActivationLayer_BackwardAutodiff()
    {
        _activationLayer!.UseAutodiff = true;
        _activationLayer.ResetState();
        _activationLayer.Forward(_activationInput!);
        return _activationLayer.Backward(_activationOutputGradient!);
    }

    #endregion

    #region BatchNormalizationLayer Benchmarks

    [Benchmark(Description = "BatchNorm_Manual")]
    public Tensor<float> BatchNormalization_BackwardManual()
    {
        _batchNormLayer!.UseAutodiff = false;
        _batchNormLayer.ResetState();
        _batchNormLayer.Forward(_batchNormInput!);
        return _batchNormLayer.Backward(_batchNormOutputGradient!);
    }

    [Benchmark(Description = "BatchNorm_Autodiff")]
    public Tensor<float> BatchNormalization_BackwardAutodiff()
    {
        _batchNormLayer!.UseAutodiff = true;
        _batchNormLayer.ResetState();
        _batchNormLayer.Forward(_batchNormInput!);
        return _batchNormLayer.Backward(_batchNormOutputGradient!);
    }

    #endregion

    #region DropoutLayer Benchmarks

    [Benchmark(Description = "Dropout_Manual")]
    public Tensor<float> Dropout_BackwardManual()
    {
        _dropoutLayer!.UseAutodiff = false;
        _dropoutLayer.ResetState();
        _dropoutLayer.Forward(_dropoutInput!);
        return _dropoutLayer.Backward(_dropoutOutputGradient!);
    }

    [Benchmark(Description = "Dropout_Autodiff")]
    public Tensor<float> Dropout_BackwardAutodiff()
    {
        _dropoutLayer!.UseAutodiff = true;
        _dropoutLayer.ResetState();
        _dropoutLayer.Forward(_dropoutInput!);
        return _dropoutLayer.Backward(_dropoutOutputGradient!);
    }

    #endregion

    /// <summary>
    /// Run all benchmarks.
    /// Usage: Use BenchmarkDotNet CLI or invoke BenchmarkRunner.Run manually
    /// Note: Main method commented out to avoid conflicts with test project entry point
    /// </summary>
    //public static void Main(string[] args)
    //{
    //    var summary = BenchmarkRunner.Run<AutodiffPerformanceBenchmarks>();
    //    Console.WriteLine(summary);
    //}
}

/// <summary>
/// Summary of expected performance characteristics based on implementation:
///
/// Expected Results:
/// - Manual backward passes: 1.0x (baseline) - highly optimized
/// - Autodiff backward passes: 3-5x slower on average
///
/// Reasons for autodiff overhead:
/// 1. Computation graph construction
/// 2. Topological sorting of graph nodes
/// 3. Dynamic dispatch through backward functions
/// 4. Gradient accumulation and memory allocation
/// 5. Less opportunity for compiler optimizations
///
/// When to use each approach:
/// - Manual: Production training, performance-critical applications
/// - Autodiff: Research, prototyping, gradient verification, custom layers
///
/// Performance optimization opportunities:
/// - Graph caching for repeated operations
/// - JIT compilation of computation graphs
/// - Graph fusion/optimization passes
/// - Static graph compilation (future work)
/// </summary>
public class PerformanceNotes { }
