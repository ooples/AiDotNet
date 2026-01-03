using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.NeuralNetworks;

/// <summary>
/// Benchmarks for DenseLayer GPU acceleration.
/// Compares GPU-optimized paths (DirectGpu, HIP, OpenCL) against CPU baseline.
/// Designed for AMD GPU testing (HIP and OpenCL backends).
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[CsvExporter]
[HtmlExporter]
public class DenseLayerGpuBenchmark
{
    private DenseLayer<float> _denseLayer = null!;
    private Tensor<float> _input = null!;
    private float[] _inputData = null!;
    private float[] _weightsData = null!;
    private float[] _biasesData = null!;

    // Test different matrix sizes to see where GPU acceleration provides benefit
    [Params(128, 256, 512, 1024, 2048)]
    public int MatrixSize { get; set; }

    [Params(32, 64, 128)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        // Report which engine/backend is being used
        var engine = AiDotNetEngine.Current;
        Console.WriteLine($"Engine: {engine.Name}");
        Console.WriteLine($"DirectGpu Available: {engine.DirectGpu?.IsAvailable ?? false}");
        if (engine.DirectGpu?.IsAvailable == true)
        {
            Console.WriteLine($"DirectGpu Backend: {engine.DirectGpu?.GetType().Name}");
        }

        // Initialize the dense layer with specified dimensions
        // Cast to IActivationFunction to avoid ambiguous constructor
        _denseLayer = new DenseLayer<float>(MatrixSize, MatrixSize, (IActivationFunction<float>)new ReLUActivation<float>());

        // Create input tensor with deterministic values
        _inputData = new float[BatchSize * MatrixSize];
        for (int i = 0; i < _inputData.Length; i++)
        {
            _inputData[i] = DeterministicValue(i);
        }

        _input = new Tensor<float>(_inputData, new[] { BatchSize, MatrixSize });

        // Pre-extract weights and biases for naive benchmark (avoid overhead in benchmark loop)
        _weightsData = _denseLayer.GetWeights().ToArray();
        _biasesData = _denseLayer.GetBiases().ToArray();

        // Warm up the GPU path (first call initializes GPU resources)
        var warmup = _denseLayer.Forward(_input);
        GC.KeepAlive(warmup);
    }

    private static float DeterministicValue(int i)
    {
        unchecked
        {
            uint x = (uint)(i * 1664525 + 1013904223);
            return (x & 0x00FFFFFF) / 16777216f;
        }
    }

    /// <summary>
    /// DenseLayer forward pass using the automatic GPU path selection.
    /// This will use DirectGpu fused, HIP, OpenCL, or cached weights depending on availability.
    /// </summary>
    [Benchmark(Baseline = true)]
    public Tensor<float> DenseLayerForward()
    {
        return _denseLayer.Forward(_input);
    }

    /// <summary>
    /// Manual naive implementation for comparison.
    /// Uses triple-nested loop without any GPU acceleration.
    /// </summary>
    [Benchmark]
    public float[] NaiveMatMulBiasRelu()
    {
        int inputSize = MatrixSize;
        int outputSize = MatrixSize;
        var output = new float[BatchSize * outputSize];

        // Naive GEMM: output = input @ weights.T
        for (int b = 0; b < BatchSize; b++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                float sum = 0.0f;
                for (int i = 0; i < inputSize; i++)
                {
                    // weights is [outputSize, inputSize], so we access [o, i]
                    sum += _inputData[b * inputSize + i] * _weightsData[o * inputSize + i];
                }
                // Add bias and apply ReLU
                sum += _biasesData[o];
                output[b * outputSize + o] = sum > 0 ? sum : 0;
            }
        }

        return output;
    }

    /// <summary>
    /// Engine-only path using TensorMatMul.
    /// This uses the current AiDotNetEngine (may be CPU or GPU depending on configuration).
    /// </summary>
    [Benchmark]
    public Tensor<float> EngineOnlyMatMul()
    {
        var weights = _denseLayer.GetWeights();
        var biases = _denseLayer.GetBiases();
        var engine = AiDotNetEngine.Current;

        // Transpose weights (since they're stored as [out, in])
        var weightsTransposed = engine.TensorTranspose(weights);

        // Standard GEMM
        var matmul = engine.TensorMatMul(_input, weightsTransposed);

        // Add bias
        var withBias = engine.TensorBroadcastAdd(matmul, biases);

        // Apply ReLU (element-wise)
        var data = withBias.ToArray();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = data[i] > 0 ? data[i] : 0;
        }

        return new Tensor<float>(data, withBias.Shape);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _denseLayer?.Dispose();
    }
}
