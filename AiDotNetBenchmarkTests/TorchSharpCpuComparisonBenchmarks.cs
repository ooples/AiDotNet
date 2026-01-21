#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNetBenchmarkTests;

[MemoryDiagnoser]
public class TorchSharpCpuComparisonBenchmarks
{
    private static readonly int[] MatrixSizes = [256, 512];
    private static readonly int[] VectorSizes = [100_000, 1_000_000];

    private CpuEngine _cpuEngine = null!;
    private readonly Dictionary<int, Tensor<float>> _aiMatricesA = new();
    private readonly Dictionary<int, Tensor<float>> _aiMatricesB = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsA = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsB = new();

    // Raw arrays for direct TensorPrimitives comparison
    private readonly Dictionary<int, float[]> _rawArraysA = new();
    private readonly Dictionary<int, float[]> _rawArraysB = new();
    private readonly Dictionary<int, float[]> _rawDestination = new();

    private readonly Dictionary<int, TorchTensor> _torchMatricesA = new();
    private readonly Dictionary<int, TorchTensor> _torchMatricesB = new();
    private readonly Dictionary<int, TorchTensor> _torchVectorsA = new();
    private readonly Dictionary<int, TorchTensor> _torchVectorsB = new();

    private Tensor<float>? _aiConvInput;
    private Tensor<float>? _aiConvKernel;
    private Tensor<float>? _aiConvOutput; // Pre-allocated for zero-allocation benchmark
    private TorchTensor? _torchConvInput;
    private TorchTensor? _torchConvKernel;

    private int _convStride;
    private int _convPadding;
    private int _convDilation;
    private long[]? _torchConvStride;
    private long[]? _torchConvPadding;
    private long[]? _torchConvDilation;

    private torch.Device _torchDevice = null!;
    private readonly Consumer _consumer = new();

    [GlobalSetup]
    public void Setup()
    {
        _cpuEngine = new CpuEngine();
        AiDotNetEngine.Current = _cpuEngine;

        torch.set_grad_enabled(false);
        _torchDevice = torch.CPU;
        Console.WriteLine("TorchSharp device: CPU (forced)");
        Console.WriteLine("AiDotNet BLAS: enabled (via AiDotNet.Native.OpenBLAS package)");

        foreach (var size in MatrixSizes)
        {
            var dataA = CreateData(size * size, seedOffset: size);
            var dataB = CreateData(size * size, seedOffset: size + 13_337);

            _aiMatricesA[size] = new Tensor<float>(dataA, new[] { size, size });
            _aiMatricesB[size] = new Tensor<float>(dataB, new[] { size, size });

            _torchMatricesA[size] = torch.tensor(dataA, new long[] { size, size }, device: _torchDevice);
            _torchMatricesB[size] = torch.tensor(dataB, new long[] { size, size }, device: _torchDevice);
        }

        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            var dataB = CreateData(size, seedOffset: size + 99_999);

            _aiVectorsA[size] = new Tensor<float>(dataA, new[] { size });
            _aiVectorsB[size] = new Tensor<float>(dataB, new[] { size });

            // Also store raw arrays for direct comparison
            _rawArraysA[size] = (float[])dataA.Clone();
            _rawArraysB[size] = (float[])dataB.Clone();
            _rawDestination[size] = new float[size]; // Separate destination to avoid modifying source data

            _torchVectorsA[size] = torch.tensor(dataA, new long[] { size }, device: _torchDevice);
            _torchVectorsB[size] = torch.tensor(dataB, new long[] { size }, device: _torchDevice);
        }

        InitializeConv2D();
        Warmup();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (var tensor in _torchMatricesA.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _torchMatricesB.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _torchVectorsA.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _torchVectorsB.Values)
        {
            tensor.Dispose();
        }

        _torchConvInput?.Dispose();
        _torchConvKernel?.Dispose();
    }

    private void Warmup()
    {
        _ = AiDotNetEngine.Current.TensorMatMul(_aiMatricesA[MatrixSizes[0]], _aiMatricesB[MatrixSizes[0]]);
        using (var result = torch.matmul(_torchMatricesA[MatrixSizes[0]], _torchMatricesB[MatrixSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        _cpuEngine.TensorAddInPlace(_aiVectorsA[VectorSizes[0]], _aiVectorsB[VectorSizes[0]]);
        torch.add_(_torchVectorsA[VectorSizes[0]], _torchVectorsB[VectorSizes[0]]);
        ConsumeTorchResult(_torchVectorsA[VectorSizes[0]]);

        _ = AiDotNetEngine.Current.Conv2D(_aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);
        using (var result = torch.nn.functional.conv2d(_torchConvInput!, _torchConvKernel!,
            strides: _torchConvStride!, padding: _torchConvPadding!, dilation: _torchConvDilation!))
        {
            ConsumeTorchResult(result);
        }
    }

    private void InitializeConv2D()
    {
        const int batch = 1;
        const int inChannels = 16;
        const int height = 64;
        const int width = 64;
        const int outChannels = 32;
        const int kernelSize = 3;

        _convStride = 1;
        _convPadding = 1;
        _convDilation = 1;

        var inputData = CreateData(batch * inChannels * height * width, seedOffset: 7_777);
        var kernelData = CreateData(outChannels * inChannels * kernelSize * kernelSize, seedOffset: 9_999);

        _aiConvInput = new Tensor<float>(inputData, new[] { batch, inChannels, height, width });
        _aiConvKernel = new Tensor<float>(kernelData, new[] { outChannels, inChannels, kernelSize, kernelSize });

        // Pre-allocate output for zero-allocation benchmark
        // Output shape: (64 + 2*1 - 3) / 1 + 1 = 64
        _aiConvOutput = new Tensor<float>(new[] { batch, outChannels, height, width });

        _torchConvInput = torch.tensor(inputData, new long[] { batch, inChannels, height, width }, device: _torchDevice);
        _torchConvKernel = torch.tensor(kernelData, new long[] { outChannels, inChannels, kernelSize, kernelSize }, device: _torchDevice);

        _torchConvStride = new[] { (long)_convStride, _convStride };
        _torchConvPadding = new[] { (long)_convPadding, _convPadding };
        _torchConvDilation = new[] { (long)_convDilation, _convDilation };
    }

    private void ConsumeTorchResult(TorchTensor result)
    {
        _consumer.Consume(result);
    }

    private static float[] CreateData(int length, int seedOffset)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = DeterministicValue(i + seedOffset);
        }

        return data;
    }

    private static float DeterministicValue(int i)
    {
        unchecked
        {
            uint x = (uint)(i * 1664525 + 1013904223);
            return (x & 0x00FFFFFF) / 16777216f;
        }
    }

    [Benchmark]
    [Arguments(256)]
    [Arguments(512)]
    public Tensor<float> AiDotNet_TensorMatMul(int size)
    {
        return AiDotNetEngine.Current.TensorMatMul(_aiMatricesA[size], _aiMatricesB[size]);
    }

    [Benchmark]
    [Arguments(256)]
    [Arguments(512)]
    public void TorchSharp_MatMul(int size)
    {
        using var result = torch.matmul(_torchMatricesA[size], _torchMatricesB[size]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public Tensor<float> AiDotNet_TensorAdd(int size)
    {
        _cpuEngine.TensorAddInPlace(_aiVectorsA[size], _aiVectorsB[size]);
        return _aiVectorsA[size];
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void RawTensorPrimitives_Add(int size)
    {
        System.Numerics.Tensors.TensorPrimitives.Add(
            _rawArraysA[size].AsSpan(),
            _rawArraysB[size].AsSpan(),
            _rawDestination[size].AsSpan());
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void TorchSharp_Add(int size)
    {
        torch.add_(_torchVectorsA[size], _torchVectorsB[size]);
        ConsumeTorchResult(_torchVectorsA[size]);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public Tensor<float> AiDotNet_TensorMultiply(int size)
    {
        _cpuEngine.TensorMultiplyInPlace(_aiVectorsA[size], _aiVectorsB[size]);
        return _aiVectorsA[size];
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void TorchSharp_Multiply(int size)
    {
        torch.mul_(_torchVectorsA[size], _torchVectorsB[size]);
        ConsumeTorchResult(_torchVectorsA[size]);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_ReLU()
    {
        _cpuEngine.ReLUInPlace(_aiVectorsA[VectorSizes[1]]);
        return _aiVectorsA[VectorSizes[1]];
    }

    [Benchmark]
    public void TorchSharp_ReLU()
    {
        _torchVectorsA[VectorSizes[1]].relu_();
        ConsumeTorchResult(_torchVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Sigmoid()
    {
        _cpuEngine.SigmoidInPlace(_aiVectorsA[VectorSizes[1]]);
        return _aiVectorsA[VectorSizes[1]];
    }

    [Benchmark]
    public void TorchSharp_Sigmoid()
    {
        _torchVectorsA[VectorSizes[1]].sigmoid_();
        ConsumeTorchResult(_torchVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public float AiDotNet_TensorSum()
    {
        return AiDotNetEngine.Current.TensorSum(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public float RawTensorPrimitives_Sum()
    {
        return System.Numerics.Tensors.TensorPrimitives.Sum(_rawArraysA[VectorSizes[1]].AsSpan());
    }

    [Benchmark]
    public void TorchSharp_Sum()
    {
        using var result = torch.sum(_torchVectorsA[VectorSizes[1]]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public float AiDotNet_TensorMean()
    {
        return AiDotNetEngine.Current.TensorMean(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public void TorchSharp_Mean()
    {
        using var result = torch.mean(_torchVectorsA[VectorSizes[1]]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Conv2D()
    {
        return AiDotNetEngine.Current.Conv2D(_aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);
    }

    [Benchmark]
    public void AiDotNet_Conv2D_ZeroAlloc()
    {
        _cpuEngine.Conv2DInto(_aiConvOutput!, _aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);
    }

    [Benchmark]
    public void TorchSharp_Conv2D()
    {
        using var result = torch.nn.functional.conv2d(_torchConvInput!, _torchConvKernel!,
            strides: _torchConvStride!, padding: _torchConvPadding!, dilation: _torchConvDilation!);
        ConsumeTorchResult(result);
    }
}
#endif
