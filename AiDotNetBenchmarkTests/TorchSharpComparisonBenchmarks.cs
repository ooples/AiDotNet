#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNetBenchmarkTests;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 5)]
public class TorchSharpComparisonBenchmarks
{
    private static readonly int[] MatrixSizes = [256, 512];
    private static readonly int[] VectorSizes = [100_000, 1_000_000];

    private readonly Dictionary<int, Tensor<float>> _aiMatricesA = new();
    private readonly Dictionary<int, Tensor<float>> _aiMatricesB = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsA = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsB = new();
    private readonly Dictionary<int, IGpuTensor<float>> _aiGpuMatricesA = new();
    private readonly Dictionary<int, IGpuTensor<float>> _aiGpuMatricesB = new();
    private readonly Dictionary<int, IGpuTensor<float>> _aiGpuVectorsA = new();
    private readonly Dictionary<int, IGpuTensor<float>> _aiGpuVectorsB = new();
    private readonly Dictionary<int, IGpuBuffer> _aiGpuAddOutputs = new();
    private readonly Dictionary<int, IGpuBuffer> _aiGpuMultiplyOutputs = new();

    private readonly Dictionary<int, TorchTensor> _torchMatricesA = new();
    private readonly Dictionary<int, TorchTensor> _torchMatricesB = new();
    private readonly Dictionary<int, TorchTensor> _torchVectorsA = new();
    private readonly Dictionary<int, TorchTensor> _torchVectorsB = new();

    private Tensor<float>? _aiConvInput;
    private Tensor<float>? _aiConvKernel;
    private IGpuTensor<float>? _aiGpuConvInput;
    private TorchTensor? _torchConvInput;
    private TorchTensor? _torchConvKernel;

    private int _convStride;
    private int _convPadding;
    private int _convDilation;
    private long[]? _torchConvStride;
    private long[]? _torchConvPadding;
    private long[]? _torchConvDilation;

    private DirectGpuTensorEngine? _gpuEngine;
    private IDirectGpuBackend? _gpuBackend;

    private torch.Device _torchDevice = torch.CUDA;
    private bool _torchUsesCuda;
    private readonly Consumer _consumer = new();

    [GlobalSetup]
    public void Setup()
    {
        AiDotNetEngine.Current = new GpuEngine(AdaptiveThresholds.AlwaysGpu);
        _gpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        if (_gpuEngine == null || !_gpuEngine.SupportsGpu)
        {
            throw new InvalidOperationException("AiDotNet GPU backend is not available. Run CPU comparison benchmarks instead.");
        }

        _gpuBackend = _gpuEngine.GetBackend();
        if (_gpuBackend == null)
        {
            throw new InvalidOperationException("AiDotNet GPU backend is not available.");
        }

        torch.set_grad_enabled(false);
        _torchUsesCuda = torch.cuda.is_available();
        if (!_torchUsesCuda)
        {
            throw new InvalidOperationException("TorchSharp CUDA is not available. Run CPU comparison benchmarks instead.");
        }

        _torchDevice = torch.CUDA;
        Console.WriteLine("TorchSharp device: CUDA");

        foreach (var size in MatrixSizes)
        {
            var dataA = CreateData(size * size, seedOffset: size);
            var dataB = CreateData(size * size, seedOffset: size + 13_337);

            _aiMatricesA[size] = new Tensor<float>(dataA, new[] { size, size });
            _aiMatricesB[size] = new Tensor<float>(dataB, new[] { size, size });
            _aiGpuMatricesA[size] = _gpuEngine.UploadToGpu(_aiMatricesA[size], GpuTensorRole.Activation);
            _aiGpuMatricesB[size] = _gpuEngine.UploadToGpu(_aiMatricesB[size], GpuTensorRole.Activation);

            _torchMatricesA[size] = torch.tensor(dataA, new long[] { size, size }, device: _torchDevice);
            _torchMatricesB[size] = torch.tensor(dataB, new long[] { size, size }, device: _torchDevice);
        }

        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            var dataB = CreateData(size, seedOffset: size + 99_999);

            _aiVectorsA[size] = new Tensor<float>(dataA, new[] { size });
            _aiVectorsB[size] = new Tensor<float>(dataB, new[] { size });
            _aiGpuVectorsA[size] = _gpuEngine.UploadToGpu(_aiVectorsA[size], GpuTensorRole.Activation);
            _aiGpuVectorsB[size] = _gpuEngine.UploadToGpu(_aiVectorsB[size], GpuTensorRole.Activation);

            _torchVectorsA[size] = torch.tensor(dataA, new long[] { size }, device: _torchDevice);
            _torchVectorsB[size] = torch.tensor(dataB, new long[] { size }, device: _torchDevice);
        }

        InitializeConv2D();
        if (_aiConvInput == null || _aiConvKernel == null)
        {
            throw new InvalidOperationException("Conv2D tensors were not initialized.");
        }

        _aiGpuConvInput = _gpuEngine.UploadToGpu(_aiConvInput, GpuTensorRole.Activation);
        if (_torchConvInput is null || _torchConvKernel is null)
        {
            throw new InvalidOperationException("TorchSharp Conv2D tensors were not initialized.");
        }

        foreach (var size in VectorSizes)
        {
            _aiGpuAddOutputs[size] = _gpuBackend.AllocateBuffer(size);
            _aiGpuMultiplyOutputs[size] = _gpuBackend.AllocateBuffer(size);
        }

        Warmup();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (var tensor in _aiGpuMatricesA.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _aiGpuMatricesB.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _aiGpuVectorsA.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _aiGpuVectorsB.Values)
        {
            tensor.Dispose();
        }

        foreach (var buffer in _aiGpuAddOutputs.Values)
        {
            buffer.Dispose();
        }

        foreach (var buffer in _aiGpuMultiplyOutputs.Values)
        {
            buffer.Dispose();
        }

        _aiGpuConvInput?.Dispose();

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
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        var gpuBackend = _gpuBackend ?? throw new InvalidOperationException("GPU backend was not initialized.");

        using var matmul = gpuEngine.MatMulGpuTensors(_aiGpuMatricesA[MatrixSizes[0]], _aiGpuMatricesB[MatrixSizes[0]]);
        matmul.Synchronize();
        using (var result = torch.matmul(_torchMatricesA[MatrixSizes[0]], _torchMatricesB[MatrixSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        using var add = gpuEngine.AddGpu(_aiGpuVectorsA[VectorSizes[0]], _aiGpuVectorsB[VectorSizes[0]]);
        add.Synchronize();
        using (var result = torch.add(_torchVectorsA[VectorSizes[0]], _torchVectorsB[VectorSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        using var multiply = gpuEngine.MultiplyGpu(_aiGpuVectorsA[VectorSizes[0]], _aiGpuVectorsB[VectorSizes[0]]);
        multiply.Synchronize();
        using (var result = torch.mul(_torchVectorsA[VectorSizes[0]], _torchVectorsB[VectorSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        using var relu = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[0]], FusedActivationType.ReLU);
        relu.Synchronize();
        using (var result = torch.nn.functional.relu(_torchVectorsA[VectorSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        using var sigmoid = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[0]], FusedActivationType.Sigmoid);
        sigmoid.Synchronize();
        using (var result = torch.nn.functional.sigmoid(_torchVectorsA[VectorSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        using var sum = gpuEngine.SumAxisGpu(_aiGpuVectorsA[VectorSizes[0]], 0);
        using var mean = gpuEngine.DivideScalarGpu(sum, VectorSizes[0]);
        mean.Synchronize();
        using (var result = torch.sum(_torchVectorsA[VectorSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        using (var result = torch.mean(_torchVectorsA[VectorSizes[0]]))
        {
            ConsumeTorchResult(result);
        }

        if (_aiGpuConvInput == null || _aiConvKernel == null)
        {
            throw new InvalidOperationException("Conv2D tensors were not initialized.");
        }

        using var conv = gpuEngine.FusedConv2DGpu(_aiGpuConvInput, _aiConvKernel, null,
            _convStride, _convStride, _convPadding, _convPadding, _convDilation, _convDilation,
            FusedActivationType.None);
        conv.Synchronize();

        var torchConvInput = _torchConvInput ?? throw new InvalidOperationException("TorchSharp Conv2D input was not initialized.");
        var torchConvKernel = _torchConvKernel ?? throw new InvalidOperationException("TorchSharp Conv2D kernel was not initialized.");
        var torchConvStride = _torchConvStride ?? throw new InvalidOperationException("TorchSharp Conv2D stride was not initialized.");
        var torchConvPadding = _torchConvPadding ?? throw new InvalidOperationException("TorchSharp Conv2D padding was not initialized.");
        var torchConvDilation = _torchConvDilation ?? throw new InvalidOperationException("TorchSharp Conv2D dilation was not initialized.");
        using (var result = torch.nn.functional.conv2d(torchConvInput, torchConvKernel,
            strides: torchConvStride, padding: torchConvPadding, dilation: torchConvDilation))
        {
            ConsumeTorchResult(result);
        }

        gpuBackend.Add(_aiGpuVectorsA[VectorSizes[0]].Buffer, _aiGpuVectorsB[VectorSizes[0]].Buffer, _aiGpuAddOutputs[VectorSizes[0]], VectorSizes[0]);
        gpuBackend.Multiply(_aiGpuVectorsA[VectorSizes[0]].Buffer, _aiGpuVectorsB[VectorSizes[0]].Buffer, _aiGpuMultiplyOutputs[VectorSizes[0]], VectorSizes[0]);
        gpuBackend.Synchronize();
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

        _torchConvInput = torch.tensor(inputData, new long[] { batch, inChannels, height, width }, device: _torchDevice);
        _torchConvKernel = torch.tensor(kernelData, new long[] { outChannels, inChannels, kernelSize, kernelSize }, device: _torchDevice);

        _torchConvStride = new[] { (long)_convStride, _convStride };
        _torchConvPadding = new[] { (long)_convPadding, _convPadding };
        _torchConvDilation = new[] { (long)_convDilation, _convDilation };
    }

    private void ConsumeTorchResult(TorchTensor result)
    {
        if (_torchUsesCuda)
        {
            torch.cuda.synchronize();
        }

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
    public void AiDotNet_TensorMatMul_GpuResident(int size)
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.MatMulGpuTensors(_aiGpuMatricesA[size], _aiGpuMatricesB[size]);
        result.Synchronize();
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
    public void AiDotNet_TensorAdd_GpuResident(int size)
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.AddGpu(_aiGpuVectorsA[size], _aiGpuVectorsB[size]);
        result.Synchronize();
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void AiDotNet_TensorAdd_GpuResident_ReusedOutput(int size)
    {
        var gpuBackend = _gpuBackend ?? throw new InvalidOperationException("GPU backend was not initialized.");
        gpuBackend.Add(_aiGpuVectorsA[size].Buffer, _aiGpuVectorsB[size].Buffer, _aiGpuAddOutputs[size], size);
        gpuBackend.Synchronize();
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void TorchSharp_Add(int size)
    {
        using var result = torch.add(_torchVectorsA[size], _torchVectorsB[size]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void AiDotNet_TensorMultiply_GpuResident(int size)
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.MultiplyGpu(_aiGpuVectorsA[size], _aiGpuVectorsB[size]);
        result.Synchronize();
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void AiDotNet_TensorMultiply_GpuResident_ReusedOutput(int size)
    {
        var gpuBackend = _gpuBackend ?? throw new InvalidOperationException("GPU backend was not initialized.");
        gpuBackend.Multiply(_aiGpuVectorsA[size].Buffer, _aiGpuVectorsB[size].Buffer, _aiGpuMultiplyOutputs[size], size);
        gpuBackend.Synchronize();
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void TorchSharp_Multiply(int size)
    {
        using var result = torch.mul(_torchVectorsA[size], _torchVectorsB[size]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public void AiDotNet_ReLU_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[1]], FusedActivationType.ReLU);
        result.Synchronize();
    }

    [Benchmark]
    public void TorchSharp_ReLU()
    {
        using var result = torch.nn.functional.relu(_torchVectorsA[VectorSizes[1]]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public void AiDotNet_Sigmoid_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[1]], FusedActivationType.Sigmoid);
        result.Synchronize();
    }

    [Benchmark]
    public void TorchSharp_Sigmoid()
    {
        using var result = torch.nn.functional.sigmoid(_torchVectorsA[VectorSizes[1]]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public void AiDotNet_TensorSum_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var sum = gpuEngine.SumAxisGpu(_aiGpuVectorsA[VectorSizes[1]], 0);
        sum.Synchronize();
    }

    [Benchmark]
    public void TorchSharp_Sum()
    {
        using var result = torch.sum(_torchVectorsA[VectorSizes[1]]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public void AiDotNet_TensorMean_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var sum = gpuEngine.SumAxisGpu(_aiGpuVectorsA[VectorSizes[1]], 0);
        using var mean = gpuEngine.DivideScalarGpu(sum, VectorSizes[1]);
        mean.Synchronize();
    }

    [Benchmark]
    public void TorchSharp_Mean()
    {
        using var result = torch.mean(_torchVectorsA[VectorSizes[1]]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public void AiDotNet_Conv2D_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        if (_aiGpuConvInput == null || _aiConvKernel == null)
        {
            throw new InvalidOperationException("Conv2D tensors were not initialized.");
        }

        using var result = gpuEngine.FusedConv2DGpu(_aiGpuConvInput, _aiConvKernel, null,
            _convStride, _convStride, _convPadding, _convPadding, _convDilation, _convDilation,
            FusedActivationType.None);
        result.Synchronize();
    }

    [Benchmark]
    public void TorchSharp_Conv2D()
    {
        var torchConvInput = _torchConvInput ?? throw new InvalidOperationException("TorchSharp Conv2D input was not initialized.");
        var torchConvKernel = _torchConvKernel ?? throw new InvalidOperationException("TorchSharp Conv2D kernel was not initialized.");
        var torchConvStride = _torchConvStride ?? throw new InvalidOperationException("TorchSharp Conv2D stride was not initialized.");
        var torchConvPadding = _torchConvPadding ?? throw new InvalidOperationException("TorchSharp Conv2D padding was not initialized.");
        var torchConvDilation = _torchConvDilation ?? throw new InvalidOperationException("TorchSharp Conv2D dilation was not initialized.");
        using var result = torch.nn.functional.conv2d(torchConvInput, torchConvKernel,
            strides: torchConvStride, padding: torchConvPadding, dilation: torchConvDilation);
        ConsumeTorchResult(result);
    }
}
#endif
