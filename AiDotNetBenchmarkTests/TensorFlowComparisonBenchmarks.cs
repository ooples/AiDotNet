#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using static Tensorflow.Binding;

namespace AiDotNetBenchmarkTests;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 5)]
public class TensorFlowComparisonBenchmarks
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

    private readonly Dictionary<int, Tensorflow.Tensor> _tfMatricesA = new();
    private readonly Dictionary<int, Tensorflow.Tensor> _tfMatricesB = new();
    private readonly Dictionary<int, Tensorflow.Tensor> _tfVectorsA = new();
    private readonly Dictionary<int, Tensorflow.Tensor> _tfVectorsB = new();

    private Tensor<float>? _aiConvInput;
    private Tensor<float>? _aiConvKernel;
    private IGpuTensor<float>? _aiGpuConvInput;
    private Tensorflow.Tensor? _tfConvInput;
    private Tensorflow.Tensor? _tfConvKernel;

    private int _convStride;
    private int _convPadding;
    private int _convDilation;
    private int[]? _tfConvStrides;
    private string _tfConvPadding = "SAME";

    private DirectGpuTensorEngine? _gpuEngine;
    private IDirectGpuBackend? _gpuBackend;

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

        var gpus = tf.config.list_physical_devices("GPU");
        if (gpus == null || gpus.Length == 0)
        {
            throw new InvalidOperationException("TensorFlow.NET GPU device is not available. Run CPU comparison benchmarks instead.");
        }

        foreach (var size in MatrixSizes)
        {
            var dataA = CreateData(size * size, seedOffset: size);
            var dataB = CreateData(size * size, seedOffset: size + 13_337);

            _aiMatricesA[size] = new Tensor<float>(dataA, new[] { size, size });
            _aiMatricesB[size] = new Tensor<float>(dataB, new[] { size, size });
            _aiGpuMatricesA[size] = _gpuEngine.UploadToGpu(_aiMatricesA[size], GpuTensorRole.Activation);
            _aiGpuMatricesB[size] = _gpuEngine.UploadToGpu(_aiMatricesB[size], GpuTensorRole.Activation);

            _tfMatricesA[size] = tf.constant(dataA, shape: new Tensorflow.Shape(size, size));
            _tfMatricesB[size] = tf.constant(dataB, shape: new Tensorflow.Shape(size, size));
        }

        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            var dataB = CreateData(size, seedOffset: size + 99_999);

            _aiVectorsA[size] = new Tensor<float>(dataA, new[] { size });
            _aiVectorsB[size] = new Tensor<float>(dataB, new[] { size });
            _aiGpuVectorsA[size] = _gpuEngine.UploadToGpu(_aiVectorsA[size], GpuTensorRole.Activation);
            _aiGpuVectorsB[size] = _gpuEngine.UploadToGpu(_aiVectorsB[size], GpuTensorRole.Activation);

            _tfVectorsA[size] = tf.constant(dataA, shape: new Tensorflow.Shape(size));
            _tfVectorsB[size] = tf.constant(dataB, shape: new Tensorflow.Shape(size));
        }

        InitializeConv2D();

        if (_aiConvInput == null || _aiConvKernel == null)
        {
            throw new InvalidOperationException("Conv2D tensors were not initialized.");
        }

        _aiGpuConvInput = _gpuEngine.UploadToGpu(_aiConvInput, GpuTensorRole.Activation);
        if (_tfConvInput == null || _tfConvKernel == null)
        {
            throw new InvalidOperationException("TensorFlow Conv2D tensors were not initialized.");
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

        foreach (var tensor in _tfMatricesA.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _tfMatricesB.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _tfVectorsA.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _tfVectorsB.Values)
        {
            tensor.Dispose();
        }

        _tfConvInput?.Dispose();
        _tfConvKernel?.Dispose();
    }

    private void Warmup()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        var gpuBackend = _gpuBackend ?? throw new InvalidOperationException("GPU backend was not initialized.");

        using var matmul = gpuEngine.MatMulGpuTensors(_aiGpuMatricesA[MatrixSizes[0]], _aiGpuMatricesB[MatrixSizes[0]]);
        matmul.Synchronize();
        var tfMatMul = tf.matmul(_tfMatricesA[MatrixSizes[0]], _tfMatricesB[MatrixSizes[0]]);
        tfMatMul.numpy();

        using var add = gpuEngine.AddGpu(_aiGpuVectorsA[VectorSizes[0]], _aiGpuVectorsB[VectorSizes[0]]);
        add.Synchronize();
        var tfAdd = tf.add(_tfVectorsA[VectorSizes[0]], _tfVectorsB[VectorSizes[0]]);
        tfAdd.numpy();

        using var multiply = gpuEngine.MultiplyGpu(_aiGpuVectorsA[VectorSizes[0]], _aiGpuVectorsB[VectorSizes[0]]);
        multiply.Synchronize();
        var tfMultiply = tf.multiply(_tfVectorsA[VectorSizes[0]], _tfVectorsB[VectorSizes[0]]);
        tfMultiply.numpy();

        using var relu = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[0]], FusedActivationType.ReLU);
        relu.Synchronize();
        var tfRelu = tf.nn.relu(_tfVectorsA[VectorSizes[0]]);
        tfRelu.numpy();

        using var sigmoid = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[0]], FusedActivationType.Sigmoid);
        sigmoid.Synchronize();
        var tfSigmoid = tf.sigmoid(_tfVectorsA[VectorSizes[0]]);
        tfSigmoid.numpy();

        using var sum = gpuEngine.SumAxisGpu(_aiGpuVectorsA[VectorSizes[0]], 0);
        using var mean = gpuEngine.DivideScalarGpu(sum, VectorSizes[0]);
        mean.Synchronize();
        var tfSum = tf.reduce_sum(_tfVectorsA[VectorSizes[0]]);
        tfSum.numpy();
        var tfMean = tf.reduce_mean(_tfVectorsA[VectorSizes[0]]);
        tfMean.numpy();

        if (_aiGpuConvInput == null || _aiConvKernel == null)
        {
            throw new InvalidOperationException("Conv2D tensors were not initialized.");
        }

        using var conv = gpuEngine.FusedConv2DGpu(_aiGpuConvInput, _aiConvKernel, null,
            _convStride, _convStride, _convPadding, _convPadding, _convDilation, _convDilation,
            FusedActivationType.None);
        conv.Synchronize();
        var tfConvInput = _tfConvInput ?? throw new InvalidOperationException("TensorFlow Conv2D input was not initialized.");
        var tfConvKernel = _tfConvKernel ?? throw new InvalidOperationException("TensorFlow Conv2D kernel was not initialized.");
        var tfConvStrides = _tfConvStrides ?? throw new InvalidOperationException("TensorFlow Conv2D strides were not initialized.");
        var tfConvResult = tf.nn.conv2d(tfConvInput, tfConvKernel, strides: tfConvStrides, padding: _tfConvPadding);
        tfConvResult.numpy();

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

        var tfInputData = CreateData(batch * height * width * inChannels, seedOffset: 7_777);
        var tfKernelData = CreateData(kernelSize * kernelSize * inChannels * outChannels, seedOffset: 9_999);

        _tfConvInput = tf.constant(tfInputData, shape: new Tensorflow.Shape(batch, height, width, inChannels));
        _tfConvKernel = tf.constant(tfKernelData, shape: new Tensorflow.Shape(kernelSize, kernelSize, inChannels, outChannels));

        _tfConvStrides = new[] { 1, _convStride, _convStride, 1 };
        _tfConvPadding = "SAME";
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
    public Tensorflow.Tensor TensorFlow_MatMul(int size)
    {
        var result = tf.matmul(_tfMatricesA[size], _tfMatricesB[size]);
        result.numpy();
        return result;
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
    public Tensorflow.Tensor TensorFlow_Add(int size)
    {
        var result = tf.add(_tfVectorsA[size], _tfVectorsB[size]);
        result.numpy();
        return result;
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
    public Tensorflow.Tensor TensorFlow_Multiply(int size)
    {
        var result = tf.multiply(_tfVectorsA[size], _tfVectorsB[size]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public void AiDotNet_ReLU_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[1]], FusedActivationType.ReLU);
        result.Synchronize();
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_ReLU()
    {
        var result = tf.nn.relu(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public void AiDotNet_Sigmoid_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var result = gpuEngine.ActivationGpu(_aiGpuVectorsA[VectorSizes[1]], FusedActivationType.Sigmoid);
        result.Synchronize();
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_Sigmoid()
    {
        var result = tf.sigmoid(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public void AiDotNet_TensorSum_GpuResident()
    {
        var gpuEngine = _gpuEngine ?? throw new InvalidOperationException("GPU engine was not initialized.");
        using var sum = gpuEngine.SumAxisGpu(_aiGpuVectorsA[VectorSizes[1]], 0);
        sum.Synchronize();
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_ReduceSum()
    {
        var result = tf.reduce_sum(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
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
    public Tensorflow.Tensor TensorFlow_ReduceMean()
    {
        var result = tf.reduce_mean(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
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
    public Tensorflow.Tensor TensorFlow_Conv2D()
    {
        var tfConvInput = _tfConvInput ?? throw new InvalidOperationException("TensorFlow Conv2D input was not initialized.");
        var tfConvKernel = _tfConvKernel ?? throw new InvalidOperationException("TensorFlow Conv2D kernel was not initialized.");
        var tfConvStrides = _tfConvStrides ?? throw new InvalidOperationException("TensorFlow Conv2D strides were not initialized.");
        var result = tf.nn.conv2d(tfConvInput, tfConvKernel, strides: tfConvStrides, padding: _tfConvPadding);
        result.numpy();
        return result;
    }
}
#endif
