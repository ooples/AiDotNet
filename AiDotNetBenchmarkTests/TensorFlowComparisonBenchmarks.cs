#if NET8_0
using AiDotNet.Tensors.Engines;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using static Tensorflow.Binding;

namespace AiDotNetBenchmarkTests;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 3, iterationCount: 5)]
public class TensorFlowComparisonBenchmarks
{
    private static readonly int[] MatrixSizes = [256, 512];
    private static readonly int[] VectorSizes = [100_000, 1_000_000];

    private readonly Dictionary<int, Tensor<float>> _aiMatricesA = new();
    private readonly Dictionary<int, Tensor<float>> _aiMatricesB = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsA = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsB = new();

    private readonly Dictionary<int, Tensorflow.Tensor> _tfMatricesA = new();
    private readonly Dictionary<int, Tensorflow.Tensor> _tfMatricesB = new();
    private readonly Dictionary<int, Tensorflow.Tensor> _tfVectorsA = new();
    private readonly Dictionary<int, Tensorflow.Tensor> _tfVectorsB = new();

    private Tensor<float>? _aiConvInput;
    private Tensor<float>? _aiConvKernel;
    private Tensorflow.Tensor? _tfConvInput;
    private Tensorflow.Tensor? _tfConvKernel;

    private int _convStride;
    private int _convPadding;
    private int _convDilation;
    private int[]? _tfConvStrides;
    private string _tfConvPadding = "SAME";

    [GlobalSetup]
    public void Setup()
    {
        AiDotNetEngine.Current = new GpuEngine(AdaptiveThresholds.AlwaysGpu);

        foreach (var size in MatrixSizes)
        {
            var dataA = CreateData(size * size, seedOffset: size);
            var dataB = CreateData(size * size, seedOffset: size + 13_337);

            _aiMatricesA[size] = new Tensor<float>(dataA, new[] { size, size });
            _aiMatricesB[size] = new Tensor<float>(dataB, new[] { size, size });

            _tfMatricesA[size] = tf.constant(dataA, shape: new Tensorflow.Shape(size, size));
            _tfMatricesB[size] = tf.constant(dataB, shape: new Tensorflow.Shape(size, size));
        }

        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            var dataB = CreateData(size, seedOffset: size + 99_999);

            _aiVectorsA[size] = new Tensor<float>(dataA, new[] { size });
            _aiVectorsB[size] = new Tensor<float>(dataB, new[] { size });

            _tfVectorsA[size] = tf.constant(dataA, shape: new Tensorflow.Shape(size));
            _tfVectorsB[size] = tf.constant(dataB, shape: new Tensorflow.Shape(size));
        }

        InitializeConv2D();

        // Warm-up to avoid first-run overhead (kernel JIT, allocator init).
        _ = AiDotNetEngine.Current.TensorMatMul(_aiMatricesA[MatrixSizes[0]], _aiMatricesB[MatrixSizes[0]]);
        _ = tf.matmul(_tfMatricesA[MatrixSizes[0]], _tfMatricesB[MatrixSizes[0]]);
        _ = AiDotNetEngine.Current.TensorAdd(_aiVectorsA[VectorSizes[0]], _aiVectorsB[VectorSizes[0]]);
        _ = tf.add(_tfVectorsA[VectorSizes[0]], _tfVectorsB[VectorSizes[0]]);
        _ = AiDotNetEngine.Current.Conv2D(_aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);
        _ = tf.nn.conv2d(_tfConvInput!, _tfConvKernel!, strides: _tfConvStrides!, padding: _tfConvPadding);
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
    public Tensor<float> AiDotNet_TensorMatMul(int size)
    {
        return AiDotNetEngine.Current.TensorMatMul(_aiMatricesA[size], _aiMatricesB[size]);
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
    public Tensor<float> AiDotNet_TensorAdd(int size)
    {
        return AiDotNetEngine.Current.TensorAdd(_aiVectorsA[size], _aiVectorsB[size]);
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
    public Tensor<float> AiDotNet_TensorMultiply(int size)
    {
        return AiDotNetEngine.Current.TensorMultiply(_aiVectorsA[size], _aiVectorsB[size]);
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
    public Tensor<float> AiDotNet_ReLU()
    {
        return AiDotNetEngine.Current.ReLU(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_ReLU()
    {
        var result = tf.nn.relu(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Sigmoid()
    {
        return AiDotNetEngine.Current.Sigmoid(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_Sigmoid()
    {
        var result = tf.sigmoid(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public float AiDotNet_TensorSum()
    {
        return AiDotNetEngine.Current.TensorSum(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_ReduceSum()
    {
        var result = tf.reduce_sum(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public float AiDotNet_TensorMean()
    {
        return AiDotNetEngine.Current.TensorMean(_aiVectorsA[VectorSizes[1]]);
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_ReduceMean()
    {
        var result = tf.reduce_mean(_tfVectorsA[VectorSizes[1]]);
        result.numpy();
        return result;
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Conv2D()
    {
        return AiDotNetEngine.Current.Conv2D(_aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);
    }

    [Benchmark]
    public Tensorflow.Tensor TensorFlow_Conv2D()
    {
        var result = tf.nn.conv2d(_tfConvInput!, _tfConvKernel!, strides: _tfConvStrides!, padding: _tfConvPadding);
        result.numpy();
        return result;
    }
}
#endif
