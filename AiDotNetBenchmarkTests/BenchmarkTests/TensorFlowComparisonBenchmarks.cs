#if NET8_0
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Tensorflow;
using static Tensorflow.Binding;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks comparing AiDotNet Tensor operations with TensorFlow.NET
/// NOTE: Only runs on .NET 8.0 due to TensorFlow.NET requirements
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, baseline: true)]
public class TensorFlowComparisonBenchmarks
{
    [Params(100, 500)]
    public int BatchSize { get; set; }

    [Params(64, 256)]
    public int FeatureSize { get; set; }

    private Tensor<double> _aiTensorA = null!;
    private Tensor<double> _aiTensorB = null!;
    private NDArray _tfTensorA = null!;
    private NDArray _tfTensorB = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize AiDotNet tensors
        _aiTensorA = new Tensor<double>(new[] { BatchSize, FeatureSize });
        _aiTensorB = new Tensor<double>(new[] { BatchSize, FeatureSize });

        for (int i = 0; i < _aiTensorA.Length; i++)
        {
            _aiTensorA[i] = random.NextDouble() * 2 - 1;
            _aiTensorB[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize TensorFlow.NET tensors
        var tfDataA = new float[BatchSize, FeatureSize];
        var tfDataB = new float[BatchSize, FeatureSize];

        for (int i = 0; i < BatchSize; i++)
        {
            for (int j = 0; j < FeatureSize; j++)
            {
                tfDataA[i, j] = (float)(random.NextDouble() * 2 - 1);
                tfDataB[i, j] = (float)(random.NextDouble() * 2 - 1);
            }
        }

        _tfTensorA = np.array(tfDataA);
        _tfTensorB = np.array(tfDataB);
    }

    #region Tensor Addition

    [Benchmark(Baseline = true)]
    public Tensor<double> AiDotNet_TensorAdd()
    {
        return _aiTensorA.Add(_aiTensorB);
    }

    [Benchmark]
    public NDArray TensorFlow_TensorAdd()
    {
        return tf.add(_tfTensorA, _tfTensorB);
    }

    #endregion

    #region Tensor Multiplication

    [Benchmark]
    public Tensor<double> AiDotNet_TensorMultiply()
    {
        return _aiTensorA.Multiply(_aiTensorB);
    }

    [Benchmark]
    public NDArray TensorFlow_TensorMultiply()
    {
        return tf.multiply(_tfTensorA, _tfTensorB);
    }

    #endregion

    #region Tensor MatMul

    [Benchmark]
    public Tensor<double> AiDotNet_MatMul()
    {
        // Transpose B for matrix multiplication
        var transposedB = _aiTensorB.Transpose(new[] { 1, 0 });
        return _aiTensorA.MatMul(transposedB);
    }

    [Benchmark]
    public NDArray TensorFlow_MatMul()
    {
        var transposedB = tf.transpose(_tfTensorB);
        return tf.matmul(_tfTensorA, transposedB);
    }

    #endregion

    #region Reduction Operations

    [Benchmark]
    public double AiDotNet_ReduceSum()
    {
        return _aiTensorA.Sum();
    }

    [Benchmark]
    public NDArray TensorFlow_ReduceSum()
    {
        return tf.reduce_sum(_tfTensorA);
    }

    [Benchmark]
    public double AiDotNet_ReduceMean()
    {
        return _aiTensorA.Mean();
    }

    [Benchmark]
    public NDArray TensorFlow_ReduceMean()
    {
        return tf.reduce_mean(_tfTensorA);
    }

    #endregion

    #region Activation Functions

    [Benchmark]
    public Tensor<double> AiDotNet_ReLU()
    {
        return _aiTensorA.Transform((x, _) => Math.Max(0, x));
    }

    [Benchmark]
    public NDArray TensorFlow_ReLU()
    {
        return tf.nn.relu(_tfTensorA);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Sigmoid()
    {
        return _aiTensorA.Transform((x, _) => 1.0 / (1.0 + Math.Exp(-x)));
    }

    [Benchmark]
    public NDArray TensorFlow_Sigmoid()
    {
        return tf.nn.sigmoid(_tfTensorA);
    }

    #endregion

    #region Tensor Reshape

    [Benchmark]
    public Tensor<double> AiDotNet_Reshape()
    {
        return _aiTensorA.Reshape(new[] { BatchSize * FeatureSize });
    }

    [Benchmark]
    public NDArray TensorFlow_Reshape()
    {
        return tf.reshape(_tfTensorA, new Shape(BatchSize * FeatureSize));
    }

    #endregion
}
#endif
