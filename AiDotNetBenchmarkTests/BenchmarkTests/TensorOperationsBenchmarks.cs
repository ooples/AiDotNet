using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Tensor operations
/// Tests performance of multi-dimensional array operations critical for deep learning
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class TensorOperationsBenchmarks
{
    [Params(32, 128)]
    public int BatchSize { get; set; }

    [Params(64, 256)]
    public int Channels { get; set; }

    [Params(28, 56)]
    public int Height { get; set; }

    private Tensor<double> _tensor4D = null!;  // Batch x Channels x Height x Width
    private Tensor<double> _tensor3D = null!;  // Batch x Sequence x Features
    private Tensor<double> _tensorA = null!;
    private Tensor<double> _tensorB = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        int width = Height; // Square for simplicity

        // Initialize 4D tensor (typical for CNNs: batch x channels x height x width)
        _tensor4D = new Tensor<double>(new[] { BatchSize, Channels, Height, width });
        for (int i = 0; i < _tensor4D.Length; i++)
        {
            _tensor4D[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize 3D tensor (typical for RNNs: batch x sequence x features)
        int seqLen = 50;
        int features = Channels;
        _tensor3D = new Tensor<double>(new[] { BatchSize, seqLen, features });
        for (int i = 0; i < _tensor3D.Length; i++)
        {
            _tensor3D[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize tensors for element-wise operations
        _tensorA = new Tensor<double>(new[] { BatchSize, Channels });
        _tensorB = new Tensor<double>(new[] { BatchSize, Channels });
        for (int i = 0; i < _tensorA.Length; i++)
        {
            _tensorA[i] = random.NextDouble() * 2 - 1;
            _tensorB[i] = random.NextDouble() * 2 - 1;
        }
    }

    #region Tensor Creation and Initialization

    [Benchmark(Baseline = true)]
    public Tensor<double> Tensor_Create_Zeros()
    {
        return Tensor<double>.Zeros(new[] { BatchSize, Channels, Height, Height });
    }

    [Benchmark]
    public Tensor<double> Tensor_Create_Ones()
    {
        return Tensor<double>.Ones(new[] { BatchSize, Channels, Height, Height });
    }

    [Benchmark]
    public Tensor<double> Tensor_Create_Random()
    {
        var tensor = new Tensor<double>(new[] { BatchSize, Channels, Height, Height });
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble();
        }
        return tensor;
    }

    #endregion

    #region Element-wise Operations

    [Benchmark]
    public Tensor<double> Tensor_Add()
    {
        return _tensorA.Add(_tensorB);
    }

    [Benchmark]
    public Tensor<double> Tensor_Subtract()
    {
        return _tensorA.Subtract(_tensorB);
    }

    [Benchmark]
    public Tensor<double> Tensor_Multiply()
    {
        return _tensorA.Multiply(_tensorB);
    }

    [Benchmark]
    public Tensor<double> Tensor_Divide()
    {
        return _tensorA.Divide(_tensorB);
    }

    #endregion

    #region Tensor Reshaping

    [Benchmark]
    public Tensor<double> Tensor_Reshape_2D()
    {
        return _tensor4D.Reshape(new[] { BatchSize, -1 });
    }

    [Benchmark]
    public Tensor<double> Tensor_Flatten()
    {
        return _tensor4D.Flatten();
    }

    [Benchmark]
    public Tensor<double> Tensor_Transpose()
    {
        // Transpose last two dimensions (common in attention mechanisms)
        return _tensor4D.Transpose(new[] { 0, 1, 3, 2 });
    }

    #endregion

    #region Reduction Operations

    [Benchmark]
    public double Tensor_Sum_All()
    {
        return _tensor4D.Sum();
    }

    [Benchmark]
    public Tensor<double> Tensor_Sum_Axis()
    {
        return _tensor4D.Sum(axis: 1); // Sum over channels
    }

    [Benchmark]
    public double Tensor_Mean_All()
    {
        return _tensor4D.Mean();
    }

    [Benchmark]
    public Tensor<double> Tensor_Mean_Axis()
    {
        return _tensor4D.Mean(axis: 1); // Mean over channels
    }

    [Benchmark]
    public double Tensor_Max()
    {
        return _tensor4D.Max();
    }

    [Benchmark]
    public double Tensor_Min()
    {
        return _tensor4D.Min();
    }

    #endregion

    #region Tensor Slicing

    [Benchmark]
    public Tensor<double> Tensor_Slice_Batch()
    {
        return _tensor4D.Slice(0, 0, BatchSize / 2);
    }

    [Benchmark]
    public Tensor<double> Tensor_Slice_Channels()
    {
        return _tensor4D.Slice(1, 0, Channels / 2);
    }

    #endregion

    #region Broadcasting Operations

    [Benchmark]
    public Tensor<double> Tensor_ScalarAdd()
    {
        return _tensor4D.Transform((x, _) => x + 1.0);
    }

    [Benchmark]
    public Tensor<double> Tensor_ScalarMultiply()
    {
        return _tensor4D.Transform((x, _) => x * 2.0);
    }

    #endregion

    #region Advanced Tensor Operations

    [Benchmark]
    public Tensor<double> Tensor_Concatenate()
    {
        return Tensor<double>.Concatenate(new[] { _tensorA, _tensorB }, axis: 0);
    }

    [Benchmark]
    public (Tensor<double>, Tensor<double>) Tensor_Split()
    {
        return _tensor3D.Split(axis: 1, splitPoint: _tensor3D.Shape[1] / 2);
    }

    [Benchmark]
    public Tensor<double> Tensor_Permute()
    {
        // Common permutation in transformers (batch, seq, features) -> (seq, batch, features)
        return _tensor3D.Transpose(new[] { 1, 0, 2 });
    }

    #endregion

    #region Memory-Intensive Operations

    [Benchmark]
    public Tensor<double> Tensor_Clone()
    {
        return _tensor4D.Clone();
    }

    [Benchmark]
    public Tensor<double> Tensor_CopyTo()
    {
        var destination = new Tensor<double>(_tensor4D.Shape);
        _tensor4D.CopyTo(destination);
        return destination;
    }

    #endregion
}
