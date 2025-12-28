using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Extensions;
using AiDotNet.Tensors.Helpers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NumSharp;
using System.Numerics.Tensors;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks comparing AiDotNet linear algebra operations against multiple baselines:
/// - MathNet.Numerics (popular .NET numerical library)
/// - NumSharp (NumPy-like library for .NET)
/// - System.Numerics.Tensors (built-in .NET SIMD primitives)
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
[RPlotExporter]
public class LinearAlgebraBenchmarks
{
    // Raw arrays for all libraries
    private double[] _data1 = null!;
    private double[] _data2 = null!;
    private double[] _result = null!;

    // AiDotNet types
    private AiDotNet.Tensors.LinearAlgebra.Vector<double> _aiVector1 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Vector<double> _aiVector2 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiMatrix1 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiMatrix2 = null!;

    // MathNet types
    private MathNet.Numerics.LinearAlgebra.Vector<double> _mnVector1 = null!;
    private MathNet.Numerics.LinearAlgebra.Vector<double> _mnVector2 = null!;
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnMatrix1 = null!;
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnMatrix2 = null!;

    // NumSharp types
    private NDArray _nsVector1 = null!;
    private NDArray _nsVector2 = null!;
    private NDArray _nsMatrix1 = null!;
    private NDArray _nsMatrix2 = null!;

    [Params(100, 500, 1000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize raw arrays
        _data1 = new double[N];
        _data2 = new double[N];
        _result = new double[N];
        for (int i = 0; i < N; i++)
        {
            _data1[i] = random.NextDouble() * 100;
            _data2[i] = random.NextDouble() * 100;
        }

        // Initialize AiDotNet vectors
        _aiVector1 = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(_data1);
        _aiVector2 = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(_data2);

        // Initialize MathNet vectors
        _mnVector1 = DenseVector.OfArray(_data1);
        _mnVector2 = DenseVector.OfArray(_data2);

        // Initialize NumSharp vectors
        _nsVector1 = np.array(_data1);
        _nsVector2 = np.array(_data2);

        // Initialize matrices
        var matData1 = new double[N, N];
        var matData2 = new double[N, N];
        var flatMat1 = new double[N * N];
        var flatMat2 = new double[N * N];

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                var val1 = random.NextDouble() * 10;
                var val2 = random.NextDouble() * 10;
                matData1[i, j] = val1;
                matData2[i, j] = val2;
                flatMat1[i * N + j] = val1;
                flatMat2[i * N + j] = val2;
            }
        }

        _aiMatrix1 = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(matData1);
        _aiMatrix2 = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(matData2);
        _mnMatrix1 = DenseMatrix.OfArray(matData1);
        _mnMatrix2 = DenseMatrix.OfArray(matData2);
        _nsMatrix1 = np.array(flatMat1).reshape(N, N);
        _nsMatrix2 = np.array(flatMat2).reshape(N, N);
    }

    #region Vector Dot Product

    [Benchmark(Description = "Dot Product - AiDotNet")]
    [BenchmarkCategory("VectorDot")]
    public double VectorDotAiDotNet()
    {
        return _aiVector1.DotProduct(_aiVector2);
    }

    [Benchmark(Description = "Dot Product - MathNet", Baseline = true)]
    [BenchmarkCategory("VectorDot")]
    public double VectorDotMathNet()
    {
        return _mnVector1.DotProduct(_mnVector2);
    }

    [Benchmark(Description = "Dot Product - NumSharp")]
    [BenchmarkCategory("VectorDot")]
    public double VectorDotNumSharp()
    {
        return (double)np.dot(_nsVector1, _nsVector2);
    }

    [Benchmark(Description = "Dot Product - TensorPrimitives")]
    [BenchmarkCategory("VectorDot")]
    public double VectorDotTensorPrimitives()
    {
        return TensorPrimitives.Dot<double>(_data1, _data2);
    }

    #endregion

    #region Vector Add

    [Benchmark(Description = "Vector Add - AiDotNet")]
    [BenchmarkCategory("VectorAdd")]
    public AiDotNet.Tensors.LinearAlgebra.Vector<double> VectorAddAiDotNet()
    {
        return (AiDotNet.Tensors.LinearAlgebra.Vector<double>)_aiVector1.Add(_aiVector2);
    }

    [Benchmark(Description = "Vector Add - MathNet")]
    [BenchmarkCategory("VectorAdd")]
    public MathNet.Numerics.LinearAlgebra.Vector<double> VectorAddMathNet()
    {
        return _mnVector1.Add(_mnVector2);
    }

    [Benchmark(Description = "Vector Add - NumSharp")]
    [BenchmarkCategory("VectorAdd")]
    public NDArray VectorAddNumSharp()
    {
        return _nsVector1 + _nsVector2;
    }

    [Benchmark(Description = "Vector Add - TensorPrimitives")]
    [BenchmarkCategory("VectorAdd")]
    public void VectorAddTensorPrimitives()
    {
        TensorPrimitives.Add<double>(_data1, _data2, _result);
    }

    #endregion

    #region Vector Add In-Place (Zero Allocation)

    [Benchmark(Description = "Vector AddInPlace - AiDotNet")]
    [BenchmarkCategory("VectorAddInPlace")]
    public void VectorAddInPlaceAiDotNet()
    {
        _aiVector1.AddInPlace(_aiVector2);
    }

    [Benchmark(Description = "Vector Add to Span - AiDotNet")]
    [BenchmarkCategory("VectorAddInPlace")]
    public void VectorAddToSpanAiDotNet()
    {
        _aiVector1.Add(_aiVector2, _result.AsSpan());
    }

    [Benchmark(Description = "Vector Add - TensorPrimitives")]
    [BenchmarkCategory("VectorAddInPlace")]
    public void VectorAddInPlaceTensorPrimitives()
    {
        TensorPrimitives.Add<double>(_data1, _data2, _result);
    }

    #endregion

    #region Vector Subtract

    [Benchmark(Description = "Vector Subtract - AiDotNet")]
    [BenchmarkCategory("VectorSubtract")]
    public AiDotNet.Tensors.LinearAlgebra.Vector<double> VectorSubtractAiDotNet()
    {
        return (AiDotNet.Tensors.LinearAlgebra.Vector<double>)_aiVector1.Subtract(_aiVector2);
    }

    [Benchmark(Description = "Vector Subtract - MathNet")]
    [BenchmarkCategory("VectorSubtract")]
    public MathNet.Numerics.LinearAlgebra.Vector<double> VectorSubtractMathNet()
    {
        return _mnVector1.Subtract(_mnVector2);
    }

    [Benchmark(Description = "Vector Subtract - TensorPrimitives")]
    [BenchmarkCategory("VectorSubtract")]
    public void VectorSubtractTensorPrimitives()
    {
        TensorPrimitives.Subtract<double>(_data1, _data2, _result);
    }

    #endregion

    #region Vector Scalar Multiply

    [Benchmark(Description = "Vector Scalar Multiply - AiDotNet")]
    [BenchmarkCategory("VectorScalarMul")]
    public AiDotNet.Tensors.LinearAlgebra.Vector<double> VectorScalarMultiplyAiDotNet()
    {
        return (AiDotNet.Tensors.LinearAlgebra.Vector<double>)_aiVector1.Multiply(2.5);
    }

    [Benchmark(Description = "Vector Scalar Multiply - MathNet")]
    [BenchmarkCategory("VectorScalarMul")]
    public MathNet.Numerics.LinearAlgebra.Vector<double> VectorScalarMultiplyMathNet()
    {
        return _mnVector1.Multiply(2.5);
    }

    [Benchmark(Description = "Vector Scalar Multiply - TensorPrimitives")]
    [BenchmarkCategory("VectorScalarMul")]
    public void VectorScalarMultiplyTensorPrimitives()
    {
        TensorPrimitives.Multiply<double>(_data1, 2.5, _result);
    }

    #endregion

    #region Vector L2 Norm

    [Benchmark(Description = "L2 Norm - AiDotNet")]
    [BenchmarkCategory("VectorNorm")]
    public double VectorNormAiDotNet()
    {
        return _aiVector1.Norm();
    }

    [Benchmark(Description = "L2 Norm - MathNet")]
    [BenchmarkCategory("VectorNorm")]
    public double VectorNormMathNet()
    {
        return _mnVector1.L2Norm();
    }

    [Benchmark(Description = "L2 Norm - TensorPrimitives")]
    [BenchmarkCategory("VectorNorm")]
    public double VectorNormTensorPrimitives()
    {
        return TensorPrimitives.Norm(_data1.AsSpan());
    }

    #endregion

    #region Matrix Multiply

    [Benchmark(Description = "Matrix Multiply - AiDotNet")]
    [BenchmarkCategory("MatrixMul")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> MatrixMultiplyAiDotNet()
    {
        return _aiMatrix1.Multiply(_aiMatrix2);
    }

    [Benchmark(Description = "Matrix Multiply - MathNet")]
    [BenchmarkCategory("MatrixMul")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> MatrixMultiplyMathNet()
    {
        return _mnMatrix1.Multiply(_mnMatrix2);
    }

    [Benchmark(Description = "Matrix Multiply - NumSharp")]
    [BenchmarkCategory("MatrixMul")]
    public NDArray MatrixMultiplyNumSharp()
    {
        return np.matmul(_nsMatrix1, _nsMatrix2);
    }

    #endregion

    #region Matrix Add

    [Benchmark(Description = "Matrix Add - AiDotNet")]
    [BenchmarkCategory("MatrixAdd")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> MatrixAddAiDotNet()
    {
        return _aiMatrix1.Add(_aiMatrix2);
    }

    [Benchmark(Description = "Matrix Add - MathNet")]
    [BenchmarkCategory("MatrixAdd")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> MatrixAddMathNet()
    {
        return _mnMatrix1.Add(_mnMatrix2);
    }

    [Benchmark(Description = "Matrix Add - NumSharp")]
    [BenchmarkCategory("MatrixAdd")]
    public NDArray MatrixAddNumSharp()
    {
        return _nsMatrix1 + _nsMatrix2;
    }

    #endregion

    #region Matrix Add In-Place (Zero Allocation)

    [Benchmark(Description = "Matrix AddInPlace - AiDotNet")]
    [BenchmarkCategory("MatrixAddInPlace")]
    public void MatrixAddInPlaceAiDotNet()
    {
        _aiMatrix1.AddInPlace(_aiMatrix2);
    }

    #endregion

    #region Matrix Subtract

    [Benchmark(Description = "Matrix Subtract - AiDotNet")]
    [BenchmarkCategory("MatrixSubtract")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> MatrixSubtractAiDotNet()
    {
        return _aiMatrix1.Subtract(_aiMatrix2);
    }

    [Benchmark(Description = "Matrix Subtract - MathNet")]
    [BenchmarkCategory("MatrixSubtract")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> MatrixSubtractMathNet()
    {
        return _mnMatrix1.Subtract(_mnMatrix2);
    }

    #endregion

    #region Matrix Scalar Multiply

    [Benchmark(Description = "Matrix Scalar Multiply - AiDotNet")]
    [BenchmarkCategory("MatrixScalarMul")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> MatrixScalarMultiplyAiDotNet()
    {
        return _aiMatrix1.Multiply(2.5);
    }

    [Benchmark(Description = "Matrix Scalar Multiply - MathNet")]
    [BenchmarkCategory("MatrixScalarMul")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> MatrixScalarMultiplyMathNet()
    {
        return _mnMatrix1.Multiply(2.5);
    }

    #endregion

    #region Matrix Transpose

    [Benchmark(Description = "Transpose - AiDotNet")]
    [BenchmarkCategory("Transpose")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> MatrixTransposeAiDotNet()
    {
        return _aiMatrix1.Transpose();
    }

    [Benchmark(Description = "Transpose - MathNet")]
    [BenchmarkCategory("Transpose")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> MatrixTransposeMathNet()
    {
        return _mnMatrix1.Transpose();
    }

    [Benchmark(Description = "Transpose - NumSharp")]
    [BenchmarkCategory("Transpose")]
    public NDArray MatrixTransposeNumSharp()
    {
        return _nsMatrix1.T;
    }

    #endregion

    #region Matrix Transpose In-Place (Zero Allocation - Square Matrices Only)

    [Benchmark(Description = "TransposeInPlace - AiDotNet")]
    [BenchmarkCategory("TransposeInPlace")]
    public void MatrixTransposeInPlaceAiDotNet()
    {
        _aiMatrix1.TransposeInPlace();
    }

    #endregion

    #region Frobenius Norm

    [Benchmark(Description = "Frobenius Norm - AiDotNet")]
    [BenchmarkCategory("FrobNorm")]
    public double FrobeniusNormAiDotNet()
    {
        return _aiMatrix1.FrobeniusNorm();
    }

    [Benchmark(Description = "Frobenius Norm - MathNet")]
    [BenchmarkCategory("FrobNorm")]
    public double FrobeniusNormMathNet()
    {
        return _mnMatrix1.FrobeniusNorm();
    }

    #endregion
}

/// <summary>
/// Benchmarks for small matrix operations where overhead matters more.
/// These sizes are typical for neural network layer computations.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class SmallMatrixBenchmarks
{
    // Small matrices (typical neural network layer sizes)
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiSmall4x4 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiSmall8x8 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiSmall16x16 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiSmall32x32 = null!;

    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnSmall4x4 = null!;
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnSmall8x8 = null!;
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnSmall16x16 = null!;
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnSmall32x32 = null!;

    private NDArray _nsSmall4x4 = null!;
    private NDArray _nsSmall8x8 = null!;
    private NDArray _nsSmall16x16 = null!;
    private NDArray _nsSmall32x32 = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _aiSmall4x4 = CreateAiMatrix(4, 4, random);
        _aiSmall8x8 = CreateAiMatrix(8, 8, random);
        _aiSmall16x16 = CreateAiMatrix(16, 16, random);
        _aiSmall32x32 = CreateAiMatrix(32, 32, random);

        _mnSmall4x4 = CreateMnMatrix(4, 4, random);
        _mnSmall8x8 = CreateMnMatrix(8, 8, random);
        _mnSmall16x16 = CreateMnMatrix(16, 16, random);
        _mnSmall32x32 = CreateMnMatrix(32, 32, random);

        _nsSmall4x4 = CreateNsMatrix(4, 4, random);
        _nsSmall8x8 = CreateNsMatrix(8, 8, random);
        _nsSmall16x16 = CreateNsMatrix(16, 16, random);
        _nsSmall32x32 = CreateNsMatrix(32, 32, random);
    }

    private static AiDotNet.Tensors.LinearAlgebra.Matrix<double> CreateAiMatrix(int rows, int cols, Random random)
    {
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = random.NextDouble() * 10;
        return new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(data);
    }

    private static MathNet.Numerics.LinearAlgebra.Matrix<double> CreateMnMatrix(int rows, int cols, Random random)
    {
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = random.NextDouble() * 10;
        return DenseMatrix.OfArray(data);
    }

    private static NDArray CreateNsMatrix(int rows, int cols, Random random)
    {
        var data = new double[rows * cols];
        for (int i = 0; i < rows * cols; i++)
            data[i] = random.NextDouble() * 10;
        return np.array(data).reshape(rows, cols);
    }

    #region 4x4 Matrix Multiply

    [Benchmark(Description = "4x4 Multiply - AiDotNet")]
    [BenchmarkCategory("Small4x4")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> Small4x4MultiplyAiDotNet()
    {
        return _aiSmall4x4.Multiply(_aiSmall4x4);
    }

    [Benchmark(Description = "4x4 Multiply - MathNet", Baseline = true)]
    [BenchmarkCategory("Small4x4")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> Small4x4MultiplyMathNet()
    {
        return _mnSmall4x4.Multiply(_mnSmall4x4);
    }

    [Benchmark(Description = "4x4 Multiply - NumSharp")]
    [BenchmarkCategory("Small4x4")]
    public NDArray Small4x4MultiplyNumSharp()
    {
        return np.matmul(_nsSmall4x4, _nsSmall4x4);
    }

    #endregion

    #region 16x16 Matrix Multiply

    [Benchmark(Description = "16x16 Multiply - AiDotNet")]
    [BenchmarkCategory("Small16x16")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> Small16x16MultiplyAiDotNet()
    {
        return _aiSmall16x16.Multiply(_aiSmall16x16);
    }

    [Benchmark(Description = "16x16 Multiply - MathNet")]
    [BenchmarkCategory("Small16x16")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> Small16x16MultiplyMathNet()
    {
        return _mnSmall16x16.Multiply(_mnSmall16x16);
    }

    [Benchmark(Description = "16x16 Multiply - NumSharp")]
    [BenchmarkCategory("Small16x16")]
    public NDArray Small16x16MultiplyNumSharp()
    {
        return np.matmul(_nsSmall16x16, _nsSmall16x16);
    }

    #endregion

    #region 32x32 Matrix Multiply

    [Benchmark(Description = "32x32 Multiply - AiDotNet")]
    [BenchmarkCategory("Small32x32")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> Small32x32MultiplyAiDotNet()
    {
        return _aiSmall32x32.Multiply(_aiSmall32x32);
    }

    [Benchmark(Description = "32x32 Multiply - MathNet")]
    [BenchmarkCategory("Small32x32")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> Small32x32MultiplyMathNet()
    {
        return _mnSmall32x32.Multiply(_mnSmall32x32);
    }

    [Benchmark(Description = "32x32 Multiply - NumSharp")]
    [BenchmarkCategory("Small32x32")]
    public NDArray Small32x32MultiplyNumSharp()
    {
        return np.matmul(_nsSmall32x32, _nsSmall32x32);
    }

    #endregion
}

/// <summary>
/// Benchmarks for element-wise operations using TensorPrimitives
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class ElementWiseBenchmarks
{
    private double[] _data1 = null!;
    private double[] _data2 = null!;
    private double[] _result = null!;

    private AiDotNet.Tensors.LinearAlgebra.Vector<double> _aiVector1 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Vector<double> _aiVector2 = null!;

    private NDArray _nsVector1 = null!;
    private NDArray _nsVector2 = null!;

    [Params(1000, 10000, 100000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _data1 = new double[N];
        _data2 = new double[N];
        _result = new double[N];

        for (int i = 0; i < N; i++)
        {
            _data1[i] = random.NextDouble() * 100 + 0.1; // Avoid zero for division
            _data2[i] = random.NextDouble() * 100 + 0.1;
        }

        _aiVector1 = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(_data1);
        _aiVector2 = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(_data2);

        _nsVector1 = np.array(_data1);
        _nsVector2 = np.array(_data2);
    }

    #region Multiply Element-wise

    [Benchmark(Description = "Multiply - TensorPrimitives", Baseline = true)]
    [BenchmarkCategory("Multiply")]
    public void MultiplyTensorPrimitives()
    {
        TensorPrimitives.Multiply<double>(_data1, _data2, _result);
    }

    [Benchmark(Description = "Multiply - NumSharp")]
    [BenchmarkCategory("Multiply")]
    public NDArray MultiplyNumSharp()
    {
        return _nsVector1 * _nsVector2;
    }

    #endregion

    #region Exp

    [Benchmark(Description = "Exp - TensorPrimitives")]
    [BenchmarkCategory("Exp")]
    public void ExpTensorPrimitives()
    {
        TensorPrimitives.Exp<double>(_data1, _result);
    }

    [Benchmark(Description = "Exp - NumSharp")]
    [BenchmarkCategory("Exp")]
    public NDArray ExpNumSharp()
    {
        return np.exp(_nsVector1);
    }

    #endregion

    #region Sum

    [Benchmark(Description = "Sum - TensorPrimitives")]
    [BenchmarkCategory("Sum")]
    public double SumTensorPrimitives()
    {
        return TensorPrimitives.Sum<double>(_data1);
    }

    [Benchmark(Description = "Sum - NumSharp")]
    [BenchmarkCategory("Sum")]
    public double SumNumSharp()
    {
        return (double)np.sum(_nsVector1);
    }

    #endregion

    #region Max

    [Benchmark(Description = "Max - TensorPrimitives")]
    [BenchmarkCategory("Max")]
    public double MaxTensorPrimitives()
    {
        return TensorPrimitives.Max<double>(_data1);
    }

    [Benchmark(Description = "Max - NumSharp")]
    [BenchmarkCategory("Max")]
    public double MaxNumSharp()
    {
        return (double)np.max(_nsVector1);
    }

    #endregion
}
