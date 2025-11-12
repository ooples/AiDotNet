using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Accord.Statistics;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Statistics operations comparing AiDotNet vs Accord.NET
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class StatisticsBenchmarks
{
    [Params(1000, 10000, 100000)]
    public int Size { get; set; }

    private Vector<double> _aiVector = null!;
    private double[] _accordArray = null!;
    private Matrix<double> _aiMatrix = null!;
    private double[,] _accordMatrix = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize vectors/arrays
        _aiVector = new Vector<double>(Size);
        _accordArray = new double[Size];

        for (int i = 0; i < Size; i++)
        {
            var value = random.NextDouble() * 100;
            _aiVector[i] = value;
            _accordArray[i] = value;
        }

        // Initialize matrices
        int rows = Math.Min(100, Size / 10);
        int cols = 10;
        _aiMatrix = new Matrix<double>(rows, cols);
        _accordMatrix = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                var value = random.NextDouble() * 100;
                _aiMatrix[i, j] = value;
                _accordMatrix[i, j] = value;
            }
        }
    }

    #region Mean

    [Benchmark]
    public double AiDotNet_Mean()
    {
        return BasicStats<double>.Mean(_aiVector);
    }

    [Benchmark(Baseline = true)]
    public double AccordNet_Mean()
    {
        return Measures.Mean(_accordArray);
    }

    #endregion

    #region Variance

    [Benchmark]
    public double AiDotNet_Variance()
    {
        return BasicStats<double>.Variance(_aiVector);
    }

    [Benchmark]
    public double AccordNet_Variance()
    {
        return Measures.Variance(_accordArray);
    }

    #endregion

    #region Standard Deviation

    [Benchmark]
    public double AiDotNet_StandardDeviation()
    {
        return BasicStats<double>.StandardDeviation(_aiVector);
    }

    [Benchmark]
    public double AccordNet_StandardDeviation()
    {
        return Measures.StandardDeviation(_accordArray);
    }

    #endregion

    #region Median

    [Benchmark]
    public double AiDotNet_Median()
    {
        return BasicStats<double>.Median(_aiVector);
    }

    [Benchmark]
    public double AccordNet_Median()
    {
        return Measures.Median(_accordArray);
    }

    #endregion

    #region Quartiles

    [Benchmark]
    public (double q1, double q2, double q3) AiDotNet_Quartiles()
    {
        return (
            BasicStats<double>.Quartile(_aiVector, 1),
            BasicStats<double>.Quartile(_aiVector, 2),
            BasicStats<double>.Quartile(_aiVector, 3)
        );
    }

    [Benchmark]
    public (double q1, double q2, double q3) AccordNet_Quartiles()
    {
        return (
            Measures.Quartiles(_accordArray, QuantileMethod.Default).Min,
            Measures.Quartiles(_accordArray, QuantileMethod.Default).Median,
            Measures.Quartiles(_accordArray, QuantileMethod.Default).Max
        );
    }

    #endregion

    #region Skewness

    [Benchmark]
    public double AiDotNet_Skewness()
    {
        return BasicStats<double>.Skewness(_aiVector);
    }

    [Benchmark]
    public double AccordNet_Skewness()
    {
        return Measures.Skewness(_accordArray);
    }

    #endregion

    #region Kurtosis

    [Benchmark]
    public double AiDotNet_Kurtosis()
    {
        return BasicStats<double>.Kurtosis(_aiVector);
    }

    [Benchmark]
    public double AccordNet_Kurtosis()
    {
        return Measures.Kurtosis(_accordArray);
    }

    #endregion

    #region Covariance

    [Benchmark]
    public double AiDotNet_Covariance()
    {
        var vec1 = new Vector<double>(Size / 2);
        var vec2 = new Vector<double>(Size / 2);
        for (int i = 0; i < Size / 2; i++)
        {
            vec1[i] = _aiVector[i];
            vec2[i] = _aiVector[i + Size / 2];
        }
        return BasicStats<double>.Covariance(vec1, vec2);
    }

    [Benchmark]
    public double AccordNet_Covariance()
    {
        var arr1 = new double[Size / 2];
        var arr2 = new double[Size / 2];
        Array.Copy(_accordArray, 0, arr1, 0, Size / 2);
        Array.Copy(_accordArray, Size / 2, arr2, 0, Size / 2);
        return Measures.Covariance(arr1, arr2);
    }

    #endregion

    #region Correlation

    [Benchmark]
    public double AiDotNet_Correlation()
    {
        var vec1 = new Vector<double>(Size / 2);
        var vec2 = new Vector<double>(Size / 2);
        for (int i = 0; i < Size / 2; i++)
        {
            vec1[i] = _aiVector[i];
            vec2[i] = _aiVector[i + Size / 2];
        }
        return BasicStats<double>.Correlation(vec1, vec2);
    }

    [Benchmark]
    public double AccordNet_Correlation()
    {
        var arr1 = new double[Size / 2];
        var arr2 = new double[Size / 2];
        Array.Copy(_accordArray, 0, arr1, 0, Size / 2);
        Array.Copy(_accordArray, Size / 2, arr2, 0, Size / 2);
        return Measures.Correlation(arr1, arr2);
    }

    #endregion
}
