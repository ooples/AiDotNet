using AiDotNet.OutlierRemoval;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for all OutlierRemoval implementations
/// Tests outlier detection and removal performance across 5 different algorithms
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class OutlierRemovalBenchmarks
{
    [Params(100, 500)]
    public int SampleSize { get; set; }

    [Params(10, 30)]
    public int FeatureCount { get; set; }

    private Matrix<double> _inputs = null!;
    private Vector<double> _outputs = null!;
    private Tensor<double> _tensorInputs = null!;
    private Tensor<double> _tensorOutputs = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Create synthetic data with outliers
        _inputs = new Matrix<double>(SampleSize, FeatureCount);
        _outputs = new Vector<double>(SampleSize);

        for (int i = 0; i < SampleSize; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                if (random.NextDouble() < 0.05) // 5% outliers
                {
                    _inputs[i, j] = random.NextDouble() * 100; // Large outlier
                }
                else
                {
                    _inputs[i, j] = random.NextDouble() * 2 - 1; // Normal data
                }
            }

            if (random.NextDouble() < 0.05) // 5% output outliers
            {
                _outputs[i] = random.NextDouble() * 100;
            }
            else
            {
                _outputs[i] = random.NextDouble() * 10;
            }
        }

        // Create tensor versions
        _tensorInputs = Tensor<double>.FromMatrix(_inputs);
        _tensorOutputs = Tensor<double>.FromVector(_outputs);
    }

    #region Matrix-Based Outlier Removal

    [Benchmark(Baseline = true)]
    public (Matrix<double>, Vector<double>) OutlierRemoval01_None_Matrix()
    {
        var remover = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval02_ZScore_Matrix()
    {
        var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval03_IQR_Matrix()
    {
        var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(multiplier: 1.5);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval04_MAD_Matrix()
    {
        var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval05_Threshold_Matrix()
    {
        var remover = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(
            lowerBound: -5.0, upperBound: 5.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    #endregion

    #region Tensor-Based Outlier Removal

    [Benchmark]
    public (Tensor<double>, Tensor<double>) OutlierRemoval06_None_Tensor()
    {
        var remover = new NoOutlierRemoval<double, Tensor<double>, Tensor<double>>();
        return remover.RemoveOutliers(_tensorInputs, _tensorOutputs);
    }

    [Benchmark]
    public (Tensor<double>, Tensor<double>) OutlierRemoval07_ZScore_Tensor()
    {
        var remover = new ZScoreOutlierRemoval<double, Tensor<double>, Tensor<double>>(threshold: 3.0);
        return remover.RemoveOutliers(_tensorInputs, _tensorOutputs);
    }

    [Benchmark]
    public (Tensor<double>, Tensor<double>) OutlierRemoval08_IQR_Tensor()
    {
        var remover = new IQROutlierRemoval<double, Tensor<double>, Tensor<double>>(multiplier: 1.5);
        return remover.RemoveOutliers(_tensorInputs, _tensorOutputs);
    }

    [Benchmark]
    public (Tensor<double>, Tensor<double>) OutlierRemoval09_MAD_Tensor()
    {
        var remover = new MADOutlierRemoval<double, Tensor<double>, Tensor<double>>(threshold: 3.5);
        return remover.RemoveOutliers(_tensorInputs, _tensorOutputs);
    }

    [Benchmark]
    public (Tensor<double>, Tensor<double>) OutlierRemoval10_Threshold_Tensor()
    {
        var remover = new ThresholdOutlierRemoval<double, Tensor<double>, Tensor<double>>(
            lowerBound: -5.0, upperBound: 5.0);
        return remover.RemoveOutliers(_tensorInputs, _tensorOutputs);
    }

    #endregion

    #region Different Threshold Configurations

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval11_ZScore_Strict()
    {
        var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval12_ZScore_Lenient()
    {
        var remover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 4.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval13_IQR_Strict()
    {
        var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(multiplier: 1.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval14_IQR_Lenient()
    {
        var remover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(multiplier: 3.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval15_MAD_Strict()
    {
        var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.5);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    [Benchmark]
    public (Matrix<double>, Vector<double>) OutlierRemoval16_MAD_Lenient()
    {
        var remover = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 5.0);
        return remover.RemoveOutliers(_inputs, _outputs);
    }

    #endregion
}
