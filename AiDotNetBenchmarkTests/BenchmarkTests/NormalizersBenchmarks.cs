using AiDotNet.Normalizers;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for data normalization methods
/// Tests performance of various scaling and normalization techniques
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class NormalizersBenchmarks
{
    [Params(1000, 10000, 50000)]
    public int SampleCount { get; set; }

    [Params(10, 50)]
    public int FeatureCount { get; set; }

    private Matrix<double> _data = null!;
    private Vector<double> _vectorData = null!;

    private MinMaxNormalizer<double> _minMax = null!;
    private ZScoreNormalizer<double> _zScore = null!;
    private LogNormalizer<double> _log = null!;
    private MeanVarianceNormalizer<double> _meanVariance = null!;
    private RobustScalingNormalizer<double> _robust = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize matrix data
        _data = new Matrix<double>(SampleCount, FeatureCount);
        for (int i = 0; i < SampleCount; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                // Mix of scales to make normalization meaningful
                _data[i, j] = random.NextDouble() * 100 + j * 10;
            }
        }

        // Initialize vector data
        _vectorData = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++)
        {
            _vectorData[i] = random.NextDouble() * 100;
        }

        // Initialize normalizers
        _minMax = new MinMaxNormalizer<double>();
        _zScore = new ZScoreNormalizer<double>();
        _log = new LogNormalizer<double>();
        _meanVariance = new MeanVarianceNormalizer<double>();
        _robust = new RobustScalingNormalizer<double>();
    }

    #region MinMax Normalization

    [Benchmark(Baseline = true)]
    public Matrix<double> MinMax_FitTransform()
    {
        return _minMax.FitTransform(_data);
    }

    [Benchmark]
    public Vector<double> MinMax_TransformVector()
    {
        _minMax.Fit(_data);
        return _minMax.Transform(_vectorData);
    }

    #endregion

    #region Z-Score Normalization

    [Benchmark]
    public Matrix<double> ZScore_FitTransform()
    {
        return _zScore.FitTransform(_data);
    }

    [Benchmark]
    public Vector<double> ZScore_TransformVector()
    {
        _zScore.Fit(_data);
        return _zScore.Transform(_vectorData);
    }

    #endregion

    #region Log Normalization

    [Benchmark]
    public Matrix<double> Log_FitTransform()
    {
        // Use positive data for log transform
        var positiveData = new Matrix<double>(SampleCount, FeatureCount);
        for (int i = 0; i < SampleCount; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                positiveData[i, j] = Math.Abs(_data[i, j]) + 1;
            }
        }
        return _log.FitTransform(positiveData);
    }

    #endregion

    #region Mean-Variance Normalization

    [Benchmark]
    public Matrix<double> MeanVariance_FitTransform()
    {
        return _meanVariance.FitTransform(_data);
    }

    [Benchmark]
    public Vector<double> MeanVariance_TransformVector()
    {
        _meanVariance.Fit(_data);
        return _meanVariance.Transform(_vectorData);
    }

    #endregion

    #region Robust Scaling

    [Benchmark]
    public Matrix<double> RobustScaling_FitTransform()
    {
        return _robust.FitTransform(_data);
    }

    [Benchmark]
    public Vector<double> RobustScaling_TransformVector()
    {
        _robust.Fit(_data);
        return _robust.Transform(_vectorData);
    }

    #endregion

    #region Inverse Transform

    [Benchmark]
    public Matrix<double> MinMax_FitTransform_InverseTransform()
    {
        var normalized = _minMax.FitTransform(_data);
        return _minMax.InverseTransform(normalized);
    }

    [Benchmark]
    public Matrix<double> ZScore_FitTransform_InverseTransform()
    {
        var normalized = _zScore.FitTransform(_data);
        return _zScore.InverseTransform(normalized);
    }

    #endregion
}
