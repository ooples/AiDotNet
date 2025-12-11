using AiDotNet.Normalizers;
using AiDotNet.Tensors.LinearAlgebra;
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
    [Params(1000, 10000)]
    public int SampleCount { get; set; }

    [Params(10, 50)]
    public int FeatureCount { get; set; }

    private Matrix<double> _data = new Matrix<double>(0, 0);
    private Vector<double> _vectorData = new Vector<double>(0);

    private MinMaxNormalizer<double, Matrix<double>, Vector<double>> _minMax = new();
    private ZScoreNormalizer<double, Matrix<double>, Vector<double>> _zScore = new();
    private LogNormalizer<double, Matrix<double>, Vector<double>> _log = new();
    private MeanVarianceNormalizer<double, Matrix<double>, Vector<double>> _meanVariance = new();
    private RobustScalingNormalizer<double, Matrix<double>, Vector<double>> _robust = new();

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
                _data[i, j] = random.NextDouble() * 100 + j * 10.0;
            }
        }

        // Initialize vector data
        _vectorData = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++)
        {
            _vectorData[i] = random.NextDouble() * 100 + 1; // Ensure positive for log
        }

        // Initialize normalizers
        _minMax = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        _zScore = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
        _log = new LogNormalizer<double, Matrix<double>, Vector<double>>();
        _meanVariance = new MeanVarianceNormalizer<double, Matrix<double>, Vector<double>>();
        _robust = new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();
    }

    #region MinMax Normalization

    [Benchmark(Baseline = true)]
    public (Vector<double>, NormalizationParameters<double>) MinMax_NormalizeOutput()
    {
        return _minMax.NormalizeOutput(_vectorData);
    }

    [Benchmark]
    public (Matrix<double>, List<NormalizationParameters<double>>) MinMax_NormalizeInput()
    {
        return _minMax.NormalizeInput(_data);
    }

    [Benchmark]
    public Vector<double> MinMax_Denormalize()
    {
        var (normalized, parameters) = _minMax.NormalizeOutput(_vectorData);
        return _minMax.Denormalize(normalized, parameters);
    }

    #endregion

    #region Z-Score Normalization

    [Benchmark]
    public (Vector<double>, NormalizationParameters<double>) ZScore_NormalizeOutput()
    {
        return _zScore.NormalizeOutput(_vectorData);
    }

    [Benchmark]
    public (Matrix<double>, List<NormalizationParameters<double>>) ZScore_NormalizeInput()
    {
        return _zScore.NormalizeInput(_data);
    }

    [Benchmark]
    public Vector<double> ZScore_Denormalize()
    {
        var (normalized, parameters) = _zScore.NormalizeOutput(_vectorData);
        return _zScore.Denormalize(normalized, parameters);
    }

    #endregion

    #region Log Normalization

    [Benchmark]
    public (Vector<double>, NormalizationParameters<double>) Log_NormalizeOutput()
    {
        return _log.NormalizeOutput(_vectorData);
    }

    [Benchmark]
    public (Matrix<double>, List<NormalizationParameters<double>>) Log_NormalizeInput()
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
        return _log.NormalizeInput(positiveData);
    }

    #endregion

    #region Mean-Variance Normalization

    [Benchmark]
    public (Vector<double>, NormalizationParameters<double>) MeanVariance_NormalizeOutput()
    {
        return _meanVariance.NormalizeOutput(_vectorData);
    }

    [Benchmark]
    public (Matrix<double>, List<NormalizationParameters<double>>) MeanVariance_NormalizeInput()
    {
        return _meanVariance.NormalizeInput(_data);
    }

    #endregion

    #region Robust Scaling

    [Benchmark]
    public (Vector<double>, NormalizationParameters<double>) RobustScaling_NormalizeOutput()
    {
        return _robust.NormalizeOutput(_vectorData);
    }

    [Benchmark]
    public (Matrix<double>, List<NormalizationParameters<double>>) RobustScaling_NormalizeInput()
    {
        return _robust.NormalizeInput(_data);
    }

    #endregion

    #region Normalizer Construction

    [Benchmark]
    public MinMaxNormalizer<double, Matrix<double>, Vector<double>> MinMax_CreateNormalizer()
    {
        return new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
    }

    [Benchmark]
    public ZScoreNormalizer<double, Matrix<double>, Vector<double>> ZScore_CreateNormalizer()
    {
        return new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
    }

    [Benchmark]
    public LogNormalizer<double, Matrix<double>, Vector<double>> Log_CreateNormalizer()
    {
        return new LogNormalizer<double, Matrix<double>, Vector<double>>();
    }

    [Benchmark]
    public MeanVarianceNormalizer<double, Matrix<double>, Vector<double>> MeanVariance_CreateNormalizer()
    {
        return new MeanVarianceNormalizer<double, Matrix<double>, Vector<double>>();
    }

    [Benchmark]
    public RobustScalingNormalizer<double, Matrix<double>, Vector<double>> RobustScaling_CreateNormalizer()
    {
        return new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();
    }

    #endregion
}
