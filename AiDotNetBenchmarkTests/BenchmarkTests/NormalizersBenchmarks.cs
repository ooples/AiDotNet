using AiDotNet.Normalizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for data normalization methods
/// Tests performance of various scaling and normalization techniques
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class NormalizersBenchmarks
{
    [Params(1000, 10000)]
    public int SampleCount { get; set; }

    [Params(10, 50)]
    public int FeatureCount { get; set; }

    private Matrix<double> _data = new Matrix<double>(0, 0);
    private Vector<double> _vectorData = new Vector<double>(0);
    private Matrix<double> _positiveData = new Matrix<double>(0, 0);

    // Pre-computed normalized data and parameters for denormalization benchmarks
    private Vector<double> _minMaxNormalized = new Vector<double>(0);
    private NormalizationParameters<double> _minMaxParams = new NormalizationParameters<double>();
    private Vector<double> _zScoreNormalized = new Vector<double>(0);
    private NormalizationParameters<double> _zScoreParams = new NormalizationParameters<double>();

    private MinMaxNormalizer<double, Matrix<double>, Vector<double>> _minMax = new();
    private ZScoreNormalizer<double, Matrix<double>, Vector<double>> _zScore = new();
    private LogNormalizer<double, Matrix<double>, Vector<double>> _log = new();
    private MeanVarianceNormalizer<double, Matrix<double>, Vector<double>> _meanVariance = new();
    private RobustScalingNormalizer<double, Matrix<double>, Vector<double>> _robust = new();

    [GlobalSetup]
    public void Setup()
    {
        // Using seeded Random for reproducible benchmark data - not used for security purposes
        var random = RandomHelper.CreateSeededRandom(42); // NOSONAR S2245 - benchmarks don't need cryptographic randomness

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

        // Initialize positive data for log transform (pre-computed to avoid allocation in benchmark)
        _positiveData = new Matrix<double>(SampleCount, FeatureCount);
        for (int i = 0; i < SampleCount; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _positiveData[i, j] = Math.Abs(_data[i, j]) + 1;
            }
        }

        // Initialize normalizers
        _minMax = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        _zScore = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
        _log = new LogNormalizer<double, Matrix<double>, Vector<double>>();
        _meanVariance = new MeanVarianceNormalizer<double, Matrix<double>, Vector<double>>();
        _robust = new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();

        // Pre-compute normalized data and parameters for denormalization benchmarks
        // This avoids measuring normalization overhead in denormalization benchmarks
        (_minMaxNormalized, _minMaxParams) = _minMax.NormalizeOutput(_vectorData);
        (_zScoreNormalized, _zScoreParams) = _zScore.NormalizeOutput(_vectorData);
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
        // Use pre-computed normalized data and parameters to measure only denormalization
        return _minMax.Denormalize(_minMaxNormalized, _minMaxParams);
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
        // Use pre-computed normalized data and parameters to measure only denormalization
        return _zScore.Denormalize(_zScoreNormalized, _zScoreParams);
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
        // Use pre-computed positive data to avoid allocation in benchmark iteration
        return _log.NormalizeInput(_positiveData);
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
