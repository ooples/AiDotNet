using AiDotNet.Caching;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models.Optimization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for Caching infrastructure
/// Tests DefaultModelCache, DefaultGradientCache, and DeterministicCacheKeyGenerator performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class CachingBenchmarks
{
    [Params(10, 100)]
    public int CacheSize { get; set; }

    [Params(50, 200)]
    public int ParameterCount { get; set; }

    private DefaultModelCache<double, Matrix<double>, Vector<double>> _modelCache = null!;
    private DefaultGradientCache<double> _gradientCache = null!;
    private List<string> _cacheKeys = null!;
    private List<OptimizationStepData<double, Matrix<double>, Vector<double>>> _stepDataList = null!;
    private List<MockGradientModel> _gradientModels = null!;
    private Vector<double> _parameters = null!;
    private Matrix<double> _inputData = null!;
    private Vector<double> _outputData = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize caches
        _modelCache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        _gradientCache = new DefaultGradientCache<double>();

        // Generate cache keys
        _cacheKeys = new List<string>();
        for (int i = 0; i < CacheSize; i++)
        {
            _cacheKeys.Add($"key_{i}");
        }

        // Generate step data
        _stepDataList = new List<OptimizationStepData<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < CacheSize; i++)
        {
            _stepDataList.Add(new OptimizationStepData<double, Matrix<double>, Vector<double>>
            {
                Iteration = i,
                LossValue = random.NextDouble() * 10,
                Parameters = new Vector<double>(ParameterCount)
            });
        }

        // Generate gradient models
        _gradientModels = new List<MockGradientModel>();
        for (int i = 0; i < CacheSize; i++)
        {
            _gradientModels.Add(new MockGradientModel { Id = i });
        }

        // Generate parameters and data for key generation
        _parameters = new Vector<double>(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            _parameters[i] = random.NextDouble() * 2 - 1;
        }

        _inputData = new Matrix<double>(100, 10);
        _outputData = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                _inputData[i, j] = random.NextDouble();
            }
            _outputData[i] = random.NextDouble();
        }
    }

    #region ModelCache Benchmarks

    [Benchmark(Baseline = true)]
    public void Cache01_ModelCache_CacheStepData()
    {
        for (int i = 0; i < CacheSize; i++)
        {
            _modelCache.CacheStepData(_cacheKeys[i], _stepDataList[i]);
        }
    }

    [Benchmark]
    public List<OptimizationStepData<double, Matrix<double>, Vector<double>>?> Cache02_ModelCache_GetCachedStepData()
    {
        var results = new List<OptimizationStepData<double, Matrix<double>, Vector<double>>?>();
        for (int i = 0; i < CacheSize; i++)
        {
            results.Add(_modelCache.GetCachedStepData(_cacheKeys[i]));
        }
        return results;
    }

    [Benchmark]
    public void Cache03_ModelCache_ClearCache()
    {
        _modelCache.ClearCache();
    }

    [Benchmark]
    public string Cache04_ModelCache_GenerateCacheKey()
    {
        var model = new SimpleTestModel();
        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = _inputData,
            YTrain = _outputData,
            XValidation = _inputData,
            YValidation = _outputData,
            XTest = _inputData,
            YTest = _outputData
        };
        return _modelCache.GenerateCacheKey(model, inputData);
    }

    #endregion

    #region GradientCache Benchmarks

    [Benchmark]
    public void Cache05_GradientCache_CacheGradient()
    {
        for (int i = 0; i < CacheSize; i++)
        {
            _gradientCache.CacheGradient(_cacheKeys[i], _gradientModels[i]);
        }
    }

    [Benchmark]
    public List<IGradientModel<double>?> Cache06_GradientCache_GetCachedGradient()
    {
        var results = new List<IGradientModel<double>?>();
        for (int i = 0; i < CacheSize; i++)
        {
            results.Add(_gradientCache.GetCachedGradient(_cacheKeys[i]));
        }
        return results;
    }

    [Benchmark]
    public void Cache07_GradientCache_ClearCache()
    {
        _gradientCache.ClearCache();
    }

    #endregion

    #region DeterministicCacheKeyGenerator Benchmarks

    [Benchmark]
    public string Cache08_GenerateKey_Parameters()
    {
        return DeterministicCacheKeyGenerator.GenerateKey(_parameters, "input_descriptor");
    }

    [Benchmark]
    public string Cache09_GenerateKey_ParametersOnly()
    {
        return DeterministicCacheKeyGenerator.GenerateKey(_parameters);
    }

    [Benchmark]
    public string Cache10_CreateInputDataDescriptor()
    {
        return DeterministicCacheKeyGenerator.CreateInputDataDescriptor<double, Matrix<double>, Vector<double>>(
            _inputData, _outputData, _inputData, _outputData, _inputData, _outputData);
    }

    #endregion

    #region Cache Hit/Miss Patterns

    [Benchmark]
    public int Cache11_ModelCache_HitRate()
    {
        // Populate cache
        for (int i = 0; i < CacheSize; i++)
        {
            _modelCache.CacheStepData(_cacheKeys[i], _stepDataList[i]);
        }

        // Measure hit rate (all hits)
        int hits = 0;
        for (int i = 0; i < CacheSize; i++)
        {
            var data = _modelCache.GetCachedStepData(_cacheKeys[i]);
            if (data != null) hits++;
        }
        return hits;
    }

    [Benchmark]
    public int Cache12_GradientCache_HitRate()
    {
        // Populate cache
        for (int i = 0; i < CacheSize; i++)
        {
            _gradientCache.CacheGradient(_cacheKeys[i], _gradientModels[i]);
        }

        // Measure hit rate (all hits)
        int hits = 0;
        for (int i = 0; i < CacheSize; i++)
        {
            var gradient = _gradientCache.GetCachedGradient(_cacheKeys[i]);
            if (gradient != null) hits++;
        }
        return hits;
    }

    #endregion

    /// <summary>
    /// Simple test model for benchmarking
    /// </summary>
    private class SimpleTestModel : IFullModel<double, Matrix<double>, Vector<double>>
    {
        private Vector<double> _parameters = new Vector<double>(10);

        public Vector<double> Predict(Matrix<double> input)
        {
            return new Vector<double>(input.Rows);
        }

        public void Train(Matrix<double> inputs, Vector<double> outputs)
        {
            // No-op for benchmark
        }

        public Vector<double> GetParameters()
        {
            return _parameters;
        }

        public void SetParameters(Vector<double> parameters)
        {
            _parameters = parameters;
        }
    }

    /// <summary>
    /// Mock gradient model for benchmarking
    /// </summary>
    private class MockGradientModel : IGradientModel<double>
    {
        public int Id { get; set; }

        public Vector<double> ComputeGradient(Vector<double> input)
        {
            return new Vector<double>(input.Length);
        }
    }
}
