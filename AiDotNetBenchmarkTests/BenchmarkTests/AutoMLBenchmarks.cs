using AiDotNet.AutoML;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for AutoML functionality
/// Tests Neural Architecture Search and hyperparameter optimization
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class AutoMLBenchmarks
{
    [Params(100, 500)]
    public int SampleCount { get; set; }

    [Params(10)]
    public int FeatureCount { get; set; }

    private Matrix<double> _trainX = null!;
    private Vector<double> _trainY = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _trainX = new Matrix<double>(SampleCount, FeatureCount);
        _trainY = new Vector<double>(SampleCount);

        for (int i = 0; i < SampleCount; i++)
        {
            double target = 0;
            for (int j = 0; j < FeatureCount; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _trainX[i, j] = value;
                target += value * (j + 1);
            }
            _trainY[i] = target > 0 ? 1 : 0;
        }
    }

    [Benchmark(Baseline = true)]
    public SearchSpace AutoML_CreateSearchSpace()
    {
        var searchSpace = new SearchSpace();
        searchSpace.AddParameter("learning_rate", new ParameterRange(0.0001, 0.1, isLogarithmic: true));
        searchSpace.AddParameter("num_layers", new ParameterRange(1, 5, isInteger: true));
        searchSpace.AddParameter("hidden_size", new ParameterRange(16, 128, isInteger: true));
        searchSpace.AddParameter("dropout_rate", new ParameterRange(0.0, 0.5));
        return searchSpace;
    }

    [Benchmark]
    public Architecture AutoML_GenerateArchitecture()
    {
        var searchSpace = new SearchSpace();
        searchSpace.AddParameter("learning_rate", new ParameterRange(0.0001, 0.1));
        searchSpace.AddParameter("num_layers", new ParameterRange(1, 5, isInteger: true));
        searchSpace.AddParameter("hidden_size", new ParameterRange(16, 128, isInteger: true));

        var architecture = new Architecture();
        architecture.AddLayer("input", FeatureCount);
        architecture.AddLayer("hidden1", 64);
        architecture.AddLayer("hidden2", 32);
        architecture.AddLayer("output", 1);

        return architecture;
    }

    [Benchmark]
    public NeuralArchitectureSearch<double> AutoML_NeuralArchitectureSearch()
    {
        var searchSpace = new SearchSpace();
        searchSpace.AddParameter("num_layers", new ParameterRange(1, 3, isInteger: true));
        searchSpace.AddParameter("hidden_size", new ParameterRange(16, 64, isInteger: true));

        var nas = new NeuralArchitectureSearch<double>(
            searchSpace,
            maxTrials: 10,
            objectiveMetric: "accuracy"
        );

        return nas;
    }

    [Benchmark]
    public TrialResult AutoML_EvaluateArchitecture()
    {
        var architecture = new Architecture();
        architecture.AddLayer("input", FeatureCount);
        architecture.AddLayer("hidden", 32);
        architecture.AddLayer("output", 1);

        var trial = new TrialResult
        {
            TrialId = 1,
            Architecture = architecture,
            Hyperparameters = new Dictionary<string, double>
            {
                { "learning_rate", 0.001 },
                { "batch_size", 32 }
            },
            Metric = 0.85,
            TrainingTime = TimeSpan.FromSeconds(10)
        };

        return trial;
    }

    [Benchmark]
    public SearchConstraint AutoML_CreateSearchConstraints()
    {
        var constraint = new SearchConstraint
        {
            MaxLayers = 5,
            MinLayerSize = 16,
            MaxLayerSize = 256,
            MaxParameters = 1000000,
            MaxTrainingTime = TimeSpan.FromMinutes(30)
        };

        return constraint;
    }
}
