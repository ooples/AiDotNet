using AiDotNet.MetaLearning.Trainers;
using AiDotNet.MetaLearning.Config;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for MetaLearning algorithms
/// Tests MAML and Reptile training performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class MetaLearningBenchmarks
{
    [Params(50, 200)]
    public int TaskSamples { get; set; }

    [Params(5)]
    public int FeatureSize { get; set; }

    private List<Matrix<double>> _tasks = null!;
    private List<Vector<double>> _taskLabels = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        int numTasks = 10;
        _tasks = new List<Matrix<double>>();
        _taskLabels = new List<Vector<double>>();

        for (int t = 0; t < numTasks; t++)
        {
            var taskData = new Matrix<double>(TaskSamples, FeatureSize);
            var labels = new Vector<double>(TaskSamples);

            for (int i = 0; i < TaskSamples; i++)
            {
                for (int j = 0; j < FeatureSize; j++)
                {
                    taskData[i, j] = random.NextDouble() * 2 - 1;
                }
                labels[i] = random.NextDouble() > 0.5 ? 1 : 0;
            }

            _tasks.Add(taskData);
            _taskLabels.Add(labels);
        }
    }

    [Benchmark(Baseline = true)]
    public MAMLTrainerConfig MetaLearning_CreateMAMLConfig()
    {
        var config = new MAMLTrainerConfig
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            InnerSteps = 5,
            MetaBatchSize = 4,
            NumEpochs = 10
        };
        return config;
    }

    [Benchmark]
    public MAMLTrainer<double> MetaLearning_InitializeMAML()
    {
        var config = new MAMLTrainerConfig
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            InnerSteps = 5,
            MetaBatchSize = 4
        };

        var trainer = new MAMLTrainer<double>(config, inputSize: FeatureSize, outputSize: 1);
        return trainer;
    }

    [Benchmark]
    public ReptileTrainerConfig MetaLearning_CreateReptileConfig()
    {
        var config = new ReptileTrainerConfig
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            InnerSteps = 10,
            MetaBatchSize = 4,
            NumEpochs = 10
        };
        return config;
    }

    [Benchmark]
    public ReptileTrainer<double> MetaLearning_InitializeReptile()
    {
        var config = new ReptileTrainerConfig
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.001,
            InnerSteps = 10,
            MetaBatchSize = 4
        };

        var trainer = new ReptileTrainer<double>(config, inputSize: FeatureSize, outputSize: 1);
        return trainer;
    }
}
