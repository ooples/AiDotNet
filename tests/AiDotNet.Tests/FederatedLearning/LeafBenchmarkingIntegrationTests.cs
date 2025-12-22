using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class LeafBenchmarkingIntegrationTests
{
    private const string TinyLeafJson = @"
{
  ""users"": [""u1"", ""u2""],
  ""num_samples"": [2, 1],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [[1, 2], [3]],
        [[4, 5], [6]]
      ],
      ""y"": [0, 1]
    },
    ""u2"": {
      ""x"": [
        [7, 8, 9]
      ],
      ""y"": [1]
    }
  }
}";

    [Fact]
    public async Task BuildAsync_WithLeafBenchmarking_AttachesBenchmarkReport()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"leaf_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"leaf_test_{Guid.NewGuid():N}.json");
        File.WriteAllText(trainPath, TinyLeafJson);
        File.WriteAllText(testPath, TinyLeafJson);

        try
        {
            var loader = DataLoaders.FromLeafFederatedJsonFiles<double>(trainPath, testPath);

            var model = CreateTrainedRegressionModel();

            var optimizer = new FederatedNoOpOptimizer(model);

            var flOptions = new FederatedLearningOptions
            {
                NumberOfClients = 1,
                MaxRounds = 1,
                ClientSelectionFraction = 1.0,
                LocalEpochs = 1,
                AggregationStrategy = FederatedAggregationStrategy.FedAvg,
                RandomSeed = 123,
                MinRoundsBeforeConvergence = 0,
                ConvergenceThreshold = 0.0
            };

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.LEAF },
                CiMode = true,
                Seed = 123,
                Leaf = new LeafFederatedBenchmarkOptions
                {
                    TrainFilePath = trainPath,
                    TestFilePath = testPath,
                    MaxSamplesPerUser = 1
                }
            };

            var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureFederatedLearning(flOptions)
                .ConfigureBenchmarking(benchOptions)
                .BuildAsync();

            Assert.NotNull(result.BenchmarkReport);
            var report = result.BenchmarkReport!;
            var leafSuite = GetSuite(report, BenchmarkSuite.LEAF);

            Assert.Equal(BenchmarkExecutionStatus.Succeeded, leafSuite.Status);
            Assert.NotNull(leafSuite.DataSelection);
            Assert.Equal(2, leafSuite.DataSelection!.ClientsUsed);
            Assert.Equal(2, leafSuite.DataSelection.TrainSamplesUsed);
            Assert.Equal(2, leafSuite.DataSelection.TestSamplesUsed);
            Assert.Equal(3, leafSuite.DataSelection.FeatureCount);

            Assert.Equal(2.0, GetMetric(leafSuite, BenchmarkMetric.TotalEvaluated));
        }
        finally
        {
            if (File.Exists(trainPath))
            {
                File.Delete(trainPath);
            }

            if (File.Exists(testPath))
            {
                File.Delete(testPath);
            }
        }
    }

    [Fact]
    public async Task EvaluateBenchmarksAsync_WithSeededSampling_IsDeterministicAndSeedAffectsSelection()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"leaf_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"leaf_test_{Guid.NewGuid():N}.json");
        File.WriteAllText(trainPath, TinyLeafJson);
        File.WriteAllText(testPath, TinyLeafJson);

        try
        {
            var loader = DataLoaders.FromLeafFederatedJsonFiles<double>(trainPath, testPath);

            var model = CreateTrainedRegressionModel();

            var optimizer = new FederatedNoOpOptimizer(model);

            var flOptions = new FederatedLearningOptions
            {
                NumberOfClients = 1,
                MaxRounds = 1,
                ClientSelectionFraction = 1.0,
                LocalEpochs = 1,
                AggregationStrategy = FederatedAggregationStrategy.FedAvg,
                RandomSeed = 123,
                MinRoundsBeforeConvergence = 0,
                ConvergenceThreshold = 0.0
            };

            var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureFederatedLearning(flOptions)
                .BuildAsync();

            int seedSelectSample0 = FindSeedForFirstDrawOfNext2(desired: 1);
            int seedSelectSample1 = FindSeedForFirstDrawOfNext2(desired: 0);

            var optionsA = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.LEAF },
                CiMode = true,
                Seed = seedSelectSample0,
                Leaf = new LeafFederatedBenchmarkOptions
                {
                    TrainFilePath = trainPath,
                    TestFilePath = testPath,
                    MaxSamplesPerUser = 1
                }
            };

            var reportA1 = await result.EvaluateBenchmarksAsync(optionsA);
            var reportA2 = await result.EvaluateBenchmarksAsync(optionsA);

            var suiteA1 = GetSuite(reportA1, BenchmarkSuite.LEAF);
            var suiteA2 = GetSuite(reportA2, BenchmarkSuite.LEAF);

            Assert.Equal(GetMetric(suiteA1, BenchmarkMetric.Accuracy), GetMetric(suiteA2, BenchmarkMetric.Accuracy), 12);
            Assert.Equal(GetMetric(suiteA1, BenchmarkMetric.MeanSquaredError), GetMetric(suiteA2, BenchmarkMetric.MeanSquaredError), 12);

            var optionsB = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.LEAF },
                CiMode = true,
                Seed = seedSelectSample1,
                Leaf = new LeafFederatedBenchmarkOptions
                {
                    TrainFilePath = trainPath,
                    TestFilePath = testPath,
                    MaxSamplesPerUser = 1
                }
            };

            var reportB = await result.EvaluateBenchmarksAsync(optionsB);
            var suiteB = GetSuite(reportB, BenchmarkSuite.LEAF);

            Assert.NotEqual(GetMetric(suiteA1, BenchmarkMetric.Accuracy), GetMetric(suiteB, BenchmarkMetric.Accuracy));
        }
        finally
        {
            if (File.Exists(trainPath))
            {
                File.Delete(trainPath);
            }

            if (File.Exists(testPath))
            {
                File.Delete(testPath);
            }
        }
    }

    private static BenchmarkSuiteReport GetSuite(BenchmarkReport report, BenchmarkSuite suite)
    {
        return report.Suites.Single(r => r.Suite == suite);
    }

    private static double GetMetric(BenchmarkSuiteReport suite, BenchmarkMetric metric)
    {
        return suite.Metrics.Single(x => x.Metric == metric).Value;
    }

    private static int FindSeedForFirstDrawOfNext2(int desired)
    {
        for (int seed = 0; seed < 10000; seed++)
        {
            var random = RandomHelper.CreateSeededRandom(unchecked(seed * 16777619));
            if (random.Next(2) == desired)
            {
                return seed;
            }
        }

        throw new InvalidOperationException("Unable to find a deterministic seed for desired sample selection.");
    }

    private static MultipleRegression<double> CreateTrainedRegressionModel()
    {
        var model = new MultipleRegression<double>();
        var x = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var y = new Vector<double>(new[] { 0.0, 1.0, 1.0 });

        model.Train(x, y);
        return model;
    }
}
