using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class CorpusBenchmarkingIntegrationTests
{
    private const string TinyRedditLeafJson = @"
{
  ""users"": [""u1""],
  ""num_samples"": [2],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [
          [""<BOS>"", ""hello"", ""world"", ""<EOS>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>""]
        ],
        [
          [""<BOS>"", ""foo"", ""bar"", ""<EOS>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>""]
        ]
      ],
      ""y"": [
        {
          ""created_utc"": 1,
          ""target_tokens"": [
            [""hello"", ""world"", ""<EOS>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>""]
          ]
        },
        {
          ""created_utc"": 2,
          ""target_tokens"": [
            [""foo"", ""bar"", ""<EOS>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>"", ""<PAD>""]
          ]
        }
      ]
    }
  }
}";

    private const string TinyStackOverflowTokenSequenceJson = @"
{
  ""users"": [""u1""],
  ""num_samples"": [2],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [""how"", ""to"", ""code""],
        [""write"", ""unit"", ""tests""]
      ],
      ""y"": [""in"", ""fast""]
    }
  }
}";

    [Fact]
    public async Task BuildAsync_WithRedditBenchmarking_AttachesBenchmarkReport()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"reddit_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"reddit_test_{Guid.NewGuid():N}.json");

        File.WriteAllText(trainPath, TinyRedditLeafJson);
        File.WriteAllText(testPath, TinyRedditLeafJson);

        try
        {
            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromArrays(new double[,] { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } }, new double[] { 0.0, 1.0 });

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.Reddit },
                CiMode = true,
                Seed = 23,
                Text = new FederatedTextBenchmarkOptions
                {
                    Reddit = new RedditFederatedBenchmarkOptions
                    {
                        TrainFilePath = trainPath,
                        TestFilePath = testPath,
                        MaxSamplesPerUser = 1,
                        SequenceLength = 10,
                        MaxVocabularySize = 128,
                        VocabularyTrainingSampleCount = 10
                    }
                }
            };

            var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureBenchmarking(benchOptions)
                .BuildAsync();

            Assert.NotNull(result.BenchmarkReport);
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.Reddit);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.Equal(1.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
            Assert.NotNull(suite.DataSelection);
            Assert.Equal(10, suite.DataSelection!.FeatureCount);
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
    public async Task BuildAsync_WithStackOverflowBenchmarking_AttachesBenchmarkReport()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"so_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"so_test_{Guid.NewGuid():N}.json");

        File.WriteAllText(trainPath, TinyStackOverflowTokenSequenceJson);
        File.WriteAllText(testPath, TinyStackOverflowTokenSequenceJson);

        try
        {
            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromArrays(new double[,] { { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 } }, new double[] { 0.0, 1.0 });

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.StackOverflow },
                CiMode = true,
                Seed = 29,
                Text = new FederatedTextBenchmarkOptions
                {
                    StackOverflow = new StackOverflowFederatedBenchmarkOptions
                    {
                        TrainFilePath = trainPath,
                        TestFilePath = testPath,
                        MaxSamplesPerUser = 1,
                        SequenceLength = 3,
                        MaxVocabularySize = 128,
                        VocabularyTrainingSampleCount = 10
                    }
                }
            };

            var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureBenchmarking(benchOptions)
                .BuildAsync();

            Assert.NotNull(result.BenchmarkReport);
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.StackOverflow);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.Equal(1.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
            Assert.NotNull(suite.DataSelection);
            Assert.Equal(3, suite.DataSelection!.FeatureCount);
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
}

