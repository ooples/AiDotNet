using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class TextBenchmarkingIntegrationTests
{
    private const string TinySent140LeafJson = @"
{
  ""users"": [""u1""],
  ""num_samples"": [2],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [""1"", ""2020-01-01"", ""query"", ""user"", ""I love this""],
        [""2"", ""2020-01-02"", ""query"", ""user"", ""I hate this""]
      ],
      ""y"": [1, 0]
    }
  }
}";

    private const string TinyShakespeareLeafJson = @"
{
  ""users"": [""u1""],
  ""num_samples"": [2],
  ""user_data"": {
    ""u1"": {
      ""x"": [""abcdefgh"", ""bcdefghi""],
      ""y"": [""i"", ""j""]
    }
  }
}";

    [Fact]
    public async Task BuildAsync_WithSent140Benchmarking_AttachesBenchmarkReport()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"sent140_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"sent140_test_{Guid.NewGuid():N}.json");

        File.WriteAllText(trainPath, TinySent140LeafJson);
        File.WriteAllText(testPath, TinySent140LeafJson);

        try
        {
            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromArrays(new double[,] { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } }, new double[] { 0.0, 1.0 });

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.Sent140 },
                CiMode = true,
                Seed = 17,
                Text = new FederatedTextBenchmarkOptions
                {
                    Sent140 = new Sent140FederatedBenchmarkOptions
                    {
                        TrainFilePath = trainPath,
                        TestFilePath = testPath,
                        MaxSamplesPerUser = 1,
                        MaxSequenceLength = 8,
                        TokenizerVocabularySize = 128,
                        TokenizerTrainingSampleCount = 10
                    }
                }
            };

            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureBenchmarking(benchOptions)
                .BuildAsync();

            Assert.NotNull(result.BenchmarkReport);
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.Sent140);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.Equal(1.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
            Assert.NotNull(suite.DataSelection);
            Assert.Equal(8, suite.DataSelection!.FeatureCount);
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
    public async Task BuildAsync_WithShakespeareBenchmarking_AttachesBenchmarkReport()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"shakespeare_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"shakespeare_test_{Guid.NewGuid():N}.json");

        File.WriteAllText(trainPath, TinyShakespeareLeafJson);
        File.WriteAllText(testPath, TinyShakespeareLeafJson);

        try
        {
            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromArrays(new double[,] { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } }, new double[] { 0.0, 1.0 });

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.Shakespeare },
                CiMode = true,
                Seed = 19,
                Text = new FederatedTextBenchmarkOptions
                {
                    Shakespeare = new ShakespeareFederatedBenchmarkOptions
                    {
                        TrainFilePath = trainPath,
                        TestFilePath = testPath,
                        MaxSamplesPerUser = 1,
                        SequenceLength = 8
                    }
                }
            };

            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureBenchmarking(benchOptions)
                .BuildAsync();

            Assert.NotNull(result.BenchmarkReport);
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.Shakespeare);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.Equal(1.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
            Assert.NotNull(suite.DataSelection);
            Assert.Equal(8, suite.DataSelection!.FeatureCount);
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

