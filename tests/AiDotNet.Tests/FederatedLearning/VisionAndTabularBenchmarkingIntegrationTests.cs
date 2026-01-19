using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class VisionAndTabularBenchmarkingIntegrationTests
{
    private const string TinyFemnistLikeLeafJson = @"
{
  ""users"": [""u1""],
  ""num_samples"": [2],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]]
      ],
      ""y"": [0, 1]
    }
  }
}";

    [Fact]
    public async Task BuildAsync_WithTabularNonIidBenchmarking_AttachesBenchmarkReport()
    {
        var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
        var optimizer = new FederatedNoOpOptimizer(model);
        var loader = DataLoaders.FromArrays(new double[,] { { 0.0 }, { 1.0 } }, new double[] { 0.0, 1.0 });

        var benchOptions = new BenchmarkingOptions
        {
            Suites = new[] { BenchmarkSuite.TabularNonIID },
            CiMode = true,
            Seed = 123,
            Tabular = new FederatedTabularBenchmarkOptions
            {
                NonIid = new SyntheticTabularFederatedBenchmarkOptions
                {
                    ClientCount = 2,
                    FeatureCount = 3,
                    TrainSamplesPerClient = 5,
                    TestSamplesPerClient = 2,
                    ClassCount = 3,
                    TaskType = SyntheticTabularTaskType.MultiClassClassification,
                    DirichletAlpha = 0.3,
                    NoiseStdDev = 0.1
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
        var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.TabularNonIID);
        Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
        Assert.Equal(4.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
        Assert.NotNull(suite.DataSelection);
        Assert.Equal(2, suite.DataSelection!.ClientsUsed);
    }

    [Fact]
    public async Task BuildAsync_WithFemnistBenchmarking_AttachesBenchmarkReport()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"femnist_train_{Guid.NewGuid():N}.json");
        string testPath = Path.Combine(Path.GetTempPath(), $"femnist_test_{Guid.NewGuid():N}.json");

        File.WriteAllText(trainPath, TinyFemnistLikeLeafJson);
        File.WriteAllText(testPath, TinyFemnistLikeLeafJson);

        try
        {
            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromLeafFederatedJsonFiles<double>(trainPath, testPath);

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.FEMNIST },
                CiMode = true,
                Seed = 7,
                Vision = new FederatedVisionBenchmarkOptions
                {
                    Femnist = new LeafFederatedBenchmarkOptions
                    {
                        TrainFilePath = trainPath,
                        TestFilePath = testPath,
                        MaxSamplesPerUser = 1
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
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.FEMNIST);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.NotNull(suite.DataSelection);
            Assert.Equal(1, suite.DataSelection!.ClientsUsed);
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
    public async Task BuildAsync_WithCifar10Benchmarking_AttachesBenchmarkReport()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"cifar10_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            WriteCifar10Files(tempDir);

            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromArrays(new double[,] { { 0.0 }, { 1.0 } }, new double[] { 0.0, 1.0 });

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.CIFAR10 },
                CiMode = true,
                Seed = 11,
                Vision = new FederatedVisionBenchmarkOptions
                {
                    Cifar10 = new CifarFederatedBenchmarkOptions
                    {
                        DataDirectoryPath = tempDir,
                        ClientCount = 2,
                        PartitioningStrategy = FederatedPartitioningStrategy.IID,
                        MaxTrainSamples = 3,
                        MaxTestSamples = 2,
                        NormalizePixels = true
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
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.CIFAR10);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.Equal(2.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public async Task BuildAsync_WithCifar100Benchmarking_AttachesBenchmarkReport()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"cifar100_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            WriteCifar100Files(tempDir);

            var model = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
            var optimizer = new FederatedNoOpOptimizer(model);
            var loader = DataLoaders.FromArrays(new double[,] { { 0.0 }, { 1.0 } }, new double[] { 0.0, 1.0 });

            var benchOptions = new BenchmarkingOptions
            {
                Suites = new[] { BenchmarkSuite.CIFAR100 },
                CiMode = true,
                Seed = 11,
                Vision = new FederatedVisionBenchmarkOptions
                {
                    Cifar100 = new CifarFederatedBenchmarkOptions
                    {
                        DataDirectoryPath = tempDir,
                        ClientCount = 2,
                        PartitioningStrategy = FederatedPartitioningStrategy.ShardByLabel,
                        ShardsPerClient = 1,
                        MaxTrainSamples = 3,
                        MaxTestSamples = 2,
                        NormalizePixels = true
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
            var suite = GetSuite(result.BenchmarkReport!, BenchmarkSuite.CIFAR100);
            Assert.Equal(BenchmarkExecutionStatus.Succeeded, suite.Status);
            Assert.Equal(2.0, GetMetric(suite, BenchmarkMetric.TotalEvaluated));
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    private static void WriteCifar10Files(string directoryPath)
    {
        File.WriteAllBytes(Path.Combine(directoryPath, "data_batch_1.bin"), CreateCifar10Records(new[]
        {
            (label: (byte)0, pixel: (byte)10),
            (label: (byte)1, pixel: (byte)20),
            (label: (byte)2, pixel: (byte)30)
        }));

        File.WriteAllBytes(Path.Combine(directoryPath, "data_batch_2.bin"), Array.Empty<byte>());
        File.WriteAllBytes(Path.Combine(directoryPath, "data_batch_3.bin"), Array.Empty<byte>());
        File.WriteAllBytes(Path.Combine(directoryPath, "data_batch_4.bin"), Array.Empty<byte>());
        File.WriteAllBytes(Path.Combine(directoryPath, "data_batch_5.bin"), Array.Empty<byte>());

        File.WriteAllBytes(Path.Combine(directoryPath, "test_batch.bin"), CreateCifar10Records(new[]
        {
            (label: (byte)0, pixel: (byte)10),
            (label: (byte)1, pixel: (byte)20)
        }));
    }

    private static void WriteCifar100Files(string directoryPath)
    {
        File.WriteAllBytes(Path.Combine(directoryPath, "train.bin"), CreateCifar100Records(new[]
        {
            (coarse: (byte)0, fine: (byte)0, pixel: (byte)10),
            (coarse: (byte)0, fine: (byte)1, pixel: (byte)20),
            (coarse: (byte)0, fine: (byte)2, pixel: (byte)30)
        }));

        File.WriteAllBytes(Path.Combine(directoryPath, "test.bin"), CreateCifar100Records(new[]
        {
            (coarse: (byte)0, fine: (byte)0, pixel: (byte)10),
            (coarse: (byte)0, fine: (byte)1, pixel: (byte)20)
        }));
    }

    private static byte[] CreateCifar10Records(IEnumerable<(byte label, byte pixel)> records)
    {
        const int recordSize = 3073;
        var list = records.ToList();
        var buffer = new byte[list.Count * recordSize];

        for (int r = 0; r < list.Count; r++)
        {
            int offset = r * recordSize;
            buffer[offset] = list[r].label;
            for (int i = 1; i < recordSize; i++)
            {
                buffer[offset + i] = list[r].pixel;
            }
        }

        return buffer;
    }

    private static byte[] CreateCifar100Records(IEnumerable<(byte coarse, byte fine, byte pixel)> records)
    {
        const int recordSize = 3074;
        var list = records.ToList();
        var buffer = new byte[list.Count * recordSize];

        for (int r = 0; r < list.Count; r++)
        {
            int offset = r * recordSize;
            buffer[offset] = list[r].coarse;
            buffer[offset + 1] = list[r].fine;
            for (int i = 2; i < recordSize; i++)
            {
                buffer[offset + i] = list[r].pixel;
            }
        }

        return buffer;
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

