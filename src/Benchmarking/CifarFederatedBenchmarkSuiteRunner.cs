using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;

namespace AiDotNet.Benchmarking;

internal static class CifarFederatedBenchmarkSuiteRunner
{
    public static Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        CancellationToken cancellationToken)
    {
        if (suite is not (BenchmarkSuite.CIFAR10 or BenchmarkSuite.CIFAR100))
        {
            throw new ArgumentOutOfRangeException(nameof(suite), suite, "CIFAR benchmark runner only supports CIFAR10/CIFAR100.");
        }

        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (model is not AiModelResult<T, Matrix<T>, Vector<T>> typedModel)
        {
            throw new NotSupportedException(
                $"CIFAR benchmarking currently requires AiModelResult<T, Matrix<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        var cifar = suite == BenchmarkSuite.CIFAR10
            ? options.Vision?.Cifar10
            : options.Vision?.Cifar100;

        if (cifar is null)
        {
            throw new InvalidOperationException($"BenchmarkingOptions.Vision must be provided when running {suite}.");
        }

        if (string.IsNullOrWhiteSpace(cifar.DataDirectoryPath))
        {
            throw new ArgumentException("CifarFederatedBenchmarkOptions.DataDirectoryPath is required for CIFAR benchmarking.", nameof(options));
        }

        int seed = options.Seed ?? 0;

        int? maxTrainSamples = cifar.MaxTrainSamples;
        int? maxTestSamples = cifar.MaxTestSamples;

        if (options.CiMode && maxTrainSamples is null)
        {
            maxTrainSamples = 200;
        }

        if (options.CiMode && maxTestSamples is null)
        {
            maxTestSamples = 100;
        }

        int clientCount = cifar.ClientCount ?? (options.CiMode ? 5 : 100);
        if (clientCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(cifar.ClientCount), "Client count must be positive when specified.");
        }

        int shardsPerClient = cifar.ShardsPerClient ?? 2;
        if (shardsPerClient <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(cifar.ShardsPerClient), "Shards per client must be positive when specified.");
        }

        var dataDirectoryPath = ResolveDatasetDirectory(cifar.DataDirectoryPath!, suite);
        var numOps = MathHelper.GetNumericOperations<T>();

        var random = RandomHelper.CreateSeededRandom(unchecked((seed * 16777619) ^ (int)suite));

        var (trainX, trainY) = suite == BenchmarkSuite.CIFAR10
            ? LoadCifar10Split(dataDirectoryPath, isTrain: true, maxSamples: maxTrainSamples, normalizePixels: cifar.NormalizePixels, numOps, cancellationToken)
            : LoadCifar100Split(dataDirectoryPath, isTrain: true, maxSamples: maxTrainSamples, normalizePixels: cifar.NormalizePixels, numOps, cancellationToken);

        var (testX, testY) = suite == BenchmarkSuite.CIFAR10
            ? LoadCifar10Split(dataDirectoryPath, isTrain: false, maxSamples: maxTestSamples, normalizePixels: cifar.NormalizePixels, numOps, cancellationToken)
            : LoadCifar100Split(dataDirectoryPath, isTrain: false, maxSamples: maxTestSamples, normalizePixels: cifar.NormalizePixels, numOps, cancellationToken);

        if (testX.Rows == 0)
        {
            return Task.FromResult(new BenchmarkSuiteReport
            {
                Suite = suite,
                Kind = BenchmarkSuiteKind.DatasetSuite,
                Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
                Status = BenchmarkExecutionStatus.Skipped,
                FailureReason = "No test samples available.",
                Metrics = new[]
                {
                    new BenchmarkMetricValue { Metric = BenchmarkMetric.TotalEvaluated, Value = 0.0 }
                },
                DataSelection = new BenchmarkDataSelectionSummary
                {
                    ClientsUsed = clientCount,
                    TrainSamplesUsed = trainX.Rows,
                    TestSamplesUsed = 0,
                    FeatureCount = trainX.Columns,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = maxTrainSamples ?? 0
                }
            });
        }

        _ = PartitionTrainingSet(trainY, clientCount, cifar.PartitioningStrategy, cifar.DirichletAlpha, shardsPerClient, random, numOps);

        var predicted = typedModel.Predict(testX);
        if (predicted.Length != testY.Length)
        {
            return Task.FromResult(new BenchmarkSuiteReport
            {
                Suite = suite,
                Kind = BenchmarkSuiteKind.DatasetSuite,
                Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
                Status = BenchmarkExecutionStatus.Failed,
                FailureReason = "Prediction output length does not match label length.",
                Metrics = new[]
                {
                    new BenchmarkMetricValue { Metric = BenchmarkMetric.TotalEvaluated, Value = 0.0 }
                },
                DataSelection = new BenchmarkDataSelectionSummary
                {
                    ClientsUsed = clientCount,
                    TrainSamplesUsed = trainX.Rows,
                    TestSamplesUsed = testX.Rows,
                    FeatureCount = testX.Columns,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = maxTrainSamples ?? 0
                }
            });
        }

        double accuracy = Convert.ToDouble(StatisticsHelper<T>.CalculateAccuracy(testY, predicted, PredictionType.MultiClass));
        var (mse, rmse) = CalculateMseAndRmse(testY, predicted, numOps);

        var metrics = new List<BenchmarkMetricValue>
        {
            new() { Metric = BenchmarkMetric.TotalEvaluated, Value = testX.Rows },
            new() { Metric = BenchmarkMetric.Accuracy, Value = accuracy },
            new() { Metric = BenchmarkMetric.MeanSquaredError, Value = mse },
            new() { Metric = BenchmarkMetric.RootMeanSquaredError, Value = rmse }
        };

        return Task.FromResult(new BenchmarkSuiteReport
        {
            Suite = suite,
            Kind = BenchmarkSuiteKind.DatasetSuite,
            Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
            Status = BenchmarkExecutionStatus.Succeeded,
            FailureReason = null,
            Metrics = metrics,
            DataSelection = new BenchmarkDataSelectionSummary
            {
                ClientsUsed = clientCount,
                TrainSamplesUsed = trainX.Rows,
                TestSamplesUsed = testX.Rows,
                FeatureCount = testX.Columns,
                CiMode = options.CiMode,
                Seed = seed,
                MaxSamplesPerUser = maxTrainSamples ?? 0
            }
        });
    }

    private static (double Mse, double Rmse) CalculateMseAndRmse<T>(Vector<T> actual, Vector<T> predicted, INumericOperations<T> numOps)
    {
        if (actual.Length == 0)
        {
            return (0.0, 0.0);
        }

        double sum = 0.0;
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = numOps.ToDouble(actual[i]) - numOps.ToDouble(predicted[i]);
            sum += diff * diff;
        }

        double mse = sum / actual.Length;
        return (mse, Math.Sqrt(mse));
    }

    private static string ResolveDatasetDirectory(string dataDirectoryPath, BenchmarkSuite suite)
    {
        if (suite == BenchmarkSuite.CIFAR10)
        {
            if (File.Exists(Path.Combine(dataDirectoryPath, "data_batch_1.bin")))
            {
                return dataDirectoryPath;
            }

            var nested = Path.Combine(dataDirectoryPath, "cifar-10-batches-bin");
            if (File.Exists(Path.Combine(nested, "data_batch_1.bin")))
            {
                return nested;
            }
        }
        else if (suite == BenchmarkSuite.CIFAR100)
        {
            if (File.Exists(Path.Combine(dataDirectoryPath, "train.bin")))
            {
                return dataDirectoryPath;
            }

            var nested = Path.Combine(dataDirectoryPath, "cifar-100-binary");
            if (File.Exists(Path.Combine(nested, "train.bin")))
            {
                return nested;
            }
        }

        return dataDirectoryPath;
    }

    private static (Matrix<T> X, Vector<T> Y) LoadCifar10Split<T>(
        string directoryPath,
        bool isTrain,
        int? maxSamples,
        bool normalizePixels,
        INumericOperations<T> numOps,
        CancellationToken cancellationToken)
    {
        var files = isTrain
            ? new[]
            {
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin"
            }
            : new[] { "test_batch.bin" };

        return LoadCifarBinarySplit(directoryPath, files, recordSize: 3073, labelOffset: 0, pixelOffset: 1, maxSamples, normalizePixels, numOps, cancellationToken);
    }

    private static (Matrix<T> X, Vector<T> Y) LoadCifar100Split<T>(
        string directoryPath,
        bool isTrain,
        int? maxSamples,
        bool normalizePixels,
        INumericOperations<T> numOps,
        CancellationToken cancellationToken)
    {
        var file = isTrain ? "train.bin" : "test.bin";
        return LoadCifarBinarySplit(directoryPath, new[] { file }, recordSize: 3074, labelOffset: 1, pixelOffset: 2, maxSamples, normalizePixels, numOps, cancellationToken);
    }

    private static (Matrix<T> X, Vector<T> Y) LoadCifarBinarySplit<T>(
        string directoryPath,
        IReadOnlyList<string> files,
        int recordSize,
        int labelOffset,
        int pixelOffset,
        int? maxSamples,
        bool normalizePixels,
        INumericOperations<T> numOps,
        CancellationToken cancellationToken)
    {
        const int featureCount = 32 * 32 * 3;

        var paths = files.Select(f => Path.Combine(directoryPath, f)).ToArray();
        foreach (var path in paths)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"CIFAR binary file not found: {path}", path);
            }
        }

        int available = 0;
        foreach (var path in paths)
        {
            long length = new FileInfo(path).Length;
            if (length % recordSize != 0)
            {
                throw new InvalidDataException($"CIFAR binary file '{path}' length is not a multiple of record size {recordSize}.");
            }

            available += (int)(length / recordSize);
        }

        int count = maxSamples.HasValue ? Math.Min(maxSamples.Value, available) : available;
        if (count < 0)
        {
            count = 0;
        }

        var x = new Matrix<T>(count, featureCount);
        var y = new Vector<T>(count);

        var record = new byte[recordSize];
        int row = 0;

        foreach (var path in paths)
        {
            using var stream = File.OpenRead(path);

            while (row < count)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int bytesRead = 0;
                while (bytesRead < recordSize)
                {
                    int n = stream.Read(record, bytesRead, recordSize - bytesRead);
                    if (n == 0)
                    {
                        break;
                    }

                    bytesRead += n;
                }

                if (bytesRead == 0)
                {
                    break;
                }

                if (bytesRead != recordSize)
                {
                    throw new InvalidDataException($"CIFAR binary file '{path}' ended with a partial record.");
                }

                byte label = record[labelOffset];
                y[row] = numOps.FromDouble(label);

                for (int i = 0; i < featureCount; i++)
                {
                    double pixel = record[pixelOffset + i];
                    x[row, i] = numOps.FromDouble(normalizePixels ? pixel / 255.0 : pixel);
                }

                row++;
            }

            if (row >= count)
            {
                break;
            }
        }

        return (x, y);
    }

    private static IReadOnlyDictionary<int, List<int>> PartitionTrainingSet<T>(
        Vector<T> labels,
        int clientCount,
        FederatedPartitioningStrategy strategy,
        double dirichletAlpha,
        int shardsPerClient,
        Random random,
        INumericOperations<T> numOps)
    {
        if (labels.Length == 0 || clientCount <= 0)
        {
            return new Dictionary<int, List<int>>();
        }

        return strategy switch
        {
            FederatedPartitioningStrategy.IID => PartitionIid(labels.Length, clientCount, random),
            FederatedPartitioningStrategy.DirichletLabel => PartitionDirichletByLabel(labels, clientCount, dirichletAlpha, random, numOps),
            FederatedPartitioningStrategy.ShardByLabel => PartitionShardByLabel(labels, clientCount, shardsPerClient, random, numOps),
            _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown partitioning strategy.")
        };
    }

    private static IReadOnlyDictionary<int, List<int>> PartitionIid(int sampleCount, int clientCount, Random random)
    {
        var indices = Enumerable.Range(0, sampleCount).ToArray();
        ShuffleInPlace(indices, random);

        var clients = new Dictionary<int, List<int>>(clientCount);
        for (int c = 0; c < clientCount; c++)
        {
            clients[c] = new List<int>();
        }

        for (int i = 0; i < indices.Length; i++)
        {
            int clientId = i % clientCount;
            clients[clientId].Add(indices[i]);
        }

        return clients;
    }

    private static IReadOnlyDictionary<int, List<int>> PartitionShardByLabel<T>(
        Vector<T> labels,
        int clientCount,
        int shardsPerClient,
        Random random,
        INumericOperations<T> numOps)
    {
        int sampleCount = labels.Length;
        int shardCount = clientCount * shardsPerClient;

        var indices = Enumerable.Range(0, sampleCount)
            .OrderBy(i => numOps.ToInt32(labels[i]))
            .ToArray();

        int shardSize = Math.Max(1, sampleCount / shardCount);

        var shards = new List<int[]>(shardCount);
        for (int s = 0; s < shardCount; s++)
        {
            int start = s * shardSize;
            if (start >= sampleCount)
            {
                shards.Add(Array.Empty<int>());
                continue;
            }

            int end = (s == shardCount - 1) ? sampleCount : Math.Min(sampleCount, start + shardSize);
            int length = end - start;
            var shard = new int[length];
            Array.Copy(indices, start, shard, 0, length);
            shards.Add(shard);
        }

        var shardOrder = Enumerable.Range(0, shardCount).ToArray();
        ShuffleInPlace(shardOrder, random);

        var clients = new Dictionary<int, List<int>>(clientCount);
        for (int c = 0; c < clientCount; c++)
        {
            clients[c] = new List<int>();
        }

        int shardIndex = 0;
        for (int c = 0; c < clientCount; c++)
        {
            for (int k = 0; k < shardsPerClient && shardIndex < shardOrder.Length; k++, shardIndex++)
            {
                clients[c].AddRange(shards[shardOrder[shardIndex]]);
            }
        }

        return clients;
    }

    private static IReadOnlyDictionary<int, List<int>> PartitionDirichletByLabel<T>(
        Vector<T> labels,
        int clientCount,
        double alpha,
        Random random,
        INumericOperations<T> numOps)
    {
        if (alpha <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Dirichlet alpha must be positive.");
        }

        var clients = new Dictionary<int, List<int>>(clientCount);
        for (int c = 0; c < clientCount; c++)
        {
            clients[c] = new List<int>();
        }

        var labelToIndices = new Dictionary<int, List<int>>();
        for (int i = 0; i < labels.Length; i++)
        {
            int label = numOps.ToInt32(labels[i]);
            if (!labelToIndices.TryGetValue(label, out var list))
            {
                list = new List<int>();
                labelToIndices[label] = list;
            }

            list.Add(i);
        }

        foreach (var indices in labelToIndices.Values)
        {
            ShuffleInPlace(indices, random);

            var proportions = SampleDirichlet(clientCount, alpha, random);
            var counts = ProportionsToCounts(indices.Count, proportions);

            int offset = 0;
            for (int c = 0; c < clientCount; c++)
            {
                int take = counts[c];
                if (take <= 0)
                {
                    continue;
                }

                clients[c].AddRange(indices.GetRange(offset, take));
                offset += take;
            }
        }

        return clients;
    }

    private static int[] ProportionsToCounts(int total, double[] proportions)
    {
        var counts = new int[proportions.Length];
        int assigned = 0;

        for (int i = 0; i < proportions.Length; i++)
        {
            int count = (int)Math.Floor(total * proportions[i]);
            counts[i] = count;
            assigned += count;
        }

        int remaining = total - assigned;
        int index = 0;
        while (remaining > 0)
        {
            counts[index % counts.Length]++;
            remaining--;
            index++;
        }

        return counts;
    }

    private static double[] SampleDirichlet(int dimension, double alpha, Random random)
    {
        var gammas = new double[dimension];
        double sum = 0.0;

        for (int i = 0; i < dimension; i++)
        {
            double g = SampleGamma(alpha, random);
            gammas[i] = g;
            sum += g;
        }

        if (sum <= 0.0)
        {
            var uniform = 1.0 / dimension;
            for (int i = 0; i < dimension; i++)
            {
                gammas[i] = uniform;
            }

            return gammas;
        }

        for (int i = 0; i < dimension; i++)
        {
            gammas[i] /= sum;
        }

        return gammas;
    }

    private static double SampleGamma(double shape, Random random)
    {
        if (shape < 1.0)
        {
            return SampleGamma(shape + 1.0, random) * Math.Pow(random.NextDouble(), 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            double x;
            double v;

            do
            {
                x = random.NextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0.0);

            v = v * v * v;
            double u = random.NextDouble();

            if (u < 1.0 - 0.0331 * x * x * x * x)
            {
                return d * v;
            }

            if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
            {
                return d * v;
            }
        }
    }

    private static void ShuffleInPlace(int[] values, Random random)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static void ShuffleInPlace(List<int> values, Random random)
    {
        for (int i = values.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }
}
