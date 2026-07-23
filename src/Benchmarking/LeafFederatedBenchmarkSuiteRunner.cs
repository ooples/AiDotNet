using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.Benchmarking;

internal static class LeafFederatedBenchmarkSuiteRunner
{
    public static async Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        LeafFederatedBenchmarkOptions leaf,
        CancellationToken cancellationToken)
    {
        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (leaf is null)
        {
            throw new ArgumentNullException(nameof(leaf));
        }

        if (model is not AiModelResult<T, Matrix<T>, Vector<T>> typedModel)
        {
            throw new NotSupportedException(
                $"Leaf-backed benchmarking currently requires AiModelResult<T, Matrix<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        if (string.IsNullOrWhiteSpace(leaf.TrainFilePath))
        {
            throw new ArgumentException("LeafFederatedBenchmarkOptions.TrainFilePath is required for leaf-backed benchmarking.", nameof(options));
        }

        int seed = options.Seed ?? 0;
        int? maxSamplesPerUser = leaf.MaxSamplesPerUser;

        if (options.CiMode && maxSamplesPerUser is null)
        {
            maxSamplesPerUser = 25;
        }

        var loadOptions = new AiDotNet.FederatedLearning.Benchmarks.Leaf.LeafFederatedDatasetLoadOptions
        {
            MaxUsers = leaf.LoadOptions.MaxUsers ?? (options.CiMode ? 5 : null),
            ValidateDeclaredSampleCounts = leaf.LoadOptions.ValidateDeclaredSampleCounts
        };

        var loader = DataLoaders.FromLeafFederatedJsonFiles<T>(
            trainFilePath: leaf.TrainFilePath!,
            testFilePath: leaf.TestFilePath,
            options: loadOptions);

        try
        {
            await loader.LoadAsync(cancellationToken).ConfigureAwait(false);

            var trainClientData = loader.TrainSplit.ToClientIdDictionary(out _);
            var testSplit = loader.TestSplit ?? loader.TrainSplit;
            var testClientData = testSplit.ToClientIdDictionary(out _);

            var sampledTrainClientData = maxSamplesPerUser.HasValue
                ? SampleClientDatasets(trainClientData, maxSamplesPerUser.Value, seed)
                : new Dictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>>(trainClientData);

            var sampledTestClientData = maxSamplesPerUser.HasValue
                ? SampleClientDatasets(testClientData, maxSamplesPerUser.Value, seed)
                : new Dictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>>(testClientData);

            var (trainX, trainY) = ConcatenateClientData(sampledTrainClientData);
            var (testX, testY) = ConcatenateClientData(sampledTestClientData);

            if (testX.Rows == 0)
            {
                return new BenchmarkSuiteReport
                {
                    Suite = suite,
                    Kind = BenchmarkSuiteKind.DatasetSuite,
                    Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
                    Status = BenchmarkExecutionStatus.Skipped,
                    FailureReason = "No test samples available after sampling.",
                    Metrics = new[]
                    {
                        new BenchmarkMetricValue { Metric = BenchmarkMetric.TotalEvaluated, Value = 0.0 }
                    },
                    DataSelection = new BenchmarkDataSelectionSummary
                    {
                        ClientsUsed = sampledTestClientData.Count,
                        TrainSamplesUsed = trainX.Rows,
                        TestSamplesUsed = 0,
                        FeatureCount = trainX.Columns,
                        CiMode = options.CiMode,
                        Seed = seed,
                        MaxSamplesPerUser = maxSamplesPerUser ?? 0
                    }
                };
            }

            var predictionType = PredictionTypeInference.InferFromTargets<T, Vector<T>>(testY);
            var predicted = typedModel.Predict(testX);

            if (predicted.Length != testY.Length)
            {
                return new BenchmarkSuiteReport
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
                        ClientsUsed = sampledTestClientData.Count,
                        TrainSamplesUsed = trainX.Rows,
                        TestSamplesUsed = testX.Rows,
                        FeatureCount = testX.Columns,
                        CiMode = options.CiMode,
                        Seed = seed,
                        MaxSamplesPerUser = maxSamplesPerUser ?? 0
                    }
                };
            }

            var numOps = MathHelper.GetNumericOperations<T>();
            double accuracy = Convert.ToDouble(StatisticsHelper<T>.CalculateAccuracy(testY, predicted, predictionType));
            var (mse, rmse) = CalculateMseAndRmse(testY, predicted, numOps);

            var metrics = new List<BenchmarkMetricValue>
            {
                new() { Metric = BenchmarkMetric.TotalEvaluated, Value = testX.Rows },
                new() { Metric = BenchmarkMetric.Accuracy, Value = accuracy },
                new() { Metric = BenchmarkMetric.MeanSquaredError, Value = mse },
                new() { Metric = BenchmarkMetric.RootMeanSquaredError, Value = rmse }
            };

            return new BenchmarkSuiteReport
            {
                Suite = suite,
                Kind = BenchmarkSuiteKind.DatasetSuite,
                Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
                Status = BenchmarkExecutionStatus.Succeeded,
                FailureReason = null,
                Metrics = metrics,
                DataSelection = new BenchmarkDataSelectionSummary
                {
                    ClientsUsed = sampledTestClientData.Count,
                    TrainSamplesUsed = trainX.Rows,
                    TestSamplesUsed = testX.Rows,
                    FeatureCount = testX.Columns,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = maxSamplesPerUser ?? 0
                }
            };
        }
        finally
        {
            loader.Unload();
        }
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

    private static Dictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>> SampleClientDatasets<T>(
        IReadOnlyDictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>> clientData,
        int maxSamplesPerUser,
        int seed)
    {
        if (maxSamplesPerUser <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamplesPerUser), "Max samples per user must be positive.");
        }

        var sampled = new Dictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>>(clientData.Count);

        foreach (var kvp in clientData)
        {
            int clientId = kvp.Key;
            var dataset = kvp.Value;

            if (dataset.SampleCount == 0)
            {
                sampled[clientId] = dataset;
                continue;
            }

            int take = Math.Min(maxSamplesPerUser, dataset.SampleCount);
            if (take == dataset.SampleCount)
            {
                sampled[clientId] = dataset;
                continue;
            }

            int combinedSeed = unchecked((seed * 16777619) ^ clientId);
            var random = RandomHelper.CreateSeededRandom(combinedSeed);

            var indices = Enumerable.Range(0, dataset.SampleCount).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            var selected = indices.Take(take).ToArray();
            Array.Sort(selected);

            var x = dataset.Features;
            var y = dataset.Labels;

            var newX = new Matrix<T>(take, x.Columns);
            var newY = new Vector<T>(take);

            for (int row = 0; row < take; row++)
            {
                int sourceRow = selected[row];
                newX.SetRow(row, x.GetRow(sourceRow));
                newY[row] = y[sourceRow];
            }

            sampled[clientId] = new FederatedClientDataset<Matrix<T>, Vector<T>>(newX, newY, take);
        }

        return sampled;
    }

    private static (Matrix<T> X, Vector<T> Y) ConcatenateClientData<T>(
        IReadOnlyDictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>> clientData)
    {
        int totalCount = 0;
        int featureCount = 0;

        foreach (var dataset in clientData.Values)
        {
            if (dataset.SampleCount <= 0)
            {
                continue;
            }

            totalCount += dataset.SampleCount;

            if (featureCount == 0 && dataset.Features.Rows > 0)
            {
                featureCount = dataset.Features.Columns;
            }
        }

        if (totalCount == 0)
        {
            return (new Matrix<T>(0, 0), new Vector<T>(0));
        }

        var allX = new Matrix<T>(totalCount, featureCount);
        var allY = new Vector<T>(totalCount);

        int rowIndex = 0;
        foreach (var dataset in clientData.Values)
        {
            var x = dataset.Features;
            var y = dataset.Labels;

            for (int i = 0; i < dataset.SampleCount; i++)
            {
                allX.SetRow(rowIndex, x.GetRow(i));
                allY[rowIndex] = y[i];
                rowIndex++;
            }
        }

        return (allX, allY);
    }
}
