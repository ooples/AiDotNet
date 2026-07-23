using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Benchmarks.Leaf;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Benchmarking;

internal static class ShakespeareFederatedBenchmarkSuiteRunner
{
    public static Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        ShakespeareFederatedBenchmarkOptions shakespeare,
        CancellationToken cancellationToken)
    {
        if (suite != BenchmarkSuite.Shakespeare)
        {
            throw new ArgumentOutOfRangeException(nameof(suite), suite, "Shakespeare runner only supports BenchmarkSuite.Shakespeare.");
        }

        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (shakespeare is null)
        {
            throw new ArgumentNullException(nameof(shakespeare));
        }

        if (model is not AiModelResult<T, Matrix<T>, Vector<T>> typedModel)
        {
            throw new NotSupportedException(
                $"Shakespeare benchmarking currently requires AiModelResult<T, Matrix<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        if (string.IsNullOrWhiteSpace(shakespeare.TrainFilePath))
        {
            throw new ArgumentException("ShakespeareFederatedBenchmarkOptions.TrainFilePath is required for Shakespeare benchmarking.", nameof(options));
        }

        int seed = options.Seed ?? 0;
        int? maxSamplesPerUser = shakespeare.MaxSamplesPerUser;

        if (options.CiMode && maxSamplesPerUser is null)
        {
            maxSamplesPerUser = 25;
        }

        var loadOptions = new LeafFederatedDatasetLoadOptions
        {
            MaxUsers = shakespeare.LoadOptions.MaxUsers ?? (options.CiMode ? 5 : null),
            ValidateDeclaredSampleCounts = shakespeare.LoadOptions.ValidateDeclaredSampleCounts
        };

        var leafLoader = new LeafShakespeareFederatedDatasetLoader();
        var dataset = leafLoader.LoadDatasetFromFiles(shakespeare.TrainFilePath!, shakespeare.TestFilePath, loadOptions);

        var trainClientData = dataset.Train.ToClientIdDictionary(out _);
        var testSplit = dataset.Test ?? dataset.Train;
        var testClientData = testSplit.ToClientIdDictionary(out _);

        var sampledTrainClientData = maxSamplesPerUser.HasValue
            ? SampleClientDatasets(trainClientData, maxSamplesPerUser.Value, seed)
            : new Dictionary<int, FederatedClientDataset<string[], string[]>>(trainClientData);

        var sampledTestClientData = maxSamplesPerUser.HasValue
            ? SampleClientDatasets(testClientData, maxSamplesPerUser.Value, seed)
            : new Dictionary<int, FederatedClientDataset<string[], string[]>>(testClientData);

        var (trainWindows, trainLabels, trainSamplesUsed) = ConcatenateClientData(sampledTrainClientData);
        var (testWindows, testLabels, testSamplesUsed) = ConcatenateClientData(sampledTestClientData);

        if (testSamplesUsed == 0)
        {
            return Task.FromResult(new BenchmarkSuiteReport
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
                    TrainSamplesUsed = trainSamplesUsed,
                    TestSamplesUsed = 0,
                    FeatureCount = 0,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = maxSamplesPerUser ?? 0
                }
            });
        }

        int sequenceLength = ResolveSequenceLength(shakespeare.SequenceLength, options.CiMode);
        var numOps = MathHelper.GetNumericOperations<T>();

        var tokenizer = typedModel.Tokenizer ?? CreateDefaultTokenizer(trainWindows, trainLabels, options);
        var encodingOptions = CreateEncodingOptions(typedModel.TokenizationConfig, sequenceLength);

        var testX = EncodeToMatrix(testWindows, tokenizer, encodingOptions, sequenceLength, numOps, cancellationToken);
        var testY = new Vector<T>(testSamplesUsed);

        var labelEncodingOptions = new EncodingOptions
        {
            AddSpecialTokens = false,
            Padding = false,
            Truncation = false,
            ReturnAttentionMask = false,
            ReturnTokenTypeIds = false,
            ReturnPositionIds = false,
            ReturnOffsets = false
        };

        for (int i = 0; i < testSamplesUsed; i++)
        {
            int labelId = EncodeSingleTokenLabel(testLabels[i], tokenizer, labelEncodingOptions);
            testY[i] = numOps.FromDouble(labelId);
        }

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
                    ClientsUsed = sampledTestClientData.Count,
                    TrainSamplesUsed = trainSamplesUsed,
                    TestSamplesUsed = testSamplesUsed,
                    FeatureCount = sequenceLength,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = maxSamplesPerUser ?? 0
                }
            });
        }

        double accuracy = Convert.ToDouble(StatisticsHelper<T>.CalculateAccuracy(testY, predicted, PredictionType.MultiClass));
        var (mse, rmse) = CalculateMseAndRmse(testY, predicted, numOps);

        var metrics = new List<BenchmarkMetricValue>
        {
            new() { Metric = BenchmarkMetric.TotalEvaluated, Value = testSamplesUsed },
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
                ClientsUsed = sampledTestClientData.Count,
                TrainSamplesUsed = trainSamplesUsed,
                TestSamplesUsed = testSamplesUsed,
                FeatureCount = sequenceLength,
                CiMode = options.CiMode,
                Seed = seed,
                MaxSamplesPerUser = maxSamplesPerUser ?? 0
            }
        });
    }

    private static int ResolveSequenceLength(int? configured, bool ciMode)
    {
        int value = configured ?? (ciMode ? 32 : 80);
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(configured), "Sequence length must be positive when specified.");
        }

        return value;
    }

    private static ITokenizer CreateDefaultTokenizer(
        IReadOnlyList<string> trainWindows,
        IReadOnlyList<string> trainLabels,
        BenchmarkingOptions options)
    {
        int sampleCount = options.CiMode ? 200 : 5000;
        var corpus = new List<string>(Math.Min(trainWindows.Count, sampleCount) + Math.Min(trainLabels.Count, sampleCount));

        foreach (var window in trainWindows.Take(sampleCount))
        {
            corpus.Add(window);
        }

        foreach (var label in trainLabels.Take(sampleCount))
        {
            corpus.Add(label);
        }

        return CharacterTokenizer.Train(corpus);
    }

    private static EncodingOptions CreateEncodingOptions(
        AiDotNet.Tokenization.Configuration.TokenizationConfig? config,
        int sequenceLength)
    {
        var options = config?.ToEncodingOptions() ?? new EncodingOptions();

        options.MaxLength = sequenceLength;
        options.Padding = true;
        options.Truncation = true;
        options.ReturnAttentionMask = false;
        options.ReturnTokenTypeIds = false;
        options.ReturnPositionIds = false;
        options.ReturnOffsets = false;

        if (config is null)
        {
            options.AddSpecialTokens = false;
            options.PaddingSide = "right";
            options.TruncationSide = "right";
        }

        return options;
    }

    private static int EncodeSingleTokenLabel(string label, ITokenizer tokenizer, EncodingOptions labelEncodingOptions)
    {
        if (string.IsNullOrEmpty(label))
        {
            return ResolvePadTokenId(tokenizer);
        }

        var result = tokenizer.Encode(label, labelEncodingOptions);
        if (result.TokenIds.Count == 0)
        {
            return ResolvePadTokenId(tokenizer);
        }

        return result.TokenIds[0];
    }

    private static Matrix<T> EncodeToMatrix<T>(
        IReadOnlyList<string> texts,
        ITokenizer tokenizer,
        EncodingOptions encodingOptions,
        int maxSequenceLength,
        INumericOperations<T> numOps,
        CancellationToken cancellationToken)
    {
        if (texts.Count == 0)
        {
            return new Matrix<T>(0, 0);
        }

        int padTokenId = ResolvePadTokenId(tokenizer);
        var matrix = new Matrix<T>(texts.Count, maxSequenceLength);

        for (int row = 0; row < texts.Count; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var text = texts[row] ?? string.Empty;
            var tokenIds = tokenizer.Encode(text, encodingOptions).TokenIds;
            WriteTokenIdsRow(matrix, row, tokenIds, maxSequenceLength, padTokenId, numOps);
        }

        return matrix;
    }

    private static int ResolvePadTokenId(ITokenizer tokenizer)
    {
        try
        {
            return tokenizer.Vocabulary.GetTokenId(tokenizer.SpecialTokens.PadToken);
        }
        catch
        {
            return 0;
        }
    }

    private static void WriteTokenIdsRow<T>(
        Matrix<T> matrix,
        int row,
        IReadOnlyList<int>? tokenIds,
        int maxSequenceLength,
        int padTokenId,
        INumericOperations<T> numOps)
    {
        int count = tokenIds?.Count ?? 0;
        if (count == 0)
        {
            for (int col = 0; col < maxSequenceLength; col++)
            {
                matrix[row, col] = numOps.FromDouble(padTokenId);
            }

            return;
        }

        var ids = tokenIds!;
        int take = Math.Min(count, maxSequenceLength);
        for (int col = 0; col < take; col++)
        {
            matrix[row, col] = numOps.FromDouble(ids[col]);
        }

        for (int col = take; col < maxSequenceLength; col++)
        {
            matrix[row, col] = numOps.FromDouble(padTokenId);
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

    private static Dictionary<int, FederatedClientDataset<string[], string[]>> SampleClientDatasets(
        IReadOnlyDictionary<int, FederatedClientDataset<string[], string[]>> clientData,
        int maxSamplesPerUser,
        int seed)
    {
        if (maxSamplesPerUser <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamplesPerUser), "Max samples per user must be positive.");
        }

        var sampled = new Dictionary<int, FederatedClientDataset<string[], string[]>>(clientData.Count);

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

            var newX = new string[take];
            var newY = new string[take];

            for (int row = 0; row < take; row++)
            {
                int sourceRow = selected[row];
                newX[row] = x[sourceRow];
                newY[row] = y[sourceRow];
            }

            sampled[clientId] = new FederatedClientDataset<string[], string[]>(newX, newY, take);
        }

        return sampled;
    }

    private static (IReadOnlyList<string> Windows, IReadOnlyList<string> Labels, int TotalCount) ConcatenateClientData(
        IReadOnlyDictionary<int, FederatedClientDataset<string[], string[]>> clientData)
    {
        int totalCount = 0;
        foreach (var dataset in clientData.Values)
        {
            if (dataset.SampleCount <= 0)
            {
                continue;
            }

            totalCount += dataset.SampleCount;
        }

        if (totalCount == 0)
        {
            return (Array.Empty<string>(), Array.Empty<string>(), 0);
        }

        var windows = new List<string>(totalCount);
        var labels = new List<string>(totalCount);

        foreach (var dataset in clientData.Values)
        {
            for (int i = 0; i < dataset.SampleCount; i++)
            {
                windows.Add(dataset.Features[i]);
                labels.Add(dataset.Labels[i]);
            }
        }

        return (windows, labels, totalCount);
    }
}
