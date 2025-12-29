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

internal static class Sent140FederatedBenchmarkSuiteRunner
{
    public static Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        PredictionModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        Sent140FederatedBenchmarkOptions sent140,
        CancellationToken cancellationToken)
    {
        if (suite != BenchmarkSuite.Sent140)
        {
            throw new ArgumentOutOfRangeException(nameof(suite), suite, "Sent140 runner only supports BenchmarkSuite.Sent140.");
        }

        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (sent140 is null)
        {
            throw new ArgumentNullException(nameof(sent140));
        }

        if (model is not PredictionModelResult<T, Matrix<T>, Vector<T>> typedModel)
        {
            throw new NotSupportedException(
                $"Sent140 benchmarking currently requires PredictionModelResult<T, Matrix<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        if (string.IsNullOrWhiteSpace(sent140.TrainFilePath))
        {
            throw new ArgumentException("Sent140FederatedBenchmarkOptions.TrainFilePath is required for Sent140 benchmarking.", nameof(options));
        }

        int seed = options.Seed ?? 0;
        int? maxSamplesPerUser = sent140.MaxSamplesPerUser;

        if (options.CiMode && maxSamplesPerUser is null)
        {
            maxSamplesPerUser = 25;
        }

        var loadOptions = new LeafFederatedDatasetLoadOptions
        {
            MaxUsers = sent140.LoadOptions.MaxUsers ?? (options.CiMode ? 5 : null),
            ValidateDeclaredSampleCounts = sent140.LoadOptions.ValidateDeclaredSampleCounts
        };

        var leafLoader = new LeafSent140FederatedDatasetLoader();
        var dataset = leafLoader.LoadDatasetFromFiles(sent140.TrainFilePath!, sent140.TestFilePath, loadOptions);

        var trainClientData = dataset.Train.ToClientIdDictionary(out _);
        var testSplit = dataset.Test ?? dataset.Train;
        var testClientData = testSplit.ToClientIdDictionary(out _);

        var sampledTrainClientData = maxSamplesPerUser.HasValue
            ? SampleClientDatasets(trainClientData, maxSamplesPerUser.Value, seed)
            : new Dictionary<int, FederatedClientDataset<string[], int[]>>(trainClientData);

        var sampledTestClientData = maxSamplesPerUser.HasValue
            ? SampleClientDatasets(testClientData, maxSamplesPerUser.Value, seed)
            : new Dictionary<int, FederatedClientDataset<string[], int[]>>(testClientData);

        var (trainTexts, _, trainSamplesUsed) = ConcatenateClientData(sampledTrainClientData);
        var (testTexts, testLabels, testSamplesUsed) = ConcatenateClientData(sampledTestClientData);

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

        int maxSequenceLength = ResolveMaxSequenceLength(sent140.MaxSequenceLength, options.CiMode);
        var numOps = MathHelper.GetNumericOperations<T>();

        var tokenizer = typedModel.Tokenizer ?? CreateDefaultTokenizer(trainTexts, options, sent140);
        var encodingOptions = CreateEncodingOptions(typedModel.TokenizationConfig, maxSequenceLength);

        var testX = EncodeToMatrix(testTexts, tokenizer, encodingOptions, maxSequenceLength, numOps, cancellationToken);
        var testY = new Vector<T>(testSamplesUsed);
        for (int i = 0; i < testSamplesUsed; i++)
        {
            testY[i] = numOps.FromDouble(testLabels[i]);
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
                    FeatureCount = maxSequenceLength,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = maxSamplesPerUser ?? 0
                }
            });
        }

        double accuracy = Convert.ToDouble(StatisticsHelper<T>.CalculateAccuracy(testY, predicted, PredictionType.BinaryClassification));
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
                FeatureCount = maxSequenceLength,
                CiMode = options.CiMode,
                Seed = seed,
                MaxSamplesPerUser = maxSamplesPerUser ?? 0
            }
        });
    }

    private static int ResolveMaxSequenceLength(int? configured, bool ciMode)
    {
        int value = configured ?? (ciMode ? 32 : 128);
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(configured), "Max sequence length must be positive when specified.");
        }

        return value;
    }

    private static ITokenizer CreateDefaultTokenizer(
        IReadOnlyList<string> trainTexts,
        BenchmarkingOptions options,
        Sent140FederatedBenchmarkOptions sent140)
    {
        int vocabSize = sent140.TokenizerVocabularySize ?? (options.CiMode ? 512 : 4096);
        if (vocabSize <= 32)
        {
            throw new ArgumentOutOfRangeException(nameof(sent140.TokenizerVocabularySize), "Tokenizer vocabulary size must be greater than 32.");
        }

        int sampleCount = sent140.TokenizerTrainingSampleCount ?? (options.CiMode ? 200 : 10000);
        if (sampleCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sent140.TokenizerTrainingSampleCount), "Tokenizer training sample count must be positive when specified.");
        }

        var corpus = trainTexts.Take(sampleCount).ToList();
        return WordPieceTokenizer.Train(corpus, vocabSize);
    }

    private static EncodingOptions CreateEncodingOptions(
        AiDotNet.Tokenization.Configuration.TokenizationConfig? config,
        int maxSequenceLength)
    {
        var options = config?.ToEncodingOptions() ?? new EncodingOptions();

        options.MaxLength = maxSequenceLength;
        options.Padding = true;
        options.Truncation = true;
        options.ReturnAttentionMask = false;
        options.ReturnTokenTypeIds = false;
        options.ReturnPositionIds = false;
        options.ReturnOffsets = false;

        if (config is null)
        {
            options.AddSpecialTokens = true;
            options.PaddingSide = "right";
            options.TruncationSide = "right";
        }

        return options;
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

    private static Dictionary<int, FederatedClientDataset<string[], int[]>> SampleClientDatasets(
        IReadOnlyDictionary<int, FederatedClientDataset<string[], int[]>> clientData,
        int maxSamplesPerUser,
        int seed)
    {
        if (maxSamplesPerUser <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamplesPerUser), "Max samples per user must be positive.");
        }

        var sampled = new Dictionary<int, FederatedClientDataset<string[], int[]>>(clientData.Count);

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
            var newY = new int[take];

            for (int row = 0; row < take; row++)
            {
                int sourceRow = selected[row];
                newX[row] = x[sourceRow];
                newY[row] = y[sourceRow];
            }

            sampled[clientId] = new FederatedClientDataset<string[], int[]>(newX, newY, take);
        }

        return sampled;
    }

    private static (IReadOnlyList<string> Texts, IReadOnlyList<int> Labels, int TotalCount) ConcatenateClientData(
        IReadOnlyDictionary<int, FederatedClientDataset<string[], int[]>> clientData)
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
            return (Array.Empty<string>(), Array.Empty<int>(), 0);
        }

        var texts = new List<string>(totalCount);
        var labels = new List<int>(totalCount);

        foreach (var dataset in clientData.Values)
        {
            for (int i = 0; i < dataset.SampleCount; i++)
            {
                texts.Add(dataset.Features[i]);
                labels.Add(dataset.Labels[i]);
            }
        }

        return (texts, labels, totalCount);
    }
}
