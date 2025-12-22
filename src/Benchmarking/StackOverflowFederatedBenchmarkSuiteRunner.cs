using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Benchmarks.Leaf;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.Benchmarking;

internal static class StackOverflowFederatedBenchmarkSuiteRunner
{
    public static Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        PredictionModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        StackOverflowFederatedBenchmarkOptions stackOverflow,
        CancellationToken cancellationToken)
    {
        if (suite != BenchmarkSuite.StackOverflow)
        {
            throw new ArgumentOutOfRangeException(nameof(suite), suite, "StackOverflow runner only supports BenchmarkSuite.StackOverflow.");
        }

        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (stackOverflow is null)
        {
            throw new ArgumentNullException(nameof(stackOverflow));
        }

        if (model is not PredictionModelResult<T, Matrix<T>, Vector<T>> typedModel)
        {
            throw new NotSupportedException(
                $"StackOverflow benchmarking currently requires PredictionModelResult<T, Matrix<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        if (string.IsNullOrWhiteSpace(stackOverflow.TrainFilePath))
        {
            throw new ArgumentException("StackOverflowFederatedBenchmarkOptions.TrainFilePath is required for StackOverflow benchmarking.", nameof(options));
        }

        int seed = options.Seed ?? 0;
        int? maxSamplesPerUser = stackOverflow.MaxSamplesPerUser;

        if (options.CiMode && maxSamplesPerUser is null)
        {
            maxSamplesPerUser = 25;
        }

        var loadOptions = new LeafFederatedDatasetLoadOptions
        {
            MaxUsers = stackOverflow.LoadOptions.MaxUsers ?? (options.CiMode ? 5 : null),
            ValidateDeclaredSampleCounts = stackOverflow.LoadOptions.ValidateDeclaredSampleCounts
        };

        var leafLoader = new LeafTokenSequenceFederatedDatasetLoader();
        var dataset = leafLoader.LoadDatasetFromFiles(stackOverflow.TrainFilePath!, stackOverflow.TestFilePath, loadOptions);

        var trainClientData = dataset.Train.ToClientIdDictionary(out _);
        var testSplit = dataset.Test ?? dataset.Train;
        var testClientData = testSplit.ToClientIdDictionary(out _);

        var sampledTrainClientData = maxSamplesPerUser.HasValue
            ? TokenSequenceFederatedBenchmarkingHelper.SampleClientDatasets(trainClientData, maxSamplesPerUser.Value, seed)
            : new Dictionary<int, FederatedClientDataset<string[][], string[]>>(trainClientData);

        var sampledTestClientData = maxSamplesPerUser.HasValue
            ? TokenSequenceFederatedBenchmarkingHelper.SampleClientDatasets(testClientData, maxSamplesPerUser.Value, seed)
            : new Dictionary<int, FederatedClientDataset<string[][], string[]>>(testClientData);

        var (trainSequences, trainLabels, trainSamplesUsed) = TokenSequenceFederatedBenchmarkingHelper.ConcatenateClientData(sampledTrainClientData);
        var (testSequences, testLabels, testSamplesUsed) = TokenSequenceFederatedBenchmarkingHelper.ConcatenateClientData(sampledTestClientData);

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

        int sequenceLength = TokenSequenceFederatedBenchmarkingHelper.ResolveSequenceLength(stackOverflow.SequenceLength, testSequences);
        if (sequenceLength <= 0)
        {
            return Task.FromResult(new BenchmarkSuiteReport
            {
                Suite = suite,
                Kind = BenchmarkSuiteKind.DatasetSuite,
                Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
                Status = BenchmarkExecutionStatus.Skipped,
                FailureReason = "Unable to infer a valid sequence length from the dataset.",
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

        int maxVocabularySize = ResolveMaxVocabularySize(stackOverflow.MaxVocabularySize, options.CiMode);
        int vocabTrainingSamples = ResolveVocabularyTrainingSampleCount(stackOverflow.VocabularyTrainingSampleCount, options.CiMode);

        var tokenToId = TokenSequenceFederatedBenchmarkingHelper.BuildVocabulary(
            sequences: trainSequences,
            labels: trainLabels,
            maxVocabularySize,
            vocabTrainingSamples);

        var numOps = MathHelper.GetNumericOperations<T>();
        var testX = TokenSequenceFederatedBenchmarkingHelper.EncodeSequencesToMatrix(
            sequences: testSequences,
            tokenToId,
            sequenceLength,
            numOps,
            cancellationToken);

        var testY = TokenSequenceFederatedBenchmarkingHelper.EncodeLabelsToVector(testLabels, tokenToId, numOps);

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

    private static int ResolveMaxVocabularySize(int? configured, bool ciMode)
    {
        int value = configured ?? (ciMode ? 512 : 20000);
        if (value <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(configured), "Max vocabulary size must be greater than 2 when specified.");
        }

        return value;
    }

    private static int ResolveVocabularyTrainingSampleCount(int? configured, bool ciMode)
    {
        int value = configured ?? (ciMode ? 500 : 20000);
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(configured), "Vocabulary training sample count must be positive when specified.");
        }

        return value;
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
}

