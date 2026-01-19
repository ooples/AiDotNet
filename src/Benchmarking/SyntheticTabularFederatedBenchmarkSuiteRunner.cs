using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;

namespace AiDotNet.Benchmarking;

internal static class SyntheticTabularFederatedBenchmarkSuiteRunner
{
    public static Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        CancellationToken cancellationToken)
    {
        if (suite != BenchmarkSuite.TabularNonIID)
        {
            throw new ArgumentOutOfRangeException(nameof(suite), suite, "Synthetic tabular benchmark runner only supports TabularNonIID.");
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
                $"Tabular benchmarking currently requires AiModelResult<T, Matrix<T>, Vector<T>>. Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }

        var tabular = options.Tabular?.NonIid ?? new SyntheticTabularFederatedBenchmarkOptions();

        int seed = options.Seed ?? 0;
        var random = RandomHelper.CreateSeededRandom(unchecked((seed * 16777619) ^ (int)suite));

        int clientCount = tabular.ClientCount ?? (options.CiMode ? 5 : 50);
        int featureCount = tabular.FeatureCount ?? 10;
        int trainSamplesPerClient = tabular.TrainSamplesPerClient ?? (options.CiMode ? 25 : 200);
        int testSamplesPerClient = tabular.TestSamplesPerClient ?? (options.CiMode ? 10 : 50);
        int classCount = tabular.ClassCount ?? 3;

        ValidatePositive(clientCount, nameof(tabular.ClientCount));
        ValidatePositive(featureCount, nameof(tabular.FeatureCount));
        ValidatePositive(trainSamplesPerClient, nameof(tabular.TrainSamplesPerClient));
        ValidatePositive(testSamplesPerClient, nameof(tabular.TestSamplesPerClient));

        if (tabular.TaskType != SyntheticTabularTaskType.Regression)
        {
            ValidatePositive(classCount, nameof(tabular.ClassCount));
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        var (trainX, trainY, testX, testY, predictionType) = GenerateSyntheticFederatedTabularDataset(
            clientCount,
            featureCount,
            trainSamplesPerClient,
            testSamplesPerClient,
            classCount,
            tabular.TaskType,
            tabular.DirichletAlpha,
            tabular.NoiseStdDev,
            random,
            numOps,
            cancellationToken);

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
                    MaxSamplesPerUser = trainSamplesPerClient
                }
            });
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
                    ClientsUsed = clientCount,
                    TrainSamplesUsed = trainX.Rows,
                    TestSamplesUsed = testX.Rows,
                    FeatureCount = featureCount,
                    CiMode = options.CiMode,
                    Seed = seed,
                    MaxSamplesPerUser = trainSamplesPerClient
                }
            });
        }

        double accuracy = Convert.ToDouble(StatisticsHelper<T>.CalculateAccuracy(testY, predicted, predictionType));
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
                FeatureCount = featureCount,
                CiMode = options.CiMode,
                Seed = seed,
                MaxSamplesPerUser = trainSamplesPerClient
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

    private static void ValidatePositive(int value, string name)
    {
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(name, "Value must be positive when specified.");
        }
    }

    private static (Matrix<T> TrainX, Vector<T> TrainY, Matrix<T> TestX, Vector<T> TestY, PredictionType PredictionType)
        GenerateSyntheticFederatedTabularDataset<T>(
            int clientCount,
            int featureCount,
            int trainSamplesPerClient,
            int testSamplesPerClient,
            int classCount,
            SyntheticTabularTaskType taskType,
            double dirichletAlpha,
            double noiseStdDev,
            Random random,
            INumericOperations<T> numOps,
            CancellationToken cancellationToken)
    {
        if (dirichletAlpha <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(dirichletAlpha), "Dirichlet alpha must be positive.");
        }

        if (noiseStdDev < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseStdDev), "Noise standard deviation cannot be negative.");
        }

        int totalTrain = checked(clientCount * trainSamplesPerClient);
        int totalTest = checked(clientCount * testSamplesPerClient);

        var trainX = new Matrix<T>(totalTrain, featureCount);
        var trainY = new Vector<T>(totalTrain);
        var testX = new Matrix<T>(totalTest, featureCount);
        var testY = new Vector<T>(totalTest);

        if (taskType == SyntheticTabularTaskType.Regression)
        {
            var weights = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                weights[f] = NextGaussian(random);
            }

            int trainRow = 0;
            int testRow = 0;

            for (int client = 0; client < clientCount; client++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                double clientBias = 0.5 * NextGaussian(random);

                for (int i = 0; i < trainSamplesPerClient; i++, trainRow++)
                {
                    double y = clientBias;
                    for (int f = 0; f < featureCount; f++)
                    {
                        double x = NextGaussian(random);
                        trainX[trainRow, f] = numOps.FromDouble(x);
                        y += weights[f] * x;
                    }

                    y += noiseStdDev * NextGaussian(random);
                    trainY[trainRow] = numOps.FromDouble(y);
                }

                for (int i = 0; i < testSamplesPerClient; i++, testRow++)
                {
                    double y = clientBias;
                    for (int f = 0; f < featureCount; f++)
                    {
                        double x = NextGaussian(random);
                        testX[testRow, f] = numOps.FromDouble(x);
                        y += weights[f] * x;
                    }

                    y += noiseStdDev * NextGaussian(random);
                    testY[testRow] = numOps.FromDouble(y);
                }
            }

            return (trainX, trainY, testX, testY, PredictionType.Regression);
        }

        int effectiveClassCount = taskType == SyntheticTabularTaskType.BinaryClassification ? 2 : classCount;
        var classMeans = new double[effectiveClassCount][];

        for (int c = 0; c < effectiveClassCount; c++)
        {
            classMeans[c] = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                classMeans[c][f] = 2.0 * NextGaussian(random);
            }
        }

        int trainIndex = 0;
        int testIndex = 0;

        for (int client = 0; client < clientCount; client++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var classProportions = SampleDirichlet(effectiveClassCount, dirichletAlpha, random);

            for (int i = 0; i < trainSamplesPerClient; i++, trainIndex++)
            {
                int label = SampleCategorical(classProportions, random);
                trainY[trainIndex] = numOps.FromDouble(label);

                for (int f = 0; f < featureCount; f++)
                {
                    double x = classMeans[label][f] + noiseStdDev * NextGaussian(random);
                    trainX[trainIndex, f] = numOps.FromDouble(x);
                }
            }

            for (int i = 0; i < testSamplesPerClient; i++, testIndex++)
            {
                int label = SampleCategorical(classProportions, random);
                testY[testIndex] = numOps.FromDouble(label);

                for (int f = 0; f < featureCount; f++)
                {
                    double x = classMeans[label][f] + noiseStdDev * NextGaussian(random);
                    testX[testIndex, f] = numOps.FromDouble(x);
                }
            }
        }

        var predictionType = taskType == SyntheticTabularTaskType.BinaryClassification
            ? PredictionType.BinaryClassification
            : PredictionType.MultiClass;

        return (trainX, trainY, testX, testY, predictionType);
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

    private static int SampleCategorical(double[] probabilities, Random random)
    {
        double u = random.NextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (u <= cumulative)
            {
                return i;
            }
        }

        return probabilities.Length - 1;
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

    private static double NextGaussian(Random random)
    {
        return random.NextGaussian();
    }
}
