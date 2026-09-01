using AiDotNet.AutoML;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AutoML;

public sealed class MapElitesAutoMLIntegrationTests
{
    [Fact(Timeout = 120000)]
    public async Task SearchAsync_SeededRunProducesDeterministicImmutableArchive()
    {
        (Matrix<double> trainX, Vector<double> trainY, Matrix<double> validationX, Vector<double> validationY) =
            CreateRegressionData();

        using var first = CreateSearch(seed: 773);
        using var second = CreateSearch(seed: 773);

        IFullModel<double, Matrix<double>, Vector<double>> firstBest = await first.SearchAsync(
            trainX, trainY, validationX, validationY, TimeSpan.FromSeconds(20));
        IFullModel<double, Matrix<double>, Vector<double>> secondBest = await second.SearchAsync(
            trainX, trainY, validationX, validationY, TimeSpan.FromSeconds(20));

        Assert.NotNull(firstBest);
        Assert.NotNull(secondBest);
        Assert.NotNull(first.BestModel);
        Assert.False(double.IsNaN(first.BestScore));
        Assert.False(double.IsInfinity(first.BestScore));
        Assert.InRange(first.GetTrialHistory().Count, 1, first.TrialLimit);
        Assert.NotEmpty(first.Archive);
        Assert.Equal(first.ArchiveStateHash, second.ArchiveStateHash);
        Assert.Equal(
            first.Archive.Select(DescribeEntry),
            second.Archive.Select(DescribeEntry));

        MapElitesAutoMLArchiveEntry elite = first.Archive[0];
        Assert.Equal(typeof(AiDotNet.Regression.PolynomialRegression<>), elite.ModelType);
        Assert.Contains("model_family", elite.Descriptors.Keys);
        Assert.Contains("configuration_complexity", elite.Descriptors.Keys);
        Assert.Throws<NotSupportedException>(() =>
            ((IDictionary<string, object>)elite.Parameters).Add("mutated", 1));
        Assert.Throws<NotSupportedException>(() =>
            ((IList<int>)elite.CellBins).Add(99));
    }

    [Fact(Timeout = 120000)]
    public async Task SearchAsync_UnsupportedInputShapeFailsExplicitlyWithoutTrials()
    {
        using var autoML = new MapElitesAutoML<double, object, object>();

        NotSupportedException exception = await Assert.ThrowsAsync<NotSupportedException>(() =>
            autoML.SearchAsync(new object(), new object(), new object(), new object(), TimeSpan.FromSeconds(1)));

        Assert.Contains("Matrix<T>/Vector<T>", exception.Message, StringComparison.Ordinal);
        Assert.Empty(autoML.GetTrialHistory());
    }

    [Fact(Timeout = 120000)]
    public async Task SearchAsync_DuplicateSpecificationsDoNotConsumeExpensiveTrialBudget()
    {
        (Matrix<double> trainX, Vector<double> trainY, Matrix<double> validationX, Vector<double> validationY) =
            CreateRegressionData();
        using var autoML = new MapElitesAutoML<double, Matrix<double>, Vector<double>>(
            new MapElitesAutoMLOptions { Seed = 91, InitialPopulationSize = 2 });
        autoML.TrialLimit = 5;
        autoML.EnsembleOptions.Enabled = false;
        autoML.SetCandidateModels(new List<Type> { typeof(AiDotNet.Regression.MultipleRegression<>) });

        _ = await autoML.SearchAsync(
            trainX, trainY, validationX, validationY, TimeSpan.FromSeconds(20));

        Assert.Single(autoML.GetTrialHistory());
        Assert.Single(autoML.Archive);
    }

    [Fact(Timeout = 120000)]
    public async Task SearchAsync_PreCanceledTokenStopsBeforeAnyTrial()
    {
        (Matrix<double> trainX, Vector<double> trainY, Matrix<double> validationX, Vector<double> validationY) =
            CreateRegressionData();
        using var autoML = CreateSearch(seed: 11);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => autoML.SearchAsync(
            trainX,
            trainY,
            validationX,
            validationY,
            TimeSpan.FromSeconds(20),
            cancellation.Token));

        Assert.Equal(AutoMLStatus.Cancelled, autoML.Status);
        Assert.Empty(autoML.GetTrialHistory());
        Assert.Empty(autoML.Archive);
    }

    [Fact]
    public void Constructor_InvalidQualityDiversityOptionsFailBeforeSearch()
    {
        var options = new MapElitesAutoMLOptions { MutationProbability = double.NaN };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MapElitesAutoML<double, Matrix<double>, Vector<double>>(options));
    }

    private static MapElitesAutoML<double, Matrix<double>, Vector<double>> CreateSearch(ulong seed)
    {
        var autoML = new MapElitesAutoML<double, Matrix<double>, Vector<double>>(
            new MapElitesAutoMLOptions
            {
                Seed = seed,
                InitialPopulationSize = 4,
                ComplexityBinCount = 4,
                ArchiveCapacity = 8,
                MutationProbability = 0.5,
                ExplorationProbability = 0
            });
        autoML.TrialLimit = 4;
        autoML.EnsembleOptions.Enabled = false;
        autoML.SetCandidateModels(new List<Type> { typeof(AiDotNet.Regression.PolynomialRegression<>) });
        return autoML;
    }

    private static string DescribeEntry(MapElitesAutoMLArchiveEntry entry)
    {
        return string.Join("|", new[]
        {
            entry.SpecificationId,
            entry.Score.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            string.Join(",", entry.CellBins),
            string.Join(",", entry.Parameters.OrderBy(item => item.Key, StringComparer.Ordinal)
                .Select(item => item.Key + "=" + item.Value))
        });
    }

    private static (Matrix<double>, Vector<double>, Matrix<double>, Vector<double>) CreateRegressionData()
    {
        var trainX = new Matrix<double>(24, 2);
        var trainY = new Vector<double>(24);
        for (int index = 0; index < trainX.Rows; index++)
        {
            double x1 = index / 4.0;
            double x2 = (index % 5) - 2;
            trainX[index, 0] = x1;
            trainX[index, 1] = x2;
            trainY[index] = 1.5 + (2 * x1) - (0.75 * x2) + (0.1 * x1 * x1);
        }

        var validationX = new Matrix<double>(10, 2);
        var validationY = new Vector<double>(10);
        for (int index = 0; index < validationX.Rows; index++)
        {
            double x1 = index / 3.0 + 0.2;
            double x2 = (index % 4) - 1.5;
            validationX[index, 0] = x1;
            validationX[index, 1] = x2;
            validationY[index] = 1.5 + (2 * x1) - (0.75 * x2) + (0.1 * x1 * x1);
        }

        return (trainX, trainY, validationX, validationY);
    }
}
