using System.Globalization;
using AiDotNet.AutoML;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.AutoML;

public sealed class MapElitesAutoMLIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public MapElitesAutoMLIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Timeout = 120000)]
    public async Task SearchAsync_SeededRunProducesDeterministicImmutableArchive()
    {
        (Matrix<double> trainX, Vector<double> trainY, Matrix<double> validationX, Vector<double> validationY) =
            CreateRegressionData();

        using var first = CreateSearch(seed: 5);
        using var second = CreateSearch(seed: 5);

        IFullModel<double, Matrix<double>, Vector<double>> firstBest = await first.SearchAsync(
            trainX, trainY, validationX, validationY, TimeSpan.FromSeconds(20));
        IFullModel<double, Matrix<double>, Vector<double>> secondBest = await second.SearchAsync(
            trainX, trainY, validationX, validationY, TimeSpan.FromSeconds(20));

        IReadOnlyList<TrialResult> history = first.GetTrialHistory();
        double initialScore = history[0].Score;
        double bestEvolvedScore = history
            .Skip(1)
            .Where(trial => trial.Success)
            .Min(trial => trial.Score);
        MapElitesAutoMLArchiveEntry bestElite = Assert.Single(
            first.Archive,
            elite => elite.Score == bestEvolvedScore);
        Assert.True(bestElite.Parameters.TryGetValue("Degree", out object? evolvedDegree));
        double improvementFactor = initialScore / bestEvolvedScore;
        _output.WriteLine(
            "metric={0}; maximize={1}; trials={2}; best-score={3:R}; archive-count={4}; initial-score={5:R}; best-evolved-score={6:R}; improvement-factor={7:R}",
            first.OptimizationMetric,
            first.MaximizeOptimizationMetric,
            history.Count,
            first.BestScore,
            first.Archive.Count,
            initialScore,
            bestEvolvedScore,
            improvementFactor);
        for (int index = 0; index < history.Count; index++)
        {
            TrialResult trial = history[index];
            _output.WriteLine(
                "trial[{0}]: id={1}; model={2}; success={3}; score={4:R}",
                index,
                trial.TrialId,
                trial.CandidateModelType?.Name ?? "unknown",
                trial.Success,
                trial.Score);
        }
        for (int index = 0; index < first.Archive.Count; index++)
        {
            MapElitesAutoMLArchiveEntry archivedElite = first.Archive[index];
            _output.WriteLine(
                "elite[{0}]: model={1}; score={2:R}; degree={3}; cell={4}",
                index,
                archivedElite.ModelType.Name,
                archivedElite.Score,
                archivedElite.Parameters["Degree"],
                string.Join(",", archivedElite.CellBins));
        }

        Assert.NotNull(firstBest);
        Assert.NotNull(secondBest);
        Assert.NotNull(first.BestModel);
        Assert.False(double.IsNaN(first.BestScore));
        Assert.False(double.IsInfinity(first.BestScore));
        Assert.Equal(first.TrialLimit, history.Count);
        Assert.True(history[0].Success, "The deliberately underfit initial model must complete successfully.");
        Assert.True(initialScore > 0.2, $"The initial degree-one model unexpectedly fit the quadratic data: RMSE={initialScore:R}.");
        Assert.True(bestEvolvedScore < 1e-10, $"Evolution did not discover the exact quadratic model: RMSE={bestEvolvedScore:R}.");
        Assert.True(
            bestEvolvedScore < initialScore / 1_000_000,
            $"Evolution did not materially improve the real validation RMSE: initial={initialScore:R}, evolved={bestEvolvedScore:R}.");
        Assert.Equal(2, Convert.ToInt32(evolvedDegree, CultureInfo.InvariantCulture));
        Assert.Equal(bestEvolvedScore, first.BestScore);
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
    public async Task FacadeBuild_CustomMapElitesImprovesRealValidationScoreAndReportsTrials()
    {
        (Matrix<double> trainX, Vector<double> trainY, _, _) = CreateRegressionData();
        using var autoML = CreateSearch(seed: 5);
        var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(trainX, trainY);

        AiModelResult<double, Matrix<double>, Vector<double>> result =
            await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureAutoML(autoML)
                .BuildAsync();

        IReadOnlyList<TrialResult> history = autoML.GetTrialHistory();
        double initialScore = history[0].Score;
        double bestEvolvedScore = history
            .Skip(1)
            .Where(trial => trial.Success)
            .Min(trial => trial.Score);
        AutoMLRunSummary summary = Assert.IsType<AutoMLRunSummary>(result.AutoMLSummary);

        _output.WriteLine(
            "facade metric={0}; trials={1}; initial-score={2:R}; best-evolved-score={3:R}; summary-best-score={4:R}",
            summary.OptimizationMetric,
            summary.Trials.Count,
            initialScore,
            bestEvolvedScore,
            summary.BestScore);

        Assert.NotNull(result.Model);
        Assert.Equal(autoML.TrialLimit, history.Count);
        Assert.Equal(history.Count, summary.Trials.Count);
        Assert.Equal(MetricType.RMSE, summary.OptimizationMetric);
        Assert.False(summary.MaximizeMetric);
        Assert.True(initialScore > 0.01, $"The deliberately underfit facade candidate unexpectedly fit the quadratic data: RMSE={initialScore:R}.");
        Assert.True(bestEvolvedScore < 1e-10, $"Facade-driven evolution did not discover the exact quadratic model: RMSE={bestEvolvedScore:R}.");
        Assert.True(
            bestEvolvedScore < initialScore / 1_000_000,
            $"Facade-driven evolution did not materially improve validation RMSE: initial={initialScore:R}, evolved={bestEvolvedScore:R}.");
        Assert.Equal(bestEvolvedScore, summary.BestScore);
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
                InitialPopulationSize = 1,
                ComplexityBinCount = 4,
                ArchiveCapacity = 8,
                MutationProbability = 0.5,
                ExplorationProbability = 0
            });
        autoML.TrialLimit = 6;
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
