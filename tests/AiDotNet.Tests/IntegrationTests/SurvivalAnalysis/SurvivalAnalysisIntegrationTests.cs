using AiDotNet.LinearAlgebra;
using AiDotNet.SurvivalAnalysis;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SurvivalAnalysis;

/// <summary>
/// Integration tests for survival analysis classes.
/// </summary>
public class SurvivalAnalysisIntegrationTests
{
    /// <summary>
    /// Creates synthetic survival data with known properties.
    /// 20 subjects, times 1-20, half censored.
    /// </summary>
    private static (Matrix<double> features, Vector<double> times, Vector<int> events) CreateSyntheticSurvivalData()
    {
        int n = 20;
        var features = new Matrix<double>(n, 2);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (i + 1) * 0.5;
            features[i, 0] = x;
            features[i, 1] = x * 0.3;
            times[i] = i + 1.0; // Times from 1 to 20
            events[i] = i % 2 == 0 ? 1 : 0; // Alternating event/censored
        }

        return (features, times, events);
    }

    #region KaplanMeierEstimator Tests

    [Fact]
    public void KaplanMeier_Construction()
    {
        var km = new KaplanMeierEstimator<double>();
        Assert.NotNull(km);
        Assert.False(km.IsTrained);
    }

    [Fact]
    public void KaplanMeier_FitSurvival_SetsTrainedState()
    {
        var km = new KaplanMeierEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        km.FitSurvival(features, times, events);
        Assert.True(km.IsTrained);
    }

    [Fact]
    public void KaplanMeier_SurvivalCurve_MonotoneDecreasing()
    {
        var km = new KaplanMeierEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        km.FitSurvival(features, times, events);

        var survProbs = km.GetSurvivalProbabilities();
        Assert.NotNull(survProbs);

        // Survival probabilities should be monotone decreasing
        for (int i = 1; i < survProbs.Length; i++)
        {
            Assert.True(survProbs[i] <= survProbs[i - 1],
                $"Survival at index {i} ({survProbs[i]}) should be <= survival at index {i - 1} ({survProbs[i - 1]})");
        }
    }

    [Fact]
    public void KaplanMeier_SurvivalStartsAtOne()
    {
        var km = new KaplanMeierEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        km.FitSurvival(features, times, events);

        // Survival before any event should be 1.0
        var queryTimes = new Vector<double>(new[] { 0.1 });
        var baseline = km.GetBaselineSurvival(queryTimes);
        Assert.Equal(1.0, baseline[0], 1e-10);
    }

    [Fact]
    public void KaplanMeier_PredictSurvival_ReturnsValidProbabilities()
    {
        var km = new KaplanMeierEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        km.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new[] { 5.0, 10.0, 15.0 });
        var result = km.PredictSurvivalProbability(features, queryTimes);
        Assert.Equal(features.Rows, result.Rows);
        Assert.Equal(3, result.Columns);

        // Survival probabilities should be between 0 and 1
        for (int r = 0; r < result.Rows; r++)
            for (int c = 0; c < result.Columns; c++)
                Assert.InRange(result[r, c], 0.0, 1.0);

        // Survival at later times should be <= survival at earlier times (monotone decreasing)
        for (int r = 0; r < result.Rows; r++)
            Assert.True(result[r, 2] <= result[r, 0],
                "Survival probability should decrease over time");
    }

    [Fact]
    public void KaplanMeier_GetNumberAtRisk_DecreasesOverTime()
    {
        var km = new KaplanMeierEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        km.FitSurvival(features, times, events);

        var atRisk = km.GetNumberAtRisk();
        Assert.NotNull(atRisk);
        Assert.True(atRisk.Length > 0);

        // Number at risk should start at full sample and decrease
        Assert.True(atRisk[0] >= atRisk[atRisk.Length - 1],
            "Number at risk should generally decrease over time");
    }

    #endregion

    #region NelsonAalenEstimator Tests

    [Fact]
    public void NelsonAalen_Construction()
    {
        var na = new NelsonAalenEstimator<double>();
        Assert.NotNull(na);
        Assert.False(na.IsTrained);
    }

    [Fact]
    public void NelsonAalen_FitSurvival_SetsTrainedState()
    {
        var na = new NelsonAalenEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        na.FitSurvival(features, times, events);
        Assert.True(na.IsTrained);
    }

    [Fact]
    public void NelsonAalen_CumulativeHazard_MonotoneIncreasing()
    {
        var na = new NelsonAalenEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        na.FitSurvival(features, times, events);

        var cumHazard = na.CumulativeHazard;
        Assert.NotNull(cumHazard);

        // Cumulative hazard should be monotone increasing
        for (int i = 1; i < cumHazard.Length; i++)
        {
            Assert.True(cumHazard[i] >= cumHazard[i - 1],
                $"Cumulative hazard at index {i} should be >= at index {i - 1}");
        }
    }

    [Fact]
    public void NelsonAalen_VarianceEstimate_NonNegative()
    {
        var na = new NelsonAalenEstimator<double>();
        var (features, times, events) = CreateSyntheticSurvivalData();
        na.FitSurvival(features, times, events);

        var variance = na.Variance;
        Assert.NotNull(variance);
        for (int i = 0; i < variance.Length; i++)
        {
            Assert.True(variance[i] >= 0, $"Variance at index {i} must be non-negative");
        }
    }

    #endregion

    #region CoxProportionalHazards Tests

    [Fact]
    public void Cox_Construction_WithDefaults()
    {
        var cox = new CoxProportionalHazards<double>();
        Assert.NotNull(cox);
        Assert.False(cox.IsTrained);
    }

    [Fact]
    public void Cox_FitSurvival_SetsTrainedState()
    {
        var cox = new CoxProportionalHazards<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        cox.FitSurvival(features, times, events);
        Assert.True(cox.IsTrained);
    }

    [Fact]
    public void Cox_PredictHazardRatio_ReturnsPositiveValues()
    {
        var cox = new CoxProportionalHazards<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        cox.FitSurvival(features, times, events);

        var hazardRatios = cox.PredictHazardRatio(features);
        Assert.Equal(features.Rows, hazardRatios.Length);
        for (int i = 0; i < hazardRatios.Length; i++)
        {
            Assert.True(hazardRatios[i] > 0, $"Hazard ratio at {i} must be positive");
        }
    }

    [Fact]
    public void Cox_GetCoefficients_ReturnsCorrectCount()
    {
        var cox = new CoxProportionalHazards<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        cox.FitSurvival(features, times, events);

        var coefficients = cox.GetCoefficients();
        Assert.NotNull(coefficients);
        Assert.Equal(features.Columns, coefficients.Length);
    }

    [Fact]
    public void Cox_GetFeatureHazardRatios_MatchesExpCoefficients()
    {
        var cox = new CoxProportionalHazards<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        cox.FitSurvival(features, times, events);

        var hrs = cox.GetFeatureHazardRatios();
        var coefficients = cox.GetCoefficients();
        Assert.NotNull(hrs);
        Assert.NotNull(coefficients);

        for (int i = 0; i < hrs.Length; i++)
        {
            Assert.Equal(Math.Exp(coefficients[i]), hrs[i], 1e-6);
        }
    }

    #endregion

    #region WeibullAFT Tests

    [Fact]
    public void WeibullAFT_Construction()
    {
        var weibull = new WeibullAFT<double>();
        Assert.NotNull(weibull);
        Assert.False(weibull.IsTrained);
    }

    [Fact]
    public void WeibullAFT_FitSurvival_SetsTrainedState()
    {
        var weibull = new WeibullAFT<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        weibull.FitSurvival(features, times, events);
        Assert.True(weibull.IsTrained);
    }

    [Fact]
    public void WeibullAFT_Predict_ReturnsPositiveSurvivalTimes()
    {
        var weibull = new WeibullAFT<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        weibull.FitSurvival(features, times, events);

        var predictions = weibull.Predict(features);
        Assert.Equal(features.Rows, predictions.Length);

        // Predicted survival times should be positive
        for (int i = 0; i < predictions.Length; i++)
            Assert.True(predictions[i] > 0, $"Predicted survival time at {i} should be positive, got {predictions[i]}");
    }

    #endregion

    #region LogNormalAFT Tests

    [Fact]
    public void LogNormalAFT_Construction()
    {
        var lognormal = new LogNormalAFT<double>();
        Assert.NotNull(lognormal);
        Assert.False(lognormal.IsTrained);
    }

    [Fact]
    public void LogNormalAFT_FitSurvival_SetsTrainedState()
    {
        var lognormal = new LogNormalAFT<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        lognormal.FitSurvival(features, times, events);
        Assert.True(lognormal.IsTrained);
    }

    [Fact]
    public void LogNormalAFT_Predict_ReturnsPositiveSurvivalTimes()
    {
        var lognormal = new LogNormalAFT<double>(maxIterations: 50);
        var (features, times, events) = CreateSyntheticSurvivalData();
        lognormal.FitSurvival(features, times, events);

        var predictions = lognormal.Predict(features);
        Assert.Equal(features.Rows, predictions.Length);

        // Predicted survival times should be positive
        for (int i = 0; i < predictions.Length; i++)
            Assert.True(predictions[i] > 0, $"Predicted survival time at {i} should be positive, got {predictions[i]}");
    }

    #endregion

    #region RandomSurvivalForest Tests

    [Fact]
    public void RandomSurvivalForest_Construction()
    {
        var rsf = new RandomSurvivalForest<double>(numTrees: 5, maxDepth: 3);
        Assert.NotNull(rsf);
        Assert.False(rsf.IsTrained);
    }

    [Fact]
    public void RandomSurvivalForest_FitSurvival_SetsTrainedState()
    {
        var rsf = new RandomSurvivalForest<double>(numTrees: 5, maxDepth: 3, seed: 42);
        var (features, times, events) = CreateSyntheticSurvivalData();
        rsf.FitSurvival(features, times, events);
        Assert.True(rsf.IsTrained);
    }

    [Fact]
    public void RandomSurvivalForest_Predict_ReturnsPositiveSurvivalTimes()
    {
        var rsf = new RandomSurvivalForest<double>(numTrees: 5, maxDepth: 3, seed: 42);
        var (features, times, events) = CreateSyntheticSurvivalData();
        rsf.FitSurvival(features, times, events);

        var predictions = rsf.Predict(features);
        Assert.Equal(features.Rows, predictions.Length);

        // Predicted survival times should be positive
        for (int i = 0; i < predictions.Length; i++)
            Assert.True(predictions[i] > 0, $"Predicted survival time at {i} should be positive, got {predictions[i]}");
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllSurvivalModels_ThrowWhenNotFitted()
    {
        var models = new SurvivalModelBase<double>[]
        {
            new KaplanMeierEstimator<double>(),
            new NelsonAalenEstimator<double>(),
            new CoxProportionalHazards<double>(),
            new WeibullAFT<double>(),
            new LogNormalAFT<double>(),
            new RandomSurvivalForest<double>(numTrees: 5, maxDepth: 3),
        };

        var features = new Matrix<double>(1, 2);

        foreach (var model in models)
        {
            Assert.False(model.IsTrained);
            Assert.Throws<InvalidOperationException>(() => model.Predict(features));
        }
    }

    [Fact]
    public void SurvivalModels_ValidateBadData_Throws()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(5, 2);
        var times = new Vector<double>(new[] { 1.0, 2.0, -1.0, 4.0, 5.0 }); // Negative time
        var events = new Vector<int>(new[] { 1, 0, 1, 0, 1 });

        Assert.Throws<ArgumentException>(() => km.FitSurvival(features, times, events));
    }

    #endregion
}
