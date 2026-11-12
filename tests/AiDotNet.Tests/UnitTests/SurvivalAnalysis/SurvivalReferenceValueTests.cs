using AiDotNet.SurvivalAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SurvivalAnalysis;

/// <summary>
/// Pins survival summaries against lifelines reference values.
/// </summary>
public class SurvivalReferenceValueTests
{
    private static KaplanMeierEstimator<double> FitKaplanMeier(double[] times, int[] events)
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(times.Length, 1);
        km.FitSurvival(features, new Vector<double>(times), new Vector<int>(events));
        return km;
    }

    [Fact]
    public void MedianSurvivalTime_IsFirstTimeSurvivalReachesHalf_LikeLifelines()
    {
        // lifelines: KaplanMeierFitter().fit([1, 2, 3]).median_survival_time_ == 2.0.
        // S = 2/3, 1/3, 0 at t = 1, 2, 3; the median is the first t with S(t) <= 0.5, not an
        // interpolation between t = 1 and t = 2 (which gives 1.5).
        var km = FitKaplanMeier(new double[] { 1, 2, 3 }, new[] { 1, 1, 1 });

        var median = km.PredictMedianSurvivalTime();

        Assert.Equal(2.0, median[0], 12);
    }

    [Fact]
    public void MedianSurvivalTime_IsExactlyHalfAtStep_LikeLifelines()
    {
        // lifelines: KaplanMeierFitter().fit([1, 2, 3, 4]).median_survival_time_ == 2.0
        // (S(2) = 0.5 exactly counts as reaching the median).
        var km = FitKaplanMeier(new double[] { 1, 2, 3, 4 }, new[] { 1, 1, 1, 1 });

        var median = km.PredictMedianSurvivalTime();

        Assert.Equal(2.0, median[0], 12);
    }

    [Fact]
    public void MedianSurvivalTime_FirstStepBelowHalf_ReturnsFirstEventTime()
    {
        // lifelines: KaplanMeierFitter().fit([1, 1, 1, 2]).median_survival_time_ == 1.0
        // (S(1) = 0.25). The old interpolation returned MaxValue when the first step was already below 0.5.
        var km = FitKaplanMeier(new double[] { 1, 1, 1, 2 }, new[] { 1, 1, 1, 1 });

        var median = km.PredictMedianSurvivalTime();

        Assert.Equal(1.0, median[0], 12);
    }

    [Fact]
    public void MedianSurvivalTime_NeverReachesHalf_IsInfinite()
    {
        // lifelines: KaplanMeierFitter().fit([1, 2, 3, 4], [1, 0, 0, 0]).median_survival_time_ == inf.
        var km = FitKaplanMeier(new double[] { 1, 2, 3, 4 }, new[] { 1, 0, 0, 0 });

        var median = km.PredictMedianSurvivalTime();

        Assert.Equal(double.MaxValue, median[0]);
    }

    private static double FitCoxCoefficient(int[] order)
    {
        double[] x = { 0, 1, 0, 1, 1, 0 };
        double[] times = { 1, 1, 2, 3, 3, 4 };
        int[] events = { 1, 1, 1, 1, 0, 1 };

        var features = new Matrix<double>(order.Length, 1);
        var t = new Vector<double>(order.Length);
        var e = new Vector<int>(order.Length);
        for (int i = 0; i < order.Length; i++)
        {
            features[i, 0] = x[order[i]];
            t[i] = times[order[i]];
            e[i] = events[order[i]];
        }

        var cox = new CoxProportionalHazards<double>(learningRate: 0.1, maxIterations: 3000, tolerance: 0.0);
        cox.FitSurvival(features, t, e);
        var coefficients = cox.GetCoefficients();
        Assert.NotNull(coefficients);
        return coefficients is null ? double.NaN : coefficients[0];
    }

    [Fact]
    public void Cox_TiedEventTimes_ConvergesToBreslowPartialLikelihoodMaximum()
    {
        // Breslow partial likelihood (statsmodels PHReg ties="breslow"): every subject with t_j >= t_i
        // is in the risk set of subject i, so tied subjects share one risk set. Newton's method on
        // this data gives beta = -0.1707048952583424.
        double beta = FitCoxCoefficient(new[] { 0, 1, 2, 3, 4, 5 });

        Assert.Equal(-0.1707048952583424, beta, 8);
    }

    [Fact]
    public void Cox_TiedEventTimes_DoesNotDependOnRowOrder()
    {
        double forward = FitCoxCoefficient(new[] { 0, 1, 2, 3, 4, 5 });
        double reversed = FitCoxCoefficient(new[] { 5, 4, 3, 2, 1, 0 });

        Assert.Equal(forward, reversed, 10);
    }
}
