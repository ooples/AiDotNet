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
}
