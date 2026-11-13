using System;
using System.Linq;
using AiDotNet.SurvivalAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SurvivalAnalysis;

/// <summary>
/// Pins survival summaries against lifelines reference values.
/// </summary>
public class SurvivalReferenceValueTests
{
    [Fact]
    public void NonparametricSurvival_HalfLargeCohort_UsesWideCounts()
    {
        const int count = 70000;
        var x = new Matrix<Half>(count, 1);
        var times = Vector<Half>.CreateDefault(count, (Half)1);
        var events = Vector<int>.CreateDefault(count, 1);
        var km = new KaplanMeierEstimator<Half>();
        km.FitSurvival(x, times, events);
        Assert.Equal((Half)0, km.GetBaselineSurvival(new Vector<Half>(new[] { (Half)1 }))[0]);
        var na = new NelsonAalenEstimator<Half>();
        na.FitSurvival(x, times, events);
        Assert.Equal((Half)1, na.CumulativeHazard![0]);
        Assert.InRange((double)na.Variance![0], 1.4e-5, 1.5e-5);
    }

    [Fact]
    public void Concordance_HalfPairCountAboveRange_PreservesTieHalfCredit()
    {
        const int count = 400;
        var x = new Matrix<Half>(count, 1);
        var times = new Vector<Half>(System.Linq.Enumerable.Range(1, count).Select(value => (Half)value));
        var events = Vector<int>.CreateDefault(count, 1);
        var model = new KaplanMeierEstimator<Half>();
        model.FitSurvival(x, times, events);
        Assert.Equal((Half)0.5, model.CalculateConcordanceIndex(x, times, events));
    }

    [Theory]
    [InlineData(1000f)]
    [InlineData(4000f)]
    [InlineData(-1000f)]
    public void Cox_FloatExtremePredictors_PreserveFiniteFitAndConditionalSurvival(float magnitude)
    {
        var model = new CoxProportionalHazards<float>(learningRate: 0.001, maxIterations: 3, tolerance: 0);
        var x = new Matrix<float>(new float[,] { { magnitude }, { 0 } });
        var times = new Vector<float>(new[] { 1f, 2f });
        model.FitSurvival(x, times, new Vector<int>(new[] { 1, 1 }));
        Assert.Single(model.GetParameters().ToArray());
        Assert.Equal(magnitude * 0.0005f, model.GetCoefficients()![0], 5);
        var queries = new Vector<float>(new[] { 0.5f, 1f, 2f });
        var survival = model.PredictSurvivalProbability(x, queries);
        Assert.Equal(1f, survival[0, 0]);
        Assert.Equal((float)Math.Exp(-1), survival[0, 1], 5);
        Assert.Equal(0f, survival[0, 2]);
        Assert.Equal((float)Math.Exp(-1), survival[1, 2], 5);
        Assert.All(model.PredictHazardRatio(x).ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        var copy = (CoxProportionalHazards<float>)model.DeepCopy();
        var copied = copy.PredictSurvivalProbability(x, queries);
        for (int i = 0; i < x.Rows; i++)
            for (int j = 0; j < queries.Length; j++)
                Assert.Equal(survival[i, j], copied[i, j]);
    }

    [Fact]
    public void Cox_AllCensored_SurvivalRemainsOne()
    {
        var model = new CoxProportionalHazards<double>(maxIterations: 2);
        var x = new Matrix<double>(new double[,] { { 1 }, { 2 } });
        model.FitSurvival(x, new Vector<double>(new[] { 1.0, 2.0 }), new Vector<int>(new[] { 0, 0 }));
        Assert.All(model.PredictSurvivalProbability(x, new Vector<double>(new[] { 0.5, 3.0 })).ToArray(),
            value => Assert.Equal(1.0, value));
    }

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

    [Fact]
    public void LogNormal_FarUpperTail_SurvivalDoesNotCancelToZero()
    {
        // S(t) = 1 - Phi(z) at z = 9 is 1.1285884059538422e-19 (scipy.stats.norm.sf(9)). Forming it as
        // 1 - Phi(z) cancels to 0 in double; the A&S 7.1.26 tail used here is within 2% of it.
        var model = new LogNormalAFT<double>();
        var features = new Matrix<double>(6, 1);
        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 6, 9 });
        var events = new Vector<int>(new[] { 1, 1, 0, 1, 1, 0 });
        model.FitSurvival(features, times, events);

        double farTime = Math.Exp(model.Intercept + 9.0 * model.Scale);
        var survival = model.GetBaselineSurvival(new Vector<double>(new[] { farTime }));

        Assert.InRange(survival[0] / 1.1285884059538422e-19, 0.98, 1.02);
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
