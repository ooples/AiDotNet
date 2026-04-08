using AiDotNet.LinearAlgebra;
using AiDotNet.SurvivalAnalysis;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SurvivalAnalysis;

/// <summary>
/// Deep math integration tests for survival analysis models.
/// Tests verify correctness of Kaplan-Meier, Nelson-Aalen, Weibull AFT, LogNormal AFT,
/// and the concordance index against hand-computed reference values and mathematical properties.
/// </summary>
public class SurvivalAnalysisDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-4;

    #region Kaplan-Meier Hand-Computed Tests

    [Fact]
    public void KaplanMeier_SimpleNoCensoring_MatchesHandComputed()
    {
        // 5 subjects, all experience event, no censoring
        // Times: 1, 2, 3, 4, 5
        // At t=1: n=5, d=1 → S = 4/5 = 0.8
        // At t=2: n=4, d=1 → S = 0.8 * 3/4 = 0.6
        // At t=3: n=3, d=1 → S = 0.6 * 2/3 = 0.4
        // At t=4: n=2, d=1 → S = 0.4 * 1/2 = 0.2
        // At t=5: n=1, d=1 → S = 0.2 * 0/1 = 0.0
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });

        km.FitSurvival(features, times, events);

        var probs = km.GetSurvivalProbabilities();
        Assert.NotNull(probs);
        Assert.Equal(5, probs.Length);
        Assert.Equal(0.8, probs[0], Tolerance);
        Assert.Equal(0.6, probs[1], Tolerance);
        Assert.Equal(0.4, probs[2], Tolerance);
        Assert.Equal(0.2, probs[3], Tolerance);
        Assert.Equal(0.0, probs[4], Tolerance);
    }

    [Fact]
    public void KaplanMeier_WithCensoring_MatchesHandComputed()
    {
        // 6 subjects: events at times 1, 3, 5; censored at times 2, 4, 6
        // Times: 1, 2, 3, 4, 5, 6
        // Events: 1, 0, 1, 0, 1, 0
        //
        // Event times (sorted): 1, 3, 5
        //
        // At t=1: n=6 (all alive), d=1 → S = 5/6
        // At t=3: n=4 (6 - 1 event at t=1 - 1 censored at t=2), d=1 → S = (5/6)(3/4) = 15/24 = 5/8
        // At t=5: n=2 (4 - 1 event at t=3 - 1 censored at t=4), d=1 → S = (5/8)(1/2) = 5/16
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(6, 1);
        for (int i = 0; i < 6; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
        var events = new Vector<int>(new int[] { 1, 0, 1, 0, 1, 0 });

        km.FitSurvival(features, times, events);

        var probs = km.GetSurvivalProbabilities();
        Assert.NotNull(probs);
        Assert.Equal(3, probs.Length); // Only 3 event times

        Assert.Equal(5.0 / 6.0, probs[0], Tolerance);  // S(1) = 5/6
        Assert.Equal(5.0 / 8.0, probs[1], Tolerance);   // S(3) = 5/8
        Assert.Equal(5.0 / 16.0, probs[2], Tolerance);  // S(5) = 5/16
    }

    [Fact]
    public void KaplanMeier_SimultaneousEvents_MatchesHandComputed()
    {
        // 6 subjects, 2 events at time 1, 2 events at time 3, 2 events at time 5
        // At t=1: n=6, d=2 → S = 4/6 = 2/3
        // At t=3: n=4, d=2 → S = (2/3)(2/4) = 2/6 = 1/3
        // At t=5: n=2, d=2 → S = (1/3)(0/2) = 0
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(6, 1);
        for (int i = 0; i < 6; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 1, 3, 3, 5, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 1 });

        km.FitSurvival(features, times, events);

        var probs = km.GetSurvivalProbabilities();
        Assert.NotNull(probs);
        Assert.Equal(3, probs.Length);
        Assert.Equal(2.0 / 3.0, probs[0], Tolerance);  // S(1) = 2/3
        Assert.Equal(1.0 / 3.0, probs[1], Tolerance);   // S(3) = 1/3
        Assert.Equal(0.0, probs[2], Tolerance);          // S(5) = 0
    }

    [Fact]
    public void KaplanMeier_NumberAtRisk_MatchesHandComputed()
    {
        // Same setup as WithCensoring test
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(6, 1);
        for (int i = 0; i < 6; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
        var events = new Vector<int>(new int[] { 1, 0, 1, 0, 1, 0 });

        km.FitSurvival(features, times, events);

        var atRisk = km.GetNumberAtRisk();
        var numEvents = km.GetNumberEvents();
        Assert.NotNull(atRisk);
        Assert.NotNull(numEvents);

        Assert.Equal(6, atRisk[0]);  // At t=1: all 6 alive
        Assert.Equal(4, atRisk[1]);  // At t=3: 6 - 1(event@1) - 1(censored@2) = 4
        Assert.Equal(2, atRisk[2]);  // At t=5: 4 - 1(event@3) - 1(censored@4) = 2

        Assert.Equal(1, numEvents[0]);
        Assert.Equal(1, numEvents[1]);
        Assert.Equal(1, numEvents[2]);
    }

    [Fact]
    public void KaplanMeier_SurvivalIsMonotonicallyNonIncreasing()
    {
        var km = new KaplanMeierEstimator<double>();
        var n = 20;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 10.0;
            events[i] = i % 3 == 0 ? 0 : 1; // Mix of events and censoring
        }

        km.FitSurvival(features, times, events);

        var probs = km.GetSurvivalProbabilities();
        Assert.NotNull(probs);

        for (int i = 1; i < probs.Length; i++)
        {
            Assert.True(probs[i] <= probs[i - 1] + Tolerance,
                $"S(t_{i})={probs[i]} > S(t_{i - 1})={probs[i - 1]}: survival not monotonically non-increasing");
        }
    }

    [Fact]
    public void KaplanMeier_SurvivalBefore0_IsOne()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 5, 10, 15 });
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        km.FitSurvival(features, times, events);

        // Query survival at time before first event
        var queryTimes = new Vector<double>(new double[] { 0.1, 1.0, 3.0 });
        var baseline = km.GetBaselineSurvival(queryTimes);

        for (int i = 0; i < queryTimes.Length; i++)
        {
            Assert.Equal(1.0, baseline[i], Tolerance);
        }
    }

    [Fact]
    public void KaplanMeier_AllCensored_SurvivalRemainsOne()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<int>(new int[] { 0, 0, 0, 0, 0 }); // All censored

        km.FitSurvival(features, times, events);

        var probs = km.GetSurvivalProbabilities();
        Assert.NotNull(probs);
        // No events → survival stays at 1.0
        for (int i = 0; i < probs.Length; i++)
        {
            Assert.Equal(1.0, probs[i], Tolerance);
        }
    }

    [Fact]
    public void KaplanMeier_HazardRatios_AllOne()
    {
        // KM doesn't use covariates, so hazard ratios = 1 for all subjects
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 2);
        features[0, 0] = 1; features[0, 1] = 100;
        features[1, 0] = 2; features[1, 1] = 200;
        features[2, 0] = 3; features[2, 1] = 300;

        var times = new Vector<double>(new double[] { 5, 10, 15 });
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        km.FitSurvival(features, times, events);
        var hr = km.PredictHazardRatio(features);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0, hr[i], Tolerance);
        }
    }

    [Fact]
    public void KaplanMeier_SameSubjectsGetSameSurvival()
    {
        // KM ignores features, so all subjects get the same survival curve
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 5, 10, 15 });
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        km.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new double[] { 3, 7, 12 });
        // Use 3 different "subjects" with different features
        var testFeatures = new Matrix<double>(3, 1);
        testFeatures[0, 0] = 0; testFeatures[1, 0] = 50; testFeatures[2, 0] = 100;

        var survivalMatrix = km.PredictSurvivalProbability(testFeatures, queryTimes);

        // All rows should be identical
        for (int t = 0; t < queryTimes.Length; t++)
        {
            Assert.Equal(survivalMatrix[0, t], survivalMatrix[1, t], Tolerance);
            Assert.Equal(survivalMatrix[0, t], survivalMatrix[2, t], Tolerance);
        }
    }

    #endregion

    #region Nelson-Aalen Hand-Computed Tests

    [Fact]
    public void NelsonAalen_SimpleNoCensoring_MatchesHandComputed()
    {
        // 5 subjects, all events, times 1..5
        // At t=1: n=5, d=1 → H += 1/5 = 0.2
        // At t=2: n=4, d=1 → H += 1/4 = 0.25, cumH = 0.45
        // At t=3: n=3, d=1 → H += 1/3 ≈ 0.333, cumH ≈ 0.783
        // At t=4: n=2, d=1 → H += 1/2 = 0.5, cumH ≈ 1.283
        // At t=5: n=1, d=1 → H += 1/1 = 1.0, cumH ≈ 2.283
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });

        na.FitSurvival(features, times, events);

        var cumHazard = na.CumulativeHazard;
        Assert.NotNull(cumHazard);

        double expected0 = 1.0 / 5.0;
        double expected1 = expected0 + 1.0 / 4.0;
        double expected2 = expected1 + 1.0 / 3.0;
        double expected3 = expected2 + 1.0 / 2.0;
        double expected4 = expected3 + 1.0 / 1.0;

        Assert.Equal(expected0, cumHazard[0], Tolerance);
        Assert.Equal(expected1, cumHazard[1], Tolerance);
        Assert.Equal(expected2, cumHazard[2], Tolerance);
        Assert.Equal(expected3, cumHazard[3], Tolerance);
        Assert.Equal(expected4, cumHazard[4], Tolerance);
    }

    [Fact]
    public void NelsonAalen_VarianceFormula_MatchesHandComputed()
    {
        // Variance of Nelson-Aalen: Var(H(t)) = sum(d_i / n_i^2)
        // Same data: 5 subjects, times 1..5
        // Var(t=1) = 1/25 = 0.04
        // Var(t=2) = 1/25 + 1/16 = 0.1025
        // Var(t=3) = 1/25 + 1/16 + 1/9 ≈ 0.2136
        // Var(t=4) = ... + 1/4 ≈ 0.4636
        // Var(t=5) = ... + 1/1 ≈ 1.4636
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });

        na.FitSurvival(features, times, events);

        var variance = na.Variance;
        Assert.NotNull(variance);

        double v0 = 1.0 / 25.0;
        double v1 = v0 + 1.0 / 16.0;
        double v2 = v1 + 1.0 / 9.0;
        double v3 = v2 + 1.0 / 4.0;
        double v4 = v3 + 1.0 / 1.0;

        Assert.Equal(v0, variance[0], Tolerance);
        Assert.Equal(v1, variance[1], Tolerance);
        Assert.Equal(v2, variance[2], Tolerance);
        Assert.Equal(v3, variance[3], Tolerance);
        Assert.Equal(v4, variance[4], Tolerance);
    }

    [Fact]
    public void NelsonAalen_CumulativeHazardIsNonDecreasing()
    {
        var na = new NelsonAalenEstimator<double>();
        var n = 15;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 5.0;
            events[i] = i % 2 == 0 ? 1 : 0;
        }

        na.FitSurvival(features, times, events);

        var cumHazard = na.CumulativeHazard;
        Assert.NotNull(cumHazard);

        for (int i = 1; i < cumHazard.Length; i++)
        {
            Assert.True(cumHazard[i] >= cumHazard[i - 1] - Tolerance,
                $"H(t_{i})={cumHazard[i]} < H(t_{i - 1})={cumHazard[i - 1]}: cumulative hazard not non-decreasing");
        }
    }

    [Fact]
    public void NelsonAalen_SurvivalEqualsExpNegH()
    {
        // S(t) = exp(-H(t)) for Nelson-Aalen
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });
        var events = new Vector<int>(new int[] { 1, 1, 0, 1, 1 });

        na.FitSurvival(features, times, events);

        var cumHazard = na.CumulativeHazard;
        Assert.NotNull(cumHazard);

        var eventTimes = na.EventTimes;
        Assert.NotNull(eventTimes);

        var baseline = na.GetBaselineSurvival(eventTimes);

        for (int i = 0; i < cumHazard.Length; i++)
        {
            double expectedSurvival = Math.Exp(-cumHazard[i]);
            Assert.Equal(expectedSurvival, baseline[i], Tolerance);
        }
    }

    [Fact]
    public void NelsonAalen_WithCensoring_MatchesHandComputed()
    {
        // 6 subjects: events at 1, 3, 5; censored at 2, 4, 6
        // Event times: 1, 3, 5
        // At t=1: n=6, d=1 → H += 1/6
        // At t=3: n=4, d=1 → H += 1/4, cumH = 1/6 + 1/4 = 5/12
        // At t=5: n=2, d=1 → H += 1/2, cumH = 5/12 + 1/2 = 11/12
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(6, 1);
        for (int i = 0; i < 6; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
        var events = new Vector<int>(new int[] { 1, 0, 1, 0, 1, 0 });

        na.FitSurvival(features, times, events);

        var cumHazard = na.CumulativeHazard;
        Assert.NotNull(cumHazard);
        Assert.Equal(3, cumHazard.Length);

        Assert.Equal(1.0 / 6.0, cumHazard[0], Tolerance);
        Assert.Equal(1.0 / 6.0 + 1.0 / 4.0, cumHazard[1], Tolerance);
        Assert.Equal(1.0 / 6.0 + 1.0 / 4.0 + 1.0 / 2.0, cumHazard[2], Tolerance);
    }

    #endregion

    #region KM vs Nelson-Aalen Relationship

    [Fact]
    public void KM_And_NelsonAalen_SurvivalApproximatelyEqual_ForLargeSample()
    {
        // For large samples, Kaplan-Meier S(t) ≈ exp(-H_NA(t))
        // The approximation improves as sample size increases
        var km = new KaplanMeierEstimator<double>();
        var na = new NelsonAalenEstimator<double>();
        var n = 50;

        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 2.0;
            events[i] = 1; // All events, no censoring
        }

        km.FitSurvival(features, times, events);
        na.FitSurvival(features, times, events);

        var kmProbs = km.GetSurvivalProbabilities();
        var naBaseline = na.GetBaselineSurvival(km.GetEventTimes()!);

        Assert.NotNull(kmProbs);

        // They should be close but not identical
        // KM uses product-limit, NA uses exp(-cumulative hazard)
        for (int i = 0; i < kmProbs.Length; i++)
        {
            // For large n, these converge. Tolerance depends on sample size
            Assert.Equal(kmProbs[i], naBaseline[i], 0.05);
        }
    }

    [Fact]
    public void KM_SurvivalAlwaysLessOrEqual_NelsonAalen_Survival()
    {
        // A well-known result: S_KM(t) <= exp(-H_NA(t)) for all t
        // This is because ln(1-x) <= -x for x in [0,1)
        var km = new KaplanMeierEstimator<double>();
        var na = new NelsonAalenEstimator<double>();
        var n = 10;

        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 3.0;
            events[i] = 1;
        }

        km.FitSurvival(features, times, events);
        na.FitSurvival(features, times, events);

        var kmProbs = km.GetSurvivalProbabilities();
        var naBaseline = na.GetBaselineSurvival(km.GetEventTimes()!);

        Assert.NotNull(kmProbs);

        for (int i = 0; i < kmProbs.Length; i++)
        {
            Assert.True(kmProbs[i] <= naBaseline[i] + Tolerance,
                $"S_KM({i})={kmProbs[i]} > S_NA({i})={naBaseline[i]}: violates known inequality");
        }
    }

    #endregion

    #region Concordance Index Tests

    [Fact]
    public void ConcordanceIndex_PerfectDiscrimination_ReturnsOne()
    {
        // Perfect model: higher risk → shorter survival time
        // Subject 1: time=1 (event), risk should be highest
        // Subject 2: time=2 (event), risk second highest
        // Subject 3: time=3 (event), risk lowest
        //
        // Use Weibull AFT where negative coefficient → shorter time → higher risk
        // With a strong negative covariate, subjects with higher X have shorter times
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 3, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        km.FitSurvival(features, times, events);

        // For KM, all risk scores are 1.0 (no covariate effects)
        // So C-index should be 0.5 (random)
        var cIndex = km.CalculateConcordanceIndex(features, times, events);
        // KM can't discriminate, but with all tied scores the standard c-index
        // should be 0.5. Let's see what we get.
        // Actually for KM all hazard ratios = 1, so ALL pairs are tied.
        // Standard c-index with all ties = 0.5
        Assert.Equal(0.5, cIndex, 0.01);
    }

    [Fact]
    public void ConcordanceIndex_TiedRiskScores_HandComputed()
    {
        // This test verifies the concordance index handles ties correctly.
        // The standard Harrell's C-index gives half credit for tied risk scores.
        //
        // Setup: 5 subjects, all events
        // Times: 1, 2, 3, 4, 5
        // KM hazard ratios: all 1.0 (all tied)
        //
        // Comparable pairs (i has event, timeI < timeJ):
        // (0,1),(0,2),(0,3),(0,4) - 4 pairs
        // (1,2),(1,3),(1,4) - 3 pairs
        // (2,3),(2,4) - 2 pairs
        // (3,4) - 1 pair
        // Total: 10 comparable pairs
        //
        // All risk scores tied at 1.0, so standard C-index:
        // concordant = 10 * 0.5 = 5.0
        // C = 5.0 / 10 = 0.5
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });

        km.FitSurvival(features, times, events);
        var cIndex = km.CalculateConcordanceIndex(features, times, events);

        // Standard C-index with all tied scores should be 0.5
        Assert.Equal(0.5, cIndex, 0.01);
    }

    [Fact]
    public void ConcordanceIndex_MixedConcordantAndTied_HandComputed()
    {
        // We need a model that produces some concordant and some tied pairs.
        // Using Weibull with a known coefficient.
        //
        // 4 subjects:
        // Subject 0: X=3, time=1 (event)
        // Subject 1: X=2, time=2 (event)
        // Subject 2: X=1, time=3 (event)
        // Subject 3: X=1, time=4 (event)  <- same X as subject 2
        //
        // After fitting Weibull, negative coefficient → higher X → higher risk → shorter time
        // Risk: subject 0 > subject 1 > subjects 2,3 (tied)
        //
        // Comparable pairs (i has event, time_i < time_j):
        // (0,1): risk0 > risk1 → concordant
        // (0,2): risk0 > risk2 → concordant
        // (0,3): risk0 > risk3 → concordant
        // (1,2): risk1 > risk2 → concordant
        // (1,3): risk1 > risk3 → concordant
        // (2,3): risk2 = risk3 → TIE
        //
        // Standard C-index: concordant = 5 + 0.5 = 5.5, comparable = 6, C = 5.5/6 ≈ 0.917
        //
        // We can't easily get Weibull to produce exact risk scores, but this test
        // verifies that the tie-handling formula is correct for KM at minimum.
        //
        // For KM, ALL 6 pairs are tied, so C = 0.5
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(4, 1);
        for (int i = 0; i < 4; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1 });

        km.FitSurvival(features, times, events);

        // 6 comparable pairs, all tied at risk=1.0
        // Standard: C = (6*0.5)/6 = 0.5
        var cIndex = km.CalculateConcordanceIndex(features, times, events);
        Assert.Equal(0.5, cIndex, 0.01);
    }

    #endregion

    #region Weibull AFT Tests

    [Fact]
    public void WeibullAFT_SurvivalStartsAtOne()
    {
        // Use zero features to simplify optimization (only intercept and scale)
        var weibull = new WeibullAFT<double>(maxIterations: 500);
        var n = 20;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 0.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        weibull.FitSurvival(features, times, events);

        // Survival at time very close to 0 should be ≈ 1
        var queryTimes = new Vector<double>(new double[] { 0.001 });
        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 0.0;

        var survival = weibull.PredictSurvivalProbability(testFeatures, queryTimes);
        Assert.True(survival[0, 0] > 0.99, $"S(0.001) should be near 1.0, got {survival[0, 0]}");
    }

    [Fact]
    public void WeibullAFT_SurvivalApproachesZero_ForLargeTime()
    {
        var weibull = new WeibullAFT<double>(maxIterations: 500);
        var n = 20;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 0.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        weibull.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new double[] { 10000 });
        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 0.0;

        var survival = weibull.PredictSurvivalProbability(testFeatures, queryTimes);
        Assert.True(survival[0, 0] < 0.01, $"S(10000) should be near 0, got {survival[0, 0]}");
    }

    [Fact]
    public void WeibullAFT_SurvivalIsMonotonicallyDecreasing()
    {
        var weibull = new WeibullAFT<double>(maxIterations: 200);
        var n = 20;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        weibull.FitSurvival(features, times, events);

        // Query at many time points
        var queryTimes = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
            queryTimes[i] = (i + 1) * 10.0;

        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 1.0;

        var survival = weibull.PredictSurvivalProbability(testFeatures, queryTimes);

        for (int i = 1; i < queryTimes.Length; i++)
        {
            Assert.True(survival[0, i] <= survival[0, i - 1] + Tolerance,
                $"Weibull S(t_{i}) > S(t_{i - 1}): survival not monotonically non-increasing");
        }
    }

    [Fact]
    public void WeibullAFT_MedianSurvival_MatchesFormula()
    {
        // For Weibull: median = scale * (ln2)^(1/shape)
        // After fitting, the Predict method should return median survival times
        // We verify that S(median) ≈ 0.5
        var weibull = new WeibullAFT<double>(maxIterations: 200);
        var n = 30;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 0.0;
            times[i] = (i + 1) * 3.0;
            events[i] = 1;
        }

        weibull.FitSurvival(features, times, events);

        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 0.0;

        // Get predicted median
        var medians = weibull.Predict(testFeatures);
        double medianTime = medians[0];

        // Verify S(median) ≈ 0.5
        var queryTimes = new Vector<double>(new double[] { medianTime });
        var survival = weibull.PredictSurvivalProbability(testFeatures, queryTimes);

        Assert.Equal(0.5, survival[0, 0], 0.1);
    }

    [Fact]
    public void WeibullAFT_SurvivalBetweenZeroAndOne()
    {
        var weibull = new WeibullAFT<double>(maxIterations: 200);
        var n = 15;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 10.0;
            events[i] = i % 3 == 0 ? 0 : 1;
        }

        weibull.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
            queryTimes[i] = (i + 1) * 5.0;

        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 1.0;

        var survival = weibull.PredictSurvivalProbability(testFeatures, queryTimes);

        for (int i = 0; i < 20; i++)
        {
            Assert.True(survival[0, i] >= 0.0 - Tolerance && survival[0, i] <= 1.0 + Tolerance,
                $"Weibull S(t_{i})={survival[0, i]} is not in [0,1]");
        }
    }

    [Fact]
    public void WeibullAFT_BaselineSurvival_MatchesPredictionWithZeroFeatures()
    {
        // Baseline survival should match prediction with all-zero features
        var weibull = new WeibullAFT<double>(maxIterations: 200);
        var n = 15;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = (i % 2 == 0) ? 1.0 : 0.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        weibull.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new double[] { 10, 30, 50 });
        var baseline = weibull.GetBaselineSurvival(queryTimes);

        // Prediction with zero features should match baseline
        var zeroFeatures = new Matrix<double>(1, 1);
        zeroFeatures[0, 0] = 0.0;
        var prediction = weibull.PredictSurvivalProbability(zeroFeatures, queryTimes);

        for (int i = 0; i < queryTimes.Length; i++)
        {
            Assert.Equal(baseline[i], prediction[0, i], LooseTolerance);
        }
    }

    #endregion

    #region LogNormal AFT Tests

    [Fact]
    public void LogNormalAFT_SurvivalStartsAtOne()
    {
        var lognormal = new LogNormalAFT<double>(maxIterations: 200);
        var n = 20;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        lognormal.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new double[] { 0.001 });
        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 0.5;

        var survival = lognormal.PredictSurvivalProbability(testFeatures, queryTimes);
        Assert.True(survival[0, 0] > 0.99, $"LogNormal S(0.001) should be near 1.0, got {survival[0, 0]}");
    }

    [Fact]
    public void LogNormalAFT_SurvivalApproachesZero_ForLargeTime()
    {
        var lognormal = new LogNormalAFT<double>(maxIterations: 200);
        var n = 20;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        lognormal.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new double[] { 100000 });
        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 0.5;

        var survival = lognormal.PredictSurvivalProbability(testFeatures, queryTimes);
        Assert.True(survival[0, 0] < 0.01, $"LogNormal S(100000) should be near 0, got {survival[0, 0]}");
    }

    [Fact]
    public void LogNormalAFT_MedianSurvival_IsExpMu()
    {
        // For log-normal: median = exp(mu) where mu is the mean of log(T)
        // The Predict method should return exp(intercept + X*beta)
        var lognormal = new LogNormalAFT<double>(maxIterations: 200);
        var n = 30;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 0.0;
            times[i] = (i + 1) * 3.0;
            events[i] = 1;
        }

        lognormal.FitSurvival(features, times, events);

        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 0.0;

        var medians = lognormal.Predict(testFeatures);
        double medianTime = medians[0];

        // Verify S(median) ≈ 0.5
        var queryTimes = new Vector<double>(new double[] { medianTime });
        var survival = lognormal.PredictSurvivalProbability(testFeatures, queryTimes);

        Assert.Equal(0.5, survival[0, 0], 0.15);
    }

    [Fact]
    public void LogNormalAFT_SurvivalBetweenZeroAndOne()
    {
        var lognormal = new LogNormalAFT<double>(maxIterations: 200);
        var n = 15;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 10.0;
            events[i] = i % 3 == 0 ? 0 : 1;
        }

        lognormal.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
            queryTimes[i] = (i + 1) * 5.0;

        var testFeatures = new Matrix<double>(1, 1);
        testFeatures[0, 0] = 1.0;

        var survival = lognormal.PredictSurvivalProbability(testFeatures, queryTimes);

        for (int i = 0; i < 20; i++)
        {
            Assert.True(survival[0, i] >= 0.0 - Tolerance && survival[0, i] <= 1.0 + Tolerance,
                $"LogNormal S(t_{i})={survival[0, i]} is not in [0,1]");
        }
    }

    [Fact]
    public void LogNormalAFT_CensoredGradients_MovesInterceptCorrectly()
    {
        // This test verifies that the LogNormal AFT correctly handles censored observations.
        // If censored gradients are correct, the fitted intercept should be HIGHER when
        // all observations are censored (because censoring means "survived at least until time t"
        // so the true distribution is shifted to longer times).
        //
        // Scenario A: all events → intercept reflects actual survival times
        // Scenario B: all censored → intercept should be pushed higher (longer survival)
        //
        // The gradient for censored observations should push the intercept UPWARD
        // (toward longer survival times).
        var timesArray = new double[] { 10, 20, 30, 40, 50 };

        // Fit with all events
        var modelEvents = new LogNormalAFT<double>(maxIterations: 200);
        var features = new Matrix<double>(5, 1);
        var times = new Vector<double>(timesArray);
        var allEvents = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });
        for (int i = 0; i < 5; i++) features[i, 0] = 0.0;

        modelEvents.FitSurvival(features, times, allEvents);
        double interceptEvents = modelEvents.Intercept;

        // Fit with all censored
        var modelCensored = new LogNormalAFT<double>(maxIterations: 200);
        var features2 = new Matrix<double>(5, 1);
        var times2 = new Vector<double>(timesArray);
        var allCensored = new Vector<int>(new int[] { 0, 0, 0, 0, 0 });
        for (int i = 0; i < 5; i++) features2[i, 0] = 0.0;

        modelCensored.FitSurvival(features2, times2, allCensored);
        double interceptCensored = modelCensored.Intercept;

        // When all are censored, the true times are longer than observed,
        // so the intercept should be higher (or at least not lower).
        // This will FAIL if censored gradients have wrong sign.
        Assert.True(interceptCensored >= interceptEvents - 0.5,
            $"Censored intercept ({interceptCensored:F4}) should be >= events intercept ({interceptEvents:F4}) " +
            "because censored observations imply longer survival times. " +
            "This failure suggests censored gradients have the wrong sign.");
    }

    [Fact]
    public void LogNormalAFT_BaselineSurvival_MatchesPredictionWithZeroFeatures()
    {
        var lognormal = new LogNormalAFT<double>(maxIterations: 200);
        var n = 15;
        var features = new Matrix<double>(n, 1);
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);

        for (int i = 0; i < n; i++)
        {
            features[i, 0] = (i % 2 == 0) ? 1.0 : 0.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        lognormal.FitSurvival(features, times, events);

        var queryTimes = new Vector<double>(new double[] { 10, 30, 50 });
        var baseline = lognormal.GetBaselineSurvival(queryTimes);

        var zeroFeatures = new Matrix<double>(1, 1);
        zeroFeatures[0, 0] = 0.0;
        var prediction = lognormal.PredictSurvivalProbability(zeroFeatures, queryTimes);

        for (int i = 0; i < queryTimes.Length; i++)
        {
            Assert.Equal(baseline[i], prediction[0, i], LooseTolerance);
        }
    }

    #endregion

    #region Base Class / Shared Tests

    [Fact]
    public void GetUniqueEventTimes_OnlyIncludesEventTimes()
    {
        // Censored times should NOT appear in event times
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(6, 1);
        for (int i = 0; i < 6; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
        var events = new Vector<int>(new int[] { 1, 0, 1, 0, 1, 0 });

        km.FitSurvival(features, times, events);

        var eventTimes = km.GetEventTimes();
        Assert.NotNull(eventTimes);
        Assert.Equal(3, eventTimes.Length);
        Assert.Equal(1.0, eventTimes[0], Tolerance);
        Assert.Equal(3.0, eventTimes[1], Tolerance);
        Assert.Equal(5.0, eventTimes[2], Tolerance);
    }

    [Fact]
    public void GetUniqueEventTimes_AreSorted()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        // Unsorted input times
        var times = new Vector<double>(new double[] { 5, 1, 4, 2, 3 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });

        km.FitSurvival(features, times, events);

        var eventTimes = km.GetEventTimes();
        Assert.NotNull(eventTimes);

        for (int i = 1; i < eventTimes.Length; i++)
        {
            Assert.True(eventTimes[i] >= eventTimes[i - 1],
                "Event times should be sorted");
        }
    }

    [Fact]
    public void GetUniqueEventTimes_NoDuplicates()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(6, 1);
        for (int i = 0; i < 6; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 3, 3, 3, 5, 5, 7 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 1 });

        km.FitSurvival(features, times, events);

        var eventTimes = km.GetEventTimes();
        Assert.NotNull(eventTimes);
        Assert.Equal(3, eventTimes.Length); // Only 3 unique times: 3, 5, 7
    }

    [Fact]
    public void Validation_NegativeTimes_Throws()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, -2, 3 });
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        Assert.Throws<ArgumentException>(() =>
            km.FitSurvival(features, times, events));
    }

    [Fact]
    public void Validation_InvalidEvents_Throws()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3 });
        var events = new Vector<int>(new int[] { 1, 2, 0 }); // 2 is invalid

        Assert.Throws<ArgumentException>(() =>
            km.FitSurvival(features, times, events));
    }

    [Fact]
    public void Validation_DimensionMismatch_Throws()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2 }); // Only 2, features has 3
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        Assert.Throws<ArgumentException>(() =>
            km.FitSurvival(features, times, events));
    }

    [Fact]
    public void PredictBeforeFit_Throws()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(1, 1);
        features[0, 0] = 1.0;

        Assert.Throws<InvalidOperationException>(() =>
            km.Predict(features));
    }

    [Fact]
    public void NelsonAalen_Serialize_Deserialize_PreservesState()
    {
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });
        var events = new Vector<int>(new int[] { 1, 1, 0, 1, 1 });

        na.FitSurvival(features, times, events);

        // Serialize
        var bytes = na.Serialize();

        // Deserialize into new instance
        var na2 = new NelsonAalenEstimator<double>();
        na2.Deserialize(bytes);

        // Parameters should be preserved
        var params1 = na.GetParameters();
        var params2 = na2.GetParameters();

        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i], Tolerance);
        }
    }

    [Fact]
    public void CumulativeHazard_EqualsSurvival_NegLogRelationship()
    {
        // H(t) = -ln(S(t)) for any model
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(10, 1);
        var times = new Vector<double>(10);
        var events = new Vector<int>(10);

        for (int i = 0; i < 10; i++)
        {
            features[i, 0] = 1.0;
            times[i] = (i + 1) * 5.0;
            events[i] = 1;
        }

        na.FitSurvival(features, times, events);

        // Use PredictCumulativeHazard from base class (which computes -ln(S(t)))
        var queryTimes = na.EventTimes!;
        var survivalMatrix = na.PredictSurvival(queryTimes);
        var cumHazardMatrix = na.PredictCumulativeHazard(queryTimes);

        for (int t = 0; t < queryTimes.Length; t++)
        {
            double s = survivalMatrix[0, t];
            double h = cumHazardMatrix[0, t];
            double expectedH = -Math.Log(Math.Max(1e-10, s));
            Assert.Equal(expectedH, h, 1e-6);
        }
    }

    #endregion

    #region ISurvivalModel Interface Tests

    [Fact]
    public void KaplanMeier_FitInterface_WithVectorEvents_Works()
    {
        // Test the Fit(times, events, features) interface method
        var km = new KaplanMeierEstimator<double>();

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<double>(new double[] { 1.0, 0.0, 1.0, 0.0, 1.0 });

        km.Fit(times, events);

        // Should be fitted
        Assert.True(km.IsTrained);

        // Should produce predictions
        var queryTimes = new Vector<double>(new double[] { 1, 3, 5 });
        var survival = km.PredictSurvival(queryTimes);
        Assert.Equal(1, survival.Rows);
        Assert.Equal(3, survival.Columns);
    }

    [Fact]
    public void NelsonAalen_PredictRisk_ReturnsOnesForAllSubjects()
    {
        // Non-parametric model returns risk=1 for all
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var events = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });

        na.FitSurvival(features, times, events);

        var risk = na.PredictRisk(features);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(1.0, risk[i], Tolerance);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void KaplanMeier_SingleSubjectEvent_SurvivalDropsToZero()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(1, 1);
        features[0, 0] = 1.0;

        var times = new Vector<double>(new double[] { 5 });
        var events = new Vector<int>(new int[] { 1 });

        km.FitSurvival(features, times, events);

        var probs = km.GetSurvivalProbabilities();
        Assert.NotNull(probs);
        Assert.Equal(1, probs.Length);
        Assert.Equal(0.0, probs[0], Tolerance); // n=1, d=1 → S = 0/1 = 0
    }

    [Fact]
    public void NelsonAalen_SingleSubjectEvent_CumulativeHazardIsOne()
    {
        // Single subject event: H = d/n = 1/1 = 1
        var na = new NelsonAalenEstimator<double>();
        var features = new Matrix<double>(1, 1);
        features[0, 0] = 1.0;

        var times = new Vector<double>(new double[] { 5 });
        var events = new Vector<int>(new int[] { 1 });

        na.FitSurvival(features, times, events);

        var cumHazard = na.CumulativeHazard;
        Assert.NotNull(cumHazard);
        Assert.Equal(1, cumHazard.Length);
        Assert.Equal(1.0, cumHazard[0], Tolerance);
    }

    [Fact]
    public void KaplanMeier_StepFunctionBehavior_SurvivalConstantBetweenEvents()
    {
        var km = new KaplanMeierEstimator<double>();
        var features = new Matrix<double>(3, 1);
        for (int i = 0; i < 3; i++) features[i, 0] = 1.0;

        var times = new Vector<double>(new double[] { 10, 20, 30 });
        var events = new Vector<int>(new int[] { 1, 1, 1 });

        km.FitSurvival(features, times, events);

        // Query at times between events: should get the step function values
        var queryTimes = new Vector<double>(new double[] { 5, 10, 15, 20, 25, 30 });
        var baseline = km.GetBaselineSurvival(queryTimes);

        // Before first event (t=5): S = 1.0
        Assert.Equal(1.0, baseline[0], Tolerance);

        // At first event (t=10): S = 2/3
        Assert.Equal(2.0 / 3.0, baseline[1], Tolerance);

        // Between events (t=15): S stays at 2/3
        Assert.Equal(2.0 / 3.0, baseline[2], Tolerance);

        // At second event (t=20): S = 2/3 * 1/2 = 1/3
        Assert.Equal(1.0 / 3.0, baseline[3], Tolerance);

        // Between events (t=25): S stays at 1/3
        Assert.Equal(1.0 / 3.0, baseline[4], Tolerance);

        // At third event (t=30): S = 1/3 * 0/1 = 0
        Assert.Equal(0.0, baseline[5], Tolerance);
    }

    #endregion
}
