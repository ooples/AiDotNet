using System;
using System.Collections.Generic;
using AiDotNet.Finance.Evaluation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

/// <summary>
/// Verifies the return-value-centric research evaluation statistical primitives: InformationCoefficient,
/// PurgedWalkForwardValidator, DeflatedSharpeRatio, BootstrapConfidenceInterval, and BenjaminiHochbergFdr.
/// </summary>
public class ResearchEvaluationStatsTests
{
    // ---------------- InformationCoefficient ----------------

    [Fact]
    public void IC_is_one_for_perfectly_correlated()
    {
        var pred = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var real = new[] { 2.0, 4.0, 6.0, 8.0, 10.0 }; // perfectly linear (and monotone)
        Assert.Equal(1.0, InformationCoefficient<double>.Pearson(pred, real), 9);
        Assert.Equal(1.0, InformationCoefficient<double>.Spearman(pred, real), 9);
    }

    [Fact]
    public void IC_is_negative_one_for_perfectly_anticorrelated()
    {
        var pred = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var real = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 };
        Assert.Equal(-1.0, InformationCoefficient<double>.Pearson(pred, real), 9);
        Assert.Equal(-1.0, InformationCoefficient<double>.Spearman(pred, real), 9);
    }

    [Fact]
    public void IC_is_near_zero_for_independent_series()
    {
        var rng = new Random(12345);
        int n = 5000;
        var pred = new double[n];
        var real = new double[n];
        for (int i = 0; i < n; i++)
        {
            pred[i] = rng.NextDouble() - 0.5;
            real[i] = rng.NextDouble() - 0.5; // independent
        }

        double ic = InformationCoefficient<double>.Pearson(pred, real);
        Assert.True(Math.Abs(ic) < 0.05, $"IC should be near 0 for independent series, was {ic}");
    }

    [Fact]
    public void IC_tstat_has_correct_sign_and_significance()
    {
        // Positive IC => positive t-stat; large n with |IC| moderate => tiny p-value.
        var (tPos, pPos) = InformationCoefficient<double>.Significance(0.3, 200);
        Assert.True(tPos > 0.0);
        Assert.True(pPos < 0.01);

        var (tNeg, _) = InformationCoefficient<double>.Significance(-0.3, 200);
        Assert.True(tNeg < 0.0);

        // IC == 0 => t == 0, p == 1.
        var (tZero, pZero) = InformationCoefficient<double>.Significance(0.0, 100);
        Assert.Equal(0.0, tZero, 9);
        Assert.Equal(1.0, pZero, 6);
    }

    [Fact]
    public void IC_tstat_matches_formula()
    {
        // t = IC*sqrt((n-2)/(1-IC^2)); IC=0.2, n=50 => 0.2*sqrt(48/0.96) = 0.2*sqrt(50) = 1.41421356
        var (t, _) = InformationCoefficient<double>.Significance(0.2, 50);
        Assert.Equal(0.2 * Math.Sqrt(48.0 / 0.96), t, 6);
    }

    [Fact]
    public void ICIR_rewards_consistent_positive_ic()
    {
        // Stable positive ICs => high ICIR; noisy ICs around zero => low |ICIR|.
        var stable = new[] { 0.04, 0.05, 0.045, 0.05, 0.04, 0.05 };
        var (meanS, stdS, icirS) = InformationCoefficient<double>.InformationRatio(stable);
        Assert.True(meanS > 0.0);
        Assert.True(stdS > 0.0);
        Assert.True(icirS > 2.0);

        var noisy = new[] { 0.1, -0.1, 0.1, -0.1, 0.1, -0.1 };
        var (_, _, icirN) = InformationCoefficient<double>.InformationRatio(noisy);
        Assert.True(Math.Abs(icirN) < Math.Abs(icirS));
    }

    // ---------------- PurgedWalkForwardValidator ----------------

    [Fact]
    public void PurgedWF_never_leaks_label_horizon_into_test_fold()
    {
        int n = 200;
        int h = 5;
        int embargo = 3;
        var folds = PurgedWalkForwardValidator.Split(n, labelHorizon: h, nSplits: 4, embargo: embargo, expanding: true);

        Assert.NotEmpty(folds);
        foreach (var fold in folds)
        {
            Assert.NotEmpty(fold.TestIndices);
            int testStart = fold.TestIndices[0];
            int testEnd = fold.TestIndices[fold.TestIndices.Count - 1] + 1; // exclusive
            int testLabelEnd = (testEnd - 1) + h;

            var testSet = new HashSet<int>(fold.TestIndices);

            foreach (int i in fold.TrainIndices)
            {
                // Train index must not be inside the test fold (walk-forward is past-only).
                Assert.DoesNotContain(i, testSet);
                Assert.True(i < testStart, $"walk-forward train index {i} must precede test start {testStart}");

                // Embargo: no train index in [testStart - embargo, testStart).
                Assert.False(i >= testStart - embargo && i < testStart, $"embargo violated by train index {i}");

                // Purge: train label window [i, i+h-1] must not overlap [testStart, testLabelEnd].
                int iLabelEnd = i + h - 1;
                bool overlaps = !(iLabelEnd < testStart || i > testLabelEnd);
                Assert.False(overlaps, $"label leakage: train {i} (label end {iLabelEnd}) overlaps test [{testStart},{testLabelEnd}]");
            }
        }
    }

    [Fact]
    public void PurgedWF_expanding_grows_and_test_folds_are_forward_ordered()
    {
        var folds = PurgedWalkForwardValidator.Split(300, labelHorizon: 1, nSplits: 5, embargo: 0, expanding: true);
        int prevTestStart = -1;
        for (int f = 0; f < folds.Count; f++)
        {
            int testStart = folds[f].TestIndices[0];
            Assert.True(testStart > prevTestStart, "test folds must walk forward");
            prevTestStart = testStart;

            // All train indices precede the test fold (h=1, no embargo, expanding).
            foreach (int i in folds[f].TrainIndices)
            {
                Assert.True(i < testStart);
            }
        }
    }

    [Fact]
    public void PurgedWF_sliding_window_is_bounded()
    {
        int nSplits = 5;
        var folds = PurgedWalkForwardValidator.Split(300, labelHorizon: 1, nSplits: nSplits, embargo: 0, expanding: false);
        // Sliding window training count should be roughly one fold-size, much smaller than expanding's last fold.
        var expanding = PurgedWalkForwardValidator.Split(300, labelHorizon: 1, nSplits: nSplits, embargo: 0, expanding: true);
        Assert.True(folds[folds.Count - 1].TrainIndices.Count < expanding[expanding.Count - 1].TrainIndices.Count);
    }

    // ---------------- DeflatedSharpeRatio ----------------

    [Fact]
    public void DSR_is_in_unit_interval()
    {
        double dsr = DeflatedSharpeRatio<double>.Compute(observedSharpe: 0.15, nObservations: 250, nTrials: 10, skew: -0.5, kurtosis: 5.0);
        Assert.InRange(dsr, 0.0, 1.0);
    }

    [Fact]
    public void DSR_decreases_with_more_trials()
    {
        double few = DeflatedSharpeRatio<double>.Compute(0.15, 250, nTrials: 2);
        double many = DeflatedSharpeRatio<double>.Compute(0.15, 250, nTrials: 1000);
        Assert.True(many < few, $"DSR should fall as trials rise (few={few}, many={many})");
    }

    [Fact]
    public void DSR_expected_max_sharpe_grows_with_trials()
    {
        double e10 = DeflatedSharpeRatio<double>.ExpectedMaxSharpe(10);
        double e1000 = DeflatedSharpeRatio<double>.ExpectedMaxSharpe(1000);
        Assert.True(e1000 > e10);
        Assert.True(e10 > 0.0);
        Assert.Equal(0.0, DeflatedSharpeRatio<double>.ExpectedMaxSharpe(1), 9); // single trial => no inflation
    }

    [Fact]
    public void DSR_high_sharpe_low_trials_is_confident()
    {
        double dsr = DeflatedSharpeRatio<double>.Compute(observedSharpe: 0.4, nObservations: 500, nTrials: 1);
        Assert.True(dsr > 0.95);
    }

    // ---------------- BootstrapConfidenceInterval ----------------

    [Fact]
    public void Bootstrap_ci_brackets_point_estimate()
    {
        var rng = new Random(42);
        var rets = new double[300];
        var r2 = new Random(7);
        for (int i = 0; i < rets.Length; i++)
        {
            rets[i] = 0.01 + 0.02 * (r2.NextDouble() - 0.5);
        }

        var ci = BootstrapConfidenceInterval<double>.Compute(
            rets, rng, BootstrapConfidenceInterval<double>.MeanStatistic, confidence: 0.95, nResamples: 2000);

        Assert.True(ci.Lower <= ci.PointEstimate);
        Assert.True(ci.PointEstimate <= ci.Upper);
    }

    [Fact]
    public void Bootstrap_ci_narrows_with_more_data()
    {
        double WidthFor(int n)
        {
            var src = new Random(99);
            var rets = new double[n];
            for (int i = 0; i < n; i++)
            {
                rets[i] = 0.01 + 0.05 * (src.NextDouble() - 0.5);
            }

            var rng = new Random(123);
            var ci = BootstrapConfidenceInterval<double>.Compute(
                rets, rng, BootstrapConfidenceInterval<double>.MeanStatistic, confidence: 0.95, nResamples: 1500);
            return ci.Upper - ci.Lower;
        }

        double wSmall = WidthFor(50);
        double wLarge = WidthFor(2000);
        Assert.True(wLarge < wSmall, $"CI should narrow with more data (small={wSmall}, large={wLarge})");
    }

    [Fact]
    public void Bootstrap_is_reproducible_with_same_seed()
    {
        var rets = new double[100];
        var src = new Random(1);
        for (int i = 0; i < rets.Length; i++)
        {
            rets[i] = src.NextDouble();
        }

        var a = BootstrapConfidenceInterval<double>.Compute(rets, new Random(555), nResamples: 500);
        var b = BootstrapConfidenceInterval<double>.Compute(rets, new Random(555), nResamples: 500);
        Assert.Equal(a.Lower, b.Lower, 12);
        Assert.Equal(a.Upper, b.Upper, 12);
    }

    // ---------------- BenjaminiHochbergFdr ----------------

    [Fact]
    public void BH_matches_hand_computed_example()
    {
        // Classic worked example: p = {0.005, 0.009, 0.019, 0.022, 0.051, 0.30}, alpha=0.05, m=6.
        // Thresholds (k/m)*alpha: 0.00833, 0.01667, 0.025, 0.03333, 0.04167, 0.05.
        // Compare sorted p to threshold: 0.005<=0.00833 (k1), 0.009<=0.01667 (k2), 0.019<=0.025 (k3),
        // 0.022<=0.03333 (k4), 0.051>0.04167 (no), 0.30>0.05 (no). Largest passing k = 4 => reject first 4.
        var p = new[] { 0.005, 0.009, 0.019, 0.022, 0.051, 0.30 };
        var res = BenjaminiHochbergFdr.Apply(p, alpha: 0.05);

        Assert.Equal(4, res.NumRejected);
        Assert.True(res.Rejected[0]);
        Assert.True(res.Rejected[1]);
        Assert.True(res.Rejected[2]);
        Assert.True(res.Rejected[3]);
        Assert.False(res.Rejected[4]);
        Assert.False(res.Rejected[5]);

        // q-values are monotone in p-rank and in [0,1]. q for smallest = min(6/1*0.005, ...) capped.
        for (int i = 0; i < p.Length; i++)
        {
            Assert.InRange(res.QValues[i], 0.0, 1.0);
        }

        // Hand q-value for the last (p=0.30): 6/6*0.30 = 0.30.
        Assert.Equal(0.30, res.QValues[5], 9);
        // q for p=0.022 (rank 4): min(6/4*0.022=0.033, 6/5*0.051=0.0612, 0.30) = 0.033.
        Assert.Equal(6.0 / 4.0 * 0.022, res.QValues[3], 9);
    }

    [Fact]
    public void BH_rejects_nothing_when_all_pvalues_large()
    {
        var p = new[] { 0.4, 0.5, 0.6, 0.9 };
        var res = BenjaminiHochbergFdr.Apply(p, 0.05);
        Assert.Equal(0, res.NumRejected);
        foreach (var rej in res.Rejected)
        {
            Assert.False(rej);
        }
    }

    [Fact]
    public void BH_qvalues_are_monotone_nondecreasing_in_pvalue_order()
    {
        var p = new[] { 0.001, 0.02, 0.04, 0.08, 0.2, 0.5 }; // already ascending
        var res = BenjaminiHochbergFdr.Apply(p, 0.1);
        for (int i = 1; i < p.Length; i++)
        {
            Assert.True(res.QValues[i] >= res.QValues[i - 1] - 1e-12);
        }
    }
}
