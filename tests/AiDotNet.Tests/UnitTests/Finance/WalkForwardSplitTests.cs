using System;
using System.Collections.Generic;
using AiDotNet.Finance.Trading.Evaluation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the chronological walk-forward train/holdout split — the honest-eval discipline for time-series RL
/// (train on the past, evaluate on a strictly-later untouched tail, with an embargo gap to block leakage).
/// Row value == index makes the boundary observable.
/// </summary>
public sealed class WalkForwardSplitTests
{
    private static double[] Index(int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Splits_chronologically_at_the_fraction()
    {
        var (train, holdout) = WalkForwardSplit.Split(new List<double[]> { Index(100) }, trainFraction: 0.7);

        Assert.Equal(70, train[0].Length);
        Assert.Equal(30, holdout[0].Length);
        Assert.Equal(0.0, train[0][0]);        // train starts at the beginning
        Assert.Equal(69.0, train[0][^1]);      // ...ends just before the split
        Assert.Equal(70.0, holdout[0][0]);     // holdout starts strictly after (chronological, no shuffle)
    }

    [Fact]
    [Trait("category", "unit")]
    public void Embargo_drops_a_gap_at_the_boundary()
    {
        var (train, holdout) = WalkForwardSplit.Split(new List<double[]> { Index(100) }, trainFraction: 0.7, embargo: 5);

        Assert.Equal(70, train[0].Length);
        Assert.Equal(70.0 - 1, train[0][^1]);        // train unchanged: rows [0,70)
        Assert.Equal(75.0, holdout[0][0]);           // holdout starts 5 rows later — the gap is dropped
        Assert.Equal(25, holdout[0].Length);         // 100 - 70 - 5
    }

    [Fact]
    [Trait("category", "unit")]
    public void All_columns_split_identically()
    {
        var (train, holdout) = WalkForwardSplit.Split(
            new List<double[]> { Index(50), Index(50) }, trainFraction: 0.6, embargo: 2);

        Assert.Equal(2, train.Count);
        Assert.Equal(2, holdout.Count);
        Assert.Equal(train[0].Length, train[1].Length);
        Assert.Equal(holdout[0][0], holdout[1][0]); // same boundary for every column
    }

    [Fact]
    [Trait("category", "unit")]
    public void Rejects_invalid_fraction_and_ragged_columns()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => WalkForwardSplit.Split(new List<double[]> { Index(10) }, 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => WalkForwardSplit.Split(new List<double[]> { Index(10) }, 1.0));
        Assert.Throws<ArgumentException>(() => WalkForwardSplit.Split(new List<double[]> { Index(10), Index(9) }, 0.5));
    }
}
