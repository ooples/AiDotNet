using System;
using System.Collections.Generic;

namespace AiDotNet.Finance.Trading.Evaluation;

/// <summary>
/// Chronological train/holdout split for portfolio experiments: train on the earlier history, evaluate on a
/// strictly-later UNTOUCHED tail, with an optional embargo gap dropped at the boundary so no information leaks
/// across it. This is the honest-eval discipline for time-series RL — a random split would let the agent train
/// on the future it is later scored on. Mirrors the purged/embargoed split used for supervised training.
/// </summary>
public static class WalkForwardSplit
{
    /// <summary>
    /// Splits each column of <paramref name="series"/> (all equal length) at <paramref name="trainFraction"/>
    /// into (train = earlier rows, holdout = later rows), dropping <paramref name="embargo"/> rows at the
    /// boundary. Feature columns should be split with the SAME parameters so they stay aligned to prices.
    /// </summary>
    public static (List<double[]> Train, List<double[]> Holdout) Split(
        IReadOnlyList<double[]> series, double trainFraction, int embargo = 0)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (series.Count == 0)
        {
            throw new ArgumentException("At least one series column is required.", nameof(series));
        }

        if (trainFraction <= 0.0 || trainFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(trainFraction), "trainFraction must be in (0, 1).");
        }

        embargo = Math.Max(0, embargo);
        int n = series[0].Length;
        for (int i = 1; i < series.Count; i++)
        {
            if (series[i].Length != n)
            {
                throw new ArgumentException("All series columns must have the same length.", nameof(series));
            }
        }

        int splitIndex = (int)(n * trainFraction);
        int holdoutStart = Math.Min(n, splitIndex + embargo);

        var train = new List<double[]>(series.Count);
        var holdout = new List<double[]>(series.Count);
        foreach (var col in series)
        {
            train.Add(col[..splitIndex]);
            holdout.Add(col[holdoutStart..]);
        }

        return (train, holdout);
    }
}
