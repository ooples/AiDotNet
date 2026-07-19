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
        if (series is null) throw new ArgumentNullException(nameof(series));
        if (series.Count == 0)
        {
            throw new ArgumentException("At least one series column is required.", nameof(series));
        }

        // NaN slips through a plain range check (every NaN comparison is false), so test finiteness explicitly.
        if ((double.IsNaN(trainFraction) || double.IsInfinity(trainFraction)) || trainFraction <= 0.0 || trainFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(trainFraction), "trainFraction must be a finite value in (0, 1).");
        }

        // Reject a negative embargo rather than silently rewriting it to 0 — a negative gap is a caller bug.
        if (embargo < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embargo), "embargo must be non-negative.");
        }

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

        // Reject a split that leaves either partition empty (series too short, or trainFraction/embargo too
        // extreme) rather than returning a zero-row train or holdout that fails downstream.
        if (splitIndex <= 0 || holdoutStart >= n)
        {
            throw new ArgumentException(
                $"Split leaves an empty partition (length {n}, train rows {splitIndex}, holdout rows {n - holdoutStart}); " +
                "reduce the embargo or adjust trainFraction / provide a longer series.",
                nameof(series));
        }

        var train = new List<double[]>(series.Count);
        var holdout = new List<double[]>(series.Count);
        int holdoutLength = n - holdoutStart;
        foreach (var col in series)
        {
            // Manual slices (net471 has no array-range/GetSubArray support): train = [0, splitIndex),
            // holdout = [holdoutStart, n).
            var trainCol = new double[splitIndex];
            Array.Copy(col, 0, trainCol, 0, splitIndex);
            var holdoutCol = new double[holdoutLength];
            Array.Copy(col, holdoutStart, holdoutCol, 0, holdoutLength);
            train.Add(trainCol);
            holdout.Add(holdoutCol);
        }

        return (train, holdout);
    }
}
