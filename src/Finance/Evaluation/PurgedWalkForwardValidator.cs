using System;
using System.Collections.Generic;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// López de Prado purged-and-embargoed walk-forward cross-validation for time-ordered financial data.
/// Generates rolling-origin train/test splits while removing training samples whose label horizon
/// overlaps the test window (purge) and a buffer of samples immediately after each test fold (embargo).
/// </summary>
/// <remarks>
/// <para>
/// Standard k-fold CV leaks information when labels are computed over a forward horizon (e.g. the label
/// for sample <c>i</c> depends on returns up to <c>i + h</c>): a training sample near a test fold can
/// "see the future" through its overlapping label, inflating measured performance. Purging drops those
/// overlapping training samples; the embargo additionally drops a gap right after the test fold to
/// neutralize serial correlation that survives purging. Walk-forward (rolling-origin) ordering keeps the
/// test fold strictly after the training data, mirroring live deployment.
/// </para>
/// <para><b>For Beginners:</b> When you test a trading model, you must never let it train on data that
/// secretly contains the answers to the test. Because each label here looks forward in time (e.g. "the
/// return over the next 5 days"), a training point sitting just before the test period overlaps with it
/// and would leak the answer. "Purging" deletes those contaminated training points; the "embargo" deletes
/// a few extra points right after the test window for good measure. The result is a list of clean
/// (train, test) splits that walk forward through time the way real trading does.</para>
/// </remarks>
public static class PurgedWalkForwardValidator
{
    /// <summary>One purged/embargoed walk-forward fold: the surviving train indices and the test indices.</summary>
    public sealed class Fold
    {
        /// <summary>Training-sample indices remaining after purge and embargo.</summary>
        public IReadOnlyList<int> TrainIndices { get; }

        /// <summary>Test-sample indices for this fold (a contiguous forward block).</summary>
        public IReadOnlyList<int> TestIndices { get; }

        /// <summary>Creates a fold from its train and test index lists.</summary>
        public Fold(IReadOnlyList<int> trainIndices, IReadOnlyList<int> testIndices)
        {
            TrainIndices = trainIndices;
            TestIndices = testIndices;
        }
    }

    /// <summary>
    /// Builds the purged + embargoed walk-forward splits.
    /// </summary>
    /// <param name="nSamples">Total number of time-ordered samples (indices 0..nSamples-1).</param>
    /// <param name="labelHorizon">
    /// Label horizon h: the label of sample <c>i</c> is assumed to depend on the window [i, i + h).
    /// A training sample i is purged from a test fold T if [i, i+h) overlaps [min(T), max(T) + h).
    /// Use 1 for point-in-time (non-overlapping) labels.
    /// </param>
    /// <param name="nSplits">Number of walk-forward test folds.</param>
    /// <param name="embargo">Number of samples immediately after each test fold to also drop from training.</param>
    /// <param name="expanding">
    /// When true, training is an expanding window (all eligible samples before the test fold). When false,
    /// it is a sliding window equal in length to one test fold (the most recent eligible block).
    /// </param>
    /// <returns>One <see cref="Fold"/> per split, in chronological order.</returns>
    public static IReadOnlyList<Fold> Split(
        int nSamples,
        int labelHorizon,
        int nSplits,
        int embargo = 0,
        bool expanding = true)
    {
        if (nSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(nSamples), "nSamples must be positive.");
        }

        if (labelHorizon < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(labelHorizon), "labelHorizon must be at least 1.");
        }

        if (nSplits < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nSplits), "nSplits must be at least 1.");
        }

        if (embargo < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embargo), "embargo must be non-negative.");
        }

        var folds = new List<Fold>(nSplits);

        // Partition the index range into nSplits contiguous, near-equal test blocks at the END of the
        // timeline, reserving the first block as the minimum training seed.
        int foldSize = Math.Max(1, nSamples / (nSplits + 1));
        int firstTestStart = nSamples - foldSize * nSplits;
        if (firstTestStart < 1)
        {
            // Not enough samples to give every split a test block and a non-empty seed; shrink folds.
            foldSize = Math.Max(1, (nSamples - 1) / nSplits);
            firstTestStart = nSamples - foldSize * nSplits;
            if (firstTestStart < 1)
            {
                firstTestStart = 1;
            }
        }

        for (int split = 0; split < nSplits; split++)
        {
            int testStart = firstTestStart + split * foldSize;
            int testEnd = split == nSplits - 1 ? nSamples : testStart + foldSize; // exclusive
            if (testStart >= nSamples)
            {
                break;
            }

            if (testEnd > nSamples)
            {
                testEnd = nSamples;
            }

            var testIndices = new List<int>(testEnd - testStart);
            for (int i = testStart; i < testEnd; i++)
            {
                testIndices.Add(i);
            }

            // Walk-forward training is strictly in the PAST of the test fold: indices in [trainLo, testStart).
            // The test fold's label window spans [testStart, testEnd - 1 + labelHorizon]; a training sample i
            // leaks if its own label window [i, i + labelHorizon - 1] reaches into the test fold (purge).
            int testLabelEnd = (testEnd - 1) + labelHorizon;

            // Embargo: drop a buffer of `embargo` samples immediately before the test fold (the most-recent,
            // most-autocorrelated training data) in addition to the purge.
            int embargoStart = testStart - embargo; // samples in [embargoStart, testStart) are embargoed

            // Candidate training universe (past-only).
            int trainLo;
            int trainHiExclusive = testStart;
            if (expanding)
            {
                trainLo = 0;
            }
            else
            {
                // Sliding: a window of foldSize eligible samples ending just before the test fold.
                trainLo = Math.Max(0, testStart - foldSize);
            }

            var trainIndices = new List<int>();
            for (int i = trainLo; i < trainHiExclusive; i++)
            {
                // Embargo: drop the buffer immediately preceding the test fold.
                if (i >= embargoStart && i < testStart)
                {
                    continue;
                }

                // Purge: drop training samples whose label window [i, i + labelHorizon - 1] overlaps the
                // test fold's index+label span [testStart, testLabelEnd]. Since i < testStart here, overlap
                // occurs exactly when the training label reaches at or past testStart.
                int iLabelEnd = i + labelHorizon - 1;
                bool overlaps = !(iLabelEnd < testStart || i > testLabelEnd);
                if (overlaps)
                {
                    continue;
                }

                trainIndices.Add(i);
            }

            folds.Add(new Fold(trainIndices, testIndices));
        }

        return folds;
    }
}
