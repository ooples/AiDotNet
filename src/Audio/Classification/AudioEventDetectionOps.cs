using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Whole-tensor building blocks shared by the windowed sound-event detectors (AST, AudioMAE, BEATs,
/// CLAP, EAT, PANNs, HTS-AT, FDY-SED, CRNN, AudioSep, AudioLDM classifier, rule-based detector).
/// </summary>
/// <remarks>
/// Every detector slices audio into overlapping windows, scores each window with a sigmoid, keeps the
/// classes at or above a threshold, merges overlapping detections and summarises them per class. These
/// helpers express each step with engine operations instead of per-element loops, with the same
/// results as the sequential code they replace.
/// </remarks>
internal static class AudioEventDetectionOps
{
    /// <summary>
    /// Element-wise logistic sigmoid, detached from the gradient tape (the detectors' post-processing
    /// always produced a fresh tensor that never joined the tape).
    /// </summary>
    internal static Tensor<T> SigmoidDetached<T>(IEngine engine, Tensor<T> logits)
    {
        using var noGrad = new NoGradScope<T>();
        return engine.Sigmoid(logits);
    }

    /// <summary>
    /// Returns <paramref name="count"/> scores: the sigmoid of the first
    /// <c>min(output.Length, count)</c> elements of <paramref name="output"/> (row-major), zero after.
    /// </summary>
    internal static T[] SigmoidScores<T>(IEngine engine, Tensor<T> output, int count)
    {
        var scores = new T[count];
        int n = Math.Min(output.Length, count);
        if (n <= 0) return scores;
        using var noGrad = new NoGradScope<T>();
        var flat = Flatten(engine, output);
        var head = n == flat.Length ? flat : engine.TensorSlice(flat, new[] { 0 }, new[] { n });
        engine.Sigmoid(head).CopyTo(scores.AsSpan(0, n));
        return scores;
    }

    /// <summary>Row-major flatten to a rank-1 tensor.</summary>
    internal static Tensor<T> Flatten<T>(IEngine engine, Tensor<T> x)
        => x.Shape.Length == 1 ? x : engine.Reshape(x, new[] { x.Length });

    /// <summary>
    /// Splits audio into windows of <paramref name="windowSamples"/> every <paramref name="hopSamples"/>
    /// samples. A trailing remainder of more than <paramref name="minTailSamples"/> samples becomes one more
    /// zero-padded window; audio shorter than one window (and not kept by that rule) becomes a single
    /// zero-padded window.
    /// </summary>
    internal static List<Tensor<T>> SplitIntoWindows<T>(IEngine engine, Tensor<T> audio, int windowSamples, int hopSamples, double minTailSamples)
    {
        var windows = new List<Tensor<T>>();
        var flat = Flatten(engine, audio);
        int lastStart = 0;
        for (int start = 0; start + windowSamples <= audio.Length; start += hopSamples)
        {
            windows.Add(windowSamples > 0
                ? engine.TensorSlice(flat, new[] { start }, new[] { windowSamples })
                : new Tensor<T>(new[] { windowSamples }));
            lastStart = start + hopSamples;
        }

        int remainingStart = windows.Count > 0 ? lastStart : 0;
        int remainingSamples = audio.Length - remainingStart;
        if (remainingSamples > minTailSamples)
        {
            int take = Math.Min(remainingSamples, windowSamples);
            windows.Add(ZeroPadTo(engine, engine.TensorSlice(flat, new[] { remainingStart }, new[] { take }), windowSamples));
        }
        else if (windows.Count == 0 && audio.Length > 0)
        {
            windows.Add(ZeroPadTo(engine, flat, windowSamples));
        }

        return windows;
    }

    /// <summary>Appends zeros to a rank-1 tensor so it has <paramref name="length"/> elements.</summary>
    internal static Tensor<T> ZeroPadTo<T>(IEngine engine, Tensor<T> x, int length)
    {
        if (x.Length >= length) return x.Length == length ? x.Clone() : engine.TensorSlice(x, new[] { 0 }, new[] { length });
        return engine.Concat(new[] { x, new Tensor<T>(new[] { length - x.Length }) }, 0);
    }

    /// <summary>
    /// Indices <c>i &lt; min(scores.Length, count)</c> with <c>scores[i] &gt;= threshold</c>, ascending.
    /// NaN scores never qualify.
    /// </summary>
    internal static int[] IndicesAtOrAbove<T>(IEngine engine, T[] scores, int count, T threshold)
    {
        int n = Math.Min(scores.Length, count);
        if (n <= 0) return Array.Empty<int>();
        var values = new Tensor<T>(n == scores.Length ? scores : scores.AsSpan(0, n).ToArray(), new[] { n });
        var mask = engine.TensorGreaterEqual(values, Tensor<T>.CreateDefault(new[] { n }, threshold));
        return engine.TensorNonzero(mask).ToArray();
    }

    /// <summary>Wraps one window's class scores as a rank-1 tensor of <paramref name="classCount"/> elements.</summary>
    internal static Tensor<T> ScoreRow<T>(T[] scores, int classCount)
    {
        var row = new T[classCount];
        Array.Copy(scores, row, Math.Min(classCount, scores.Length));
        return new Tensor<T>(row, new[] { classCount });
    }

    /// <summary>Stacks per-window score rows into a [windows, classCount] tensor.</summary>
    internal static Tensor<T> StackScoreRows<T>(IEngine engine, List<Tensor<T>> rows, int classCount)
        => rows.Count == 0 ? new Tensor<T>(new[] { 0, classCount }) : engine.TensorStack(rows.ToArray(), 0);

    /// <summary>
    /// Merges detections of the same class that overlap or are separated by at most
    /// <paramref name="gap"/> seconds. A merged event spans the first start to the furthest end and takes
    /// the confidence and peak time of its highest-confidence member (the latest one on ties), exactly
    /// as the pairwise sequential merge did. Output is ordered by start time (stable).
    /// </summary>
    internal static List<AudioEvent<T>> MergeEvents<T>(IEngine engine, List<AudioEvent<T>> events, double gap)
    {
        if (events.Count == 0) return events;
        var merged = new List<AudioEvent<T>>();
        foreach (var group in events.GroupBy(e => e.EventType))
        {
            var sorted = group.OrderBy(e => e.StartTime).ToList();
            var segmentOf = new int[sorted.Count];
            var segmentStarts = new List<int> { 0 };
            var segmentEnds = new List<double> { sorted[0].EndTime };
            for (int i = 1; i < sorted.Count; i++)
            {
                int s = segmentStarts.Count - 1;
                if (sorted[i].StartTime <= segmentEnds[s] + gap)
                {
                    segmentEnds[s] = Math.Max(segmentEnds[s], sorted[i].EndTime);
                }
                else
                {
                    segmentStarts.Add(i);
                    segmentEnds.Add(sorted[i].EndTime);
                }
                segmentOf[i] = segmentStarts.Count - 1;
            }

            int[] winners = LastArgMaxPerSegment(engine, sorted.Select(e => e.Confidence).ToArray(), segmentOf, segmentStarts.Count);
            for (int s = 0; s < segmentStarts.Count; s++)
            {
                int first = segmentStarts[s];
                int end = s + 1 < segmentStarts.Count ? segmentStarts[s + 1] : sorted.Count;
                if (end - first == 1)
                {
                    merged.Add(sorted[first]);
                    continue;
                }
                var winner = sorted[winners[s]];
                merged.Add(new AudioEvent<T>
                {
                    EventType = sorted[first].EventType,
                    StartTime = sorted[first].StartTime,
                    EndTime = segmentEnds[s],
                    Confidence = winner.Confidence,
                    PeakTime = winner.PeakTime
                });
            }
        }
        return merged.OrderBy(e => e.StartTime).ToList();
    }

    /// <summary>
    /// For each segment id in <c>[0, segmentCount)</c>, the index of the LAST element holding the segment's
    /// maximum value. Computed with scatter-max over the values and then over 1-based positions masked to
    /// the maxima, so it does not depend on any tie order inside the engine.
    /// </summary>
    internal static int[] LastArgMaxPerSegment<T>(IEngine engine, T[] values, int[] segmentOf, int segmentCount)
    {
        int n = values.Length;
        var v = new Tensor<T>(values, new[] { n });
        var seg = new Tensor<int>(segmentOf, new[] { n });
        var segMax = engine.ScatterMax(v, seg, out _, 0, segmentCount);
        var isMax = engine.TensorEquals(v, engine.TensorIndexSelect(segMax, seg, 0));
        var position1 = engine.TensorMultiply(isMax, engine.TensorArange<T>(1, n + 1));
        int[] last1 = engine.ScatterMax(position1, seg, out _, 0, segmentCount).Cast<int>().ToArray();
        var result = new int[segmentCount];
        for (int s = 0; s < segmentCount; s++) result[s] = last1[s] - 1;
        return result;
    }

    /// <summary>
    /// Per-class statistics: count, total duration, mean and maximum confidence.
    /// </summary>
    internal static Dictionary<string, EventStatistics<T>> ComputeEventStatistics<T>(IEngine engine, IReadOnlyList<AudioEvent<T>> events)
    {
        var stats = new Dictionary<string, EventStatistics<T>>();
        if (events.Count == 0) return stats;
        var groups = events.GroupBy(e => e.EventType).Select(g => g.ToList()).ToList();
        var confidences = groups.SelectMany(g => g.Select(e => e.Confidence)).ToArray();
        var groupOf = groups.SelectMany((g, gi) => Enumerable.Repeat(gi, g.Count)).ToArray();
        var c = new Tensor<T>(confidences, new[] { confidences.Length });
        var ids = new Tensor<int>(groupOf, new[] { groupOf.Length });
        T[] means = engine.ScatterMean(c, ids, out _, 0, groups.Count).ToArray();
        T[] maxima = engine.ScatterMax(c, ids, out _, 0, groups.Count).ToArray();
        for (int g = 0; g < groups.Count; g++)
        {
            var list = groups[g];
            stats[list[0].EventType] = new EventStatistics<T>
            {
                Count = list.Count,
                TotalDuration = list.Sum(e => e.Duration),
                AverageConfidence = means[g],
                MaxConfidence = maxima[g]
            };
        }
        return stats;
    }
}
