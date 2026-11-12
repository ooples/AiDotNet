using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio;

/// <summary>
/// Whole-tensor helpers shared by the audio models (ordering, windows, framing), written with engine
/// operations so callers do not loop over elements.
/// </summary>
internal static class AudioEngineOps
{
    /// <summary>Row-major flatten to a rank-1 tensor.</summary>
    internal static Tensor<T> Flatten<T>(IEngine engine, Tensor<T> x)
        => x.Shape.Length == 1 ? x : engine.Reshape(x, new[] { x.Length });

    /// <summary>
    /// Indices of <paramref name="values"/> (row-major) ordered by descending value, ties in ascending
    /// index order and NaN last - the order a stable <c>OrderByDescending</c> over the values gives.
    /// </summary>
    /// <remarks>
    /// The engine's argsort is not documented as stable, so each element gets a distinct composite key
    /// <c>(rank, index)</c>: its rank is the number of elements strictly greater than it (NaN elements
    /// rank after everything), and the key <c>rank * n + index</c> is sorted ascending.
    /// </remarks>
    internal static int[] StableDescendingOrder<T>(IEngine engine, Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0) return Array.Empty<int>();
        if (n == 1) return new[] { 0 };
        var v = Flatten(engine, values);
        // greater[i, j] = 1 when v[j] > v[i] (false for any NaN operand).
        var others = engine.TensorTile(engine.Reshape(v, new[] { 1, n }), new[] { n, 1 });
        var self = engine.TensorTile(engine.Reshape(v, new[] { n, 1 }), new[] { 1, n });
        var rank = engine.ReduceSum(engine.TensorGreaterThan(others, self), new[] { 1 }, keepDims: false);
        var nanRank = engine.TensorWhere(engine.TensorIsNan(v),
            Tensor<T>.CreateDefault(new[] { n }, MathHelper.GetNumericOperations<T>().FromDouble(n)),
            new Tensor<T>(new[] { n }));
        var rankInt = engine.TensorAdd(rank, nanRank).Cast<int>();
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(rankInt, n), engine.TensorArange<int>(0, n));
        return engine.TensorArgsort(key, 0, descending: false).ToArray();
    }

    /// <summary>
    /// Index of the first maximum of <paramref name="values"/> (row-major) under the sequential scan
    /// <c>best = v[0]; if (v[i] &gt; best) best = v[i]</c>: later ties never replace, NaN never wins
    /// against a number, and a leading NaN is kept.
    /// </summary>
    internal static int FirstArgMaxSequential<T>(IEngine engine, Tensor<T> values)
    {
        if (values.Length == 0) throw new ArgumentException("Cannot take the maximum of an empty tensor.", nameof(values));
        var numOps = MathHelper.GetNumericOperations<T>();
        var v = Flatten(engine, values);
        T first = v[0];
        if (numOps.IsNaN(first)) return 0;
        return StableDescendingOrder(engine, v)[0];
    }

    /// <summary>
    /// Number of adjacent sample pairs (row-major) whose signs differ: one is <c>&gt;= 0</c> and the
    /// other <c>&lt; 0</c>. A NaN sample is neither, so pairs containing NaN never count.
    /// </summary>
    internal static int ZeroCrossingCount<T>(IEngine engine, Tensor<T> audio)
    {
        int n = audio.Length;
        if (n < 2) return 0;
        var numOps = MathHelper.GetNumericOperations<T>();
        var flat = Flatten(engine, audio);
        var nonNegative = engine.TensorGreaterEqual(flat, Tensor<T>.CreateDefault(new[] { n }, numOps.Zero));
        var negative = engine.TensorLessThan(flat, numOps.Zero);
        var prevNonNeg = engine.TensorSlice(nonNegative, new[] { 0 }, new[] { n - 1 });
        var prevNeg = engine.TensorSlice(negative, new[] { 0 }, new[] { n - 1 });
        var nextNonNeg = engine.TensorSlice(nonNegative, new[] { 1 }, new[] { n - 1 });
        var nextNeg = engine.TensorSlice(negative, new[] { 1 }, new[] { n - 1 });
        var crossings = engine.TensorAdd(engine.TensorMultiply(prevNonNeg, nextNeg), engine.TensorMultiply(prevNeg, nextNonNeg));
        return engine.TensorCountNonzero(crossings);
    }

    /// <summary>Sum of squared samples.</summary>
    internal static T SumOfSquares<T>(IEngine engine, Tensor<T> x) => engine.TensorSum(engine.TensorSquare(x));

    /// <summary>
    /// For a [frames, bins] spectrogram: <c>(sum_f f * sum_t mag[t, f], sum of all magnitudes)</c>.
    /// </summary>
    internal static (T Weighted, T Total) BinMoment<T>(IEngine engine, Tensor<T> spec)
    {
        var binTotals = engine.ReduceSum(spec, new[] { 0 }, keepDims: false);
        var weighted = engine.TensorSum(engine.TensorMultiply(binTotals, engine.TensorArange<T>(0, spec.Shape[1])));
        return (weighted, engine.TensorSum(binTotals));
    }

    /// <summary>
    /// Spectral centroid in bins of a [frames, bins] spectrogram over all frames, 0 when the total
    /// magnitude is not positive.
    /// </summary>
    internal static double SpectralCentroidBins<T>(IEngine engine, Tensor<T> spec)
    {
        if (spec.Shape[0] == 0 || spec.Shape[1] == 0) return 0;
        var numOps = MathHelper.GetNumericOperations<T>();
        var (weighted, total) = BinMoment(engine, spec);
        double totalSum = numOps.ToDouble(total);
        return totalSum > 0 ? numOps.ToDouble(weighted) / totalSum : 0;
    }

    /// <summary>
    /// Spectral flatness (geometric / arithmetic mean of the magnitudes floored at 1e-10) of a
    /// [frames, bins] spectrogram over all frames; 0 for an empty spectrogram or a non-positive mean.
    /// </summary>
    internal static double SpectralFlatness<T>(IEngine engine, Tensor<T> spec)
    {
        int count = spec.Shape[0] * spec.Shape[1];
        if (count == 0) return 0;
        var numOps = MathHelper.GetNumericOperations<T>();
        var mag = engine.TensorMax(spec, numOps.FromDouble(1e-10));
        double logSum = numOps.ToDouble(engine.TensorSum(engine.TensorLog(mag)));
        double sum = numOps.ToDouble(engine.TensorSum(mag));
        double geometricMean = Math.Exp(logSum / count);
        double arithmeticMean = sum / count;
        return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
    }

    /// <summary>
    /// Per-column population mean and standard deviation of a [rows, cols] tensor (divisor = rows).
    /// </summary>
    internal static (Tensor<T> Mean, Tensor<T> Std) ColumnMeanStd<T>(IEngine engine, Tensor<T> x)
    {
        var mean = engine.ReduceMean(x, new[] { 0 }, keepDims: true);
        var deviation = engine.TensorSubtract(x, engine.TensorBroadcastTo(mean, x.Shape.ToArray()));
        var variance = engine.ReduceMean(engine.TensorSquare(deviation), new[] { 0 }, keepDims: false);
        return (engine.Reshape(mean, new[] { x.Shape[1] }), engine.TensorSqrt(variance));
    }

    /// <summary>
    /// Per-column sum of absolute first differences down the rows, <c>sum_t |x[t, c] - x[t-1, c]|</c>,
    /// for a [rows, cols] tensor with at least two rows.
    /// </summary>
    internal static Tensor<T> ColumnAbsDiffSum<T>(IEngine engine, Tensor<T> x)
    {
        int rows = x.Shape[0], cols = x.Shape[1];
        var next = engine.TensorSlice(x, new[] { 1, 0 }, new[] { rows - 1, cols });
        var prev = engine.TensorSlice(x, new[] { 0, 0 }, new[] { rows - 1, cols });
        return engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(next, prev)), new[] { 0 }, keepDims: false);
    }

    /// <summary>
    /// Softmax over the first <c>min(classCount, logits.Length)</c> logits (row-major), returned as
    /// <paramref name="classCount"/> probabilities with zeros past the logits. When the exponentials do
    /// not sum to a positive number (no logits, or a NaN logit) every class gets <c>1 / classCount</c>.
    /// </summary>
    /// <remarks>
    /// The maximum is subtracted before exponentiating, which leaves every finite result unchanged
    /// (softmax is shift-invariant) and keeps large logits from overflowing.
    /// </remarks>
    internal static T[] SoftmaxOrUniform<T>(IEngine engine, Tensor<T> logits, int classCount)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[Math.Max(classCount, 0)];
        int n = Math.Min(classCount, logits.Length);
        if (n > 0)
        {
            var flat = Flatten(engine, logits);
            var head = n == flat.Length ? flat : engine.TensorSlice(flat, new[] { 0 }, new[] { n });
            var exp = engine.TensorExp(engine.TensorSubtractScalar(head, engine.TensorMaxValue(head)));
            T sum = engine.TensorSum(exp);
            if (numOps.ToDouble(sum) > 0)
            {
                engine.TensorDivideScalar(exp, sum).CopyTo(result.AsSpan(0, n));
                return result;
            }
        }
        T uniform = numOps.FromDouble(1.0 / classCount);
        result.AsSpan().Fill(uniform);
        return result;
    }

    /// <summary>The first <paramref name="count"/> elements (row-major) of <paramref name="x"/>.</summary>
    internal static T[] Head<T>(IEngine engine, Tensor<T> x, int count)
    {
        int n = Math.Min(Math.Max(count, 0), x.Length);
        var result = new T[n];
        if (n == 0) return result;
        var flat = Flatten(engine, x);
        (n == flat.Length ? flat : engine.TensorSlice(flat, new[] { 0 }, new[] { n })).CopyTo(result.AsSpan());
        return result;
    }
}
