using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio;

/// <summary>
/// Whole-tensor helpers shared by the audio models (ordering, windows, framing), written with engine
/// operations so callers do not loop over elements.
/// </summary>
internal static class AudioEngineOps
{
    /// <summary>True for the element types that can hold NaN and infinities.</summary>
    internal static bool IsFloating<T>()
        => typeof(T) == typeof(double) || typeof(T) == typeof(float) || typeof(T) == typeof(Half);

    /// <summary>
    /// -infinity for floating element types, otherwise the type's minimum value (decimal and the
    /// integer types have no infinity, and converting double -infinity to them throws).
    /// </summary>
    internal static T NegativeInfinityOrMin<T>()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return IsFloating<T>() ? numOps.FromDouble(double.NegativeInfinity) : numOps.MinValue;
    }

    /// <summary>NaN for floating element types, otherwise zero (only reachable where no NaN exists).</summary>
    internal static T NaNOrZero<T>()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return IsFloating<T>() ? numOps.FromDouble(double.NaN) : numOps.Zero;
    }

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
    /// rank after everything), and the key <c>rank * n + index</c> is sorted ascending. The rank comes
    /// from one sort plus a binary search, so time is O(n log n) and memory O(n).
    /// </remarks>
    internal static int[] StableDescendingOrder<T>(IEngine engine, Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0) return Array.Empty<int>();
        if (n == 1) return new[] { 0 };
        var rank = GreaterCountTensor(engine, values);
        var key = engine.TensorAdd(
            engine.TensorMultiplyScalar(rank.Cast<long>(), (long)n),
            engine.TensorArange<int>(0, n).Cast<long>());
        return engine.TensorArgsort(key, 0, descending: false).ToArray();
    }

    /// <summary>
    /// For each element of <paramref name="values"/> (row-major), the number of elements strictly
    /// greater than it; a NaN element gets <c>n</c> (it ranks after everything). O(n log n) time, O(n)
    /// memory.
    /// </summary>
    internal static int[] GreaterCounts<T>(IEngine engine, Tensor<T> values)
        => values.Length == 0 ? Array.Empty<int>() : GreaterCountTensor(engine, values).ToArray();

    private static Tensor<int> GreaterCountTensor<T>(IEngine engine, Tensor<T> values)
    {
        int n = values.Length;
        var v = Flatten(engine, values);
        var isNan = engine.TensorIsNan(v);
        bool anyNan = engine.TensorCountNonzero(isNan) > 0;
        // NaN compares false with everything, so for ranking it behaves like a value no number exceeds
        // from below: substitute -inf, which every number is >= (only floating types can hold NaN).
        var comparable = anyNan
            ? engine.TensorWhere(isNan, Tensor<T>.CreateDefault(new[] { n }, NegativeInfinityOrMin<T>()), v)
            : v;
        var sorted = engine.TensorSort(comparable, 0, descending: false).Values;
        // Elements strictly greater than v[i] = n - (elements <= v[i]).
        var atMost = engine.TensorSearchSorted(sorted, comparable, right: true);
        var rank = engine.TensorSubtract(Tensor<int>.CreateDefault(new[] { n }, n), atMost);
        return anyNan ? engine.TensorWhere(isNan, Tensor<int>.CreateDefault(new[] { n }, n), rank) : rank;
    }

    /// <summary>
    /// Index of the first maximum of <paramref name="values"/> (row-major) under the sequential scan
    /// <c>best = v[0]; if (v[i] &gt; best) best = v[i]</c>: later ties never replace, NaN never wins
    /// against a number, and a leading NaN is kept. Linear time and memory.
    /// </summary>
    internal static int FirstArgMaxSequential<T>(IEngine engine, Tensor<T> values)
    {
        if (values.Length == 0) throw new ArgumentException("Cannot take the maximum of an empty tensor.", nameof(values));
        var numOps = MathHelper.GetNumericOperations<T>();
        var v = Flatten(engine, values);
        int n = v.Length;
        T first = v[0];
        if (numOps.IsNaN(first)) return 0;
        var isNan = engine.TensorIsNan(v);
        var comparable = engine.TensorCountNonzero(isNan) > 0
            ? engine.TensorWhere(isNan, Tensor<T>.CreateDefault(new[] { n }, NegativeInfinityOrMin<T>()), v)
            : v;
        // v[0] is a number, so the maximum is a number and its first occurrence is the scan's answer.
        T best = engine.TensorMaxValue(comparable);
        var hit = engine.TensorEquals(comparable, best).Cast<int>();
        var candidates = engine.TensorWhere(hit, engine.TensorArange<int>(0, n), Tensor<int>.CreateDefault(new[] { n }, n));
        return engine.TensorMinValue(candidates);
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

    /// <summary>
    /// Number of adjacent sample pairs (row-major) where one is <c>&gt; 0</c> and the other
    /// <c>&lt; 0</c>; zeros and NaN never count.
    /// </summary>
    internal static int StrictSignChangeCount<T>(IEngine engine, Tensor<T> audio)
    {
        int n = audio.Length;
        if (n < 2) return 0;
        var numOps = MathHelper.GetNumericOperations<T>();
        var flat = Flatten(engine, audio);
        var positive = engine.TensorGreaterThan(flat, numOps.Zero);
        var negative = engine.TensorLessThan(flat, numOps.Zero);
        var prevPos = engine.TensorSlice(positive, new[] { 0 }, new[] { n - 1 });
        var prevNeg = engine.TensorSlice(negative, new[] { 0 }, new[] { n - 1 });
        var nextPos = engine.TensorSlice(positive, new[] { 1 }, new[] { n - 1 });
        var nextNeg = engine.TensorSlice(negative, new[] { 1 }, new[] { n - 1 });
        var crossings = engine.TensorAdd(engine.TensorMultiply(prevPos, nextNeg), engine.TensorMultiply(prevNeg, nextPos));
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

    /// <summary>Truncates or zero-pads a rank-1 tensor to exactly <paramref name="length"/> elements.</summary>
    internal static Tensor<T> FitLength<T>(IEngine engine, Tensor<T> x, int length)
    {
        var flat = Flatten(engine, x);
        if (flat.Length == length) return flat;
        if (flat.Length > length) return engine.TensorSlice(flat, new[] { 0 }, new[] { length });
        if (flat.Length == 0) return new Tensor<T>(new[] { length });
        return engine.Concat(new[] { flat, new Tensor<T>(new[] { length - flat.Length }) }, 0);
    }

    /// <summary>
    /// Magnitude and phase (<c>atan2(im, re)</c>) of the first <c>fftSize / 2 + 1</c> bins of the
    /// complex FFT of <paramref name="frame"/>, truncated or zero-padded to <paramref name="fftSize"/>
    /// samples (a power of two).
    /// </summary>
    internal static (Tensor<T> Magnitude, Tensor<T> Phase) HalfSpectrum<T>(IEngine engine, Tensor<T> frame, int fftSize)
    {
        var spectrum = engine.NativeComplexFFT(FitLength(engine, frame, fftSize));
        var magnitude = engine.NativeMagnitudeAndPhase(spectrum, out var phase);
        int bins = fftSize / 2 + 1;
        return (engine.TensorSlice(magnitude, new[] { 0 }, new[] { bins }), engine.TensorSlice(phase, new[] { 0 }, new[] { bins }));
    }

    /// <summary>
    /// Real part of the inverse FFT (1/N normalised) of the length-<paramref name="fftSize"/> spectrum
    /// whose bins <c>k &lt; m</c> are <c>polar(magnitude[k], phase[k])</c> and whose bins
    /// <c>fftSize - k</c> for <c>1 &lt;= k &lt;= m - 2</c> are their conjugates (m = magnitude length).
    /// </summary>
    internal static Tensor<T> InverseHalfSpectrum<T>(IEngine engine, Tensor<T> magnitude, Tensor<T> phase, int fftSize)
    {
        var mag = Flatten(engine, magnitude);
        var ph = Flatten(engine, phase);
        int m = mag.Length;
        int mirrored = Math.Max(0, m - 2);
        int gap = fftSize - m - mirrored;
        var magParts = new List<Tensor<T>> { mag };
        var phaseParts = new List<Tensor<T>> { ph };
        if (gap > 0)
        {
            magParts.Add(new Tensor<T>(new[] { gap }));
            phaseParts.Add(new Tensor<T>(new[] { gap }));
        }
        if (mirrored > 0)
        {
            // Conjugate symmetry X[N - k] = conj(X[k]): same magnitude, negated phase, reversed order.
            magParts.Add(engine.TensorFlip(engine.TensorSlice(mag, new[] { 1 }, new[] { mirrored }), new[] { 0 }));
            phaseParts.Add(engine.TensorNegate(engine.TensorFlip(engine.TensorSlice(ph, new[] { 1 }, new[] { mirrored }), new[] { 0 })));
        }
        var fullMag = magParts.Count == 1 ? mag : engine.Concat(magParts, 0);
        var fullPhase = phaseParts.Count == 1 ? ph : engine.Concat(phaseParts, 0);
        return engine.NativeComplexIFFTReal(engine.NativeComplexFromPolar(fullMag, fullPhase));
    }

    /// <summary>
    /// Overlap-adds rank-1 frames into a signal of <paramref name="length"/> samples: frame f lands at
    /// <c>starts[f]</c>, and samples falling at or past the end are dropped. One scatter-add.
    /// </summary>
    internal static Tensor<T> OverlapAdd<T>(IEngine engine, IReadOnlyList<Tensor<T>> frames, IReadOnlyList<int> starts, int length)
    {
        var values = new List<Tensor<T>>(frames.Count);
        var positions = new List<Tensor<int>>(frames.Count);
        for (int f = 0; f < frames.Count; f++)
        {
            var frame = Flatten(engine, frames[f]);
            int keep = Math.Min(frame.Length, length - starts[f]);
            if (keep <= 0) continue;
            values.Add(keep == frame.Length ? frame : engine.TensorSlice(frame, new[] { 0 }, new[] { keep }));
            positions.Add(engine.TensorArange<int>(starts[f], starts[f] + keep));
        }
        if (values.Count == 0) return new Tensor<T>(new[] { length });
        return engine.ScatterAdd(engine.Concat(values, 0), engine.Concat(positions, 0), 0, length);
    }

    /// <summary>
    /// The symmetric Hann window <c>0.5 * (1 - cos(2 pi i / (size - 1)))</c>; a one-sample window
    /// evaluates 0/0 like the closed form and is NaN.
    /// </summary>
    internal static Tensor<T> SymmetricHannWindow<T>(IEngine engine, int size)
    {
        if (size == 1) return Tensor<T>.CreateDefault(new[] { 1 }, MathHelper.GetNumericOperations<T>().FromDouble(double.NaN));
        if (size < 0) throw new OverflowException("Window size must be non-negative.");
        if (size == 0) return new Tensor<T>(new[] { 0 });
        return engine.TensorHannWindow<T>(size, periodic: false);
    }

    /// <summary>
    /// <paramref name="x"/> (same shape) with its first <paramref name="count"/> row-major elements
    /// replaced by <paramref name="map"/> applied to them; the remaining elements are unchanged.
    /// </summary>
    internal static Tensor<T> MapPrefix<T>(IEngine engine, Tensor<T> x, int count, Func<Tensor<T>, Tensor<T>> map)
    {
        if (count <= 0) return x;
        var flat = Flatten(engine, x);
        var head = map(engine.TensorSlice(flat, new[] { 0 }, new[] { count }));
        var joined = count >= flat.Length
            ? head
            : engine.Concat(new[] { head, engine.TensorSlice(flat, new[] { count }, new[] { flat.Length - count }) }, 0);
        return engine.Reshape(joined, x.Shape.ToArray());
    }

    /// <summary>
    /// Spectral subtraction over the overlapping prefix: <c>x[i] - noise[i]</c> where that is greater
    /// than zero, else zero (NaN maps to zero), for <c>i &lt; min(|x|, |noise|)</c>.
    /// </summary>
    internal static Tensor<T> SubtractNoiseClampPrefix<T>(IEngine engine, Tensor<T> x, Tensor<T> noise)
    {
        int count = Math.Min(x.Length, noise.Length);
        return MapPrefix(engine, x, count, head =>
        {
            var subtracted = engine.TensorSubtract(head, FitLength(engine, noise, count));
            return engine.TensorWhere(engine.TensorGreaterThan(subtracted, MathHelper.GetNumericOperations<T>().Zero), subtracted, new Tensor<T>(new[] { count }));
        });
    }

    /// <summary>
    /// <c>max(0, x[i] - noiseScale * noise[i])</c> (NaN propagates, as <c>Math.Max</c> does) for
    /// <c>i &lt; min(|x|, |noise|)</c>; the remaining elements are unchanged.
    /// </summary>
    internal static Tensor<T> SubtractScaledNoiseFloorPrefix<T>(IEngine engine, Tensor<T> x, Tensor<T> noise, double noiseScale)
    {
        int count = Math.Min(x.Length, noise.Length);
        var numOps = MathHelper.GetNumericOperations<T>();
        return MapPrefix(engine, x, count, head =>
        {
            var value = engine.TensorSubtract(head, engine.TensorMultiplyScalar(FitLength(engine, noise, count), numOps.FromDouble(noiseScale)));
            return engine.TensorWhere(engine.TensorLessThan(value, numOps.Zero), new Tensor<T>(new[] { count }), value);
        });
    }

    /// <summary>
    /// <c>s * result[i] + inv * other[i]</c> for <c>i &lt; min(|result|, |other|)</c>; the remaining
    /// elements of <paramref name="result"/> are unchanged.
    /// </summary>
    internal static Tensor<T> BlendPrefix<T>(IEngine engine, Tensor<T> result, Tensor<T> other, T s, T inv)
    {
        int count = Math.Min(result.Length, other.Length);
        return MapPrefix(engine, result, count, head =>
            engine.TensorAdd(engine.TensorMultiplyScalar(head, s), engine.TensorMultiplyScalar(FitLength(engine, other, count), inv)));
    }

    /// <summary>
    /// A tensor shaped like <paramref name="a"/> holding <c>a[i] * b[i]</c> for
    /// <c>i &lt; min(|a|, |b|)</c> and zero after.
    /// </summary>
    internal static Tensor<T> MultiplyPrefix<T>(IEngine engine, Tensor<T> a, Tensor<T> b)
    {
        int count = Math.Min(a.Length, b.Length);
        if (count == 0) return new Tensor<T>(a.Shape.ToArray());
        var product = engine.TensorMultiply(FitLength(engine, a, count), FitLength(engine, b, count));
        return engine.Reshape(FitLength(engine, product, a.Length), a.Shape.ToArray());
    }

    /// <summary><c>sign(v) * |v|^exponent</c> element-wise (a negative value keeps its sign).</summary>
    internal static Tensor<T> SignedPower<T>(IEngine engine, Tensor<T> values, double exponent)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var magnitude = engine.TensorPower(engine.TensorAbs(values), numOps.FromDouble(exponent));
        return engine.TensorWhere(engine.TensorLessThan(values, numOps.Zero), engine.TensorNegate(magnitude), magnitude);
    }

    /// <summary>
    /// <paramref name="embedding"/> divided by its L2 norm when that norm is positive (otherwise
    /// unchanged, including a NaN norm), as a fresh detached array.
    /// </summary>
    internal static T[] L2NormalizedData<T>(IEngine engine, Tensor<T> embedding)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var numOps = MathHelper.GetNumericOperations<T>();
        if (embedding.Length == 0) return new T[0];
        T norm = numOps.Sqrt(engine.TensorSum(engine.TensorSquare(embedding)));
        return numOps.GreaterThan(norm, numOps.Zero)
            ? engine.TensorDivideScalar(embedding, norm).ToArray()
            : embedding.ToArray();
    }

    /// <summary>Most gathered values one block of <see cref="SlidingCosineScores"/> materializes.</summary>
    private const long AlignedBlockBudget = 1L << 22;

    /// <summary>
    /// Cosine similarity of the query embedding frames against the reference frames starting at
    /// each offset <c>rStart = 0 .. refFrames - min(queryFrames, minMatchLength)</c>, over the
    /// <c>min(queryFrames, refFrames - rStart)</c> overlapping frames; 0 where
    /// <c>sqrt(|q|^2) * sqrt(|r|^2)</c> is not above 1e-8. Frames are <paramref name="embDim"/>
    /// consecutive values; a trailing partial frame is ignored.
    /// </summary>
    internal static double[] SlidingCosineScores<T>(IEngine engine, T[] query, T[] reference, int embDim, int minMatchLength)
    {
        int frameDim = Math.Max(1, embDim);
        int queryFrames = query.Length / frameDim;
        int refFrames = reference.Length / frameDim;
        int starts = refFrames - Math.Min(queryFrames, minMatchLength) + 1;
        if (queryFrames <= 0 || refFrames <= 0 || starts <= 0) return new double[0];
        if (embDim <= 0) return new double[starts];

        // Only the aligned pairs (start, f) -> reference frame start + f are needed, so the frame
        // products are gathered per block of starts instead of forming the full Gram matrix; each
        // block holds at most AlignedBlockBudget values (or one start when a single start is larger).
        var numOps = MathHelper.GetNumericOperations<T>();
        var q = engine.Reshape(engine.TensorSlice(new Tensor<T>(query, new[] { query.Length }), new[] { 0 }, new[] { queryFrames * embDim }), new[] { queryFrames, embDim });
        var r = engine.Reshape(engine.TensorSlice(new Tensor<T>(reference, new[] { reference.Length }), new[] { 0 }, new[] { refFrames * embDim }), new[] { refFrames, embDim });
        var qNorm = engine.ReduceSum(engine.TensorSquare(q), new[] { 1 }, keepDims: false);
        var rNorm = engine.ReduceSum(engine.TensorSquare(r), new[] { 1 }, keepDims: false);
        T cosineFloor = numOps.FromDouble(1e-8);
        long perStart = (long)queryFrames * embDim;
        int block = (int)Math.Max(1L, Math.Min(starts, AlignedBlockBudget / perStart));

        var result = new double[starts];
        for (int blockStart = 0; blockStart < starts; blockStart += block)
        {
            int count = Math.Min(block, starts - blockStart);
            var queryIndex = new int[count * queryFrames];
            var refIndex = new int[count * queryFrames];
            var valid = new int[count * queryFrames];
            for (int s = 0; s < count; s++)
            {
                for (int f = 0; f < queryFrames; f++)
                {
                    int k = s * queryFrames + f;
                    int j = blockStart + s + f;
                    queryIndex[k] = f;
                    if (j < refFrames)
                    {
                        refIndex[k] = j;
                        valid[k] = 1;
                    }
                }
            }

            var pairShape = new[] { count, queryFrames };
            var mask = new Tensor<int>(valid, pairShape).Cast<T>();
            var zeros = new Tensor<T>(pairShape);
            var qIdx = new Tensor<int>(queryIndex, new[] { queryIndex.Length });
            var rIdx = new Tensor<int>(refIndex, new[] { refIndex.Length });
            Tensor<T> Gather(Tensor<T> source, Tensor<int> index) => engine.TensorWhere(
                mask, engine.Reshape(engine.TensorIndexSelect(source, index, 0), pairShape), zeros);

            // Row products q_f . r_(start+f), zeroed where the reference frame does not exist
            var frameDots = engine.ReduceSum(
                engine.TensorMultiply(engine.TensorIndexSelect(q, qIdx, 0), engine.TensorIndexSelect(r, rIdx, 0)),
                new[] { 1 }, keepDims: false);
            var sim = engine.ReduceSum(engine.TensorWhere(mask, engine.Reshape(frameDots, pairShape), zeros), new[] { 1 }, keepDims: false);
            var normQ = engine.ReduceSum(Gather(qNorm, qIdx), new[] { 1 }, keepDims: false);
            var normR = engine.ReduceSum(Gather(rNorm, rIdx), new[] { 1 }, keepDims: false);
            var denom = engine.TensorMultiply(engine.TensorSqrt(normQ), engine.TensorSqrt(normR));
            // Invalid denominators divide by one (then score 0), so decimal never divides by zero
            var validDenom = engine.TensorGreaterThan(denom, cosineFloor);
            var safeDenom = engine.TensorWhere(validDenom, denom, Tensor<T>.CreateDefault(new[] { count }, numOps.One));
            var cosine = engine.TensorWhere(validDenom, engine.TensorDivide(sim, safeDenom), new Tensor<T>(new[] { count }));
            Array.Copy(cosine.Cast<double>().ToArray(), 0, result, blockStart, count);
        }

        return result;
    }

    /// <summary>Block length of one level of <see cref="Mean"/>.</summary>
    private const int MeanBlock = 64;

    /// <summary>
    /// Arithmetic mean of all elements without ever forming the element count or the full sum in
    /// <typeparamref name="T"/>: each level averages blocks of <see cref="MeanBlock"/> values (each
    /// value pre-scaled by <c>1 / MeanBlock</c>), and the zero padding of a partial last block is
    /// corrected by the ratio <c>blocks * MeanBlock / n</c>, which lies in [1, 2). Every intermediate
    /// stays within the range of the inputs, so a narrow type such as Half averages more than
    /// 65,504 samples correctly. NaN propagates; an empty tensor has no mean.
    /// </summary>
    internal static T Mean<T>(IEngine engine, Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0) throw new ArgumentException("Cannot average an empty tensor.", nameof(values));
        var numOps = MathHelper.GetNumericOperations<T>();
        var v = Flatten(engine, values);
        if (n <= MeanBlock)
            return engine.TensorSum(engine.TensorMultiplyScalar(v, numOps.FromDouble(1.0 / n)));

        int blocks = (n + MeanBlock - 1) / MeanBlock;
        var padded = FitLength(engine, v, blocks * MeanBlock);
        var blockMeans = engine.ReduceSum(
            engine.TensorMultiplyScalar(engine.Reshape(padded, new[] { blocks, MeanBlock }), numOps.FromDouble(1.0 / MeanBlock)),
            new[] { 1 }, keepDims: false);
        T meanOfBlocks = Mean(engine, blockMeans);
        return numOps.Multiply(meanOfBlocks, numOps.FromDouble((double)blocks * MeanBlock / n));
    }

    /// <summary>
    /// The mean of <paramref name="values"/> as a one-element tensor, by the same blocking as <see cref="Mean"/>, so
    /// no sample count or sum has to fit in <typeparamref name="T"/>. Nothing is read back to the host: a compiled
    /// training plan recomputes it from each step's samples instead of freezing the traced step's value.
    /// </summary>
    internal static Tensor<T> MeanTensor<T>(IEngine engine, Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0) throw new ArgumentException("Cannot average an empty tensor.", nameof(values));
        var numOps = MathHelper.GetNumericOperations<T>();
        var v = Flatten(engine, values);
        if (n <= MeanBlock)
            return engine.ReduceSum(engine.TensorMultiplyScalar(v, numOps.FromDouble(1.0 / n)), new[] { 0 }, keepDims: true);

        int blocks = (n + MeanBlock - 1) / MeanBlock;
        var padded = FitLength(engine, v, blocks * MeanBlock);
        var blockMeans = engine.ReduceSum(
            engine.TensorMultiplyScalar(engine.Reshape(padded, new[] { blocks, MeanBlock }), numOps.FromDouble(1.0 / MeanBlock)),
            new[] { 1 }, keepDims: false);
        return engine.TensorMultiplyScalar(MeanTensor(engine, blockMeans), numOps.FromDouble((double)blocks * MeanBlock / n));
    }

    /// <summary>
    /// <c>(x - mean) / max(std, floor)</c> with the population standard deviation, as a fresh tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The samples are first divided by their largest magnitude <c>s</c>, so every intermediate (the
    /// unit samples, their mean through <see cref="MeanTensor"/>, the deviations, at most 2 in magnitude,
    /// and their squares) stays small; neither the sample count, a sum, nor a deviation of the raw
    /// samples has to fit in <typeparamref name="T"/>. Since the standardized value is scale-free,
    /// <c>(x - mean) / std = (u - mean_u) / std_u</c>; the floor is applied as <c>std = s * std_u</c>.
    /// Any NaN sample makes every result NaN, as the direct formula does; an all-zero input gives
    /// zeros and an empty input an empty result.
    /// </para>
    /// <para>
    /// Every statistic stays a tensor and every branch is a selection, so the whole standardization is
    /// graph ops: a compiled training plan recomputes the statistics from each step's samples. Reading
    /// them back to the host would freeze the traced step's scale, mean and spread into the plan.
    /// </para>
    /// </remarks>
    internal static Tensor<T> Standardize<T>(IEngine engine, Tensor<T> x, double floor)
    {
        if (x.Length == 0) return new Tensor<T>(x.Shape.ToArray());
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        var flat = Flatten(engine, x);
        int n = flat.Length;

        // The largest magnitude, or 1 for an all-zero input, whose deviations are then all 0. A NaN
        // sample makes the unit samples' mean NaN and so every deviation NaN, whatever the maximum is.
        var magnitude = engine.ReduceMax(engine.TensorAbs(flat), new[] { 0 }, keepDims: true);
        var one = Tensor<T>.CreateDefault(new[] { 1 }, numOps.One);
        var scale = engine.TensorWhere(engine.TensorEquals(magnitude, numOps.Zero), one, magnitude);

        var unit = engine.TensorDivide(flat, scale);
        var deviation = engine.TensorSubtract(unit, MeanTensor(engine, unit));
        var unitStd = engine.TensorSqrt(MeanTensor(engine, engine.TensorSquare(deviation)));

        // std = scale * unitStd; below the floor the divisor is the floor itself, i.e. the unit
        // deviations are multiplied by scale / floor. Zero deviations stay 0 in either branch, even
        // where the other branch's factor is not representable in T.
        T floorValue = numOps.FromDouble(floor);
        var belowFloor = engine.TensorBroadcastTo(
            engine.TensorLessThan(engine.TensorMultiply(scale, unitStd), floorValue), new[] { n });
        var standardized = engine.TensorWhere(belowFloor,
            engine.TensorMultiply(deviation, engine.TensorDivideScalar(scale, floorValue)),
            engine.TensorDivide(deviation, unitStd));
        var result = engine.TensorWhere(engine.TensorEquals(deviation, numOps.Zero), new Tensor<T>(new[] { n }), standardized);
        return engine.Reshape(result, shape);
    }

    /// <summary>
    /// The value of the running pairwise average <c>c = v[0]; c = (c + v[i]) / 2</c> over
    /// <paramref name="values"/>, i.e. <c>v[0] / 2^m + sum_{i&gt;=1} v[i] / 2^(m - i + 1)</c> for
    /// <c>m = |v| - 1</c>.
    /// </summary>
    internal static T RunningPairAverage<T>(IEngine engine, T[] values)
    {
        if (values.Length == 0) throw new ArgumentException("At least one value is required.", nameof(values));
        if (values.Length == 1) return values[0];
        int m = values.Length - 1;
        var exponents = new int[values.Length];
        exponents[0] = m;
        for (int i = 1; i < values.Length; i++) exponents[i] = m - i + 1;
        var numOps = MathHelper.GetNumericOperations<T>();
        var weights = engine.TensorPower(
            Tensor<T>.CreateDefault(new[] { values.Length }, numOps.FromDouble(0.5)),
            new Tensor<int>(exponents, new[] { values.Length }).Cast<T>());
        return engine.TensorSum(engine.TensorMultiply(new Tensor<T>(values, new[] { values.Length }), weights));
    }

    /// <summary><c>exp(x - max) / sum(exp(x - max))</c> over <paramref name="logits"/>.</summary>
    internal static T[] Softmax<T>(IEngine engine, T[] logits)
    {
        if (logits.Length == 0) throw new InvalidOperationException("Sequence contains no elements");
        var x = new Tensor<T>(logits, new[] { logits.Length });
        var shifted = engine.TensorExp(engine.TensorSubtractScalar(x, engine.TensorMaxValue(x)));
        return engine.TensorDivideScalar(shifted, engine.TensorSum(shifted)).ToArray();
    }

    /// <summary>
    /// The entries of <paramref name="scores"/> ordered by descending value (stable: equal values keep
    /// their order, NaN last), as a stable <c>OrderByDescending</c> over the converted values gives.
    /// </summary>
    internal static List<KeyValuePair<TKey, T>> OrderByDescendingValue<TKey, T>(IEngine engine, IEnumerable<KeyValuePair<TKey, T>> scores)
    {
        var entries = scores.ToList();
        if (entries.Count == 0) return entries;
        var values = new Tensor<T>(entries.Select(e => e.Value).ToArray(), new[] { entries.Count });
        return StableDescendingOrder(engine, values).Select(i => entries[i]).ToList();
    }

    /// <summary>
    /// <c>sigmoid(x[offset + i])</c> for <c>i &lt; count</c> where that element exists, zero after (a
    /// fresh, detached tensor).
    /// </summary>
    internal static Tensor<T> SigmoidSegment<T>(IEngine engine, Tensor<T> x, int offset, int count)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int available = Math.Max(0, Math.Min(count, x.Length - offset));
        if (available == 0) return new Tensor<T>(new[] { count });
        var segment = engine.TensorSlice(Flatten(engine, x), new[] { offset }, new[] { available });
        return FitLength(engine, engine.Sigmoid(segment), count);
    }

    /// <summary>
    /// Indices <c>i &lt; count</c> (ascending) where <c>a[i] &gt;= aThreshold</c> and
    /// <c>b[i] &gt;= bThreshold</c>; NaN never qualifies.
    /// </summary>
    internal static int[] IndicesBothAtLeast<T>(IEngine engine, Tensor<T> a, double aThreshold, Tensor<T> b, double bThreshold, int count)
    {
        if (count <= 0) return new int[0];
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = new[] { count };
        var aHead = FitLength(engine, a, count);
        var bHead = FitLength(engine, b, count);
        var both = engine.TensorMultiply(
            engine.TensorGreaterEqual(aHead, Tensor<T>.CreateDefault(shape, numOps.FromDouble(aThreshold))),
            engine.TensorGreaterEqual(bHead, Tensor<T>.CreateDefault(shape, numOps.FromDouble(bThreshold))));
        return engine.TensorNonzero(both).ToArray();
    }

    /// <summary>
    /// The scan <c>best = -inf; for i &lt; count: if (v[i] &gt; best) { best = v[i]; bin = i; }</c>
    /// (first strict maximum, NaN never wins; bin 0 with value -inf when nothing beats -inf) and
    /// the softmax confidence <c>1 / sum_i exp(v[i] - best)</c> over the same prefix.
    /// </summary>
    internal static (int Bin, T Confidence) PeakBinWithSoftmaxConfidence<T>(IEngine engine, Tensor<T> logits, int count)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = Math.Max(0, Math.Min(count, logits.Length));
        if (n == 0) return (0, numOps.Divide(numOps.One, numOps.Zero));
        var head = FitLength(engine, logits, n);
        var candidates = engine.Concat(new[] { Tensor<T>.CreateDefault(new[] { 1 }, NegativeInfinityOrMin<T>()), head }, 0);
        int best = FirstArgMaxSequential(engine, candidates);
        int bin = best > 0 ? best - 1 : 0;
        T bestValue = best > 0 ? Head(engine, engine.TensorSlice(head, new[] { bin }, new[] { 1 }), 1)[0] : NegativeInfinityOrMin<T>();
        T sumExp = engine.TensorSum(engine.TensorExp(engine.TensorSubtractScalar(head, bestValue)));
        return (bin, numOps.Divide(numOps.One, sumExp));
    }

    /// <summary>
    /// For each row of a <c>[rows, cols]</c> tensor, the scan <c>best = -inf; for c: if (v[c] &gt; best)
    /// { best = v[c]; arg = c; }</c>: the first strict maximum, NaN never wins, 0 when nothing beats -inf.
    /// </summary>
    internal static int[] RowFirstArgMax<T>(IEngine engine, Tensor<T> matrix)
    {
        int rows = matrix.Shape[0];
        int cols = matrix.Shape[1];
        if (rows == 0) return new int[0];
        if (cols == 0) return new int[rows];
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = new[] { rows, cols };
        var ranked = engine.TensorWhere(
            engine.TensorIsNan(matrix), Tensor<T>.CreateDefault(shape, NegativeInfinityOrMin<T>()), matrix);
        var best = engine.TensorBroadcastTo(engine.ReduceMax(ranked, new[] { 1 }, keepDims: true), shape);
        var column = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, cols, 1), new[] { 1, cols }), shape);
        var firstBest = engine.TensorWhere(
            engine.TensorEquals(ranked, best), column, Tensor<T>.CreateDefault(shape, numOps.FromDouble(cols)));
        return engine.TensorAmin(firstBest, new[] { 1 }, keepDims: false).Cast<int>().ToArray();
    }

    /// <summary>
    /// <c>x[i] + other[i]</c> for <c>i &lt; min(|x|, |other|)</c> (row-major); the remaining elements of
    /// <paramref name="x"/> are unchanged.
    /// </summary>
    internal static Tensor<T> AddPrefix<T>(IEngine engine, Tensor<T> x, Tensor<T> other)
    {
        int count = Math.Min(x.Length, other.Length);
        return MapPrefix(engine, x, count, head => engine.TensorAdd(head, FitLength(engine, other, count)));
    }

    /// <summary>
    /// A fresh <c>[frames, bins]</c> tensor holding <c>source[f * stride + offset + b]</c> (row-major).
    /// An element is read when that index is below <c>|source|</c>, or, with
    /// <paramref name="rowZeroBound"/>, when <c>offset + b</c> is (a later row then reading past the end
    /// throws, as the indexer does); elements not read are zero.
    /// </summary>
    internal static Tensor<T> GatherFrameBins<T>(IEngine engine, Tensor<T> source, int frames, int bins, int stride, int offset, bool rowZeroBound = false)
    {
        var result = new Tensor<T>(new[] { frames, bins });
        int length = source.Length;
        if (frames == 0 || bins == 0 || length == 0) return result;
        int n = frames * bins;
        var index = new int[n];
        for (int f = 0; f < frames; f++)
        {
            for (int b = 0; b < bins; b++)
            {
                int mi = f * stride + offset + b;
                bool read = rowZeroBound ? offset + b < length : mi < length;
                if (read && (mi < 0 || mi >= length))
                    throw new ArgumentOutOfRangeException("flatIndex", "Flat index is out of range.");
                index[f * bins + b] = read ? mi : length;
            }
        }

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var padded = engine.Concat(new[] { Flatten(engine, source), new Tensor<T>(new[] { 1 }) }, 0);
        return engine.Reshape(engine.TensorIndexSelect(padded, new Tensor<int>(index, new[] { n }), 0), new[] { frames, bins });
    }

    /// <summary>
    /// A fresh tensor shaped like the <c>[frames, bins]</c> <paramref name="magnitude"/> holding
    /// <c>magnitude[f, b] * mask</c>, where mask is <c>masks[f * bins * stems + stem * bins + b]</c>
    /// (zero past the end), clamped to [0, 1] with NaN kept when <paramref name="clamp"/> is set.
    /// </summary>
    internal static Tensor<T> MaskedMagnitude<T>(IEngine engine, Tensor<T> magnitude, Tensor<T> masks, int stems, int stem, bool clamp)
    {
        int frames = magnitude.Shape[0];
        int bins = magnitude.Shape[1];
        var shape = magnitude.Shape.ToArray();
        if (frames == 0 || bins == 0) return new Tensor<T>(shape);
        var mask = GatherFrameBins(engine, masks, frames, bins, bins * stems, stem * bins);
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        if (clamp)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var frameBins = new[] { frames, bins };
            mask = engine.TensorWhere(engine.TensorGreaterThan(mask, numOps.One), Tensor<T>.CreateDefault(frameBins, numOps.One), mask);
            mask = engine.TensorWhere(engine.TensorLessThan(mask, numOps.Zero), new Tensor<T>(frameBins), mask);
        }
        var product = engine.TensorMultiply(engine.Reshape(magnitude, new[] { frames, bins }), mask);
        return engine.Reshape(product, shape);
    }
}
