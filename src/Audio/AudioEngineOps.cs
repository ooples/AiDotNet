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

        // Gram matrix G[f, j] = q_f . r_j and the per-frame squared norms
        var q = engine.Reshape(engine.TensorSlice(new Tensor<T>(query, new[] { query.Length }), new[] { 0 }, new[] { queryFrames * embDim }), new[] { queryFrames, embDim });
        var r = engine.Reshape(engine.TensorSlice(new Tensor<T>(reference, new[] { reference.Length }), new[] { 0 }, new[] { refFrames * embDim }), new[] { refFrames, embDim });
        var gram = Flatten(engine, engine.TensorMatMul(q, engine.TensorTranspose(r)));
        var qNorm = engine.ReduceSum(engine.TensorSquare(q), new[] { 1 }, keepDims: false);
        var rNorm = engine.ReduceSum(engine.TensorSquare(r), new[] { 1 }, keepDims: false);

        // Pair (rStart, f) covers reference frame rStart + f while that frame exists
        var gramIndex = new int[starts * queryFrames];
        var queryIndex = new int[starts * queryFrames];
        var refIndex = new int[starts * queryFrames];
        var valid = new int[starts * queryFrames];
        for (int start = 0; start < starts; start++)
        {
            for (int f = 0; f < queryFrames; f++)
            {
                int k = start * queryFrames + f;
                int j = start + f;
                queryIndex[k] = f;
                if (j < refFrames)
                {
                    gramIndex[k] = f * refFrames + j;
                    refIndex[k] = j;
                    valid[k] = 1;
                }
            }
        }

        var shape = new[] { starts, queryFrames };
        var mask = new Tensor<int>(valid, shape).Cast<T>();
        var zeros = new Tensor<T>(shape);
        Tensor<T> Gather(Tensor<T> source, int[] index) => engine.TensorWhere(
            mask, engine.Reshape(engine.TensorIndexSelect(source, new Tensor<int>(index, new[] { index.Length }), 0), shape), zeros);

        var sim = engine.ReduceSum(Gather(gram, gramIndex), new[] { 1 }, keepDims: false);
        var normQ = engine.ReduceSum(Gather(qNorm, queryIndex), new[] { 1 }, keepDims: false);
        var normR = engine.ReduceSum(Gather(rNorm, refIndex), new[] { 1 }, keepDims: false);
        var denom = engine.TensorMultiply(engine.TensorSqrt(normQ), engine.TensorSqrt(normR));
        var numOps = MathHelper.GetNumericOperations<T>();
        var cosine = engine.TensorWhere(
            engine.TensorGreaterThan(denom, numOps.FromDouble(1e-8)),
            engine.TensorDivide(sim, denom),
            new Tensor<T>(new[] { starts }));

        return cosine.Cast<double>().ToArray();
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
}
