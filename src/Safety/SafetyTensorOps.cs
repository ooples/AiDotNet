using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety;

/// <summary>
/// Whole-tensor building blocks shared by the heuristic safety detectors (framing, windowing,
/// spectra, autocorrelation, NaN-preserving clamps). Every helper is expressed as engine operations
/// generic in <typeparamref name="T"/>; none of them reads or writes one element at a time.
/// </summary>
/// <typeparam name="T">The numeric element type.</typeparam>
internal static class SafetyTensorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The active engine.</summary>
    internal static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Converts a literal constant to <typeparamref name="T"/>.</summary>
    internal static T C(double value) => NumOps.FromDouble(value);

    /// <summary>A tensor of the given shape filled with <paramref name="value"/>.</summary>
    internal static Tensor<T> Full(int[] shape, T value) => Tensor<T>.CreateDefault(shape, value);

    /// <summary>A tensor shaped like <paramref name="like"/> filled with <paramref name="value"/>.</summary>
    internal static Tensor<T> FullLike(Tensor<T> like, double value) => Full(like.Shape.ToArray(), C(value));

    /// <summary>Element-wise select: <paramref name="mask"/> non-zero picks <paramref name="whenTrue"/>.</summary>
    internal static Tensor<T> Where(Tensor<T> mask, Tensor<T> whenTrue, Tensor<T> whenFalse)
        => Engine.TensorWhere(mask, whenTrue, whenFalse);

    /// <summary>Element-wise select against a scalar for the false branch.</summary>
    internal static Tensor<T> Where(Tensor<T> mask, Tensor<T> whenTrue, double whenFalse)
        => Engine.TensorWhere(mask, whenTrue, FullLike(whenTrue, whenFalse));

    /// <summary>Element-wise select against a scalar for the true branch.</summary>
    internal static Tensor<T> Where(Tensor<T> mask, double whenTrue, Tensor<T> whenFalse)
        => Engine.TensorWhere(mask, FullLike(whenFalse, whenTrue), whenFalse);

    /// <summary>Element-wise select between two scalars.</summary>
    internal static Tensor<T> Where(Tensor<T> mask, double whenTrue, double whenFalse)
        => Engine.TensorWhere(mask, FullLike(mask, whenTrue), FullLike(mask, whenFalse));

    /// <summary>1 where <c>x &gt; v</c> (NaN gives 0).</summary>
    internal static Tensor<T> Gt(Tensor<T> x, double v) => Engine.TensorGreaterThan(x, C(v));

    /// <summary>1 where <c>x &lt; v</c> (NaN gives 0).</summary>
    internal static Tensor<T> Lt(Tensor<T> x, double v) => Engine.TensorLessThan(x, C(v));

    /// <summary>1 where <c>x &gt;= v</c> (NaN gives 0).</summary>
    internal static Tensor<T> Ge(Tensor<T> x, double v) => Engine.TensorGreaterEqual(x, FullLike(x, v));

    /// <summary>1 where <c>x &lt;= v</c> (NaN gives 0).</summary>
    internal static Tensor<T> Le(Tensor<T> x, double v) => Engine.TensorLessEqual(x, FullLike(x, v));

    /// <summary>Logical AND of two 0/1 masks.</summary>
    internal static Tensor<T> And(Tensor<T> a, Tensor<T> b) => Engine.TensorMultiply(a, b);

    /// <summary>Logical OR of two 0/1 masks.</summary>
    internal static Tensor<T> Or(Tensor<T> a, Tensor<T> b) => Engine.TensorMax(a, b);

    /// <summary>Logical NOT of a 0/1 mask.</summary>
    internal static Tensor<T> Not(Tensor<T> a) => Engine.ScalarMinusTensor(NumOps.One, a);

    /// <summary>Converts a 0/1 mask to a <see cref="Bit"/> mask.</summary>
    internal static Tensor<AiDotNet.Tensors.Bit> ToBit(Tensor<T> mask)
        => Engine.TensorEqScalar(Engine.TensorNotEquals(mask, NumOps.Zero), NumOps.One);

    /// <summary>Elements of <paramref name="x"/> where <paramref name="mask"/> is non-zero, in order.</summary>
    internal static Tensor<T> Compact(Tensor<T> x, Tensor<T> mask)
        => Engine.TensorMaskedSelect(x, ToBit(mask));

    /// <summary><c>Math.Min(1, x)</c> per element (NaN passes through).</summary>
    internal static Tensor<T> MinOne(Tensor<T> x) => Where(Gt(x, 1.0), 1.0, x);

    /// <summary><c>Math.Max(0, x)</c> per element (NaN passes through).</summary>
    internal static Tensor<T> MaxZero(Tensor<T> x) => Where(Lt(x, 0.0), 0.0, x);

    /// <summary><c>Math.Min(1, Math.Max(0, x))</c> per element (NaN passes through).</summary>
    internal static Tensor<T> ClampUnit(Tensor<T> x) => MinOne(MaxZero(x));

    /// <summary><c>a * x + b</c> per element.</summary>
    internal static Tensor<T> Affine(Tensor<T> x, double a, double b)
        => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, C(a)), C(b));

    /// <summary>The integers <c>[0, n)</c> as <typeparamref name="T"/>.</summary>
    internal static Tensor<T> Range(int n) => Engine.TensorArange<T>(0, n, 1);

    /// <summary>Wraps a vector as a rank-1 tensor (copy).</summary>
    internal static Tensor<T> AsTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    /// <summary>
    /// Overlapping frames <c>[count, frameSize]</c> of a signal: row <c>f</c> holds
    /// <c>signal[f * hop .. f * hop + frameSize)</c>.
    /// </summary>
    internal static Tensor<T> Frames(Tensor<T> signal, int frameSize, int hop, int count)
    {
        var unfolded = Engine.TensorUnfold(signal, 0, frameSize, hop);
        if (unfolded.Shape[0] != count)
            unfolded = Engine.TensorNarrow(unfolded, 0, 0, count);
        return unfolded.Contiguous();
    }

    /// <summary>Symmetric Hann window <c>0.5 * (1 - cos(2 pi i / (n - 1)))</c>, <c>i = 0..n-1</c>.</summary>
    internal static Tensor<T> SymmetricHann(int n)
    {
        var angle = Engine.TensorMultiplyScalar(Range(n), C(2.0 * Math.PI / (n - 1)));
        return Engine.TensorMultiplyScalar(Engine.ScalarMinusTensor(NumOps.One, Engine.TensorCos(angle)), C(0.5));
    }

    /// <summary>
    /// Magnitudes of the full complex DFT of each row of <paramref name="frames"/> (last axis),
    /// shape unchanged.
    /// </summary>
    internal static Tensor<T> FftMagnitude(Tensor<T> frames)
    {
        Dft(frames, out var re, out var im);
        return Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(re), Engine.TensorSquare(im)));
    }

    /// <summary>
    /// Lagged inner products of each row of <paramref name="rows"/> (<c>[F, N]</c>) for
    /// <c>lag = minLag .. maxLagExclusive-1</c>, each over the overlap <c>len = N - lag</c>:
    /// <c>corr = sum x[i] x[i+lag]</c>, <c>head = sum x[i]^2</c>, <c>tail = sum x[i+lag]^2</c>
    /// (<c>i &lt; len</c>). Every output is <c>[F, maxLagExclusive - minLag]</c>.
    /// </summary>
    internal static void LaggedProducts(Tensor<T> rows, int minLag, int maxLagExclusive,
        out Tensor<T> corr, out Tensor<T> head, out Tensor<T> tail)
    {
        int n = rows.Shape[1];
        var corrs = new List<Tensor<T>>();
        var heads = new List<Tensor<T>>();
        var tails = new List<Tensor<T>>();
        var axis = new[] { 1 };
        for (int lag = minLag; lag < maxLagExclusive; lag++)
        {
            int len = n - lag;
            var a = Engine.TensorNarrow(rows, 1, 0, len);
            var b = Engine.TensorNarrow(rows, 1, lag, len);
            corrs.Add(Engine.ReduceSum(Engine.TensorMultiply(a, b), axis, false));
            heads.Add(Engine.ReduceSum(Engine.TensorSquare(a), axis, false));
            tails.Add(Engine.ReduceSum(Engine.TensorSquare(b), axis, false));
        }

        corr = Engine.TensorStack(corrs.ToArray(), 1);
        head = Engine.TensorStack(heads.ToArray(), 1);
        tail = Engine.TensorStack(tails.ToArray(), 1);
    }

    /// <summary>
    /// Running "strictly greater than the best so far, starting from <paramref name="floor"/>" scan
    /// along axis 1 of <paramref name="values"/> (<c>[F, L]</c>), as a scalar loop
    /// <c>if (v &gt; best) { best = v; idx = j; }</c> computes it: NaN never wins, ties keep the
    /// lowest index. Returns the best value (<paramref name="floor"/> when nothing beat it) and
    /// the winning index (-1 when nothing beat it), both <c>[F]</c>.
    /// </summary>
    internal static (Tensor<T> Best, Tensor<T> Index) FirstStrictMax(Tensor<T> values, double floor)
    {
        int f = values.Shape[0];
        int l = values.Shape[1];
        // Entries that cannot beat the floor (including NaN) never update the scalar scan.
        var eligible = Gt(values, floor);
        var candidates = Where(eligible, values, floor);
        var best = Engine.ReduceMax(candidates, new[] { 1 }, true);
        var isBest = And(eligible, Engine.TensorEquals(candidates, Engine.TensorBroadcastTo(best, new[] { f, l })));
        // Score l - j for winning columns so the max picks the lowest j.
        var reversed = Engine.ScalarMinusTensor(C(l), Engine.Reshape(Range(l), new[] { 1, l }));
        var score = Engine.TensorMultiply(isBest, Engine.TensorBroadcastTo(reversed, new[] { f, l }));
        var top = Engine.ReduceMax(score, new[] { 1 }, false);
        var anyWin = Gt(top, 0.0);
        var index = Where(anyWin, Engine.ScalarMinusTensor(C(l), top), -1.0);
        return (Engine.Reshape(best, new[] { f }), index);
    }

    /// <summary>
    /// Normalised-autocorrelation pitch of each frame (<c>[F, N]</c>), as the per-frame scalar
    /// estimator computes it: lags <c>sampleRate/500 .. maxLag-1</c>, <c>normalized = corr /
    /// sqrt(head * tail)</c> (0 when the denominator is at most 1e-10), the first strictly largest
    /// positive value wins, and the pitch is <c>sampleRate / bestLag</c> when <c>bestLag &gt; 0</c>
    /// and the peak exceeds 0.3 (otherwise 0).
    /// </summary>
    internal static Tensor<T> AutocorrelationPitch(Tensor<T> frames, int sampleRate, int maxLag)
    {
        int f = frames.Shape[0];
        int minLag = sampleRate / 500;
        if (minLag >= maxLag)
            return Tensor<T>.CreateZeros(f);

        LaggedProducts(frames, minLag, maxLag, out var corr, out var head, out var tail);
        var denom = Engine.TensorSqrt(Engine.TensorMultiply(head, tail));
        var valid = Gt(denom, 1e-10);
        var safeDenom = Where(valid, denom, 1.0);
        var normalized = Where(valid, Engine.TensorDivide(corr, safeDenom), 0.0);
        var (best, index) = FirstStrictMax(normalized, 0.0);
        var bestLag = Engine.TensorAddScalar(index, C(minLag));
        var hasPitch = And(Gt(index, -0.5), And(Gt(bestLag, 0.0), Gt(best, 0.3)));
        var safeLag = Where(hasPitch, bestLag, 1.0);
        var pitch = Engine.TensorDivide(FullLike(safeLag, sampleRate), safeLag);
        return Where(hasPitch, pitch, 0.0);
    }

    /// <summary>
    /// Number of zero crossings along axis 1 of <paramref name="rows"/>, returned as <c>[F]</c>.
    /// With <paramref name="nanIsNegative"/> a crossing is <c>(x[i] &gt;= 0) != (x[i-1] &gt;= 0)</c>
    /// (NaN fails <c>&gt;= 0</c>); otherwise it is <c>(x[i] &gt;= 0 &amp;&amp; x[i-1] &lt; 0) ||
    /// (x[i] &lt; 0 &amp;&amp; x[i-1] &gt;= 0)</c>, so a NaN on either side never counts.
    /// </summary>
    internal static Tensor<T> ZeroCrossings(Tensor<T> rows, bool nanIsNegative)
    {
        int n = rows.Shape[1];
        if (n < 2)
            return Tensor<T>.CreateZeros(rows.Shape[0]);
        var nonNegative = Ge(rows, 0.0);
        var currentNonNegative = Engine.TensorNarrow(nonNegative, 1, 1, n - 1);
        var previousNonNegative = Engine.TensorNarrow(nonNegative, 1, 0, n - 1);
        Tensor<T> change;
        if (nanIsNegative)
        {
            change = Engine.TensorAbs(Engine.TensorSubtract(currentNonNegative, previousNonNegative));
        }
        else
        {
            var negative = Lt(rows, 0.0);
            var currentNegative = Engine.TensorNarrow(negative, 1, 1, n - 1);
            var previousNegative = Engine.TensorNarrow(negative, 1, 0, n - 1);
            change = Or(And(currentNonNegative, previousNegative), And(currentNegative, previousNonNegative));
        }

        return Engine.ReduceSum(change, new[] { 1 }, false);
    }

    /// <summary><c>Math.Min(1, x)</c> for a scalar (NaN passes through).</summary>
    internal static T MinOne(T x) => NumOps.GreaterThan(x, NumOps.One) ? NumOps.One : x;

    /// <summary><c>Math.Max(0, x)</c> for a scalar (NaN passes through).</summary>
    internal static T MaxZero(T x) => NumOps.LessThan(x, NumOps.Zero) ? NumOps.Zero : x;

    /// <summary><c>Math.Max(a, b)</c> for scalars: NaN if either operand is NaN.</summary>
    internal static T MaxNaN(T a, T b)
    {
        if (NumOps.IsNaN(a)) return a;
        if (NumOps.IsNaN(b)) return b;
        return NumOps.GreaterThan(a, b) ? a : b;
    }

    /// <summary><c>Math.Min(1, Math.Max(0, x))</c> for a scalar (NaN passes through).</summary>
    internal static T ClampUnit(T x) => MinOne(MaxZero(x));

    /// <summary>
    /// The integer finalizer hash the protectors use, <c>x = ((x &gt;&gt; 16) ^ x) * 0x45d9f3b</c> twice,
    /// <c>x = (x &gt;&gt; 16) ^ x</c>, then <c>x &amp; 0x7FFFFFFF</c> (wrapping arithmetic), per element.
    /// </summary>
    internal static Tensor<int> HashInt(Tensor<int> x)
    {
        var engine = Engine;
        var shape = x.Shape.ToArray();
        var sixteen = Tensor<int>.CreateDefault(shape, 16);
        x = engine.TensorMultiplyScalar(engine.TensorBitwiseXor(engine.TensorBitwiseRightShift(x, sixteen), x), 0x45d9f3b);
        x = engine.TensorMultiplyScalar(engine.TensorBitwiseXor(engine.TensorBitwiseRightShift(x, sixteen), x), 0x45d9f3b);
        x = engine.TensorBitwiseXor(engine.TensorBitwiseRightShift(x, sixteen), x);
        return engine.TensorBitwiseAnd(x, Tensor<int>.CreateDefault(shape, 0x7FFFFFFF));
    }

    /// <summary>Hash keys <c>row * rowStride + colStart + col</c> as a <c>[rows, cols]</c> grid.</summary>
    internal static Tensor<int> HashKeys(int rows, int rowStride, int colStart, int cols)
    {
        var engine = Engine;
        var grid = new[] { rows, cols };
        var rowPart = engine.Reshape(engine.TensorMultiplyScalar(engine.TensorArange<int>(0, rows, 1), rowStride),
            new[] { rows, 1 });
        var colPart = engine.Reshape(engine.TensorArange<int>(colStart, colStart + cols, 1), new[] { 1, cols });
        return engine.TensorAdd(engine.TensorBroadcastTo(rowPart, grid), engine.TensorBroadcastTo(colPart, grid));
    }

    /// <summary><c>value % modulus</c> of non-negative integers, as <typeparamref name="T"/>.</summary>
    internal static Tensor<T> ModuloAsT(Tensor<int> values, int modulus)
    {
        var remainder = Engine.TensorRemainder(values, Tensor<int>.CreateDefault(values.Shape.ToArray(), modulus));
        return Engine.TensorTake(Range(modulus), remainder);
    }

    /// <summary>
    /// Overlap-add placement of half-overlapping frames (<c>[F, N]</c>, hop <c>N / 2</c>) onto a
    /// signal of <paramref name="length"/> samples: <c>Current</c> holds each frame's first half at
    /// its own position, <c>Previous</c> each frame's second half (which lands on the next frame's
    /// first half). Positions no frame covers are zero.
    /// </summary>
    internal static (Tensor<T> Current, Tensor<T> Previous) HalfOverlapParts(Tensor<T> frames, int length)
    {
        var engine = Engine;
        int count = frames.Shape[0];
        int half = frames.Shape[1] / 2;
        var zeroRow = Tensor<T>.CreateZeros(1, half);
        var first = engine.TensorNarrow(frames, 1, 0, half);
        var second = engine.TensorNarrow(frames, 1, half, half);
        int covered = (count + 1) * half;
        var current = engine.Reshape(engine.TensorConcatenate(new[] { first, zeroRow }, 0), new[] { covered });
        var previous = engine.Reshape(engine.TensorConcatenate(new[] { zeroRow, second }, 0), new[] { covered });
        if (length > covered)
        {
            var tail = Tensor<T>.CreateZeros(length - covered);
            current = engine.TensorConcatenate(new[] { current, tail }, 0);
            previous = engine.TensorConcatenate(new[] { previous, tail }, 0);
        }

        return (current, previous);
    }

    /// <summary>
    /// Full complex DFT of each row (last axis): the radix-2 FFT for power-of-two lengths and the
    /// exact DFT (twiddle matrix) otherwise, so non-power-of-two frames are not zero-padded.
    /// </summary>
    internal static void Dft(Tensor<T> rows, out Tensor<T> real, out Tensor<T> imag)
    {
        var engine = Engine;
        int n = rows.Shape[rows.Rank - 1];
        if (n > 0 && (n & (n - 1)) == 0)
        {
            engine.FFT(rows, Tensor<T>.CreateZeros(rows.Shape.ToArray()), out real, out imag);
            return;
        }

        // angle[j, k] = -2 pi ((j * k) mod n) / n, reduced before scaling to keep the argument small.
        var index = Range(n);
        var product = engine.TensorOuter(index, index);
        var reduced = engine.TensorRemainder(product, Full(new[] { n, n }, C(n)));
        var angle = engine.TensorMultiplyScalar(reduced, C(-2.0 * Math.PI / n));
        real = engine.TensorMatMul(rows, engine.TensorCos(angle));
        imag = engine.TensorMatMul(rows, engine.TensorSin(angle));
    }

    /// <summary>
    /// Decodes a vector of character codes into text: each value is rounded half-to-even
    /// (<c>Math.Round</c>) and codes outside <c>[0, 65535]</c> become <c>'?'</c>; a NaN code
    /// follows this runtime's <c>(int)double.NaN</c> conversion, as the scalar decoder did.
    /// </summary>
    internal static string DecodeCharCodes(Vector<T> content)
    {
        if (content.Length == 0) return string.Empty;

        var engine = Engine;
        var rounded = engine.TensorRound(AsTensor(content));
        double nan = double.NaN;
        int nanCode = unchecked((int)nan);
        T nanReplacement = C(nanCode is >= 0 and <= 65535 ? nanCode : '?');

        var inRange = And(Ge(rounded, 0.0), Le(rounded, 65535.0));
        var codes = Where(inRange, rounded, (double)'?');
        codes = engine.TensorWhere(engine.TensorIsNan(rounded), Full(rounded.Shape.ToArray(), nanReplacement), codes);

        // Integral codes 0..65535 -> int: the number of boundaries 0..65534 strictly below each code.
        var codePoints = engine.TensorBucketize(codes, engine.TensorArange<T>(0, 65535, 1)).ToArray();
        return new string(Array.ConvertAll(codePoints, code => (char)code));
    }

    /// <summary>FNV-1a hash of <c>text[start .. start + length)</c> (clipped to the text), as the text detectors use.</summary>
    internal static int HashNgram(string text, int start, int length)
    {
        unchecked
        {
            int hash = (int)2166136261;
            for (int i = start; i < start + length && i < text.Length; i++)
            {
                hash ^= text[i];
                hash *= 16777619;
            }

            return hash;
        }
    }

    /// <summary>
    /// Hashing-trick accumulation into a <paramref name="dim"/>-vector: every hash adds
    /// <paramref name="signed"/> ? (+1 when bit 0x80000 is set, else -1) : +1 at
    /// <c>((hash % dim) + dim) % dim</c>.
    /// </summary>
    internal static Tensor<T> HashedCounts(int dim, IReadOnlyList<int> hashes, bool signed)
    {
        var engine = Engine;
        var counts = Tensor<T>.CreateZeros(dim);
        if (hashes.Count == 0) return counts;

        var buckets = new int[hashes.Count];
        var signBits = new int[hashes.Count];
        for (int i = 0; i < buckets.Length; i++)
        {
            int hash = hashes[i];
            buckets[i] = ((hash % dim) + dim) % dim;
            signBits[i] = signed && (hash & 0x80000) == 0 ? 0 : 1;
        }

        // signBits picks -1 (0) or +1 (1) from a two-entry table.
        var signTable = Engine.TensorArange<T>(-1, 2, 2);
        var deltas = engine.TensorTake(signTable, new Tensor<int>(new[] { signBits.Length }, new Vector<int>(signBits)));
        return engine.TensorIndexAdd(counts, 0, new Tensor<int>(new[] { buckets.Length }, new Vector<int>(buckets)), deltas);
    }

    /// <summary>Hashes of every character n-gram of length <paramref name="n"/>, left to right.</summary>
    internal static List<int> NgramHashes(string text, int n)
    {
        var hashes = new List<int>(Math.Max(0, text.Length - n + 1));
        for (int i = 0; i <= text.Length - n; i++)
            hashes.Add(HashNgram(text, i, n));
        return hashes;
    }

    /// <summary>
    /// <c>v / sqrt(v . v)</c> when that norm exceeds 1e-10, otherwise <paramref name="v"/> unchanged.
    /// </summary>
    internal static Tensor<T> DivideByNorm(Tensor<T> v)
    {
        T norm = NumOps.Sqrt(Engine.TensorSumOfSquares(v));
        return NumOps.GreaterThan(norm, C(1e-10)) ? Engine.TensorDivideScalar(v, norm) : v;
    }

    /// <summary>
    /// <c>VectorHelper.NormalizeInPlace</c>: scale by <c>1 / sqrt(max(0, v . v))</c> unless that
    /// norm is below 1e-10.
    /// </summary>
    internal static Tensor<T> NormalizeUnlessTiny(Tensor<T> v)
    {
        T norm = NumOps.Sqrt(MaxZero(Engine.TensorSumOfSquares(v)));
        if (NumOps.LessThan(norm, C(1e-10))) return v;
        return Engine.TensorMultiplyScalar(v, NumOps.Divide(NumOps.One, norm));
    }

    /// <summary>
    /// <c>VectorHelper.CosineSimilarity(query, row)</c> for every row of <paramref name="rows"/>
    /// (<c>[count, dim]</c>): <c>clamp(dot / sqrt(|q|^2 |r|^2), -1, 1)</c>, 0 when that
    /// denominator is at most 1e-10.
    /// </summary>
    internal static Tensor<T> CosineSimilarities(Tensor<T> query, Tensor<T> rows)
    {
        var engine = Engine;
        int count = rows.Shape[0];
        int dim = rows.Shape[1];
        var dot = engine.Reshape(engine.TensorMatMul(rows, engine.Reshape(query, new[] { dim, 1 })), new[] { count });
        T queryNorm = engine.TensorSumOfSquares(query);
        var rowNorms = engine.ReduceSum(engine.TensorSquare(rows), new[] { 1 }, false);
        var denom = engine.TensorSqrt(engine.TensorMultiplyScalar(rowNorms, queryNorm));
        var valid = Gt(denom, 1e-10);
        var ratio = engine.TensorDivide(dot, Where(valid, denom, 1.0));
        ratio = Where(Lt(ratio, -1.0), -1.0, MinOne(ratio));
        return Where(valid, ratio, 0.0);
    }

    /// <summary>Positions (ascending) where <c>values &gt;= threshold</c> (NaN never qualifies).</summary>
    internal static int[] IndicesAtLeast(Tensor<T> values, T threshold)
    {
        var engine = Engine;
        var hit = engine.TensorGreaterEqual(values, Full(values.Shape.ToArray(), threshold));
        return engine.TensorNonzero(hit).ToArray();
    }

    /// <summary>Population mean and variance (two-pass, like the scalar code) of a rank-1 tensor.</summary>
    internal static (T Mean, T Variance) MeanVariance(Tensor<T> x)
    {
        T mean = Engine.TensorMean(x);
        var centered = Engine.TensorSubtractScalar(x, mean);
        T variance = Engine.TensorMean(Engine.TensorSquare(centered));
        return (mean, variance);
    }

    /// <summary>
    /// The largest element, scanning as <c>if (v &gt; max) max = v</c> from <c>x[0]</c>:
    /// NaN elements after the first are skipped, a NaN first element stays the result.
    /// </summary>
    internal static T ScanMax(Tensor<T> x)
    {
        T first = x.GetFlat(0);
        if (NumOps.IsNaN(first)) return first;
        var clean = Engine.TensorWhere(Engine.TensorIsNan(x), Full(x.Shape.ToArray(), first), x);
        return Engine.TensorMaxValue(clean);
    }

    /// <summary>
    /// The smallest element, scanning as <c>if (v &lt; min) min = v</c> from <c>x[0]</c>:
    /// NaN elements after the first are skipped, a NaN first element stays the result.
    /// </summary>
    internal static T ScanMin(Tensor<T> x)
    {
        T first = x.GetFlat(0);
        if (NumOps.IsNaN(first)) return first;
        var clean = Engine.TensorWhere(Engine.TensorIsNan(x), Full(x.Shape.ToArray(), first), x);
        return Engine.TensorMinValue(clean);
    }
}
