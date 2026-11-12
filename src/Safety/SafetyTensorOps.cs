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
        Engine.FFT(frames, Tensor<T>.CreateZeros(frames.Shape.ToArray()), out var re, out var im);
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

    /// <summary><c>Math.Min(1, Math.Max(0, x))</c> for a scalar (NaN passes through).</summary>
    internal static T ClampUnit(T x) => MinOne(MaxZero(x));

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
