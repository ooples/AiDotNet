namespace AiDotNet.TimeSeries;

/// <summary>
/// Whole-series engine operations shared by the time-series models: lag matrices, differencing and
/// their inverses, expressed as <see cref="IEngine"/> ops in T.
/// </summary>
/// <typeparam name="T">The numeric type of the series.</typeparam>
internal static class TimeSeriesEngineOps<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>The elements of <paramref name="v"/> in reverse order.</summary>
    public static Vector<T> Reverse(Vector<T> v)
    {
        if (v.Length <= 1)
        {
            return v.Clone();
        }

        var order = new Tensor<int>(new[] { v.Length }, new Vector<int>(Enumerable.Range(0, v.Length).Reverse()));
        return Engine.TensorIndexSelect(Tensor<T>.FromVector(v), order, 0).ToVector();
    }

    /// <summary>
    /// Lag matrix of <paramref name="y"/> for the time steps <c>start .. start + count - 1</c>: row r holds
    /// <c>[y[t-s], y[t-2s], ..., y[t-p·s]]</c> for <c>t = start + r</c> and stride s. Requires
    /// <c>start &gt;= p·s</c>.
    /// </summary>
    public static Matrix<T> LagMatrix(Vector<T> y, int p, int start, int count, int stride = 1)
    {
        if (count <= 0 || p <= 0)
        {
            return new Matrix<T>(Math.Max(count, 0), Math.Max(p, 0));
        }

        // Windows y[t-p·s .. t-s] in forward order; column i (lag (i + 1)·s) sits at offset (p - 1 - i)·s.
        int span = (p - 1) * stride + 1;
        var segment = Tensor<T>.FromVector(y.GetSubVector(start - p * stride, count + span - 1));
        var windows = Engine.TensorUnfold(segment, 0, span, 1);
        var columns = new Tensor<int>(new[] { p },
            new Vector<int>(Enumerable.Range(0, p).Select(i => (p - 1 - i) * stride)));
        return Engine.TensorIndexSelect(windows, columns, 1).ToMatrix();
    }

    /// <summary>The elements of <paramref name="v"/> at <paramref name="indices"/>, in that order.</summary>
    public static Vector<T> Gather(Vector<T> v, IEnumerable<int> indices)
    {
        var index = indices.ToArray();
        if (index.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorIndexSelect(Tensor<T>.FromVector(v), new Tensor<int>(new[] { index.Length }, new Vector<int>(index)), 0)
            .ToVector();
    }

    /// <summary>Dot product of the first <paramref name="count"/> elements of two vectors (zero when count is 0).</summary>
    public static T DotPrefix(Vector<T> a, Vector<T> b, int count)
        => count > 0 ? Engine.DotProduct(a.GetSubVector(0, count), b.GetSubVector(0, count)) : NumOps.Zero;

    /// <summary>
    /// The most recent <paramref name="count"/> values of <paramref name="history"/>, newest first
    /// (<c>[y[n-1], ..., y[n-count]]</c>), zero-filled where the history is shorter.
    /// </summary>
    public static Vector<T> NewestFirst(Vector<T> history, int count)
    {
        int take = Math.Min(count, history.Length);
        var recent = take > 0 ? Reverse(history.GetSubVector(history.Length - take, take)) : new Vector<T>(0);
        return take < count ? Vector<T>.Concatenate(recent, new Vector<T>(count - take)) : recent;
    }

    /// <summary>
    /// Applies <paramref name="predict"/> to every row of <paramref name="rows"/> in order and gathers the
    /// results. Keeps per-sample (possibly overridden) prediction calls while the inputs move in bulk.
    /// </summary>
    public static Vector<T> PredictRows(Matrix<T> rows, Func<Vector<T>, T> predict)
    {
        var predictions = new List<T>(rows.Rows);
        for (int r = 0; r < rows.Rows; r++)
        {
            predictions.Add(predict(rows.GetRow(r)));
        }

        return new Vector<T>(predictions);
    }

    /// <summary>Element-wise <c>v[i] &gt; threshold</c> as a bool mask (NaN compares false).</summary>
    public static bool[] GreaterThan(Vector<T> v, T threshold)
    {
        if (v.Length == 0)
        {
            return Array.Empty<bool>();
        }

        var above = Engine.TensorGreaterThan(Tensor<T>.FromVector(v), threshold);
        var bits = Engine.TensorEqScalar(above, NumOps.One);
        return Array.ConvertAll(bits.ToArray(), bit => (bool)bit);
    }

    /// <summary>Lag-<paramref name="lag"/> difference <c>y[t] - y[t-lag]</c> for t = lag .. n-1.</summary>
    public static Vector<T> Difference(Vector<T> y, int lag)
    {
        int n = y.Length - lag;
        if (n <= 0)
        {
            return new Vector<T>(0);
        }

        return Engine.Subtract(y.GetSubVector(lag, n), y.GetSubVector(0, n));
    }

    /// <summary>Applies the lag-1 difference <paramref name="order"/> times.</summary>
    public static Vector<T> Difference(Vector<T> y, int lag, int order)
    {
        var result = y;
        for (int d = 0; d < order; d++)
        {
            result = Difference(result, lag);
        }

        return result;
    }

    /// <summary>Running sum of <paramref name="v"/> (<c>out[i] = v[0] + ... + v[i]</c>).</summary>
    public static Vector<T> CumulativeSum(Vector<T> v)
    {
        if (v.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorCumSum(Tensor<T>.FromVector(v), 0).ToVector();
    }

    /// <summary>
    /// Inverse of one lag-1 difference: <c>out[i] = ((start + d[0]) + d[1]) + ... + d[i]</c>, summed in that
    /// order (a running sum seeded with <paramref name="start"/>).
    /// </summary>
    public static Vector<T> Integrate(Vector<T> differences, T start)
        => Integrate(differences, new Vector<T>(new[] { start }));

    /// <summary>
    /// Inverse of one lag-1 difference seeded with the single value in <paramref name="start"/>.
    /// </summary>
    public static Vector<T> Integrate(Vector<T> differences, Vector<T> start)
    {
        if (differences.Length == 0)
        {
            return new Vector<T>(0);
        }

        var seeded = Tensor<T>.FromVector(Vector<T>.Concatenate(start.GetSubVector(0, 1), differences));
        var sums = Engine.TensorCumSum(seeded, 0);
        return Engine.TensorSlice(sums, new[] { 1 }, new[] { differences.Length }).ToVector();
    }

    /// <summary>
    /// The (index, actual, predicted, score) entries whose <paramref name="flagged"/> bit is set, in order,
    /// with indices offset by <paramref name="firstIndex"/>.
    /// </summary>
    public static List<(int Index, T Actual, T Predicted, T Score)> SelectFlagged(
        Vector<T> actuals, Vector<T> predictions, Vector<T> scores, bool[] flagged, int firstIndex)
    {
        return actuals
            .Zip(predictions, (actual, predicted) => (actual, predicted))
            .Zip(scores, (pair, score) => (pair.actual, pair.predicted, score))
            .Select((entry, offset) => (Index: firstIndex + offset, Actual: entry.actual, Predicted: entry.predicted, Score: entry.score))
            .Where(entry => flagged[entry.Index - firstIndex])
            .ToList();
    }

    /// <summary>
    /// Inverse of one lag-<paramref name="lag"/> difference continuing after <paramref name="tail"/> (the
    /// last <paramref name="lag"/> undifferenced values): <c>out[i] = out[i - lag] + d[i]</c>, with
    /// <c>out[i - lag]</c> taken from the tail for i &lt; lag.
    /// </summary>
    public static Vector<T> IntegrateSeasonal(Vector<T> differences, Vector<T> tail, int lag)
    {
        int h = differences.Length;
        if (h == 0)
        {
            return new Vector<T>(0);
        }

        // Lay the differences out as a [rows, lag] phase grid (zero-padded) under a first row holding the
        // tail, so a cumulative sum down each phase column continues from that phase's tail value in the
        // same left-to-right order as the scalar recursion; the tail row is dropped afterwards.
        int rows = (h + lag - 1) / lag;
        var padded = Vector<T>.Concatenate(tail.GetSubVector(0, lag), differences, new Vector<T>(rows * lag - h));
        var grid = Tensor<T>.FromVector(padded, new[] { rows + 1, lag });
        var values = Engine.TensorCumSum(grid, 0).Reshape(new[] { (rows + 1) * lag });
        return Engine.TensorSlice(values, new[] { lag }, new[] { h }).ToVector();
    }

    /// <summary>
    /// Lagged product sums Σ_t x[t] x[t-lag] of a (centered) series for lags 0..maxLag; lags at or beyond
    /// the series length give zero.
    /// </summary>
    public static Vector<T> LagProductSums(Vector<T> x, int maxLag)
    {
        int n = x.Length;
        int computed = Math.Min(maxLag + 1, n);
        var sums = new List<T>(maxLag + 1);
        for (int lag = 0; lag < computed; lag++)
        {
            sums.Add(Engine.DotProduct(x.GetSubVector(lag, n - lag), x.GetSubVector(0, n - lag)));
        }

        var head = new Vector<T>(sums);
        return computed < maxLag + 1 ? Vector<T>.Concatenate(head, new Vector<T>(maxLag + 1 - computed)) : head;
    }

    /// <summary>
    /// Autocorrelations ρ(lag) = Σ_t (y[t] - mean)(y[t + lag] - mean) / ((n - lag) · variance) for lags
    /// 0..maxLag, with the sample mean and variance of <see cref="StatisticsHelper{T}"/>.
    /// </summary>
    public static Vector<T> Autocorrelations(Vector<T> y, int maxLag)
    {
        T mean = StatisticsHelper<T>.CalculateMean(y);
        T variance = StatisticsHelper<T>.CalculateVariance(y);

        var centered = Engine.Subtract(y, Engine.Fill(y.Length, mean));
        var sums = LagProductSums(centered, maxLag);
        var lags = Engine.TensorArange<T>(0, maxLag + 1, 1);
        var counts = Engine.ScalarMinusTensor(NumOps.FromDouble(y.Length), lags);
        var denominators = Engine.TensorMultiplyScalar(counts, variance).ToVector();
        return Engine.Divide(sums, denominators);
    }

    /// <summary>
    /// The index a running "keep the first strictly smaller value" scan starting from
    /// <paramref name="start"/> would end on: the first index of the smallest value below
    /// <paramref name="start"/> (NaN never qualifies), or -1 when no value is below it.
    /// </summary>
    public static int FirstStrictMinimumBelow(Vector<T> values, T start)
    {
        if (values.Length == 0)
        {
            return -1;
        }

        var tensor = Tensor<T>.FromVector(values);
        var below = Engine.TensorLessThan(tensor, start);
        var candidates = Engine.TensorWhere(below, tensor, Engine.TensorFullLike(tensor, start));
        T minimum = Engine.TensorMinValue(candidates);
        if (!NumOps.LessThan(minimum, start))
        {
            return -1;
        }

        var hits = Engine.TensorEqScalar(candidates, minimum);
        return Array.FindIndex(hits.ToArray(), bit => (bool)bit);
    }

    /// <summary>
    /// Index of the first element that is the largest of those strictly above <paramref name="start"/>
    /// (a running strict-greater scan seeded with <paramref name="start"/>; NaN never wins), or -1.
    /// </summary>
    public static int FirstStrictMaximumAbove(Vector<T> values, T start)
    {
        if (values.Length == 0)
        {
            return -1;
        }

        var tensor = Tensor<T>.FromVector(values);
        var above = Engine.TensorGreaterThan(tensor, start);
        var candidates = Engine.TensorWhere(above, tensor, Engine.TensorFullLike(tensor, start));
        T maximum = Engine.TensorMaxValue(candidates);
        if (!NumOps.GreaterThan(maximum, start))
        {
            return -1;
        }

        var hits = Engine.TensorEqScalar(candidates, maximum);
        return Array.FindIndex(hits.ToArray(), bit => (bool)bit);
    }

    /// <summary>The first <paramref name="length"/> elements of <paramref name="pattern"/> repeated periodically.</summary>
    public static Vector<T> RepeatPeriodic(Vector<T> pattern, int length)
    {
        if (length <= 0 || pattern.Length == 0)
        {
            return new Vector<T>(Math.Max(length, 0));
        }

        int copies = (length + pattern.Length - 1) / pattern.Length;
        var tiled = Engine.TensorTile(Tensor<T>.FromVector(pattern), new[] { copies });
        return Engine.TensorSlice(tiled, new[] { 0 }, new[] { length }).ToVector();
    }

    /// <summary>Symmetric Toeplitz matrix <c>R[i, j] = values[|i - j|]</c> of size p × p.</summary>
    public static Matrix<T> Toeplitz(Vector<T> values, int p)
    {
        if (p <= 0)
        {
            return new Matrix<T>(0, 0);
        }

        var index = Enumerable.Range(0, p * p).Select(cell => Math.Abs(cell / p - cell % p));
        return Tensor<T>.FromVector(Gather(values, index), new[] { p, p }).ToMatrix();
    }

    /// <summary>
    /// Scales <paramref name="coefficients"/> in place by <c>target / Σ|c|</c> when <c>Σ|c|</c> exceeds
    /// <paramref name="limit"/>.
    /// </summary>
    public static void ShrinkAbsSum(Vector<T> coefficients, T limit, T target)
    {
        if (coefficients.Length == 0)
        {
            return;
        }

        T absSum = Engine.Sum(Engine.Abs(coefficients));
        if (NumOps.GreaterThan(absSum, limit))
        {
            Engine.Multiply(coefficients, NumOps.Divide(target, absSum)).CopyTo(coefficients.AsWritableSpan());
        }
    }

    /// <summary>Formats the elements as "  {label}[{i + 1}] = {value:F4}" lines.</summary>
    public static IEnumerable<string> FormatCoefficients(Vector<T> coefficients, string label)
        => coefficients.Select((c, i) => $"  {label}[{i + 1}] = {FormatValue(c)}");

    /// <summary>
    /// Sums of every contiguous window of <paramref name="size"/> elements: element k is
    /// <c>Σ v[k .. k + size - 1]</c> (length <c>v.Length - size + 1</c>, empty when no window fits).
    /// </summary>
    public static Vector<T> WindowSums(Vector<T> v, int size)
    {
        if (size <= 0 || size > v.Length)
        {
            return new Vector<T>(0);
        }

        var windows = Engine.TensorUnfold(Tensor<T>.FromVector(v), 0, size, 1);
        return Engine.ReduceSum(windows, new[] { 1 }, keepDims: false).ToVector();
    }

    /// <summary>
    /// Per-phase sums <c>Σ_{i ≡ phase (mod period)} v[i]</c> (ascending i) and element counts for
    /// phases 0..period-1. Phases with no element have sum and count zero. Empty when period ≤ 0.
    /// </summary>
    public static (Vector<T> Sums, Vector<T> Counts) PhaseSums(Vector<T> v, int period)
    {
        if (period <= 0)
        {
            return (new Vector<T>(0), new Vector<T>(0));
        }

        int rows = (v.Length + period - 1) / period;
        if (rows == 0)
        {
            return (new Vector<T>(period), new Vector<T>(period));
        }

        // [rows, period] phase grid, zero-padded at the end; the padding adds exact zeros.
        int padding = rows * period - v.Length;
        var values = Tensor<T>.FromVector(Vector<T>.Concatenate(v, new Vector<T>(padding)), new[] { rows, period });
        var ones = Tensor<T>.FromVector(
            Vector<T>.Concatenate(Engine.Fill(v.Length, NumOps.One), new Vector<T>(padding)), new[] { rows, period });
        return (Engine.ReduceSum(values, new[] { 0 }, keepDims: false).ToVector(),
                Engine.ReduceSum(ones, new[] { 0 }, keepDims: false).ToVector());
    }

    /// <summary>Element-wise <c>(v[i] - mean) / scale</c>.</summary>
    public static Vector<T> Standardize(Vector<T> v, T mean, T scale)
        => Engine.TensorDivideScalar(Engine.TensorSubtractScalar(Tensor<T>.FromVector(v), mean), scale).ToVector();

    /// <summary>Element-wise <c>v[i] · scale + mean</c>.</summary>
    public static Vector<T> Destandardize(Vector<T> v, T scale, T mean)
        => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Tensor<T>.FromVector(v), scale), mean).ToVector();

    /// <summary>Scalar <c>value · scale + mean</c> (one draw of a sequential sampler).</summary>
    public static T Destandardize(T value, T scale, T mean)
        => NumOps.Add(NumOps.Multiply(value, scale), mean);

    /// <summary>
    /// The <paramref name="length"/> values of <paramref name="y"/> ending just before index <paramref name="end"/>:
    /// element j is <c>y[end - length + j]</c>, or zero where that index is negative.
    /// </summary>
    public static Vector<T> PrecedingWindow(Vector<T> y, int end, int length)
    {
        int available = Math.Max(0, Math.Min(length, end));
        if (available == 0)
        {
            return new Vector<T>(Math.Max(length, 0));
        }

        return Vector<T>.Concatenate(new Vector<T>(length - available), y.GetSubVector(end - available, available));
    }

    /// <summary>
    /// A copy of <paramref name="target"/> with <c>result[positions[k]] = values[k]</c> for every k
    /// (positions must be distinct and in range).
    /// </summary>
    public static Vector<T> Scatter(Vector<T> target, int[] positions, Vector<T> values)
    {
        var result = Tensor<T>.FromVector(target.Clone());
        if (positions.Length > 0)
        {
            var index = new Tensor<int>(new[] { positions.Length }, new Vector<int>(positions));
            Engine.TensorIndexPutInPlace(result, new[] { index }, Tensor<T>.FromVector(values.GetSubVector(0, positions.Length)));
        }

        return result.ToVector();
    }

    /// <summary>
    /// Per-step empirical quantiles of sampled forecast paths: for each q, step h is the
    /// <c>clamp(floor(q · count), 0, count - 1)</c>-th smallest of the samples' values at h, with NaN
    /// samples ranked before every number (the order of a double sort).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">No samples while a quantile and a step are requested.</exception>
    public static Dictionary<double, Vector<T>> SampleQuantiles(IReadOnlyList<Vector<T>> samples, double[] quantiles, int horizon)
    {
        var result = new Dictionary<double, Vector<T>>();
        if (horizon <= 0)
        {
            foreach (var q in quantiles)
                result[q] = new Vector<T>(horizon);
            return result;
        }

        int count = samples.Count;
        if (count == 0)
        {
            if (quantiles.Length > 0)
                throw new ArgumentOutOfRangeException(nameof(samples), "At least one sample is required to compute quantiles.");
            return result;
        }

        // [count, horizon] sample matrix; each step's column is ranked ascending
        var stacked = Engine.TensorSlice(
            Engine.TensorStack(samples.Select(sample => Tensor<T>.FromVector(sample)).ToArray(), 0),
            new[] { 0, 0 }, new[] { count, horizon });

        // Master ranked the samples as doubles, where NaN sorts before every number. NaN is split out
        // explicitly so no sort path (including the SIMD float sort) ever sees it: NaN samples become +inf
        // for the sort, so each column's numbers fill its first (count - nanCount) ranks, and rank r maps
        // to NaN when r < nanCount and to sorted rank r - nanCount otherwise.
        var nanMask = Engine.TensorIsNan(stacked);
        var shape = new[] { count, horizon };
        var nanFlags = Engine.TensorWhere(nanMask, Engine.TensorFullLike(new Tensor<int>(shape), 1), new Tensor<int>(shape));
        int[] nanCounts = Engine.ReduceSum(nanFlags, new[] { 0 }, keepDims: false).ToArray();
        bool anyNaN = nanCounts.Any(nanCount => nanCount > 0);

        // Only a type that holds NaN reaches the infinity/NaN constants
        var ranked = anyNaN
            ? Engine.TensorWhere(nanMask, Engine.TensorFullLike(stacked, NumOps.FromDouble(double.PositiveInfinity)), stacked)
            : stacked;
        var (sorted, _) = Engine.TensorSort(ranked, 0);
        var flatSorted = sorted.Reshape(new[] { count * horizon });
        foreach (var q in quantiles)
        {
            int idx = Math.Max(0, Math.Min((int)(q * count), count - 1));
            if (!anyNaN)
            {
                result[q] = Engine.TensorSlice(sorted, new[] { idx, 0 }, new[] { 1, horizon }).ToVector();
                continue;
            }

            int[] flatIndices = Enumerable.Range(0, horizon)
                .Select(h => Math.Max(idx - nanCounts[h], 0) * horizon + h)
                .ToArray();
            var picked = Engine.TensorIndexSelect(
                flatSorted, new Tensor<int>(new[] { horizon }, new Vector<int>(flatIndices)), 0);
            int[] pickNaN = Enumerable.Range(0, horizon).Select(h => idx < nanCounts[h] ? 1 : 0).ToArray();
            var pickNaNMask = Engine.TensorEqScalar(new Tensor<int>(new[] { horizon }, new Vector<int>(pickNaN)), 1);
            result[q] = Engine.TensorWhere(
                pickNaNMask, Engine.TensorFullLike(picked, NumOps.FromDouble(double.NaN)), picked).ToVector();
        }

        return result;
    }

    private static string FormatValue(T value)
        => value is IFormattable formattable
            ? formattable.ToString("F4", System.Globalization.CultureInfo.CurrentCulture)
            : value?.ToString() ?? string.Empty;
}
