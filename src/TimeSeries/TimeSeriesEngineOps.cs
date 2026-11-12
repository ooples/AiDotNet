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
    /// <c>[y[t-1], y[t-2], ..., y[t-p]]</c> for <c>t = start + r</c>. Requires <c>start &gt;= p</c>.
    /// </summary>
    public static Matrix<T> LagMatrix(Vector<T> y, int p, int start, int count)
    {
        if (count <= 0 || p <= 0)
        {
            return new Matrix<T>(Math.Max(count, 0), Math.Max(p, 0));
        }

        // Windows y[t-p .. t-1] in forward order, then columns reversed so column i is lag i + 1.
        var segment = Tensor<T>.FromVector(y.GetSubVector(start - p, count + p - 1));
        var windows = Engine.TensorUnfold(segment, 0, p, 1);
        var reversed = new Tensor<int>(new[] { p }, new Vector<int>(Enumerable.Range(0, p).Reverse()));
        return Engine.TensorIndexSelect(windows, reversed, 1).ToMatrix();
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
    /// Inverse of one lag-1 difference: <c>out[i] = start + d[0] + ... + d[i]</c>.
    /// </summary>
    public static Vector<T> Integrate(Vector<T> differences, T start)
    {
        if (differences.Length == 0)
        {
            return new Vector<T>(0);
        }

        var sums = Engine.TensorCumSum(Tensor<T>.FromVector(differences), 0);
        return Engine.TensorAddScalar(sums, start).ToVector();
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

        // Lay the differences out as a [rows, lag] phase grid (zero-padded), cumulative-sum each phase column
        // and add the tail value of that phase.
        int rows = (h + lag - 1) / lag;
        var padded = Vector<T>.Concatenate(differences, new Vector<T>(rows * lag - h));
        var grid = Tensor<T>.FromVector(padded, new[] { rows, lag });
        var sums = Engine.TensorCumSum(grid, 0);
        var start = Engine.TensorBroadcastTo(Tensor<T>.FromVector(tail, new[] { 1, lag }), new[] { rows, lag });
        var values = Engine.TensorAdd(sums, start).Reshape(new[] { rows * lag });
        return Engine.TensorSlice(values, new[] { 0 }, new[] { h }).ToVector();
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

    /// <summary>Formats the elements as "  {label}[{i + 1}] = {value:F4}" lines.</summary>
    public static IEnumerable<string> FormatCoefficients(Vector<T> coefficients, string label)
        => coefficients.Select((c, i) => $"  {label}[{i + 1}] = {FormatValue(c)}");

    private static string FormatValue(T value)
        => value is IFormattable formattable
            ? formattable.ToString("F4", System.Globalization.CultureInfo.CurrentCulture)
            : value?.ToString() ?? string.Empty;
}
