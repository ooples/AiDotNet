namespace AiDotNet.Finance.Volatility;

/// <summary>
/// Whole-tensor return statistics shared by the neural volatility models: the sample covariance and
/// correlation of the asset columns, per-asset realized volatility, and horizon tiling of a forecast.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class VolatilityStatistics<T>
{
    /// <summary>
    /// Views a 2D <c>[samples, assets]</c> or 3D <c>[batch, sequence, assets]</c> return tensor as a
    /// <c>[samples, assets]</c> matrix (a 3D batch contributes <c>batch * sequence</c> samples).
    /// </summary>
    internal static Tensor<T> ReturnMatrix(Tensor<T> returns)
    {
        if (returns.Rank == 2)
        {
            return returns;
        }

        if (returns.Rank == 3)
        {
            return AiDotNetEngine.Current.Reshape(
                returns, new[] { returns.Shape[0] * returns.Shape[1], returns.Shape[2] });
        }

        throw new ArgumentException("Returns tensor must be 2D or 3D.");
    }

    /// <summary>Sample covariance <c>[assets, assets]</c> with the <c>n - 1</c> denominator (at least 1).</summary>
    /// <remarks>
    /// The centring, the scatter products and their sums run in double and only the covariance is rounded to
    /// <typeparamref name="T"/>: in Half a return of 300 already squares past 65504, and a scatter sum over many
    /// samples overflows long before the averaged covariance does.
    /// </remarks>
    internal static Tensor<T> Covariance(Tensor<T> returns) => WideCovariance(returns).Cast<T>();

    /// <summary>Correlation <c>cov_ij / (sqrt(cov_ii) * sqrt(cov_jj))</c>, from the double covariance.</summary>
    internal static Tensor<T> Correlation(Tensor<T> returns)
    {
        var engine = AiDotNetEngine.Current;
        var covariance = WideCovariance(returns);
        var std = engine.TensorSqrt(engine.TensorDiagonal(covariance));
        return engine.TensorDivide(covariance, engine.TensorOuter(std, std)).Cast<T>();
    }

    /// <summary>Per-asset realized volatility <c>sqrt(mean(r^2))</c>, <c>[assets]</c>.</summary>
    /// <remarks>The squares and their sum run in double; only the volatility is rounded to
    /// <typeparamref name="T"/>, so a Half or float sum of squares cannot overflow first.</remarks>
    internal static Tensor<T> RealizedVolatility(Tensor<T> returns)
    {
        var engine = AiDotNetEngine.Current;
        var x = ReturnMatrix(returns).Cast<double>();
        var meanSquare = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorMultiply(x, x), new[] { 0 }, keepDims: false), (double)x.Shape[0]);
        return engine.TensorSqrt(meanSquare).Cast<T>();
    }

    private static Tensor<double> WideCovariance(Tensor<T> returns)
    {
        var engine = AiDotNetEngine.Current;
        var x = ReturnMatrix(returns).Cast<double>();
        int samples = x.Shape[0];
        var means = engine.TensorDivideScalar(engine.ReduceSum(x, new[] { 0 }, keepDims: true), (double)samples);
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(means, x.Shape.ToArray()));
        var scatter = engine.TensorMatMul(engine.TensorTranspose(centered), centered);
        return engine.TensorDivideScalar(scatter, (double)Math.Max(1, samples - 1));
    }

    /// <summary>
    /// Returns <paramref name="forecast"/> when it is already <c>[horizon, assets]</c>; otherwise repeats its
    /// leading values (cycling when it holds fewer than <paramref name="assets"/>) across every step.
    /// </summary>
    internal static Tensor<T> TileToHorizon(Tensor<T> forecast, int horizon, int assets)
    {
        if (forecast.Rank == 2 && forecast.Shape[0] == horizon && forecast.Shape[1] == assets)
            return forecast;

        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(forecast, new[] { forecast.Length });
        int count = Math.Min(assets, forecast.Length);
        var cycle = Enumerable.Range(0, assets).Select(a => a % count).ToArray();
        var row = engine.Reshape(engine.TensorTake(flat, new Tensor<int>(cycle, new[] { assets })), new[] { 1, assets });
        return engine.TensorTile(row, new[] { horizon, 1 });
    }
}
