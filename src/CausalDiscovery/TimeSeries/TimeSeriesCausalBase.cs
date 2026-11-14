using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// Base class for time series causal discovery algorithms (Granger, PCMCI, DYNOTEARS, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Time series causal discovery extends standard methods by considering temporal relationships.
/// Variable X is said to Granger-cause Y if past values of X help predict Y beyond what
/// Y's own past values can predict.
/// </para>
/// <para>
/// <b>For Beginners:</b> In time series, the order of events matters. These algorithms figure
/// out which variables help predict other variables' future values. For example, does yesterday's
/// stock price of company A help predict today's stock price of company B?
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TimeSeriesCausalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.TimeSeries;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    /// <summary>
    /// Maximum lag order for temporal relationships.
    /// </summary>
    protected int MaxLag { get; set; } = 3;

    /// <summary>
    /// Applies time-series-specific options.
    /// </summary>
    protected void ApplyTimeSeriesOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxLag.HasValue)
        {
            if (options.MaxLag.Value <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxLag must be > 0.");
            MaxLag = options.MaxLag.Value;
        }
    }

    /// <summary>
    /// Creates lagged data matrix from time series: for each time step t,
    /// includes values at lags t-1, t-2, ..., t-maxLag.
    /// </summary>
    protected (Matrix<T> LaggedX, Vector<T> Target) CreateLaggedData(
        Matrix<T> data, int targetCol, int maxLag)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (targetCol < 0 || targetCol >= d)
            throw new ArgumentOutOfRangeException(nameof(targetCol), $"targetCol must be in [0, {d - 1}].");
        if (maxLag <= 0 || maxLag >= n)
            throw new ArgumentOutOfRangeException(nameof(maxLag), $"maxLag must be in [1, {n - 1}].");

        // laggedX[t, lag * d + col] = data[t + maxLag - lag - 1, col]; target[t] = data[t + maxLag, targetCol]
        int effectiveN = n - maxLag;
        var x = Tensor<T>.FromMatrix(data);
        var lagBlocks = new Tensor<T>[maxLag];
        for (int lag = 0; lag < maxLag; lag++)
            lagBlocks[lag] = Engine.TensorSlice(x, new[] { maxLag - lag - 1, 0 }, new[] { effectiveN, d });
        var laggedX = Engine.TensorConcatenate(lagBlocks, 1).ToMatrix();
        var target = Engine.TensorSlice(x, new[] { maxLag, targetCol }, new[] { effectiveN, 1 })
            .Reshape(new[] { effectiveN }).ToVector();

        return (laggedX, target);
    }

    /// <summary>
    /// The [effectiveN] series x[offset + t - lag, variable] for t = 0..effectiveN-1, with rows outside
    /// the data read as zero.
    /// </summary>
    private protected Tensor<T> LaggedSeries(Tensor<T> x, int variable, int lag, int offset, int effectiveN)
    {
        int n = x.Shape[0];
        int start = offset - lag;
        int first = Math.Max(0, start);
        int last = Math.Min(n, start + effectiveN);
        var column = Engine.TensorSliceAxis(x, 1, variable);
        if (first == start && last == start + effectiveN)
            return Engine.TensorSlice(column, new[] { start }, new[] { effectiveN });

        var parts = new List<Tensor<T>>();
        if (first > start) parts.Add(new Tensor<T>(new[] { Math.Min(first - start, effectiveN) }));
        if (last > first) parts.Add(Engine.TensorSlice(column, new[] { first }, new[] { last - first }));
        int tail = effectiveN - parts.Sum(p => p.Length);
        if (tail > 0) parts.Add(new Tensor<T>(new[] { tail }));
        return parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>
    /// Residuals y - Z beta of the least-squares fit (Z'Z + ridge I) beta = Z'y over Z [n x p] and y [n];
    /// with <paramref name="center"/> both are centered first, which is the residual of a fit with an
    /// intercept. Z'Z is mirrored from its upper triangle so it is exactly symmetric. With the Cholesky
    /// solver, a factorization that fails (a non-positive pivot from collinear columns whose rounding
    /// error exceeds the ridge) falls back to partial-pivoting LU, and a system LU also reports singular
    /// gets beta = 0 (the residual is the response), so callers see residuals instead of an exception.
    /// </summary>
    private protected Tensor<T> OlsResidualSeries(Tensor<T> z, Tensor<T> y, bool center, double ridge,
        MatrixDecompositionType solver)
    {
        int n = z.Shape[0];
        int p = z.Shape[1];
        var design = center ? CenterColumns(z) : z;
        var response = (center ? Engine.TensorSubtractScalar(y, Engine.TensorMean(y)) : y).Reshape(new[] { n, 1 });
        var designT = Engine.TensorTranspose(design);
        var gram = CausalTensorHelpers.Symmetrize(Engine.TensorMatMul(designT, design));
        gram = Engine.TensorAdd(gram, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), NumOps.FromDouble(ridge)));
        var ztyColumn = Engine.TensorMatMul(designT, response);

        Tensor<T> beta;
        if (solver == MatrixDecompositionType.Cholesky)
        {
            var (factor, info) = LinalgDispatch<T>.CholeskyEx(gram);
            if (Engine.TensorCountNonzero(info) == 0)
            {
                beta = LinalgDispatch<T>.CholeskySolve(ztyColumn, factor);
            }
            else
            {
                var (solution, luInfo) = LinalgDispatch<T>.SolveEx(gram, ztyColumn);
                beta = Engine.TensorCountNonzero(luInfo) == 0 ? solution : new Tensor<T>(new[] { p, 1 });
            }
        }
        else
        {
            var zty = ztyColumn.Reshape(new[] { p }).ToVector();
            beta = MatrixSolutionHelper.SolveLinearSystem<T>(gram.ToMatrix(), zty, solver).AsTensor();
        }

        return Engine.TensorSubtract(response, Engine.TensorMatMul(design, beta.Reshape(new[] { p, 1 })))
            .Reshape(new[] { n });
    }

    /// <summary>
    /// Pearson correlation of two series (0 unless both centered sums of squares are above 1e-10).
    /// </summary>
    private protected double SeriesCorrelation(Tensor<T> x, Tensor<T> y)
        => NumOps.ToDouble(CausalTensorHelpers.PearsonCorrelation(x, y));

    /// <summary>
    /// Computes RSS (Residual Sum of Squares) for OLS regression using generic operations.
    /// </summary>
    protected double ComputeRSS(Matrix<T> X, Vector<T> y, int n, int p)
    {
        // Build normal equations: XtX * beta = Xty over the first n rows and p columns
        var design = Engine.TensorSlice(Tensor<T>.FromMatrix(X), new[] { 0, 0 }, new[] { n, p });
        var response = Engine.TensorSlice(y.AsTensor(), new[] { 0 }, new[] { n }).Reshape(new[] { n, 1 });
        var designT = Engine.TensorTranspose(design);
        var XtX = CausalTensorHelpers.Symmetrize(Engine.TensorMatMul(designT, design));

        // Ridge regularization
        XtX = Engine.TensorAdd(XtX, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), NumOps.FromDouble(1e-6)));
        var Xty = Engine.TensorMatMul(designT, response).Reshape(new[] { p }).ToVector();

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX.ToMatrix(), Xty, MatrixDecompositionType.Lu);

        var residual = Engine.TensorSubtract(response, Engine.TensorMatMul(design, beta.AsTensor().Reshape(new[] { p, 1 })));
        return NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(residual)));
    }
}
