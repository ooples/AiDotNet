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
