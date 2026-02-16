using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// VAR-LiNGAM — Vector Autoregressive LiNGAM for time series causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// VAR-LiNGAM combines VAR (Vector Autoregression) with LiNGAM to discover both
/// contemporaneous (same time-step) and lagged (across time-steps) causal relationships.
/// </para>
/// <para>
/// <b>Model:</b> X(t) = B₀ X(t) + B₁ X(t-1) + ... + Bₖ X(t-k) + e(t)
/// where B₀ encodes contemporaneous effects and B₁...Bₖ encode lagged effects.
/// </para>
/// <para>
/// <b>For Beginners:</b> This algorithm finds causal relationships in time series data
/// that work at different time scales. It can detect both "X causes Y right now"
/// (contemporaneous) and "yesterday's X causes today's Y" (lagged) relationships.
/// </para>
/// <para>
/// Reference: Hyvarinen et al. (2010), "Estimation of a Structural Vector Autoregression
/// Model Using Non-Gaussianity", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class VARLiNGAMAlgorithm<T> : FunctionalBase<T>
{
    private int _maxLag = 3;
    private double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "VAR-LiNGAM";

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public VARLiNGAMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
        if (options?.MaxIterations.HasValue == true) _maxLag = options.MaxIterations.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        int effectiveN = n - _maxLag;
        if (effectiveN < d + 1) return DoubleArrayToMatrix(new double[d, d]);

        // Step 1: Fit VAR model to get residuals and lagged coefficients
        var (residuals, laggedCoefs) = FitVARAndGetResiduals(X, n, d, _maxLag);

        // Step 2: Apply DirectLiNGAM on residuals to get B₀ (contemporaneous effects)
        var residMatrix = new Matrix<T>(effectiveN, d);
        for (int i = 0; i < effectiveN; i++)
            for (int j = 0; j < d; j++)
                residMatrix[i, j] = NumOps.FromDouble(residuals[i, j]);

        var directLiNGAM = new DirectLiNGAMAlgorithm<T>(
            new CausalDiscoveryOptions { EdgeThreshold = _threshold });
        var B0Graph = directLiNGAM.DiscoverStructure(residMatrix);

        // Step 3: Merge lagged effects into the adjacency matrix
        var result = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                result[i, j] = NumOps.ToDouble(B0Graph.AdjacencyMatrix[i, j]);

        // Aggregate lagged coefficients: take max absolute lagged effect across all lags
        for (int target = 0; target < d; target++)
        {
            for (int lag = 0; lag < _maxLag; lag++)
            {
                for (int source = 0; source < d; source++)
                {
                    double lagWeight = Math.Abs(laggedCoefs[target][lag * d + source]);
                    if (lagWeight >= _threshold && lagWeight > Math.Abs(result[source, target]))
                    {
                        result[source, target] = laggedCoefs[target][lag * d + source];
                    }
                }
            }
        }

        return DoubleArrayToMatrix(result);
    }

    private (double[,] Residuals, double[][] LaggedCoefs) FitVARAndGetResiduals(double[,] X, int n, int d, int maxLag)
    {
        int effectiveN = n - maxLag;
        var residuals = new double[effectiveN, d];
        var laggedCoefs = new double[d][];

        for (int target = 0; target < d; target++)
        {
            int p = d * maxLag;
            var design = new double[effectiveN, p];
            var y = new double[effectiveN];

            for (int t = 0; t < effectiveN; t++)
            {
                y[t] = X[t + maxLag, target];
                for (int lag = 0; lag < maxLag; lag++)
                    for (int col = 0; col < d; col++)
                        design[t, lag * d + col] = X[t + maxLag - lag - 1, col];
            }

            // Solve OLS
            var XtX = new double[p, p];
            var Xty = new double[p];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                    for (int k = 0; k < effectiveN; k++) XtX[i, j] += design[k, i] * design[k, j];
                for (int k = 0; k < effectiveN; k++) Xty[i] += design[k, i] * y[k];
            }

            for (int i = 0; i < p; i++) XtX[i, i] += 1e-4;

            var beta = SolveSystem(XtX, Xty, p);

            laggedCoefs[target] = beta;

            for (int t = 0; t < effectiveN; t++)
            {
                double pred = 0;
                for (int j = 0; j < p; j++) pred += beta[j] * design[t, j];
                residuals[t, target] = y[t] - pred;
            }
        }

        return (residuals, laggedCoefs);
    }

    private static double[] SolveSystem(double[,] A, double[] b, int size)
    {
        var aug = new double[size, size + 1];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++) aug[i, j] = A[i, j];
            aug[i, size] = b[i];
        }

        for (int col = 0; col < size; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < size; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col])) maxRow = row;
            for (int j = 0; j <= size; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            if (Math.Abs(aug[col, col]) < 1e-10) continue;
            for (int row = col + 1; row < size; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= size; j++) aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[size];
        for (int i = size - 1; i >= 0; i--)
        {
            x[i] = aug[i, size];
            for (int j = i + 1; j < size; j++) x[i] -= aug[i, j] * x[j];
            if (Math.Abs(aug[i, i]) > 1e-10)
            {
                x[i] /= aug[i, i];
            }
            else
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"VAR-LiNGAM: near-singular pivot at index {i} (value={aug[i, i]:E2}); coefficient set to zero (possible collinearity).");
                x[i] = 0;
            }
        }

        return x;
    }
}
