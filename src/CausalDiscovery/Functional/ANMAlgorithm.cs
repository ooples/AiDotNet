using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// ANM (Additive Noise Model) — pairwise causal discovery via independence of residuals.
/// </summary>
/// <remarks>
/// <para>
/// ANM determines causal direction between variable pairs by fitting Y = f(X) + N
/// in both directions and checking which direction yields residuals N that are independent
/// of the cause. If the residuals from X → Y are more independent of X than the residuals
/// from Y → X are of Y, then X → Y is the inferred causal direction.
/// </para>
/// <para>
/// <b>For Beginners:</b> If X truly causes Y, then the "noise" left over after predicting
/// Y from X should have nothing to do with X. But if you try predicting X from Y, the
/// leftover noise will still be related to Y. ANM uses this asymmetry to figure out
/// which variable causes which.
/// </para>
/// <para>
/// Reference: Hoyer et al. (2008), "Nonlinear Causal Discovery with Additive Noise Models",
/// NIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ANMAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "ANM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public ANMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
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

        X = StandardizeData(X, n, d);

        var W = new double[d, d];

        // Pairwise ANM test for all variable pairs
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                var xi = GetColumn(X, i, n);
                var xj = GetColumn(X, j, n);

                // Test i → j: fit xj = f(xi) + noise, check independence of noise and xi
                var residIJ = RegressOut(xj, xi, n);
                double depIJ = Math.Abs(GaussianMI(residIJ, xi, n));

                // Test j → i: fit xi = f(xj) + noise, check independence of noise and xj
                var residJI = RegressOut(xi, xj, n);
                double depJI = Math.Abs(GaussianMI(residJI, xj, n));

                // The direction with lower residual dependence is preferred
                double asymmetry = depJI - depIJ; // positive means i → j

                if (Math.Abs(asymmetry) > _threshold * 0.1) // weak threshold for direction
                {
                    double weight = Math.Abs(ComputeCorrelation(xi, xj, n));
                    if (weight < _threshold) continue;

                    if (asymmetry > 0)
                        W[i, j] = weight;
                    else
                        W[j, i] = weight;
                }
            }
        }

        return DoubleArrayToMatrix(W);
    }

    private static double[] GetColumn(double[,] X, int col, int n)
    {
        var result = new double[n];
        for (int i = 0; i < n; i++) result[i] = X[i, col];
        return result;
    }

    private static double ComputeCorrelation(double[] x, double[] y, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx, dy = y[i] - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }
}
