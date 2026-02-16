using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// PNL (Post-Nonlinear Causal Model) — Y = g(f(X) + N).
/// </summary>
/// <remarks>
/// <para>
/// PNL extends the Additive Noise Model by allowing a post-nonlinear distortion g.
/// The model is Y = g(f(X) + N) where f is the causal mechanism, N is independent noise,
/// and g is an invertible post-nonlinear transformation. The algorithm:
/// <list type="number">
/// <item>For each variable pair, fits the inner function f via kernel regression.</item>
/// <item>Estimates the post-nonlinear function g by applying a rank-based CDF transform
/// (probability integral transform) to approximate g^(-1).</item>
/// <item>Computes residuals in the "linearized" space after inverting g.</item>
/// <item>Tests independence of the residuals from the cause using mutual information.</item>
/// <item>Orients the edge in the direction where residuals are more independent.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes the relationship between cause and effect has two layers
/// of nonlinearity: first the cause produces an effect through some function f, then
/// that result gets distorted by another function g (like a sensor with nonlinear response).
/// PNL can handle this double-nonlinearity by "undoing" the outer distortion before
/// testing for the cause-effect direction.
/// </para>
/// <para>
/// Reference: Zhang and Hyvarinen (2009), "On the Identifiability of the
/// Post-Nonlinear Causal Model", UAI.
/// </para>
/// </remarks>
internal class PNLAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "PNL";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public PNLAlgorithm(CausalDiscoveryOptions? options = null)
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

        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                var xi = GetColumn(X, i, n);
                var xj = GetColumn(X, j, n);

                double weight = Math.Abs(ComputeCorrelation(xi, xj, n));
                if (weight < _threshold) continue;

                // Test direction i → j: Y=xj, X=xi => Y = g(f(X) + N)
                double depIJ = PNLResidualDependence(xi, xj, n);

                // Test direction j → i: Y=xi, X=xj => Y = g(f(X) + N)
                double depJI = PNLResidualDependence(xj, xi, n);

                double asymmetry = depJI - depIJ;
                if (Math.Abs(asymmetry) > _threshold * 0.1)
                {
                    if (asymmetry > 0)
                        W[i, j] = weight;
                    else
                        W[j, i] = weight;
                }
            }
        }

        return DoubleArrayToMatrix(W);
    }

    /// <summary>
    /// Computes residual dependence for the PNL model Y = g(f(X) + N).
    /// <list type="number">
    /// <item>Fits inner function f via kernel regression: f_hat(X)</item>
    /// <item>Inverts outer function g using rank-based CDF transform on Y</item>
    /// <item>Computes residuals: N_hat = g^(-1)(Y) - f_hat(X)</item>
    /// <item>Returns dependence between residuals and cause X</item>
    /// </list>
    /// </summary>
    private double PNLResidualDependence(double[] cause, double[] effect, int n)
    {
        // Step 1: Estimate inner function f via Nadaraya-Watson kernel regression
        double h = Math.Pow(n, -1.0 / 5.0);
        var fHat = new double[n];
        for (int i = 0; i < n; i++)
        {
            double num = 0, den = 0;
            for (int j = 0; j < n; j++)
            {
                double diff = (cause[i] - cause[j]) / h;
                double kernel = Math.Exp(-0.5 * diff * diff);
                num += kernel * effect[j];
                den += kernel;
            }
            fHat[i] = den > 1e-15 ? num / den : 0;
        }

        // Step 2: Invert g using rank-based probability integral transform
        // g^(-1)(Y) is approximated by Φ^(-1)(F_Y(Y)) where F_Y is empirical CDF
        var gInvY = RankTransform(effect, n);

        // Step 3: Also transform f_hat to the same scale
        var gInvFHat = RankTransform(fHat, n);

        // Step 4: Compute residuals in the linearized space
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
            residuals[i] = gInvY[i] - gInvFHat[i];

        // Step 5: Measure independence of residuals from cause
        return Math.Abs(GaussianMI(residuals, cause, n));
    }

    /// <summary>
    /// Rank-based transform: maps values to approximate standard normal via empirical CDF.
    /// This serves as the inverse of the unknown post-nonlinear function g.
    /// </summary>
    private static double[] RankTransform(double[] values, int n)
    {
        var indexed = new (double Value, int Index)[n];
        for (int i = 0; i < n; i++)
            indexed[i] = (values[i], i);
        Array.Sort(indexed, (a, b) => a.Value.CompareTo(b.Value));

        var transformed = new double[n];
        for (int rank = 0; rank < n; rank++)
        {
            // Map rank to (0,1) then to standard normal via probit approximation
            double u = (rank + 0.5) / n;
            transformed[indexed[rank].Index] = ProbitApproximation(u);
        }
        return transformed;
    }

    /// <summary>
    /// Approximation of the probit function (inverse normal CDF) using rational approximation.
    /// </summary>
    private static double ProbitApproximation(double p)
    {
        // Abramowitz and Stegun 26.2.23 rational approximation
        p = Math.Max(1e-6, Math.Min(1 - 1e-6, p));
        double sign = p < 0.5 ? -1.0 : 1.0;
        double q = p < 0.5 ? p : 1 - p;
        double t = Math.Sqrt(-2.0 * Math.Log(q));

        const double c0 = 2.515517;
        const double c1 = 0.802853;
        const double c2 = 0.010328;
        const double d1 = 1.432788;
        const double d2 = 0.189269;
        const double d3 = 0.001308;

        double result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
        return sign * result;
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
