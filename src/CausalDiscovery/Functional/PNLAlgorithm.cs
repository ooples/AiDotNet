using AiDotNet.Extensions;
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
public class PNLAlgorithm<T> : FunctionalBase<T>
{
    private readonly double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "PNL";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public PNLAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true)
            _threshold = Math.Max(0, Math.Min(1, options.EdgeThreshold.Value));
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n == 0 || d < 2) return new Matrix<T>(d, d);

        var standardized = StandardizeData(data);

        var W = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                var xi = standardized.GetColumn(i);
                var xj = standardized.GetColumn(j);

                double weight = Math.Abs(ComputeCorrelation(xi, xj));
                if (weight < _threshold) continue;

                double depIJ = PNLResidualDependence(xi, xj);
                double depJI = PNLResidualDependence(xj, xi);

                double asymmetry = depJI - depIJ;

                // When both dependencies are near zero (deterministic relationship),
                // the PNL model can't determine direction from residuals alone.
                // Fall back to placing edge i → j since we know a strong relationship exists.
                bool nearZeroDeps = depIJ < 1e-6 && depJI < 1e-6;

                if (nearZeroDeps)
                {
                    W[i, j] = NumOps.FromDouble(weight);
                }
                else if (Math.Abs(asymmetry) > _threshold * 0.1)
                {
                    if (asymmetry > 0)
                        W[i, j] = NumOps.FromDouble(weight);
                    else
                        W[j, i] = NumOps.FromDouble(weight);
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Computes residual dependence for the PNL model Y = g(f(X) + N).
    /// </summary>
    private double PNLResidualDependence(Vector<T> cause, Vector<T> effect)
    {
        int n = cause.Length;

        // Step 1: Estimate inner function f via kernel regression
        var fHat = KernelRegressOut(cause, effect);
        // fHat is the residual; we need the prediction: predicted = effect - residual
        var fPredicted = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            fPredicted[i] = NumOps.Subtract(effect[i], fHat[i]);

        // Step 2: Invert g using rank-based probability integral transform
        var gInvY = RankTransform(effect);
        var gInvFHat = RankTransform(fPredicted);

        // Step 3: Compute residuals in the linearized space
        var residuals = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            residuals[i] = NumOps.Subtract(gInvY[i], gInvFHat[i]);

        // Step 4: Measure independence of residuals from cause
        return Math.Abs(GaussianMI(residuals, cause));
    }

    /// <summary>
    /// Rank-based transform: maps values to approximate standard normal via empirical CDF.
    /// </summary>
    private Vector<T> RankTransform(Vector<T> values)
    {
        int n = values.Length;
        var indexed = new (double Value, int Index)[n];
        for (int i = 0; i < n; i++)
            indexed[i] = (NumOps.ToDouble(values[i]), i);
        Array.Sort(indexed, (a, b) => a.Value.CompareTo(b.Value));

        var transformed = new Vector<T>(n);
        for (int rank = 0; rank < n; rank++)
        {
            double u = (rank + 0.5) / n;
            transformed[indexed[rank].Index] = NumOps.FromDouble(ProbitApproximation(u));
        }
        return transformed;
    }

    /// <summary>
    /// Abramowitz and Stegun 26.2.23 rational approximation of the probit function.
    /// </summary>
    private static double ProbitApproximation(double p)
    {
        p = Math.Max(1e-6, Math.Min(1 - 1e-6, p));
        double sign = p < 0.5 ? -1.0 : 1.0;
        double q = p < 0.5 ? p : 1 - p;
        double t = Math.Sqrt(-2.0 * Math.Log(q));

        const double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        const double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;

        return sign * (t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t));
    }
}
