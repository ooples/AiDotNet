using AiDotNet.Extensions;
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
        int d = data.Columns;
        var standardized = StandardizeData(data);

        var W = new Matrix<T>(d, d);

        // Pairwise ANM test for all variable pairs
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                var xi = standardized.GetColumn(i);
                var xj = standardized.GetColumn(j);

                // Test i → j: fit xj = f(xi) + noise, check independence of noise and xi
                var residIJ = RegressOut(xj, xi);
                double depIJ = Math.Abs(GaussianMI(residIJ, xi));

                // Test j → i: fit xi = f(xj) + noise, check independence of noise and xj
                var residJI = RegressOut(xi, xj);
                double depJI = Math.Abs(GaussianMI(residJI, xj));

                // The direction with lower residual dependence is preferred
                double asymmetry = depJI - depIJ; // positive means i → j

                // Asymmetry threshold: a fraction of _threshold to detect directional preference
                double asymmetryThreshold = _threshold * 0.1;
                if (Math.Abs(asymmetry) > asymmetryThreshold)
                {
                    double weight = Math.Abs(ComputeCorrelation(xi, xj));
                    if (weight < _threshold) continue;

                    if (asymmetry > 0)
                        W[i, j] = NumOps.FromDouble(weight);
                    else
                        W[j, i] = NumOps.FromDouble(weight);
                }
            }
        }

        return W;
    }
}
