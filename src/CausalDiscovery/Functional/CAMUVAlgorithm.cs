using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM-UV — Causal Additive Model with Unobserved Variables.
/// </summary>
/// <remarks>
/// <para>
/// CAM-UV extends CAM to handle latent (unobserved) confounders. It discovers the causal
/// structure among observed variables even when some common causes are hidden. The algorithm:
/// <list type="number">
/// <item>Fits pairwise additive noise models between all variable pairs.</item>
/// <item>Identifies potential latent confounders by detecting pairs where residuals in
/// both directions show high dependence (neither direction fits well).</item>
/// <item>Marks bidirectional edges for pairs with suspected latent confounders.</item>
/// <item>Orients remaining edges using the standard ANM asymmetry criterion.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes two variables appear related not because one causes the
/// other, but because a hidden third variable causes both. CAM-UV detects these situations
/// by checking: if neither direction X→Y nor Y→X fits cleanly, there might be a hidden
/// common cause. It marks such pairs as "confounded" rather than forcing a causal direction.
/// </para>
/// <para>Reference: Maeda and Shimizu (2021), "Causal Additive Models with Unobserved Variables".</para>
/// </remarks>
internal class CAMUVAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;
    private double _confoundingThreshold = 0.3;

    /// <inheritdoc/>
    public override string Name => "CAM-UV";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CAMUVAlgorithm(CausalDiscoveryOptions? options = null)
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

        // Pairwise analysis: for each pair, test both causal directions
        // and detect latent confounders via bidirectional residual dependence
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                var xi = GetColumn(X, i, n);
                var xj = GetColumn(X, j, n);

                double weight = Math.Abs(ComputeCorrelation(xi, xj, n));
                if (weight < _threshold) continue;

                // Test i → j: fit xj = f(xi) + noise
                var residIJ = KernelRegressOut(xi, xj, n);
                double depIJ = Math.Abs(GaussianMI(residIJ, xi, n));

                // Test j → i: fit xi = f(xj) + noise
                var residJI = KernelRegressOut(xj, xi, n);
                double depJI = Math.Abs(GaussianMI(residJI, xj, n));

                // Detect latent confounder: both directions show high residual dependence
                bool suspectConfounder = depIJ > _confoundingThreshold && depJI > _confoundingThreshold;

                if (suspectConfounder)
                {
                    // Bidirectional edge indicates suspected latent confounder
                    W[i, j] = weight;
                    W[j, i] = weight;
                }
                else
                {
                    // Orient using asymmetry: lower dependence = better fit
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
        }

        return DoubleArrayToMatrix(W);
    }

    /// <summary>
    /// Kernel regression residuals: fits y = f(x) + ε using Nadaraya–Watson
    /// kernel smoothing, returns ε.
    /// </summary>
    private static double[] KernelRegressOut(double[] x, double[] y, int n)
    {
        double h = Math.Pow(n, -1.0 / 5.0);
        var residuals = new double[n];

        for (int i = 0; i < n; i++)
        {
            double numerator = 0;
            double denominator = 0;

            for (int j = 0; j < n; j++)
            {
                double diff = (x[i] - x[j]) / h;
                double kernel = Math.Exp(-0.5 * diff * diff);
                numerator += kernel * y[j];
                denominator += kernel;
            }

            double predicted = denominator > 1e-15 ? numerator / denominator : 0;
            residuals[i] = y[i] - predicted;
        }

        return residuals;
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
