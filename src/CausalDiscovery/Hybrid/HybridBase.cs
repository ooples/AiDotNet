using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// Base class for hybrid causal discovery algorithms that combine constraint-based and score-based methods.
/// </summary>
/// <remarks>
/// <para>
/// Hybrid methods first use constraint-based tests to restrict the search space (e.g., finding
/// candidate parents via conditional independence tests), then use score-based search to find the
/// optimal DAG within the restricted space.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hybrid algorithms get the best of both worlds. Constraint-based methods
/// are good at ruling out edges quickly, while score-based methods are good at finding the best
/// structure among remaining options. Combining them gives faster AND more accurate results.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class HybridBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.Hybrid;

    /// <summary>
    /// Significance level for constraint-based phase.
    /// </summary>
    protected double Alpha { get; set; } = 0.05;

    /// <summary>
    /// Maximum number of parents per variable (restricts search space).
    /// </summary>
    protected int MaxParents { get; set; } = 5;

    /// <summary>
    /// Applies options from CausalDiscoveryOptions.
    /// </summary>
    protected void ApplyHybridOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.SignificanceLevel.HasValue) Alpha = options.SignificanceLevel.Value;
        if (options.MaxParents.HasValue) MaxParents = options.MaxParents.Value;
    }

    /// <summary>
    /// Computes Pearson correlation between two columns of data.
    /// </summary>
    protected static double ComputeCorrelation(double[,] X, int n, int col1, int col2)
    {
        double mean1 = 0, mean2 = 0;
        for (int i = 0; i < n; i++) { mean1 += X[i, col1]; mean2 += X[i, col2]; }
        mean1 /= n; mean2 /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double d1 = X[i, col1] - mean1, d2 = X[i, col2] - mean2;
            sxy += d1 * d2; sxx += d1 * d1; syy += d2 * d2;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Computes BIC score for a variable given its parents.
    /// </summary>
    protected static double ComputeBIC(double[,] X, int n, int d, int variable, List<int> parents)
    {
        if (parents.Count == 0)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += X[i, variable];
            mean /= n;
            double rss = 0;
            for (int i = 0; i < n; i++) { double e = X[i, variable] - mean; rss += e * e; }
            return n * Math.Log(rss / n + 1e-15) + Math.Log(n);
        }

        int p = parents.Count;
        var Z = new double[n, p + 1];
        for (int i = 0; i < n; i++)
        {
            Z[i, 0] = 1.0;
            for (int j = 0; j < p; j++) Z[i, j + 1] = X[i, parents[j]];
        }

        var ZtZ = new double[p + 1, p + 1];
        var Zty = new double[p + 1];
        for (int i = 0; i <= p; i++)
        {
            for (int j = 0; j <= p; j++)
                for (int k = 0; k < n; k++) ZtZ[i, j] += Z[k, i] * Z[k, j];
            for (int k = 0; k < n; k++) Zty[i] += Z[k, i] * X[k, variable];
        }

        for (int i = 0; i <= p; i++) ZtZ[i, i] += 1e-6;

        var beta = SolveSystem(ZtZ, Zty, p + 1);
        double rss2 = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            for (int j = 0; j <= p; j++) pred += beta[j] * Z[i, j];
            double err = X[i, variable] - pred;
            rss2 += err * err;
        }

        return n * Math.Log(rss2 / n + 1e-15) + (p + 1) * Math.Log(n);
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
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;
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
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
    }

}
