using AiDotNet.Enums;
using AiDotNet.Helpers;

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
    protected double ComputeCorrelation(Matrix<T> data, int col1, int col2)
    {
        int n = data.Rows;
        double mean1 = 0, mean2 = 0;
        for (int i = 0; i < n; i++)
        {
            mean1 += NumOps.ToDouble(data[i, col1]);
            mean2 += NumOps.ToDouble(data[i, col2]);
        }
        mean1 /= n; mean2 /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double d1 = NumOps.ToDouble(data[i, col1]) - mean1;
            double d2 = NumOps.ToDouble(data[i, col2]) - mean2;
            sxy += d1 * d2; sxx += d1 * d1; syy += d2 * d2;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Computes BIC score for a variable given its parents.
    /// </summary>
    protected double ComputeBIC(Matrix<T> data, int variable, List<int> parents)
    {
        int n = data.Rows;
        if (parents.Count == 0)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += NumOps.ToDouble(data[i, variable]);
            mean /= n;
            double rss = 0;
            for (int i = 0; i < n; i++)
            {
                double e = NumOps.ToDouble(data[i, variable]) - mean;
                rss += e * e;
            }
            return n * Math.Log(rss / n + 1e-15) + Math.Log(n);
        }

        int p = parents.Count;
        int dim = p + 1; // +1 for intercept

        // Build normal equations using generic operations
        var ZtZ = new Matrix<T>(dim, dim);
        var Zty = new Vector<T>(dim);
        for (int a = 0; a < dim; a++)
        {
            for (int b = 0; b < dim; b++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    T va = a == 0 ? NumOps.One : data[k, parents[a - 1]];
                    T vb = b == 0 ? NumOps.One : data[k, parents[b - 1]];
                    sum = NumOps.Add(sum, NumOps.Multiply(va, vb));
                }
                ZtZ[a, b] = sum;
            }
            T sumY = NumOps.Zero;
            for (int k = 0; k < n; k++)
            {
                T va = a == 0 ? NumOps.One : data[k, parents[a - 1]];
                sumY = NumOps.Add(sumY, NumOps.Multiply(va, data[k, variable]));
            }
            Zty[a] = sumY;
        }

        T ridge = NumOps.FromDouble(1e-6);
        for (int a = 0; a < dim; a++)
            ZtZ[a, a] = NumOps.Add(ZtZ[a, a], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(ZtZ, Zty, MatrixDecompositionType.Lu);

        double rss2 = 0;
        for (int i = 0; i < n; i++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                T vj = j == 0 ? NumOps.One : data[i, parents[j - 1]];
                pred = NumOps.Add(pred, NumOps.Multiply(beta[j], vj));
            }
            double err = NumOps.ToDouble(data[i, variable]) - NumOps.ToDouble(pred);
            rss2 += err * err;
        }

        return n * Math.Log(rss2 / n + 1e-15) + (p + 1) * Math.Log(n);
    }
}
