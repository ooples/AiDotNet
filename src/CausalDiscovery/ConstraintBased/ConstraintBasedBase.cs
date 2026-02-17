using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// Base class for constraint-based causal discovery algorithms (PC, FCI, MMPC, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Constraint-based methods learn causal structure by performing conditional independence (CI) tests.
/// They start with a complete graph and remove edges between variables that are found to be
/// conditionally independent given some set of other variables.
/// </para>
/// <para>
/// <b>For Beginners:</b> These algorithms work by asking "Are variables X and Y still related
/// after we account for other variables?" If not, there's no direct causal link between them.
/// By systematically testing all variable pairs, the algorithm builds a causal graph.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ConstraintBasedBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.ConstraintBased;

    /// <summary>
    /// Significance level (alpha) for conditional independence tests.
    /// </summary>
    protected double Alpha { get; set; } = 0.05;

    /// <summary>
    /// Maximum size of conditioning sets to test.
    /// </summary>
    protected int MaxConditioningSetSize { get; set; } = 3;

    /// <summary>
    /// Applies options from CausalDiscoveryOptions.
    /// </summary>
    protected void ApplyConstraintOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.SignificanceLevel.HasValue) Alpha = Math.Max(0.001, Math.Min(0.5, options.SignificanceLevel.Value));
        if (options.MaxConditioningSetSize.HasValue) MaxConditioningSetSize = Math.Max(0, options.MaxConditioningSetSize.Value);
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
            sxy += d1 * d2;
            sxx += d1 * d1;
            syy += d2 * d2;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Tests conditional independence between variables i and j given conditioning set,
    /// using Fisher's z-transform of partial correlation.
    /// </summary>
    protected bool TestCI(Matrix<T> data, int i, int j, List<int> condSet, double alpha)
    {
        double partialCorr = ComputePartialCorr(data, i, j, condSet);
        int n = data.Rows;
        int dof = n - condSet.Count - 3;
        if (dof <= 0) return true; // insufficient data — cannot reject independence

        double clampedCorr = Math.Max(-0.999999, Math.Min(partialCorr, 0.999999));
        double z = Math.Sqrt(dof) * 0.5 * Math.Log((1 + clampedCorr) / (1 - clampedCorr));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
        return pValue > alpha;
    }

    /// <summary>
    /// Computes partial correlation between variables i and j given conditioning set.
    /// </summary>
    protected double ComputePartialCorr(Matrix<T> data, int i, int j, List<int> condSet)
    {
        if (condSet.Count == 0)
            return ComputeCorrelation(data, i, j);

        var residI = ComputeResiduals(data, i, condSet);
        var residJ = ComputeResiduals(data, j, condSet);

        int n = data.Rows;
        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++)
        {
            meanI += NumOps.ToDouble(residI[k]);
            meanJ += NumOps.ToDouble(residJ[k]);
        }
        meanI /= n; meanJ /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int k = 0; k < n; k++)
        {
            double di = NumOps.ToDouble(residI[k]) - meanI;
            double dj = NumOps.ToDouble(residJ[k]) - meanJ;
            sxy += di * dj;
            sxx += di * di;
            syy += dj * dj;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Computes residuals of column target after regressing on predictor columns.
    /// </summary>
    private Vector<T> ComputeResiduals(Matrix<T> data, int target, List<int> predictors)
    {
        int n = data.Rows;
        var y = data.GetColumn(target);

        if (predictors.Count == 0) return y;

        int p = predictors.Count;
        int dim = p + 1;
        var Z = new Matrix<T>(n, dim);
        for (int i = 0; i < n; i++)
        {
            Z[i, 0] = NumOps.One;
            for (int j = 0; j < p; j++)
                Z[i, j + 1] = data[i, predictors[j]];
        }

        // Build normal equations: ZtZ * beta = Zty
        var ZtZ = new Matrix<T>(dim, dim);
        var Zty = new Vector<T>(dim);
        for (int a = 0; a < dim; a++)
        {
            for (int b = 0; b < dim; b++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                    sum = NumOps.Add(sum, NumOps.Multiply(Z[k, a], Z[k, b]));
                ZtZ[a, b] = sum;
            }
            T sumY = NumOps.Zero;
            for (int k = 0; k < n; k++)
                sumY = NumOps.Add(sumY, NumOps.Multiply(Z[k, a], y[k]));
            Zty[a] = sumY;
        }

        T ridge = NumOps.FromDouble(1e-6);
        for (int a = 0; a < dim; a++)
            ZtZ[a, a] = NumOps.Add(ZtZ[a, a], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(ZtZ, Zty, MatrixDecompositionType.Lu);

        var residuals = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < dim; j++)
                pred = NumOps.Add(pred, NumOps.Multiply(beta[j], Z[i, j]));
            residuals[i] = NumOps.Subtract(y[i], pred);
        }

        return residuals;
    }

    /// <summary>
    /// Standard normal CDF approximation.
    /// </summary>
    protected static double NormalCDF(double x)
    {
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x) / Math.Sqrt(2);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * y);
    }

    /// <summary>
    /// Generates all combinations of size k from the given list.
    /// </summary>
    protected static IEnumerable<List<int>> GetCombinations(List<int> items, int size)
    {
        if (size == 0) { yield return []; yield break; }
        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var rest in GetCombinations(items.Skip(i + 1).ToList(), size - 1))
            {
                var combination = new List<int> { items[i] };
                combination.AddRange(rest);
                yield return combination;
            }
        }
    }
}
