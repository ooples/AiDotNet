using AiDotNet.Enums;

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
        if (options.SignificanceLevel.HasValue) Alpha = options.SignificanceLevel.Value;
        if (options.MaxConditioningSetSize.HasValue) MaxConditioningSetSize = options.MaxConditioningSetSize.Value;
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
            double d1 = X[i, col1] - mean1;
            double d2 = X[i, col2] - mean2;
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
    protected bool TestCI(double[,] X, int n, int i, int j, List<int> condSet, double alpha)
    {
        double partialCorr = ComputePartialCorr(X, n, i, j, condSet);
        int dof = n - condSet.Count - 3;
        if (dof <= 0) return false;

        double clampedCorr = Math.Max(-0.999999, Math.Min(partialCorr, 0.999999));
        double z = Math.Sqrt(dof) * 0.5 * Math.Log((1 + clampedCorr) / (1 - clampedCorr));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
        return pValue > alpha;
    }

    /// <summary>
    /// Computes partial correlation between variables i and j given conditioning set.
    /// </summary>
    protected static double ComputePartialCorr(double[,] X, int n, int i, int j, List<int> condSet)
    {
        if (condSet.Count == 0)
            return ComputeCorrelation(X, n, i, j);

        var residI = ComputeResiduals(X, n, i, condSet);
        var residJ = ComputeResiduals(X, n, j, condSet);

        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++) { meanI += residI[k]; meanJ += residJ[k]; }
        meanI /= n; meanJ /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int k = 0; k < n; k++)
        {
            double di = residI[k] - meanI;
            double dj = residJ[k] - meanJ;
            sxy += di * dj;
            sxx += di * di;
            syy += dj * dj;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Computes residuals of column target after regressing on predictor columns.
    /// </summary>
    private static double[] ComputeResiduals(double[,] X, int n, int target, List<int> predictors)
    {
        var residuals = new double[n];
        for (int i = 0; i < n; i++) residuals[i] = X[i, target];

        if (predictors.Count == 0) return residuals;

        int p = predictors.Count;
        var Z = new double[n, p + 1];
        for (int i = 0; i < n; i++)
        {
            Z[i, 0] = 1.0;
            for (int j = 0; j < p; j++)
                Z[i, j + 1] = X[i, predictors[j]];
        }

        var beta = SolveOLS(Z, residuals, n, p + 1);
        for (int i = 0; i < n; i++)
        {
            double predicted = beta[0];
            for (int j = 0; j < p; j++)
                predicted += beta[j + 1] * X[i, predictors[j]];
            residuals[i] -= predicted;
        }

        return residuals;
    }

    private static double[] SolveOLS(double[,] Z, double[] y, int n, int p)
    {
        var ZtZ = new double[p, p];
        var Zty = new double[p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++) sum += Z[k, i] * Z[k, j];
                ZtZ[i, j] = sum;
            }
            double sumY = 0;
            for (int k = 0; k < n; k++) sumY += Z[k, i] * y[k];
            Zty[i] = sumY;
        }

        for (int i = 0; i < p; i++) ZtZ[i, i] += 1e-6;

        var aug = new double[p, p + 1];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++) aug[i, j] = ZtZ[i, j];
            aug[i, p] = Zty[i];
        }

        for (int col = 0; col < p; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < p; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;
            for (int j = 0; j <= p; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            if (Math.Abs(aug[col, col]) < 1e-10) continue;
            for (int row = col + 1; row < p; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= p; j++) aug[row, j] -= factor * aug[col, j];
            }
        }

        var beta = new double[p];
        for (int i = p - 1; i >= 0; i--)
        {
            beta[i] = aug[i, p];
            for (int j = i + 1; j < p; j++) beta[i] -= aug[i, j] * beta[j];
            beta[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return beta;
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
