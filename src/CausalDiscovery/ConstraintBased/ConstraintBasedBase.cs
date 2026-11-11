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
        var source = Tensor<T>.FromMatrix(data);
        var (sxyT, sxxT, syyT) = CenteredCrossProducts(
            SelectColumns(source, new[] { col1 }), SelectColumns(source, new[] { col2 }));
        double sxy = NumOps.ToDouble(sxyT), sxx = NumOps.ToDouble(sxxT), syy = NumOps.ToDouble(syyT);

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    /// <summary>
    /// Keeps the entries of <paramref name="weights"/> where <paramref name="mask"/> is set and
    /// zeroes the rest.
    /// </summary>
    /// <param name="mask">Row-major [d x d] edge mask.</param>
    /// <param name="weights">Edge weights [d x d].</param>
    private protected Matrix<T> ApplyEdgeMask(bool[] mask, Matrix<T> weights)
    {
        var values = Tensor<T>.FromMatrix(weights);
        return Engine.TensorWhere(mask, values, Engine.TensorZerosLike(values)).ToMatrix();
    }

    /// <summary>
    /// Sums of centered cross products (sxy, sxx, syy) of two [n x 1] columns.
    /// </summary>
    private (T Sxy, T Sxx, T Syy) CenteredCrossProducts(Tensor<T> x, Tensor<T> y)
    {
        var dx = CenterColumns(x);
        var dy = CenterColumns(y);
        return (Engine.TensorSum(Engine.TensorMultiply(dx, dy)),
            Engine.TensorSum(Engine.TensorSquare(dx)),
            Engine.TensorSum(Engine.TensorSquare(dy)));
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

        var (sxyT, sxxT, syyT) = CenteredCrossProducts(residI, residJ);
        double sxy = NumOps.ToDouble(sxyT), sxx = NumOps.ToDouble(sxxT), syy = NumOps.ToDouble(syyT);

        if (sxx < 1e-10 || syy < 1e-10)
        {
            // Near-zero residual variance means the conditioning set perfectly predicts
            // one or both variables. Partial correlation is undefined — return 0 (no evidence
            // of direct association beyond what the conditioning set explains).
            return 0;
        }

        return sxy / Math.Sqrt(sxx * syy);
    }

    /// <summary>
    /// Computes residuals of column target after regressing on predictor columns.
    /// </summary>
    private Tensor<T> ComputeResiduals(Matrix<T> data, int target, List<int> predictors)
    {
        var source = Tensor<T>.FromMatrix(data);
        var y = SelectColumns(source, new[] { target });

        if (predictors.Count == 0) return y;

        int dim = predictors.Count + 1;
        var Z = Engine.TensorConcatenate(
            new[] { Engine.TensorOnesLike(y), SelectColumns(source, predictors.ToArray()) }, 1);
        var Zt = Engine.TensorTranspose(Z);

        // Normal equations with ridge: (ZtZ + ridge*I) * beta = Zty
        var ZtZ = Engine.TensorAdd(Engine.TensorMatMul(Zt, Z),
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(dim), NumOps.FromDouble(1e-6)));
        var Zty = Engine.TensorMatMul(Zt, y).Reshape(new[] { dim }).ToVector();

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(ZtZ.ToMatrix(), Zty, MatrixDecompositionType.Lu);

        var prediction = Engine.TensorMatMul(Z, Tensor<T>.FromVector(beta).Reshape(new[] { dim, 1 }));
        return Engine.TensorSubtract(y, prediction);
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
