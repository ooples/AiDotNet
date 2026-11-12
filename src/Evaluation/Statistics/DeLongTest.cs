using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// DeLong's test for comparing two ROC curves.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DeLong's test compares the AUC of two classifiers:
/// <list type="bullet">
/// <item>Tests if the difference in AUC is statistically significant</item>
/// <item>Specifically designed for comparing ROC curves</item>
/// <item>Non-parametric and doesn't assume normal distribution</item>
/// <item>Can handle correlated data (same test set for both models)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Comparing two binary classifiers on the same dataset</item>
/// <item>When you want to know if AUC improvement is significant</item>
/// <item>Medical diagnostic test comparisons</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DeLongTest<T> : IStatisticalTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public string Name => "DeLong Test";
    /// <inheritdoc />
    public string Description => "Compares two ROC curves to test if AUC difference is significant.";
    /// <inheritdoc />
    public bool IsPaired => true;
    /// <inheritdoc />
    public bool IsNonParametric => true;

    /// <summary>
    /// Performs DeLong's test comparing two sets of predicted probabilities.
    /// </summary>
    /// <param name="predictions1">Probabilities from first classifier.</param>
    /// <param name="predictions2">Probabilities from second classifier.</param>
    /// <param name="actuals">True binary labels.</param>
    /// <returns>Statistical test result with p-value.</returns>
    public StatisticalTestResult<T> Test(T[] predictions1, T[] predictions2, T[] actuals)
    {
        if (predictions1.Length != predictions2.Length || predictions1.Length != actuals.Length)
            throw new ArgumentException("All arrays must have the same length.");

        int n = predictions1.Length;

        // Separate positives and negatives
        using var noGrad = new NoGradScope<T>();
        var positiveMask = EvaluationTensorOps<T>.EqualMask(new Tensor<T>(actuals, new[] { n }), NumOps.One);
        var negativeMask = EvaluationTensorOps<T>.Not(positiveMask);

        int nPos = EvaluationTensorOps<T>.Count(positiveMask);
        int nNeg = n - nPos;

        if (nPos == 0 || nNeg == 0)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                EffectSize = NumOps.Zero,
                Description = "Cannot compute: need both positive and negative samples."
            };
        }

        // Reduce pair kernels in bounded blocks; retain only the per-positive and per-negative components.
        var (auc1, V10_1, V01_1) = ComputeComponents(predictions1, positiveMask, negativeMask);
        var (auc2, V10_2, V01_2) = ComputeComponents(predictions2, positiveMask, negativeMask);

        // Compute variance-covariance matrix
        double s10_11 = Covariance(V10_1, V10_1);
        double s10_22 = Covariance(V10_2, V10_2);
        double s10_12 = Covariance(V10_1, V10_2);

        double s01_11 = Covariance(V01_1, V01_1);
        double s01_22 = Covariance(V01_2, V01_2);
        double s01_12 = Covariance(V01_1, V01_2);

        double var1 = s10_11 / nPos + s01_11 / nNeg;
        double var2 = s10_22 / nPos + s01_22 / nNeg;
        double covar = s10_12 / nPos + s01_12 / nNeg;

        double varDiff = var1 + var2 - 2 * covar;
        if (varDiff < 1e-10)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                EffectSize = NumOps.FromDouble(auc1 - auc2),
                Description = "Variance is zero; cannot compute test statistic."
            };
        }

        double z = (auc1 - auc2) / Math.Sqrt(varDiff);
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(z),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < 0.05,
            EffectSize = NumOps.FromDouble(auc1 - auc2),
            Interpretation = pValue < 0.05
                ? $"Significant difference in AUC (AUC1={auc1:F4}, AUC2={auc2:F4}, Δ={auc1 - auc2:F4})"
                : $"No significant difference in AUC (AUC1={auc1:F4}, AUC2={auc2:F4})",
            Description = "DeLong's test for comparing AUC of two ROC curves."
        };
    }

    // psi(pos, neg) = 1 when pos > neg, else 0.5 when |pos - neg| < 1e-10, else 0.
    private static (double Auc, Tensor<double> V10, Tensor<double> V01) ComputeComponents(
        T[] predictions, Tensor<T> positiveMask, Tensor<T> negativeMask)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<double>();
        var scores = new Tensor<T>(predictions, new[] { predictions.Length });
        var pos = EvaluationTensorOps<T>.Select(scores, positiveMask).Cast<double>();
        var neg = EvaluationTensorOps<T>.Select(scores, negativeMask).Cast<double>();
        var rowSums = new Tensor<double>(new[] { pos.Length });
        var columnSums = new Tensor<double>(new[] { neg.Length });
        for (int row = 0; row < pos.Length; row += 256)
        {
            int rows = Math.Min(256, pos.Length - row);
            var rowTotal = new Tensor<double>(new[] { rows });
            var positives = engine.Reshape(engine.TensorNarrow(pos, 0, row, rows), new[] { rows, 1 });
            for (int column = 0; column < neg.Length; column += 256)
            {
                int columns = Math.Min(256, neg.Length - column);
                var negatives = engine.Reshape(engine.TensorNarrow(neg, 0, column, columns), new[] { 1, columns });
                var shape = new[] { rows, columns };
                var left = engine.TensorBroadcastTo(positives, shape);
                var right = engine.TensorBroadcastTo(negatives, shape);
                var above = engine.TensorGreaterThan(left, right);
                var tied = engine.TensorMultiply(EvaluationTensorOps<double>.Not(above),
                    engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtract(left, right)), 1e-10));
                var kernel = engine.TensorAdd(above, engine.TensorMultiplyScalar(tied, 0.5));
                rowTotal = engine.TensorAdd(rowTotal, engine.ReduceSum(kernel, new[] { 1 }, false));
                var columnTotal = engine.TensorAdd(engine.TensorNarrow(columnSums, 0, column, columns),
                    engine.ReduceSum(kernel, new[] { 0 }, false));
                columnTotal.AsSpan().CopyTo(columnSums.AsWritableSpan().Slice(column, columns));
            }
            rowTotal.AsSpan().CopyTo(rowSums.AsWritableSpan().Slice(row, rows));
        }
        double auc = engine.TensorSum(rowSums) / ((double)pos.Length * neg.Length);
        return (auc, engine.TensorDivideScalar(rowSums, (double)neg.Length),
            engine.TensorDivideScalar(columnSums, (double)pos.Length));
    }

    private static double Covariance(Tensor<double> x, Tensor<double> y)
    {
        if (x.Length != y.Length || x.Length < 2) return 0;
        var engine = AiDotNetEngine.Current;
        var dx = engine.TensorSubtractScalar(x, engine.TensorMean(x));
        var dy = engine.TensorSubtractScalar(y, engine.TensorMean(y));
        return engine.TensorSum(engine.TensorMultiply(dx, dy)) / (x.Length - 1);
    }

    private double NormalCDF(double z)
    {
        // Approximation of standard normal CDF
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(z));
        double d = 0.3989423 * Math.Exp(-z * z / 2);
        double prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return z > 0 ? 1 - prob : prob;
    }
}
