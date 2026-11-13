using AiDotNet.Enums;
using AiDotNet.Extensions;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// Base class for functional/ICA-based causal discovery algorithms (LiNGAM, ANM, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Functional methods exploit properties of the data-generating process to determine
/// causal direction. Unlike constraint-based methods (which only learn equivalence classes),
/// functional methods can often identify the unique causal DAG.
/// </para>
/// <para>
/// <b>Key assumption families:</b>
/// <list type="bullet">
/// <item><b>LiNGAM:</b> Linear model with non-Gaussian noise → uses ICA to identify structure</item>
/// <item><b>ANM:</b> Additive noise model Y = f(X) + N → exploits asymmetry in residuals</item>
/// <item><b>PNL:</b> Post-nonlinear model Y = g(f(X) + N) → generalizes ANM</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> These methods use clever mathematical properties to figure out
/// which variable causes which. For example, if X causes Y with some noise added, the
/// noise pattern looks different depending on which direction you assume the causation
/// goes. These methods detect that asymmetry.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class FunctionalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.Functional;

    /// <summary>
    /// Computes kurtosis (fourth standardized moment) of a data vector.
    /// Non-Gaussian distributions have kurtosis != 3.
    /// </summary>
    protected double ComputeKurtosis(Vector<T> x)
    {
        var d2 = Engine.TensorSquare(CenterVector(Tensor<T>.FromVector(x)));
        T m2 = Engine.TensorMean(d2);
        T m4 = Engine.TensorMean(Engine.TensorSquare(d2));
        T m2Sq = NumOps.Multiply(m2, m2);

        double m2Sq_d = NumOps.ToDouble(m2Sq);
        return m2Sq_d > 1e-15 ? NumOps.ToDouble(m4) / m2Sq_d : 3.0;
    }

    /// <summary>
    /// Standardizes data to zero mean and unit variance using generic Matrix operations.
    /// </summary>
    protected Matrix<T> StandardizeData(Matrix<T> data)
    {
        var centered = CenterColumns(Tensor<T>.FromMatrix(data));
        var variance = Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, true);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-15)));
        return Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, centered.Shape.ToArray())).ToMatrix();
    }

    /// <summary>
    /// Computes residuals of y after regressing on x using generic Vector operations.
    /// </summary>
    protected Vector<T> RegressOut(Vector<T> y, Vector<T> x)
    {
        var centX = CenterVector(Tensor<T>.FromVector(x));
        var centY = CenterVector(Tensor<T>.FromVector(y));
        T sxy = Engine.TensorSum(Engine.TensorMultiply(centX, centY));
        T sxx = Engine.TensorSum(Engine.TensorSquare(centX));

        T beta = NumOps.GreaterThan(sxx, NumOps.FromDouble(1e-10))
            ? NumOps.Divide(sxy, sxx)
            : NumOps.Zero;

        return Engine.TensorSubtract(centY, Engine.TensorMultiplyScalar(centX, beta)).ToVector();
    }

    /// <summary>
    /// Computes mutual information (Gaussian approximation) between two vectors.
    /// Returns a double since MI is a statistical metric used for comparisons.
    /// </summary>
    protected double GaussianMI(Vector<T> x, Vector<T> y)
    {
        var cX = CenterVector(Tensor<T>.FromVector(x));
        var cY = CenterVector(Tensor<T>.FromVector(y));
        T sxx = Engine.TensorMean(Engine.TensorSquare(cX));
        T syy = Engine.TensorMean(Engine.TensorSquare(cY));
        T sxy = Engine.TensorMean(Engine.TensorMultiply(cX, cY));

        double sxx_d = NumOps.ToDouble(sxx), syy_d = NumOps.ToDouble(syy), sxy_d = NumOps.ToDouble(sxy);
        double corr = (sxx_d > 1e-15 && syy_d > 1e-15) ? sxy_d / Math.Sqrt(sxx_d * syy_d) : 0;
        corr = Math.Max(-0.9999, Math.Min(0.9999, corr));
        return -0.5 * Math.Log(1 - corr * corr);
    }

    /// <summary>
    /// Computes Pearson correlation between two vectors.
    /// Returns double since it's a statistical metric.
    /// </summary>
    protected double ComputeCorrelation(Vector<T> x, Vector<T> y)
    {
        var dx = CenterVector(Tensor<T>.FromVector(x));
        var dy = CenterVector(Tensor<T>.FromVector(y));
        T sxy = Engine.TensorSum(Engine.TensorMultiply(dx, dy));
        T sxx = Engine.TensorSum(Engine.TensorSquare(dx));
        T syy = Engine.TensorSum(Engine.TensorSquare(dy));

        double sxx_d = NumOps.ToDouble(sxx), syy_d = NumOps.ToDouble(syy);
        return (sxx_d > 1e-10 && syy_d > 1e-10)
            ? NumOps.ToDouble(sxy) / Math.Sqrt(sxx_d * syy_d) : 0;
    }

    /// <summary>
    /// Computes column variance from a Matrix using generic operations.
    /// </summary>
    protected T ComputeColumnVariance(Matrix<T> data, int col)
    {
        var centered = CenterVector(Tensor<T>.FromVector(Engine.GetColumn(data, col)));
        return Engine.TensorMean(Engine.TensorSquare(centered));
    }

    /// <summary>
    /// Nadaraya–Watson kernel smoother for a single predictor column against a response vector.
    /// Uses Gaussian kernel with Scott's rule bandwidth.
    /// </summary>
    protected Vector<T> KernelSmooth(Matrix<T> data, int predictorCol, Vector<T> response)
    {
        var predictor = Tensor<T>.FromVector(Engine.GetColumn(data, predictorCol));
        return NadarayaWatson(predictor, Tensor<T>.FromVector(response)).ToVector();
    }

    /// <summary>
    /// Kernel regression residuals: fits response = f(predictor) + ε using Nadaraya–Watson,
    /// returns the residual vector ε.
    /// </summary>
    protected Vector<T> KernelRegressOut(Vector<T> predictor, Vector<T> response)
    {
        var y = Tensor<T>.FromVector(response);
        return Engine.TensorSubtract(y, NadarayaWatson(Tensor<T>.FromVector(predictor), y)).ToVector();
    }

    /// <summary>
    /// Subtracts the mean from a 1-D tensor.
    /// </summary>
    private protected Tensor<T> CenterVector(Tensor<T> x)
        => Engine.TensorSubtractScalar(x, Engine.TensorMean(x));

    /// <summary>
    /// Gaussian-kernel Nadaraya–Watson fit of <paramref name="response"/> on <paramref name="predictor"/>
    /// (both [n]) with Scott's rule bandwidth h = max(sigma, 1e-10) * n^(-1/5); an entry whose kernel
    /// weights sum to at most 1e-15 is 0.
    /// </summary>
    private protected Tensor<T> NadarayaWatson(Tensor<T> predictor, Tensor<T> response)
    {
        int n = predictor.Length;
        if (n == 0)
            return new Tensor<T>(new[] { 0 });

        T sigma = NumOps.Sqrt(Engine.TensorMean(Engine.TensorSquare(CenterVector(predictor))));
        T minSigma = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(sigma, minSigma))
            sigma = minSigma;
        T h = NumOps.Multiply(sigma, NumOps.FromDouble(Math.Pow(n, -1.0 / 5.0)));
        T minusHalf = NumOps.FromDouble(-0.5);
        T tiny = NumOps.FromDouble(1e-15);

        // Kernel rows are formed in blocks so the [rows x n] weight matrix stays bounded.
        var column = predictor.Reshape(new[] { n, 1 });
        var row = predictor.Reshape(new[] { 1, n });
        var y = response.Reshape(new[] { n, 1 });
        int block = Math.Max(1, Math.Min(n, (1 << 22) / Math.Max(1, n)));
        var parts = new List<Tensor<T>>();
        for (int start = 0; start < n; start += block)
        {
            int len = Math.Min(block, n - start);
            var xi = Engine.TensorSlice(column, new[] { start, 0 }, new[] { len, 1 });
            var diff = Engine.TensorDivideScalar(Engine.TensorSubtract(
                Engine.TensorBroadcastTo(xi, new[] { len, n }), Engine.TensorBroadcastTo(row, new[] { len, n })), h);
            var kernel = Engine.TensorExp(Engine.TensorMultiplyScalar(Engine.TensorSquare(diff), minusHalf));
            var numerator = Engine.TensorMatMul(kernel, y).Reshape(new[] { len });
            var denominator = Engine.ReduceSum(kernel, new[] { 1 }, false);
            var valid = Engine.TensorGreaterThan(denominator, tiny);
            var safe = Engine.TensorWhere(valid, denominator, Engine.TensorOnesLike(denominator));
            parts.Add(Engine.TensorWhere(valid, Engine.TensorDivide(numerator, safe), Engine.TensorZerosLike(numerator)));
        }

        return parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), 0);
    }
}
