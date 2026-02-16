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
        int n = x.Length;
        T mean = NumOps.Zero;
        for (int i = 0; i < n; i++) mean = NumOps.Add(mean, x[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(n));

        T m2 = NumOps.Zero, m4 = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T d = NumOps.Subtract(x[i], mean);
            T d2 = NumOps.Multiply(d, d);
            m2 = NumOps.Add(m2, d2);
            m4 = NumOps.Add(m4, NumOps.Multiply(d2, d2));
        }

        T nT = NumOps.FromDouble(n);
        m2 = NumOps.Divide(m2, nT);
        m4 = NumOps.Divide(m4, nT);
        T m2Sq = NumOps.Multiply(m2, m2);

        double m2Sq_d = NumOps.ToDouble(m2Sq);
        return m2Sq_d > 1e-15 ? NumOps.ToDouble(m4) / m2Sq_d : 3.0;
    }

    /// <summary>
    /// Standardizes data to zero mean and unit variance using generic Matrix operations.
    /// </summary>
    protected Matrix<T> StandardizeData(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var result = new Matrix<T>(n, d);
        T nT = NumOps.FromDouble(n);

        for (int j = 0; j < d; j++)
        {
            T mean = NumOps.Zero;
            for (int i = 0; i < n; i++) mean = NumOps.Add(mean, data[i, j]);
            mean = NumOps.Divide(mean, nT);

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, nT);
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-15)));

            for (int i = 0; i < n; i++)
                result[i, j] = NumOps.Divide(NumOps.Subtract(data[i, j], mean), std);
        }

        return result;
    }

    /// <summary>
    /// Computes residuals of y after regressing on x using generic Vector operations.
    /// </summary>
    protected Vector<T> RegressOut(Vector<T> y, Vector<T> x)
    {
        int n = x.Length;
        T nT = NumOps.FromDouble(n);
        T mx = NumOps.Zero, my = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            mx = NumOps.Add(mx, x[i]);
            my = NumOps.Add(my, y[i]);
        }
        mx = NumOps.Divide(mx, nT);
        my = NumOps.Divide(my, nT);

        T sxy = NumOps.Zero, sxx = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T dx = NumOps.Subtract(x[i], mx);
            sxy = NumOps.Add(sxy, NumOps.Multiply(dx, NumOps.Subtract(y[i], my)));
            sxx = NumOps.Add(sxx, NumOps.Multiply(dx, dx));
        }

        T beta = NumOps.ToDouble(sxx) > 1e-10
            ? NumOps.Divide(sxy, sxx)
            : NumOps.Zero;

        var residuals = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            residuals[i] = NumOps.Subtract(
                NumOps.Subtract(y[i], my),
                NumOps.Multiply(beta, NumOps.Subtract(x[i], mx)));

        return residuals;
    }

    /// <summary>
    /// Computes mutual information (Gaussian approximation) between two vectors.
    /// Returns a double since MI is a statistical metric used for comparisons.
    /// </summary>
    protected double GaussianMI(Vector<T> x, Vector<T> y)
    {
        int n = x.Length;
        T nT = NumOps.FromDouble(n);
        T mx = NumOps.Zero, my = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            mx = NumOps.Add(mx, x[i]);
            my = NumOps.Add(my, y[i]);
        }
        mx = NumOps.Divide(mx, nT);
        my = NumOps.Divide(my, nT);

        T sxx = NumOps.Zero, syy = NumOps.Zero, sxy = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T dx = NumOps.Subtract(x[i], mx);
            T dy = NumOps.Subtract(y[i], my);
            sxx = NumOps.Add(sxx, NumOps.Multiply(dx, dx));
            syy = NumOps.Add(syy, NumOps.Multiply(dy, dy));
            sxy = NumOps.Add(sxy, NumOps.Multiply(dx, dy));
        }

        sxx = NumOps.Divide(sxx, nT);
        syy = NumOps.Divide(syy, nT);
        sxy = NumOps.Divide(sxy, nT);

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
        int n = x.Length;
        T nT = NumOps.FromDouble(n);
        T mx = NumOps.Zero, my = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            mx = NumOps.Add(mx, x[i]);
            my = NumOps.Add(my, y[i]);
        }
        mx = NumOps.Divide(mx, nT);
        my = NumOps.Divide(my, nT);

        T sxy = NumOps.Zero, sxx = NumOps.Zero, syy = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T dx = NumOps.Subtract(x[i], mx);
            T dy = NumOps.Subtract(y[i], my);
            sxy = NumOps.Add(sxy, NumOps.Multiply(dx, dy));
            sxx = NumOps.Add(sxx, NumOps.Multiply(dx, dx));
            syy = NumOps.Add(syy, NumOps.Multiply(dy, dy));
        }

        double sxx_d = NumOps.ToDouble(sxx), syy_d = NumOps.ToDouble(syy);
        return (sxx_d > 1e-10 && syy_d > 1e-10)
            ? NumOps.ToDouble(sxy) / Math.Sqrt(sxx_d * syy_d) : 0;
    }

    /// <summary>
    /// Computes column variance from a Matrix using generic operations.
    /// </summary>
    protected T ComputeColumnVariance(Matrix<T> data, int col)
    {
        int n = data.Rows;
        T nT = NumOps.FromDouble(n);
        T mean = NumOps.Zero;
        for (int i = 0; i < n; i++) mean = NumOps.Add(mean, data[i, col]);
        mean = NumOps.Divide(mean, nT);

        T variance = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T d = NumOps.Subtract(data[i, col], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(d, d));
        }

        return NumOps.Divide(variance, nT);
    }

    /// <summary>
    /// Nadaraya–Watson kernel smoother for a single predictor column against a response vector.
    /// Uses Gaussian kernel with Scott's rule bandwidth.
    /// </summary>
    protected Vector<T> KernelSmooth(Matrix<T> data, int predictorCol, Vector<T> response)
    {
        int n = data.Rows;

        // Scott's rule: h = sigma * n^(-1/5)
        double mean = 0;
        for (int i = 0; i < n; i++) mean += NumOps.ToDouble(data[i, predictorCol]);
        mean /= n;
        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(data[i, predictorCol]) - mean;
            variance += diff * diff;
        }
        double sigma = Math.Max(Math.Sqrt(variance / n), 1e-10);
        T h = NumOps.FromDouble(sigma * Math.Pow(n, -1.0 / 5.0));
        var smoothed = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T xi = data[i, predictorCol];
            T numerator = NumOps.Zero;
            T denominator = NumOps.Zero;

            for (int j = 0; j < n; j++)
            {
                T diff = NumOps.Divide(NumOps.Subtract(xi, data[j, predictorCol]), h);
                double diffD = NumOps.ToDouble(diff);
                T kernel = NumOps.FromDouble(Math.Exp(-0.5 * diffD * diffD));
                numerator = NumOps.Add(numerator, NumOps.Multiply(kernel, response[j]));
                denominator = NumOps.Add(denominator, kernel);
            }

            smoothed[i] = NumOps.ToDouble(denominator) > 1e-15
                ? NumOps.Divide(numerator, denominator)
                : NumOps.Zero;
        }

        return smoothed;
    }

    /// <summary>
    /// Kernel regression residuals: fits response = f(predictor) + ε using Nadaraya–Watson,
    /// returns the residual vector ε.
    /// </summary>
    protected Vector<T> KernelRegressOut(Vector<T> predictor, Vector<T> response)
    {
        int n = predictor.Length;

        // Scott's rule: h = sigma * n^(-1/5)
        double mean = 0;
        for (int i = 0; i < n; i++) mean += NumOps.ToDouble(predictor[i]);
        mean /= n;
        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(predictor[i]) - mean;
            variance += diff * diff;
        }
        double sigma = Math.Max(Math.Sqrt(variance / n), 1e-10);
        double h = sigma * Math.Pow(n, -1.0 / 5.0);
        var residuals = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T xi = predictor[i];
            T numerator = NumOps.Zero;
            T denominator = NumOps.Zero;

            for (int j = 0; j < n; j++)
            {
                double diffD = NumOps.ToDouble(NumOps.Subtract(xi, predictor[j])) / h;
                T kernel = NumOps.FromDouble(Math.Exp(-0.5 * diffD * diffD));
                numerator = NumOps.Add(numerator, NumOps.Multiply(kernel, response[j]));
                denominator = NumOps.Add(denominator, kernel);
            }

            T predicted = NumOps.ToDouble(denominator) > 1e-15
                ? NumOps.Divide(numerator, denominator)
                : NumOps.Zero;
            residuals[i] = NumOps.Subtract(response[i], predicted);
        }

        return residuals;
    }
}
