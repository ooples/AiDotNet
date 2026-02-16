using AiDotNet.Enums;

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
    protected static double ComputeKurtosis(double[] x, int n)
    {
        double mean = 0;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;

        double m2 = 0, m4 = 0;
        for (int i = 0; i < n; i++)
        {
            double d = x[i] - mean;
            m2 += d * d;
            m4 += d * d * d * d;
        }

        m2 /= n;
        m4 /= n;
        return m2 > 1e-15 ? m4 / (m2 * m2) : 3.0;
    }

    /// <summary>
    /// Standardizes data to zero mean and unit variance.
    /// </summary>
    protected static double[,] StandardizeData(double[,] X, int n, int d)
    {
        var result = new double[n, d];
        for (int j = 0; j < d; j++)
        {
            double mean = 0, variance = 0;
            for (int i = 0; i < n; i++) mean += X[i, j];
            mean /= n;
            for (int i = 0; i < n; i++) variance += (X[i, j] - mean) * (X[i, j] - mean);
            variance /= n;
            double std = Math.Sqrt(variance + 1e-15);
            for (int i = 0; i < n; i++) result[i, j] = (X[i, j] - mean) / std;
        }

        return result;
    }

    /// <summary>
    /// Computes residuals of y after regressing on x.
    /// </summary>
    protected static double[] RegressOut(double[] y, double[] x, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;

        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx;
            sxy += dx * (y[i] - my);
            sxx += dx * dx;
        }

        double beta = sxx > 1e-10 ? sxy / sxx : 0;
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
            residuals[i] = y[i] - my - beta * (x[i] - mx);

        return residuals;
    }

    /// <summary>
    /// Computes mutual information (Gaussian approximation) between two variables.
    /// </summary>
    protected static double GaussianMI(double[] x, double[] y, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;

        double sxx = 0, syy = 0, sxy = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx, dy = y[i] - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }

        sxx /= n; syy /= n; sxy /= n;
        double corr = (sxx > 1e-15 && syy > 1e-15) ? sxy / Math.Sqrt(sxx * syy) : 0;
        corr = Math.Max(-0.9999, Math.Min(0.9999, corr));
        return -0.5 * Math.Log(1 - corr * corr);
    }

}
