using AiDotNet.Distributions;
using AiDotNet.Helpers;

namespace AiDotNet.Scoring;

/// <summary>
/// Continuous Ranked Probability Score (CRPS) scoring rule.
/// </summary>
/// <remarks>
/// <para>
/// CRPS measures the integral of the squared difference between the predicted
/// cumulative distribution function (CDF) and the empirical CDF of the observation.
/// It generalizes the mean absolute error to probabilistic predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> CRPS rewards predictions that are both accurate (close to
/// the true value) and confident (narrow distributions). Unlike log score, CRPS:
/// - Has the same units as the predicted variable (like MAE)
/// - Is robust to outliers
/// - Considers the full shape of the distribution, not just probability at one point
///
/// For point forecasts, CRPS reduces to mean absolute error.
/// </para>
/// <para>
/// CRPS = ∫ (F(x) - 1(x ≥ y))² dx
/// where F is the predicted CDF and y is the observation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CRPSScore<T> : ScoringRuleBase<T>
{
    private readonly int _numIntegrationPoints;

    /// <summary>
    /// Initializes a new CRPS scoring rule.
    /// </summary>
    /// <param name="numIntegrationPoints">Number of points for numerical integration (default: 100).</param>
    public CRPSScore(int numIntegrationPoints = 100)
    {
        if (numIntegrationPoints < 10)
            throw new ArgumentOutOfRangeException(nameof(numIntegrationPoints), "At least 10 integration points required.");

        _numIntegrationPoints = numIntegrationPoints;
    }

    /// <inheritdoc/>
    public override string Name => "CRPS";

    /// <inheritdoc/>
    public override bool IsMinimized => true;  // Lower is better

    /// <inheritdoc/>
    public override T Score(IParametricDistribution<T> distribution, T observation)
    {
        // For Normal distribution, there's a closed-form solution
        if (distribution is NormalDistribution<T> normal)
        {
            return ScoreNormal(normal, observation);
        }

        // For Laplace distribution, there's also a closed-form solution
        if (distribution is LaplaceDistribution<T> laplace)
        {
            return ScoreLaplace(laplace, observation);
        }

        // Fall back to numerical integration for other distributions
        return ScoreNumerical(distribution, observation);
    }

    /// <summary>
    /// Closed-form CRPS for Normal distribution.
    /// </summary>
    private T ScoreNormal(NormalDistribution<T> distribution, T observation)
    {
        double y = NumOps.ToDouble(observation);
        double mu = NumOps.ToDouble(distribution.Mean);
        double sigma = Math.Sqrt(NumOps.ToDouble(distribution.Variance));

        double z = (y - mu) / sigma;
        double phi = StandardNormalPdf(z);
        double Phi = StandardNormalCdf(z);

        // CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
        double crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / Math.Sqrt(Math.PI));

        return NumOps.FromDouble(crps);
    }

    /// <summary>
    /// Closed-form CRPS for Laplace distribution.
    /// </summary>
    private T ScoreLaplace(LaplaceDistribution<T> distribution, T observation)
    {
        double y = NumOps.ToDouble(observation);
        double mu = NumOps.ToDouble(distribution.Location);
        double b = NumOps.ToDouble(distribution.Scale);

        double absDiff = Math.Abs(y - mu);
        double expTerm = Math.Exp(-absDiff / b);

        // CRPS = |y - μ| + b * exp(-|y - μ|/b) - 3b/4
        double crps = absDiff + b * expTerm - 0.75 * b;

        return NumOps.FromDouble(crps);
    }

    /// <summary>
    /// Numerical CRPS computation using integration.
    /// </summary>
    private T ScoreNumerical(IParametricDistribution<T> distribution, T observation)
    {
        double y = NumOps.ToDouble(observation);
        double mean = NumOps.ToDouble(distribution.Mean);
        double stdDev = Math.Sqrt(NumOps.ToDouble(distribution.Variance));

        // Integration bounds: mean ± 6 standard deviations
        double lower = mean - 6 * stdDev;
        double upper = mean + 6 * stdDev;
        double dx = (upper - lower) / _numIntegrationPoints;

        double crps = 0;
        for (int i = 0; i < _numIntegrationPoints; i++)
        {
            double x = lower + (i + 0.5) * dx;
            double cdf = NumOps.ToDouble(distribution.Cdf(NumOps.FromDouble(x)));
            double indicator = x >= y ? 1.0 : 0.0;
            double diff = cdf - indicator;
            crps += diff * diff * dx;
        }

        return NumOps.FromDouble(crps);
    }

    /// <inheritdoc/>
    public override T[] ScoreGradient(IParametricDistribution<T> distribution, T observation)
    {
        // For Normal distribution, use analytical gradients
        if (distribution is NormalDistribution<T> normal)
        {
            return ScoreGradientNormal(normal, observation);
        }

        // For other distributions, use numerical differentiation
        return ScoreGradientNumerical(distribution, observation);
    }

    /// <summary>
    /// Analytical CRPS gradient for Normal distribution.
    /// </summary>
    private T[] ScoreGradientNormal(NormalDistribution<T> distribution, T observation)
    {
        double y = NumOps.ToDouble(observation);
        double mu = NumOps.ToDouble(distribution.Mean);
        double variance = NumOps.ToDouble(distribution.Variance);
        double sigma = Math.Sqrt(variance);

        double z = (y - mu) / sigma;
        double phi = StandardNormalPdf(z);
        double Phi = StandardNormalCdf(z);

        // d(CRPS)/d(μ) = -(2Φ(z) - 1)
        double gradMean = -(2 * Phi - 1);

        // d(CRPS)/d(σ²) = [z * (2Φ(z) - 1) + 2φ(z) - 1/√π] / (2σ)
        double gradVariance = (z * (2 * Phi - 1) + 2 * phi - 1 / Math.Sqrt(Math.PI)) / (2 * sigma);

        return [NumOps.FromDouble(gradMean), NumOps.FromDouble(gradVariance)];
    }

    /// <summary>
    /// Numerical CRPS gradient computation.
    /// </summary>
    private T[] ScoreGradientNumerical(IParametricDistribution<T> distribution, T observation)
    {
        const double epsilon = 1e-6;
        T[] parameters = distribution.Parameters;
        T[] gradients = new T[parameters.Length];

        T baseScore = Score(distribution, observation);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Perturb parameter
            T[] perturbedParams = (T[])parameters.Clone();
            double paramVal = NumOps.ToDouble(perturbedParams[i]);
            double perturbedVal = paramVal + epsilon;
            perturbedParams[i] = NumOps.FromDouble(perturbedVal);

            // Compute score with perturbed parameter
            var perturbedDist = distribution.Clone();
            perturbedDist.Parameters = perturbedParams;
            T perturbedScore = Score(perturbedDist, observation);

            // Finite difference gradient
            double grad = (NumOps.ToDouble(perturbedScore) - NumOps.ToDouble(baseScore)) / epsilon;
            gradients[i] = NumOps.FromDouble(grad);
        }

        return gradients;
    }

    private static double StandardNormalPdf(double z)
    {
        return Math.Exp(-0.5 * z * z) / Math.Sqrt(2 * Math.PI);
    }

    private static double StandardNormalCdf(double z)
    {
        return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
    }

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);

        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        double t = 1.0 / (1.0 + p * x);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        double y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.Exp(-x * x);
        return sign * y;
    }
}
