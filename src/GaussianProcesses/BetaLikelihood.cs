namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Beta Likelihood for Gaussian Processes with bounded outputs in [0, 1].
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard GP regression assumes Gaussian noise: y ~ N(f(x), σ²)
/// But what if your outputs are bounded between 0 and 1? (e.g., proportions, probabilities)
///
/// The Beta likelihood handles this:
/// 1. GP models a latent function f(x) (unbounded)
/// 2. Sigmoid transformation: μ = sigmoid(f) ∈ (0, 1)
/// 3. Beta distribution: y ~ Beta(μ × ν, (1-μ) × ν)
///
/// Where:
/// - μ is the mean of the Beta (determined by sigmoid(f))
/// - ν is the "precision" (higher = less variance)
///
/// This is useful for:
/// - Modeling proportions (e.g., click-through rates)
/// - Modeling probabilities
/// - Any response bounded in [0, 1]
///
/// The Beta distribution naturally handles the bounded nature:
/// - Values near 0 or 1 have appropriately skewed distributions
/// - Variance is heteroscedastic (depends on mean)
/// </para>
/// </remarks>
public class BetaLikelihood<T>
{
    /// <summary>
    /// Precision parameter ν.
    /// </summary>
    private readonly double _precision;

    /// <summary>
    /// Small constant for numerical stability.
    /// </summary>
    private const double Epsilon = 1e-10;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a Beta Likelihood.
    /// </summary>
    /// <param name="precision">Precision parameter ν (default 10). Higher = less variance.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The precision controls how "noisy" the observations are.
    ///
    /// For Beta(α, β) with mean μ:
    /// - α = μ × ν
    /// - β = (1-μ) × ν
    /// - Variance = μ(1-μ)/(ν+1)
    ///
    /// Examples:
    /// - ν = 2: High variance (very noisy data)
    /// - ν = 10: Moderate variance
    /// - ν = 100: Low variance (precise observations)
    ///
    /// Choose based on your data's inherent noise level.
    /// </para>
    /// </remarks>
    public BetaLikelihood(double precision = 10.0)
    {
        if (precision <= 0)
            throw new ArgumentException("Precision must be positive.", nameof(precision));

        _precision = precision;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the precision parameter.
    /// </summary>
    public double Precision => _precision;

    /// <summary>
    /// Transforms latent function values to Beta means via sigmoid.
    /// </summary>
    /// <param name="f">Latent function values.</param>
    /// <returns>Mean parameters μ ∈ (0, 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sigmoid function squashes any real number to (0, 1):
    /// μ = 1 / (1 + exp(-f))
    ///
    /// Properties:
    /// - f = 0 → μ = 0.5
    /// - f → +∞ → μ → 1
    /// - f → -∞ → μ → 0
    /// </para>
    /// </remarks>
    public Vector<T> GetMeans(Vector<T> f)
    {
        var mu = new Vector<T>(f.Length);
        for (int i = 0; i < f.Length; i++)
        {
            double fi = _numOps.ToDouble(f[i]);
            double mui = Sigmoid(fi);
            mu[i] = _numOps.FromDouble(mui);
        }
        return mu;
    }

    /// <summary>
    /// Computes Beta parameters (α, β) from mean μ.
    /// </summary>
    /// <param name="mu">Mean parameter ∈ (0, 1).</param>
    /// <returns>Tuple (α, β) for Beta distribution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given mean μ and precision ν:
    /// - α = μ × ν (shape parameter for values near 1)
    /// - β = (1 - μ) × ν (shape parameter for values near 0)
    ///
    /// This parameterization ensures E[Y] = μ.
    /// </para>
    /// </remarks>
    public (double Alpha, double Beta) GetBetaParameters(double mu)
    {
        mu = Math.Max(Epsilon, Math.Min(1 - Epsilon, mu));
        double alpha = mu * _precision;
        double beta = (1 - mu) * _precision;
        return (alpha, beta);
    }

    /// <summary>
    /// Computes log-likelihood of observations given latent values.
    /// </summary>
    /// <param name="y">Observations in [0, 1].</param>
    /// <param name="f">Latent function values.</param>
    /// <returns>Log-likelihood.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes how likely the observations are given the latent function.
    ///
    /// log p(y | f) = sum_i log Beta(y_i; α_i, β_i)
    ///
    /// Where α_i, β_i come from sigmoid(f_i).
    /// </para>
    /// </remarks>
    public double LogLikelihood(Vector<T> y, Vector<T> f)
    {
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have same length.");

        double logLik = 0;

        for (int i = 0; i < y.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = _numOps.ToDouble(f[i]);

            // Clamp y to valid range
            yi = Math.Max(Epsilon, Math.Min(1 - Epsilon, yi));

            double mu = Sigmoid(fi);
            var (alpha, beta) = GetBetaParameters(mu);

            // Log Beta PDF: log Γ(α+β) - log Γ(α) - log Γ(β) + (α-1)log(y) + (β-1)log(1-y)
            logLik += LogGamma(alpha + beta) - LogGamma(alpha) - LogGamma(beta);
            logLik += (alpha - 1) * Math.Log(yi);
            logLik += (beta - 1) * Math.Log(1 - yi);
        }

        return logLik;
    }

    /// <summary>
    /// Computes gradient of log-likelihood with respect to f.
    /// </summary>
    /// <param name="y">Observations in [0, 1].</param>
    /// <param name="f">Latent function values.</param>
    /// <returns>Gradient ∂ log p(y|f) / ∂f.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient is needed for optimizing the latent function.
    ///
    /// Uses chain rule: ∂/∂f = ∂/∂μ × ∂μ/∂f
    /// Where ∂μ/∂f = μ(1-μ) (sigmoid derivative).
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodGradient(Vector<T> y, Vector<T> f)
    {
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have same length.");

        var grad = new Vector<T>(f.Length);

        for (int i = 0; i < f.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = _numOps.ToDouble(f[i]);

            yi = Math.Max(Epsilon, Math.Min(1 - Epsilon, yi));

            double mu = Sigmoid(fi);
            var (alpha, beta) = GetBetaParameters(mu);

            // ∂log p / ∂μ
            double dLogP_dMu = _precision * (Digamma(beta) - Digamma(alpha));
            dLogP_dMu += _precision * Math.Log(yi);
            dLogP_dMu -= _precision * Math.Log(1 - yi);

            // ∂μ / ∂f = μ(1-μ)
            double dMu_dF = mu * (1 - mu);

            grad[i] = _numOps.FromDouble(dLogP_dMu * dMu_dF);
        }

        return grad;
    }

    /// <summary>
    /// Computes Hessian (second derivative) of log-likelihood.
    /// </summary>
    /// <param name="y">Observations in [0, 1].</param>
    /// <param name="f">Latent function values.</param>
    /// <returns>Diagonal of Hessian (assuming diagonal approximation).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hessian is the matrix of second derivatives.
    /// It's used in Laplace approximation for approximate inference.
    ///
    /// We return only the diagonal (assuming observations are conditionally independent).
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodHessianDiag(Vector<T> y, Vector<T> f)
    {
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have same length.");

        var hess = new Vector<T>(f.Length);

        for (int i = 0; i < f.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = _numOps.ToDouble(f[i]);

            yi = Math.Max(Epsilon, Math.Min(1 - Epsilon, yi));

            double mu = Sigmoid(fi);
            var (alpha, beta) = GetBetaParameters(mu);

            // Derivatives of sigmoid: μ = sigmoid(f)
            double dMu_dF = mu * (1 - mu);           // ∂μ/∂f
            double d2Mu_dF2 = dMu_dF * (1 - 2 * mu); // ∂²μ/∂f²

            // First derivative: ∂logP/∂μ (needed for chain rule)
            double dLogP_dMu = _precision * (Digamma(beta) - Digamma(alpha));
            dLogP_dMu += _precision * Math.Log(yi);
            dLogP_dMu -= _precision * Math.Log(1 - yi);

            // Second derivative: ∂²logP/∂μ² (from trigamma functions)
            double d2LogP_dMu2 = -_precision * _precision * (Trigamma(alpha) + Trigamma(beta));

            // Full chain rule for second derivative:
            // ∂²logP/∂f² = (∂²logP/∂μ²)(∂μ/∂f)² + (∂logP/∂μ)(∂²μ/∂f²)
            double d2 = d2LogP_dMu2 * dMu_dF * dMu_dF + dLogP_dMu * d2Mu_dF2;

            hess[i] = _numOps.FromDouble(d2);
        }

        return hess;
    }

    /// <summary>
    /// Samples from the Beta distribution given mean.
    /// </summary>
    /// <param name="mu">Mean parameter.</param>
    /// <param name="random">Random generator.</param>
    /// <returns>Sample from Beta(μν, (1-μ)ν).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generates a random sample from the Beta distribution.
    /// Useful for predictive sampling and uncertainty quantification.
    /// </para>
    /// </remarks>
    public double Sample(double mu, Random random)
    {
        mu = Math.Max(Epsilon, Math.Min(1 - Epsilon, mu));
        var (alpha, beta) = GetBetaParameters(mu);
        return SampleBeta(alpha, beta, random);
    }

    /// <summary>
    /// Computes predictive mean and variance for a new point.
    /// </summary>
    /// <param name="fMean">Predictive mean of latent function.</param>
    /// <param name="fVariance">Predictive variance of latent function.</param>
    /// <returns>Tuple of (mean, variance) for observed variable.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given uncertainty in f, computes uncertainty in y.
    ///
    /// Uses moment matching:
    /// - E[y] ≈ sigmoid(f_mean) with correction for variance
    /// - Var[y] ≈ propagated uncertainty + Beta variance
    ///
    /// This is an approximation since the exact integral is intractable.
    /// </para>
    /// </remarks>
    public (double Mean, double Variance) PredictiveMoments(double fMean, double fVariance)
    {
        // Use probit approximation for expectation of sigmoid
        double kappa = 1.0 / Math.Sqrt(1 + Math.PI * fVariance / 8);
        double muPred = Sigmoid(kappa * fMean);

        // Variance from both latent uncertainty and Beta noise
        double betaVar = muPred * (1 - muPred) / (_precision + 1);

        // Additional variance from uncertainty in f (delta method approximation)
        double sigmoidDeriv = muPred * (1 - muPred);
        double latentVar = sigmoidDeriv * sigmoidDeriv * fVariance;

        double totalVar = betaVar + latentVar;

        return (muPred, totalVar);
    }

    /// <summary>
    /// Creates a default Beta Likelihood with automatic precision estimation.
    /// </summary>
    /// <param name="y">Sample of observations to estimate precision from.</param>
    /// <returns>A BetaLikelihood with estimated precision.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Estimates an appropriate precision from data.
    ///
    /// Uses method of moments:
    /// Given sample mean m and variance v, precision ν ≈ m(1-m)/v - 1
    ///
    /// This is a rough estimate; for better results, optimize precision via marginal likelihood.
    /// </para>
    /// </remarks>
    public static BetaLikelihood<T> FromData(Vector<T> y)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute sample mean and variance
        double sum = 0, sumSq = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = numOps.ToDouble(y[i]);
            sum += yi;
            sumSq += yi * yi;
        }

        double mean = sum / y.Length;
        double variance = sumSq / y.Length - mean * mean;

        // Estimate precision via method of moments
        // Var(Y) = μ(1-μ)/(ν+1) → ν = μ(1-μ)/Var - 1
        variance = Math.Max(variance, 1e-6); // Avoid division by zero
        mean = Math.Max(0.01, Math.Min(0.99, mean)); // Avoid extreme means

        double precision = mean * (1 - mean) / variance - 1;
        precision = Math.Max(2.0, Math.Min(1000.0, precision)); // Reasonable bounds

        return new BetaLikelihood<T>(precision);
    }

    #region Private Methods

    /// <summary>
    /// Sigmoid function.
    /// </summary>
    private static double Sigmoid(double x)
    {
        if (x >= 0)
            return 1.0 / (1.0 + Math.Exp(-x));
        else
        {
            double ex = Math.Exp(x);
            return ex / (1.0 + ex);
        }
    }

    /// <summary>
    /// Log Gamma function.
    /// </summary>
    private static double LogGamma(double x)
    {
        // Lanczos approximation
        double[] coeffs = {
            76.18009172947146, -86.50532032941677, 24.01409824083091,
            -1.231739572450155, 0.001208650973866179, -0.000005395239384953
        };

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;

        for (int j = 0; j < 6; j++)
        {
            y += 1;
            ser += coeffs[j] / y;
        }

        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    /// <summary>
    /// Digamma function (derivative of log Gamma).
    /// </summary>
    private static double Digamma(double x)
    {
        // Asymptotic expansion for large x (use >= to avoid infinite recursion at boundary)
        if (x >= 6)
        {
            double result = Math.Log(x) - 1.0 / (2 * x);
            double x2 = x * x;
            result -= 1.0 / (12 * x2);
            result += 1.0 / (120 * x2 * x2);
            return result;
        }

        // Recurrence relation for small x
        if (x < 1)
        {
            return Digamma(x + 1) - 1.0 / x;
        }

        // Iterative approach for moderate x (1 <= x < 6)
        double sum = 0;
        while (x < 6)
        {
            sum -= 1.0 / x;
            x += 1;
        }
        // Now x >= 6, use asymptotic expansion directly
        double result2 = Math.Log(x) - 1.0 / (2 * x);
        double x2_2 = x * x;
        result2 -= 1.0 / (12 * x2_2);
        result2 += 1.0 / (120 * x2_2 * x2_2);
        return sum + result2;
    }

    /// <summary>
    /// Trigamma function (second derivative of log Gamma).
    /// </summary>
    private static double Trigamma(double x)
    {
        // Asymptotic expansion (use >= to avoid infinite recursion at boundary)
        if (x >= 6)
        {
            double result = 1.0 / x + 1.0 / (2 * x * x);
            double x3 = x * x * x;
            result += 1.0 / (6 * x3);
            return result;
        }

        // Recurrence
        if (x < 1)
        {
            return Trigamma(x + 1) + 1.0 / (x * x);
        }

        // Iterative approach for moderate x (1 <= x < 6)
        double sum = 0;
        while (x < 6)
        {
            sum += 1.0 / (x * x);
            x += 1;
        }
        // Now x >= 6, use asymptotic expansion directly
        double result2 = 1.0 / x + 1.0 / (2 * x * x);
        double x3_2 = x * x * x;
        result2 += 1.0 / (6 * x3_2);
        return sum + result2;
    }

    /// <summary>
    /// Samples from Beta distribution using rejection sampling.
    /// </summary>
    private static double SampleBeta(double alpha, double beta, Random random)
    {
        // Use Gamma distribution: if X ~ Gamma(α), Y ~ Gamma(β), then X/(X+Y) ~ Beta(α,β)
        double x = SampleGamma(alpha, random);
        double y = SampleGamma(beta, random);
        return x / (x + y + Epsilon);
    }

    /// <summary>
    /// Samples from Gamma distribution using Marsaglia and Tsang's method.
    /// </summary>
    private static double SampleGamma(double shape, Random random)
    {
        if (shape < 1)
        {
            // Use transformation for shape < 1
            double u = random.NextDouble();
            return SampleGamma(shape + 1, random) * Math.Pow(u, 1.0 / shape);
        }

        // Marsaglia and Tsang's method
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            double x, v;
            do
            {
                x = SampleNormal(random);
                v = 1.0 + c * x;
            } while (v <= 0);

            v = v * v * v;
            double u = random.NextDouble();

            if (u < 1 - 0.0331 * (x * x) * (x * x))
                return d * v;

            if (Math.Log(u) < 0.5 * x * x + d * (1 - v + Math.Log(v)))
                return d * v;
        }
    }

    /// <summary>
    /// Samples from standard normal using Box-Muller.
    /// </summary>
    private static double SampleNormal(Random random)
    {
        double u1 = random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    #endregion
}
