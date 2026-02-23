namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Poisson probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Poisson distribution models the number of events occurring in a fixed interval of time
/// or space when these events occur with a known constant mean rate and independently of each other.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Poisson distribution is for counting rare events, like:
/// - Number of customers arriving per hour
/// - Number of typos on a page
/// - Number of earthquakes per year
/// It's characterized by just one parameter λ (lambda), which is both the mean and variance.
/// If the variance of your count data is much larger than the mean, consider Negative Binomial instead.
/// </para>
/// <para>
/// PMF: P(X = k) = (λ^k * e^(-λ)) / k!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class PoissonDistribution<T> : DistributionBase<T>
{
    private T _lambda;

    /// <summary>
    /// Initializes a new Poisson distribution with specified rate.
    /// </summary>
    /// <param name="lambda">The rate parameter (λ, must be positive).</param>
    public PoissonDistribution(T lambda)
    {
        if (NumOps.Compare(lambda, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(lambda), "Lambda must be positive.");

        _lambda = lambda;
    }

    /// <inheritdoc/>
    public override int NumParameters => 1;

    /// <inheritdoc/>
    public override Vector<T> Parameters
    {
        get => new Vector<T>(new[] { _lambda });
        set
        {
            if (value.Length != 1)
                throw new ArgumentException("Poisson distribution requires exactly 1 parameter.");
            if (NumOps.Compare(value[0], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Lambda must be positive.");

            _lambda = value[0];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["lambda"];

    /// <summary>
    /// Gets the rate parameter (λ).
    /// </summary>
    public T Lambda => _lambda;

    /// <inheritdoc/>
    public override T Mean => _lambda;

    /// <inheritdoc/>
    public override T Variance => _lambda;

    /// <summary>
    /// Computes the probability mass function (PMF) at integer k.
    /// </summary>
    /// <param name="k">The number of events (must be non-negative integer).</param>
    /// <returns>The probability of exactly k events.</returns>
    public T Pmf(int k)
    {
        if (k < 0) return Zero;

        double lambda = NumOps.ToDouble(_lambda);
        double logPmf = k * Math.Log(lambda) - lambda - LogFactorial(k);
        return NumOps.FromDouble(Math.Exp(logPmf));
    }

    /// <summary>
    /// Computes the log probability mass function at integer k.
    /// </summary>
    public T LogPmf(int k)
    {
        if (k < 0) return NumOps.FromDouble(double.NegativeInfinity);

        double lambda = NumOps.ToDouble(_lambda);
        return NumOps.FromDouble(k * Math.Log(lambda) - lambda - LogFactorial(k));
    }

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        // For discrete distributions, PDF is the PMF at the nearest integer
        int k = (int)Math.Round(NumOps.ToDouble(x));
        return Pmf(k);
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        int k = (int)Math.Round(NumOps.ToDouble(x));
        return LogPmf(k);
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        int k = (int)Math.Floor(NumOps.ToDouble(x));
        if (k < 0) return Zero;

        double lambda = NumOps.ToDouble(_lambda);

        // CDF = regularized incomplete gamma function Q(k+1, λ)
        return NumOps.FromDouble(RegularizedUpperGamma(k + 1, lambda));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        double lambda = NumOps.ToDouble(_lambda);

        // For large λ, use normal approximation as initial guess then refine
        int low, high;
        if (lambda >= 30)
        {
            // Normal approximation: X ~ N(λ, λ) for large λ
            double z = NormalQuantile(pVal);
            double guess = lambda + z * Math.Sqrt(lambda);
            int center = Math.Max(0, (int)Math.Round(guess));
            // Search in a narrow band around the guess
            int band = Math.Max(10, (int)(4 * Math.Sqrt(lambda)));
            low = Math.Max(0, center - band);
            high = center + band;
            // Expand if needed
            while (high < int.MaxValue && RegularizedUpperGamma(high + 1, lambda) < pVal)
            {
                long next = (long)high * 2;
                high = next > int.MaxValue ? int.MaxValue : (int)next;
            }
        }
        else
        {
            low = 0;
            double rawHigh = lambda + 10 * Math.Sqrt(lambda) + 10;
            high = rawHigh >= int.MaxValue || double.IsNaN(rawHigh) || double.IsInfinity(rawHigh)
                ? int.MaxValue
                : (int)rawHigh;

            while (high < int.MaxValue && RegularizedUpperGamma(high + 1, lambda) < pVal)
            {
                long next = (long)high * 2;
                high = next > int.MaxValue ? int.MaxValue : (int)next;
            }
        }

        while (low < high)
        {
            int mid = (low + high) / 2;
            double cdf = RegularizedUpperGamma(mid + 1, lambda);

            if (cdf < pVal)
                low = mid + 1;
            else
                high = mid;
        }

        return NumOps.FromDouble(low);
    }

    /// <inheritdoc/>
    public override Vector<T> GradLogPdf(T x)
    {
        int k = (int)Math.Round(NumOps.ToDouble(x));
        if (k < 0)
            return new Vector<T>(new[] { NumOps.FromDouble(double.NaN) });

        double lambda = NumOps.ToDouble(_lambda);

        // d/d(lambda) log(pmf) = k/λ - 1
        double gradLambda = k / lambda - 1;
        return new Vector<T>(new[] { NumOps.FromDouble(gradLambda) });
    }

    /// <inheritdoc/>
    public override Matrix<T> FisherInformation()
    {
        // Fisher Information for Poisson(λ):
        // I = 1/λ
        return new Matrix<T>(new T[,]
        {
            { NumOps.Divide(One, _lambda) }
        });
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new PoissonDistribution<T>(_lambda);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double lambda = NumOps.ToDouble(_lambda);

        if (lambda < 30)
        {
            // Direct method for small lambda
            double L = Math.Exp(-lambda);
            int k = 0;
            double p = 1;

            do
            {
                k++;
                p *= random.NextDouble();
            } while (p > L);

            return NumOps.FromDouble(k - 1);
        }
        else
        {
            // Normal approximation for large lambda
            double z = BoxMullerNormal(random);
            return NumOps.FromDouble(Math.Max(0, Math.Round(lambda + Math.Sqrt(lambda) * z)));
        }
    }

    private static double LogFactorial(int n)
    {
        if (n <= 1) return 0;
        return LogGamma(n + 1);
    }

    /// <summary>
    /// Computes the regularized upper incomplete gamma function Q(a, x) = P(X &lt;= a-1)
    /// for Poisson with rate x. For Poisson CDF: P(X &lt;= k) = Q(k+1, lambda).
    /// Uses series summation for small a and continued fraction for large a.
    /// </summary>
    private static double RegularizedUpperGamma(int a, double x)
    {
        if (x <= 0) return 1.0;
        if (a <= 0) return 0.0;

        // For large a, use continued fraction (Lentz's method) for the
        // upper incomplete gamma Γ(a,x)/Γ(a) = 1 - P(a,x), then
        // Q(a,x) = P(a,x) = 1 - Γ(a,x)/Γ(a).
        // The series is O(a) but with early stopping on convergence.
        if (a > 200 && x < a + 1)
        {
            // Use regularized lower gamma via series: P(a,x) = e^{-x} x^a / Gamma(a) * sum
            return LowerGammaSeries(a, x);
        }
        else if (a > 200)
        {
            // Use upper gamma via continued fraction: Q(a,x) = 1 - P(a,x)
            return 1.0 - UpperGammaCF(a, x);
        }

        // For moderate a, use log-space summation with early stopping
        double logSum = double.NegativeInfinity;
        double logFactorial = 0;
        double logX = Math.Log(x);

        for (int i = 0; i < a; i++)
        {
            double logTerm = -x + i * logX - logFactorial;

            // Early stop: if terms become negligibly small
            if (i > 0 && logTerm < logSum - 50) break;

            if (double.IsNegativeInfinity(logSum))
                logSum = logTerm;
            else
            {
                double max = Math.Max(logSum, logTerm);
                logSum = max + Math.Log(Math.Exp(logSum - max) + Math.Exp(logTerm - max));
            }

            logFactorial += Math.Log(i + 1);
        }

        return Math.Exp(logSum);
    }

    /// <summary>
    /// Regularized lower incomplete gamma P(a,x) via series expansion.
    /// P(a,x) = e^{-x} * x^a / Gamma(a) * sum_{n=0}^inf x^n / (a*(a+1)*...*(a+n))
    /// </summary>
    private static double LowerGammaSeries(int a, double x)
    {
        double logPrefix = -x + a * Math.Log(x) - LogGamma(a);
        double sum = 1.0;
        double term = 1.0;

        for (int n = 1; n < 1000; n++)
        {
            term *= x / (a + n);
            sum += term;
            if (Math.Abs(term) < sum * 1e-15) break;
        }

        return Math.Exp(logPrefix) * sum / a;
    }

    /// <summary>
    /// Upper incomplete gamma Γ(a,x)/Γ(a) via Lentz continued fraction.
    /// Returns 1 - P(a,x).
    /// </summary>
    private static double UpperGammaCF(int a, double x)
    {
        double logPrefix = -x + a * Math.Log(x) - LogGamma(a);

        // Lentz's method for continued fraction
        double f = 1e-30;
        double c = 1e-30;
        double d = 1.0 / (x + 1 - a);
        f = d;

        for (int n = 1; n < 1000; n++)
        {
            double an = n * ((double)a - n);
            double bn = x + 2 * n + 1 - a;
            d = bn + an * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = bn + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            if (Math.Abs(delta - 1.0) < 1e-15) break;
        }

        return 1.0 - Math.Exp(logPrefix) * f;
    }

    /// <summary>
    /// Approximate normal quantile using rational approximation (Abramowitz and Stegun).
    /// </summary>
    private static double NormalQuantile(double p)
    {
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;
        if (Math.Abs(p - 0.5) < 1e-15) return 0.0;

        // Rational approximation
        double t = p < 0.5 ? Math.Sqrt(-2.0 * Math.Log(p)) : Math.Sqrt(-2.0 * Math.Log(1.0 - p));
        double z = t - (2.515517 + t * (0.802853 + t * 0.010328))
                       / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
        return p < 0.5 ? -z : z;
    }
}
