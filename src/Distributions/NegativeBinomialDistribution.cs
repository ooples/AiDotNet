namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Negative Binomial probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Negative Binomial distribution models the number of failures before r successes occur,
/// or equivalently, overdispersed count data where variance exceeds the mean.
/// It generalizes the Poisson distribution to handle overdispersion.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use Negative Binomial instead of Poisson when your count data
/// has variance larger than its mean (overdispersion). This is common in real-world data:
/// - Number of accidents (some people are accident-prone)
/// - Disease counts (clustering in certain areas)
/// - Customer purchases (some customers buy much more)
/// The extra parameter r controls the degree of overdispersion.
/// </para>
/// <para>
/// PMF: P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
/// Alternative parametrization with mean μ and dispersion r:
/// P(X = k) = C(k+r-1, k) * (r/(r+μ))^r * (μ/(r+μ))^k
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NegativeBinomialDistribution<T> : DistributionBase<T>
{
    private T _r;      // Number of successes (dispersion parameter)
    private T _prob;   // Probability of success

    /// <summary>
    /// Initializes a new Negative Binomial distribution.
    /// </summary>
    /// <param name="r">The number of successes / dispersion parameter (must be positive).</param>
    /// <param name="prob">The probability of success (must be in (0, 1]).</param>
    public NegativeBinomialDistribution(T r, T prob)
    {
        if (NumOps.Compare(r, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(r), "r must be positive.");
        if (NumOps.Compare(prob, Zero) <= 0 || NumOps.Compare(prob, One) > 0)
            throw new ArgumentOutOfRangeException(nameof(prob), "Probability must be in (0, 1].");

        _r = r;
        _prob = prob;
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override T[] Parameters
    {
        get => [_r, _prob];
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Negative Binomial distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[0], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "r must be positive.");
            if (NumOps.Compare(value[1], Zero) <= 0 || NumOps.Compare(value[1], One) > 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Probability must be in (0, 1].");

            _r = value[0];
            _prob = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["r", "probability"];

    /// <summary>
    /// Gets the dispersion parameter (r).
    /// </summary>
    public T R => _r;

    /// <summary>
    /// Gets the success probability.
    /// </summary>
    public T Probability => _prob;

    /// <inheritdoc/>
    public override T Mean
    {
        get
        {
            // mean = r(1-p)/p
            T oneMinusP = NumOps.Subtract(One, _prob);
            return NumOps.Divide(NumOps.Multiply(_r, oneMinusP), _prob);
        }
    }

    /// <inheritdoc/>
    public override T Variance
    {
        get
        {
            // variance = r(1-p)/p²
            T oneMinusP = NumOps.Subtract(One, _prob);
            T pSquared = NumOps.Multiply(_prob, _prob);
            return NumOps.Divide(NumOps.Multiply(_r, oneMinusP), pSquared);
        }
    }

    /// <summary>
    /// Computes the probability mass function (PMF) at integer k.
    /// </summary>
    public T Pmf(int k)
    {
        if (k < 0) return Zero;

        double r = NumOps.ToDouble(_r);
        double p = NumOps.ToDouble(_prob);

        double logPmf = LogGamma(k + r) - LogGamma(k + 1) - LogGamma(r) +
                        r * Math.Log(p) + k * Math.Log(1 - p);

        return NumOps.FromDouble(Math.Exp(logPmf));
    }

    /// <summary>
    /// Computes the log probability mass function at integer k.
    /// </summary>
    public T LogPmf(int k)
    {
        if (k < 0) return NumOps.FromDouble(double.NegativeInfinity);

        double r = NumOps.ToDouble(_r);
        double p = NumOps.ToDouble(_prob);

        return NumOps.FromDouble(
            LogGamma(k + r) - LogGamma(k + 1) - LogGamma(r) +
            r * Math.Log(p) + k * Math.Log(1 - p));
    }

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
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

        double r = NumOps.ToDouble(_r);
        double p = NumOps.ToDouble(_prob);

        // CDF = I_p(r, k+1) where I is the regularized incomplete beta function
        return NumOps.FromDouble(BetaIncomplete(r, k + 1, p));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T pQuantile)
    {
        double pVal = NumOps.ToDouble(pQuantile);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(pQuantile), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        double mean = NumOps.ToDouble(Mean);

        // Binary search
        int low = 0;
        int high = (int)(mean + 10 * Math.Sqrt(NumOps.ToDouble(Variance)) + 10);

        while (low < high)
        {
            int mid = (low + high) / 2;
            double cdf = NumOps.ToDouble(Cdf(NumOps.FromDouble(mid)));

            if (cdf < pVal)
                low = mid + 1;
            else
                high = mid;
        }

        return NumOps.FromDouble(low);
    }

    /// <inheritdoc/>
    public override T[] GradLogPdf(T x)
    {
        int k = (int)Math.Round(NumOps.ToDouble(x));
        double r = NumOps.ToDouble(_r);
        double p = NumOps.ToDouble(_prob);

        // d/d(r) log(pmf) = digamma(k+r) - digamma(r) + log(p)
        double gradR = Digamma(k + r) - Digamma(r) + Math.Log(p);

        // d/d(p) log(pmf) = r/p - k/(1-p)
        double gradP = r / p - k / (1 - p);

        return [NumOps.FromDouble(gradR), NumOps.FromDouble(gradP)];
    }

    /// <inheritdoc/>
    public override T[,] FisherInformation()
    {
        double r = NumOps.ToDouble(_r);
        double p = NumOps.ToDouble(_prob);

        // Fisher Information Matrix
        double iR = Trigamma(r) - Trigamma(r + r * (1 - p) / p);
        double iP = r / (p * p * (1 - p));
        double iRP = 1 / p;

        return new T[,]
        {
            { NumOps.FromDouble(Math.Max(iR, 1e-10)), NumOps.FromDouble(iRP) },
            { NumOps.FromDouble(iRP), NumOps.FromDouble(iP) }
        };
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new NegativeBinomialDistribution<T>(_r, _prob);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double r = NumOps.ToDouble(_r);
        double p = NumOps.ToDouble(_prob);

        // Sample using Gamma-Poisson mixture
        // If X ~ Gamma(r, p/(1-p)) then Y|X ~ Poisson(X) has Y ~ NegBinom(r, p)
        double gamma = SampleGamma(random, r, p / (1 - p));
        return NumOps.FromDouble(SamplePoisson(random, gamma));
    }

    private static double SampleGamma(Random random, double shape, double rate)
    {
        if (shape < 1)
        {
            return SampleGamma(random, shape + 1, rate) * Math.Pow(random.NextDouble(), 1 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            double x, v;
            do
            {
                x = BoxMullerNormal(random);
                v = 1 + c * x;
            } while (v <= 0);

            v = v * v * v;
            double u = random.NextDouble();

            if (u < 1 - 0.0331 * x * x * x * x)
                return d * v / rate;

            if (Math.Log(u) < 0.5 * x * x + d * (1 - v + Math.Log(v)))
                return d * v / rate;
        }
    }

    private static int SamplePoisson(Random random, double lambda)
    {
        if (lambda < 30)
        {
            double L = Math.Exp(-lambda);
            int k = 0;
            double pProduct = 1;

            do
            {
                k++;
                pProduct *= random.NextDouble();
            } while (pProduct > L);

            return k - 1;
        }
        else
        {
            double z = BoxMullerNormal(random);
            return Math.Max(0, (int)Math.Round(lambda + Math.Sqrt(lambda) * z));
        }
    }

    /// <summary>
    /// Creates a Negative Binomial distribution from mean and variance.
    /// </summary>
    public static NegativeBinomialDistribution<T> FromMeanVariance(T mean, T variance)
    {
        double m = NumOps.ToDouble(mean);
        double v = NumOps.ToDouble(variance);

        if (v <= m)
            throw new ArgumentException("Variance must be greater than mean for Negative Binomial.");

        // r = μ²/(σ² - μ), p = μ/σ² = r/(r + μ)
        double r = m * m / (v - m);
        double p = r / (r + m);

        return new NegativeBinomialDistribution<T>(NumOps.FromDouble(r), NumOps.FromDouble(p));
    }

    /// <summary>
    /// Creates a Negative Binomial distribution from mean and dispersion.
    /// </summary>
    public static NegativeBinomialDistribution<T> FromMeanDispersion(T mean, T dispersion)
    {
        double m = NumOps.ToDouble(mean);
        double r = NumOps.ToDouble(dispersion);

        double p = r / (r + m);
        return new NegativeBinomialDistribution<T>(dispersion, NumOps.FromDouble(p));
    }
}
