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
public class PoissonDistribution<T> : DistributionBase<T>
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
    public override T[] Parameters
    {
        get => [_lambda];
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
        return NumOps.FromDouble(GammaUpperIncomplete(k + 1, lambda));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        // Binary search for the quantile
        double lambda = NumOps.ToDouble(_lambda);
        int low = 0;
        int high = (int)(lambda + 10 * Math.Sqrt(lambda) + 10);

        while (low < high)
        {
            int mid = (low + high) / 2;
            double cdf = GammaUpperIncomplete(mid + 1, lambda);

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
        double lambda = NumOps.ToDouble(_lambda);

        // d/d(lambda) log(pmf) = k/λ - 1
        double gradLambda = k / lambda - 1;
        return [NumOps.FromDouble(gradLambda)];
    }

    /// <inheritdoc/>
    public override T[,] FisherInformation()
    {
        // Fisher Information for Poisson(λ):
        // I = 1/λ
        return new T[,]
        {
            { NumOps.Divide(One, _lambda) }
        };
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

    private static double GammaUpperIncomplete(int a, double x)
    {
        // For integer a, this is the regularized upper incomplete gamma function
        // P(a, x) = 1 - Q(a, x) where Q is the regularized lower incomplete gamma

        double sum = 0;
        double term = Math.Exp(-x);

        for (int i = 0; i < a; i++)
        {
            sum += term;
            term *= x / (i + 1);
        }

        return sum;
    }
}
