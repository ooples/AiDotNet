using AiDotNet.Helpers;

namespace AiDotNet.Distributions;

/// <summary>
/// Abstract base class for parametric probability distributions.
/// </summary>
/// <remarks>
/// <para>
/// This base class provides common functionality for all distributions including
/// numeric operations through INumericOperations, default implementations for
/// sampling, and helper methods for numerical computations.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation that all specific distributions
/// (Normal, Laplace, Gamma, etc.) build upon. It handles the common math operations
/// so each distribution only needs to implement its specific formulas.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal abstract class DistributionBase<T> : ISamplingDistribution<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Default random number generator for sampling.
    /// </summary>
    private static readonly ThreadLocal<Random> DefaultRandom = new(() => RandomHelper.CreateSecureRandom());

    /// <summary>
    /// Mathematical constants.
    /// </summary>
    protected static readonly T Pi = NumOps.FromDouble(Math.PI);
    protected static readonly T TwoPi = NumOps.FromDouble(2.0 * Math.PI);
    protected static readonly T SqrtTwoPi = NumOps.FromDouble(Math.Sqrt(2.0 * Math.PI));
    protected static readonly T LogTwoPi = NumOps.FromDouble(Math.Log(2.0 * Math.PI));
    protected static readonly T Half = NumOps.FromDouble(0.5);
    protected static readonly T One = NumOps.One;
    protected static readonly T Two = NumOps.FromDouble(2.0);
    protected static readonly T Zero = NumOps.Zero;

    /// <inheritdoc/>
    public abstract int NumParameters { get; }

    /// <inheritdoc/>
    public abstract Vector<T> Parameters { get; set; }

    /// <inheritdoc/>
    public abstract string[] ParameterNames { get; }

    /// <inheritdoc/>
    public abstract T Mean { get; }

    /// <inheritdoc/>
    public abstract T Variance { get; }

    /// <inheritdoc/>
    public virtual T StdDev => NumOps.Sqrt(Variance);

    /// <inheritdoc/>
    public abstract T Pdf(T x);

    /// <inheritdoc/>
    public abstract T LogPdf(T x);

    /// <inheritdoc/>
    public abstract T Cdf(T x);

    /// <inheritdoc/>
    public abstract T InverseCdf(T p);

    /// <inheritdoc/>
    public abstract Vector<T> GradLogPdf(T x);

    /// <inheritdoc/>
    public abstract Matrix<T> FisherInformation();

    /// <inheritdoc/>
    public abstract IParametricDistribution<T> Clone();

    /// <inheritdoc/>
    public abstract T Sample(Random random);

    /// <inheritdoc/>
    public virtual Vector<T> Sample(Random random, int count)
    {
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count must be positive.");

        var samples = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            samples[i] = Sample(random);
        }
        return samples;
    }

    /// <inheritdoc/>
    public T Sample() => Sample(DefaultRandom.Value!);

    /// <inheritdoc/>
    public Vector<T> Sample(int count) => Sample(DefaultRandom.Value!, count);

    /// <summary>
    /// Generates a standard normal random variate using the Box-Muller transform.
    /// </summary>
    /// <param name="random">The random number generator.</param>
    /// <returns>A standard normal random variate.</returns>
    protected static double BoxMullerNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble(); // Avoid log(0)
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Computes the standard normal CDF using the error function approximation.
    /// </summary>
    /// <param name="x">The value at which to compute the CDF.</param>
    /// <returns>The standard normal CDF at x.</returns>
    protected static double StandardNormalCdf(double x)
    {
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    /// <summary>
    /// Computes the standard normal inverse CDF (probit function).
    /// </summary>
    /// <param name="p">The probability.</param>
    /// <returns>The inverse CDF value.</returns>
    protected static double StandardNormalInverseCdf(double p)
    {
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        // Rational approximation for central region
        if (p > 0.5)
        {
            return -StandardNormalInverseCdf(1.0 - p);
        }

        double t = Math.Sqrt(-2.0 * Math.Log(p));

        // Coefficients for rational approximation
        const double c0 = 2.515517;
        const double c1 = 0.802853;
        const double c2 = 0.010328;
        const double d1 = 1.432788;
        const double d2 = 0.189269;
        const double d3 = 0.001308;

        return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t));
    }

    /// <summary>
    /// Computes the error function using a polynomial approximation.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The error function value.</returns>
    protected static double Erf(double x)
    {
        // Approximation with maximum error < 1.5e-7
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

    /// <summary>
    /// Computes the log of the gamma function using Stirling's approximation.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The log gamma value.</returns>
    protected static double LogGamma(double x)
    {
        if (x <= 0)
            throw new ArgumentOutOfRangeException(nameof(x), "Argument must be positive.");

        // Lanczos approximation coefficients
        double[] coefficients =
        [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5
        ];

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);

        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
        {
            y += 1.0;
            ser += coefficients[j] / y;
        }

        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    /// <summary>
    /// Computes the gamma function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The gamma function value.</returns>
    protected static double Gamma(double x)
    {
        return Math.Exp(LogGamma(x));
    }

    /// <summary>
    /// Computes the digamma (psi) function - the derivative of log gamma.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The digamma function value.</returns>
    protected static double Digamma(double x)
    {
        if (x <= 0)
            throw new ArgumentOutOfRangeException(nameof(x), "Argument must be positive.");

        double result = 0.0;

        // Use recurrence relation for small x
        while (x < 6)
        {
            result -= 1.0 / x;
            x += 1.0;
        }

        // Asymptotic expansion for large x
        double x2 = 1.0 / (x * x);
        result += Math.Log(x) - 0.5 / x
            - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0));

        return result;
    }

    /// <summary>
    /// Computes the trigamma function - the second derivative of log gamma.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The trigamma function value.</returns>
    protected static double Trigamma(double x)
    {
        if (x <= 0)
            throw new ArgumentOutOfRangeException(nameof(x), "Argument must be positive.");

        double result = 0.0;

        // Use recurrence relation for small x
        while (x < 6)
        {
            result += 1.0 / (x * x);
            x += 1.0;
        }

        // Asymptotic expansion for large x
        double x2 = 1.0 / (x * x);
        result += 1.0 / x + x2 * (0.5 + x2 * (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 / 42.0)));

        return result;
    }

    /// <summary>
    /// Computes the regularized incomplete beta function.
    /// </summary>
    protected static double BetaIncomplete(double a, double b, double x)
    {
        if (x < 0 || x > 1)
            throw new ArgumentOutOfRangeException(nameof(x), "x must be in [0, 1].");

        if (x == 0) return 0;
        if (x == 1) return 1;

        double bt = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) +
            a * Math.Log(x) + b * Math.Log(1 - x));

        if (x < (a + 1) / (a + b + 2))
        {
            return bt * BetaContinuedFraction(a, b, x) / a;
        }
        else
        {
            return 1 - bt * BetaContinuedFraction(b, a, 1 - x) / b;
        }
    }

    private static double BetaContinuedFraction(double a, double b, double x)
    {
        const int maxIterations = 100;
        const double epsilon = 3e-7;

        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.Abs(d) < 1e-30) d = 1e-30;
        d = 1 / d;
        double h = d;

        for (int m = 1; m <= maxIterations; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < epsilon)
                return h;
        }

        throw new InvalidOperationException("Beta continued fraction did not converge.");
    }
}
