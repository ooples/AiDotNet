namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Gamma probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Gamma distribution is a two-parameter family of continuous probability distributions
/// commonly used to model waiting times, insurance claim amounts, and other positive-valued phenomena.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Gamma distribution is for modeling positive values only (like time
/// until an event, or amounts of money). It's very flexible - with shape=1 it becomes an
/// exponential distribution, and with large shape it approaches a Normal distribution.
/// The shape controls how "peaked" the distribution is, while rate controls how spread out.
/// </para>
/// <para>
/// PDF: f(x) = (β^α / Γ(α)) * x^(α-1) * exp(-βx) for x > 0
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class GammaDistribution<T> : DistributionBase<T>
{
    private T _shape;  // α (alpha)
    private T _rate;   // β (beta)

    /// <summary>
    /// Initializes a new Gamma distribution with specified shape and rate.
    /// </summary>
    /// <param name="shape">The shape parameter (α, must be positive).</param>
    /// <param name="rate">The rate parameter (β, must be positive).</param>
    public GammaDistribution(T shape, T rate)
    {
        if (NumOps.Compare(shape, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be positive.");
        if (NumOps.Compare(rate, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(rate), "Rate must be positive.");

        _shape = shape;
        _rate = rate;
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override Vector<T> Parameters
    {
        get => new Vector<T>(new[] { _shape, _rate });
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Gamma distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[0], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Shape must be positive.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Rate must be positive.");

            _shape = value[0];
            _rate = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["shape", "rate"];

    /// <summary>
    /// Gets the shape parameter (α).
    /// </summary>
    public T Shape => _shape;

    /// <summary>
    /// Gets the rate parameter (β).
    /// </summary>
    public T Rate => _rate;

    /// <summary>
    /// Gets the scale parameter (1/β).
    /// </summary>
    public T Scale => NumOps.Divide(One, _rate);

    /// <inheritdoc/>
    public override T Mean => NumOps.Divide(_shape, _rate);

    /// <inheritdoc/>
    public override T Variance => NumOps.Divide(_shape, NumOps.Multiply(_rate, _rate));

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return Zero;

        double xVal = NumOps.ToDouble(x);
        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        double logPdf = alpha * Math.Log(beta) - LogGamma(alpha) +
                        (alpha - 1) * Math.Log(xVal) - beta * xVal;

        return NumOps.FromDouble(Math.Exp(logPdf));
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        double xVal = NumOps.ToDouble(x);
        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        return NumOps.FromDouble(alpha * Math.Log(beta) - LogGamma(alpha) +
                                 (alpha - 1) * Math.Log(xVal) - beta * xVal);
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return Zero;

        double xVal = NumOps.ToDouble(x);
        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        return NumOps.FromDouble(GammaIncomplete(alpha, beta * xVal));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        // Newton-Raphson iteration
        double x = alpha / beta; // Initial guess = mean
        for (int i = 0; i < 100; i++)
        {
            double currentCdf = GammaIncomplete(alpha, beta * x);
            double error = currentCdf - pVal;

            if (Math.Abs(error) < 1e-10)
                break;

            double pdf = Math.Exp(alpha * Math.Log(beta) - LogGamma(alpha) +
                                  (alpha - 1) * Math.Log(x) - beta * x);

            x = Math.Max(1e-10, x - error / pdf);
        }

        return NumOps.FromDouble(x);
    }

    /// <inheritdoc/>
    public override Vector<T> GradLogPdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return new Vector<T>(new[] { NumOps.FromDouble(double.NaN), NumOps.FromDouble(double.NaN) });

        double xVal = NumOps.ToDouble(x);
        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        // d/d(shape) log(pdf) = log(beta) - digamma(alpha) + log(x)
        double gradShape = Math.Log(beta) - Digamma(alpha) + Math.Log(xVal);

        // d/d(rate) log(pdf) = alpha/beta - x
        double gradRate = alpha / beta - xVal;

        return new Vector<T>(new[] { NumOps.FromDouble(gradShape), NumOps.FromDouble(gradRate) });
    }

    /// <inheritdoc/>
    public override Matrix<T> FisherInformation()
    {
        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        // Fisher Information for Gamma(α, β):
        // I = [[trigamma(α), -1/β], [-1/β, α/β²]]

        double iAlpha = Trigamma(alpha);
        double iAlphaBeta = -1.0 / beta;
        double iBeta = alpha / (beta * beta);

        return new Matrix<T>(new T[,]
        {
            { NumOps.FromDouble(iAlpha), NumOps.FromDouble(iAlphaBeta) },
            { NumOps.FromDouble(iAlphaBeta), NumOps.FromDouble(iBeta) }
        });
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new GammaDistribution<T>(_shape, _rate);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double alpha = NumOps.ToDouble(_shape);
        double beta = NumOps.ToDouble(_rate);

        return NumOps.FromDouble(SampleGamma(random, alpha, beta));
    }

    private static double SampleGamma(Random random, double shape, double rate)
    {
        // Marsaglia and Tsang's method
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

    /// <summary>
    /// Computes the regularized lower incomplete gamma function.
    /// </summary>
    private static double GammaIncomplete(double a, double x)
    {
        if (x < 0 || a <= 0)
            throw new ArgumentException("Invalid arguments for incomplete gamma function.");

        if (x == 0)
            return 0;

        if (x < a + 1)
        {
            // Use series expansion
            return GammaIncompleteSeriesExpansion(a, x);
        }
        else
        {
            // Use continued fraction
            return 1 - GammaIncompleteContinuedFraction(a, x);
        }
    }

    private static double GammaIncompleteSeriesExpansion(double a, double x)
    {
        const int maxIterations = 100;
        const double epsilon = 3e-7;

        double ap = a;
        double sum = 1.0 / a;
        double del = sum;

        for (int n = 1; n <= maxIterations; n++)
        {
            ap += 1;
            del *= x / ap;
            sum += del;

            if (Math.Abs(del) < Math.Abs(sum) * epsilon)
            {
                return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
            }
        }

        throw new InvalidOperationException("Incomplete gamma series did not converge.");
    }

    private static double GammaIncompleteContinuedFraction(double a, double x)
    {
        const int maxIterations = 100;
        const double epsilon = 3e-7;

        double b = x + 1 - a;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;

        for (int i = 1; i <= maxIterations; i++)
        {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < epsilon)
            {
                return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * h;
            }
        }

        throw new InvalidOperationException("Incomplete gamma continued fraction did not converge.");
    }

    /// <summary>
    /// Creates a Gamma distribution from mean and variance.
    /// </summary>
    public static GammaDistribution<T> FromMeanVariance(T mean, T variance)
    {
        // mean = α/β, variance = α/β²
        // => α = mean²/variance, β = mean/variance
        T meanSquared = NumOps.Multiply(mean, mean);
        T shape = NumOps.Divide(meanSquared, variance);
        T rate = NumOps.Divide(mean, variance);
        return new GammaDistribution<T>(shape, rate);
    }
}
