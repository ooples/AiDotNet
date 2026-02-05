namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Beta probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Beta distribution is defined on the interval [0, 1] and is commonly used to model
/// random variables that represent proportions, probabilities, or bounded rates.
/// It's conjugate prior to the Binomial distribution in Bayesian statistics.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Beta distribution is perfect for modeling probabilities or
/// percentages (values between 0 and 1). Its shape is very flexible:
/// - α = β = 1: Uniform distribution
/// - α = β &gt; 1: Bell-shaped, peaked at 0.5
/// - α &gt; β: Skewed toward 1 (high probabilities more likely)
/// - α &lt; β: Skewed toward 0 (low probabilities more likely)
/// It's used extensively in A/B testing and Bayesian inference.
/// </para>
/// <para>
/// PDF: f(x) = (x^(α-1) * (1-x)^(β-1)) / B(α,β) for x ∈ [0,1]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class BetaDistribution<T> : DistributionBase<T>
{
    private T _alpha;
    private T _beta;

    /// <summary>
    /// Initializes a new Beta distribution.
    /// </summary>
    /// <param name="alpha">The first shape parameter (α, must be positive).</param>
    /// <param name="beta">The second shape parameter (β, must be positive).</param>
    public BetaDistribution(T alpha, T beta)
    {
        if (NumOps.Compare(alpha, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be positive.");
        if (NumOps.Compare(beta, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be positive.");

        _alpha = alpha;
        _beta = beta;
    }

    /// <summary>
    /// Initializes a uniform Beta distribution (α=1, β=1).
    /// </summary>
    public BetaDistribution() : this(NumOps.One, NumOps.One)
    {
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override T[] Parameters
    {
        get => [_alpha, _beta];
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Beta distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[0], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Alpha must be positive.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Beta must be positive.");

            _alpha = value[0];
            _beta = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["alpha", "beta"];

    /// <summary>
    /// Gets the alpha shape parameter.
    /// </summary>
    public T Alpha => _alpha;

    /// <summary>
    /// Gets the beta shape parameter.
    /// </summary>
    public T Beta => _beta;

    /// <inheritdoc/>
    public override T Mean
    {
        get
        {
            // mean = α / (α + β)
            return NumOps.Divide(_alpha, NumOps.Add(_alpha, _beta));
        }
    }

    /// <inheritdoc/>
    public override T Variance
    {
        get
        {
            double a = NumOps.ToDouble(_alpha);
            double b = NumOps.ToDouble(_beta);
            double sum = a + b;
            return NumOps.FromDouble(a * b / (sum * sum * (sum + 1)));
        }
    }

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        if (xVal < 0 || xVal > 1)
            return Zero;

        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        // Handle edge cases
        if (xVal == 0)
        {
            if (a < 1) return NumOps.FromDouble(double.PositiveInfinity);
            if (a == 1) return NumOps.FromDouble(Math.Exp(LogBeta(a, b)));
            return Zero;
        }
        if (xVal == 1)
        {
            if (b < 1) return NumOps.FromDouble(double.PositiveInfinity);
            if (b == 1) return NumOps.FromDouble(Math.Exp(LogBeta(a, b)));
            return Zero;
        }

        double logPdf = (a - 1) * Math.Log(xVal) + (b - 1) * Math.Log(1 - xVal) - LogBeta(a, b);
        return NumOps.FromDouble(Math.Exp(logPdf));
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        if (xVal <= 0 || xVal >= 1)
            return NumOps.FromDouble(double.NegativeInfinity);

        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        return NumOps.FromDouble((a - 1) * Math.Log(xVal) + (b - 1) * Math.Log(1 - xVal) - LogBeta(a, b));
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        if (xVal <= 0) return Zero;
        if (xVal >= 1) return One;

        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        return NumOps.FromDouble(BetaIncomplete(a, b, xVal));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return One;

        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        // Newton-Raphson iteration
        double x = a / (a + b); // Initial guess = mean

        for (int i = 0; i < 100; i++)
        {
            double currentCdf = BetaIncomplete(a, b, x);
            double error = currentCdf - pVal;

            if (Math.Abs(error) < 1e-10)
                break;

            double pdf = Math.Exp((a - 1) * Math.Log(x) + (b - 1) * Math.Log(1 - x) - LogBeta(a, b));
            x = Math.Max(1e-10, Math.Min(1 - 1e-10, x - error / pdf));
        }

        return NumOps.FromDouble(x);
    }

    /// <inheritdoc/>
    public override T[] GradLogPdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        double psiA = Digamma(a);
        double psiB = Digamma(b);
        double psiAB = Digamma(a + b);

        // d/d(alpha) log(pdf) = log(x) - digamma(α) + digamma(α+β)
        double gradAlpha = Math.Log(xVal) - psiA + psiAB;

        // d/d(beta) log(pdf) = log(1-x) - digamma(β) + digamma(α+β)
        double gradBeta = Math.Log(1 - xVal) - psiB + psiAB;

        return [NumOps.FromDouble(gradAlpha), NumOps.FromDouble(gradBeta)];
    }

    /// <inheritdoc/>
    public override T[,] FisherInformation()
    {
        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        double psiA = Trigamma(a);
        double psiB = Trigamma(b);
        double psiAB = Trigamma(a + b);

        double iAlpha = psiA - psiAB;
        double iBeta = psiB - psiAB;
        double iAlphaBeta = -psiAB;

        return new T[,]
        {
            { NumOps.FromDouble(iAlpha), NumOps.FromDouble(iAlphaBeta) },
            { NumOps.FromDouble(iAlphaBeta), NumOps.FromDouble(iBeta) }
        };
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new BetaDistribution<T>(_alpha, _beta);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double a = NumOps.ToDouble(_alpha);
        double b = NumOps.ToDouble(_beta);

        // Generate using ratio of Gamma variates
        double x = SampleGamma(random, a, 1);
        double y = SampleGamma(random, b, 1);

        return NumOps.FromDouble(x / (x + y));
    }

    private static double LogBeta(double a, double b)
    {
        return LogGamma(a) + LogGamma(b) - LogGamma(a + b);
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

    /// <summary>
    /// Creates a Beta distribution from mean and variance.
    /// </summary>
    public static BetaDistribution<T> FromMeanVariance(T mean, T variance)
    {
        double m = NumOps.ToDouble(mean);
        double v = NumOps.ToDouble(variance);

        // α = m * ((m*(1-m)/v) - 1)
        // β = (1-m) * ((m*(1-m)/v) - 1)
        double common = m * (1 - m) / v - 1;
        double alpha = m * common;
        double beta = (1 - m) * common;

        return new BetaDistribution<T>(NumOps.FromDouble(alpha), NumOps.FromDouble(beta));
    }

    /// <summary>
    /// Creates a Beta distribution from a number of successes and failures (Bayesian posterior).
    /// </summary>
    public static BetaDistribution<T> FromSuccessFailure(int successes, int failures, T priorAlpha, T priorBeta)
    {
        T alpha = NumOps.Add(priorAlpha, NumOps.FromDouble(successes));
        T beta = NumOps.Add(priorBeta, NumOps.FromDouble(failures));
        return new BetaDistribution<T>(alpha, beta);
    }
}
