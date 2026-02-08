namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Log-Normal probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Log-Normal distribution is used for modeling positive-valued data where the logarithm
/// of the variable follows a Normal distribution. Common applications include stock prices,
/// income distributions, and particle sizes.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you take the natural log of values from a Log-Normal distribution,
/// you get a Normal distribution. This is useful for data that's always positive and tends to
/// be right-skewed (long tail on the right). Think of things that grow by percentages, like
/// investment returns or biological growth.
/// </para>
/// <para>
/// PDF: f(x) = (1 / (xσ√(2π))) * exp(-(ln(x) - μ)² / (2σ²)) for x > 0
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LogNormalDistribution<T> : DistributionBase<T>
{
    private T _mu;      // μ - mean of log(X)
    private T _sigma;   // σ - std dev of log(X)

    /// <summary>
    /// Initializes a new Log-Normal distribution.
    /// </summary>
    /// <param name="mu">The mean of the underlying Normal distribution (location of log(X)).</param>
    /// <param name="sigma">The standard deviation of the underlying Normal distribution (must be positive).</param>
    public LogNormalDistribution(T mu, T sigma)
    {
        if (NumOps.Compare(sigma, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(sigma), "Sigma must be positive.");

        _mu = mu;
        _sigma = sigma;
    }

    /// <summary>
    /// Initializes a standard Log-Normal distribution (μ=0, σ=1).
    /// </summary>
    public LogNormalDistribution() : this(NumOps.Zero, NumOps.One)
    {
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override Vector<T> Parameters
    {
        get => new Vector<T>(new[] { _mu, _sigma });
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Log-Normal distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Sigma must be positive.");

            _mu = value[0];
            _sigma = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["mu", "sigma"];

    /// <summary>
    /// Gets the mu parameter (mean of log(X)).
    /// </summary>
    public T Mu => _mu;

    /// <summary>
    /// Gets the sigma parameter (std dev of log(X)).
    /// </summary>
    public T Sigma => _sigma;

    /// <inheritdoc/>
    public override T Mean
    {
        get
        {
            double mu = NumOps.ToDouble(_mu);
            double sigma = NumOps.ToDouble(_sigma);
            return NumOps.FromDouble(Math.Exp(mu + sigma * sigma / 2));
        }
    }

    /// <inheritdoc/>
    public override T Variance
    {
        get
        {
            double mu = NumOps.ToDouble(_mu);
            double sigma = NumOps.ToDouble(_sigma);
            double sigmaSquared = sigma * sigma;
            return NumOps.FromDouble((Math.Exp(sigmaSquared) - 1) * Math.Exp(2 * mu + sigmaSquared));
        }
    }

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return Zero;

        double xVal = NumOps.ToDouble(x);
        double mu = NumOps.ToDouble(_mu);
        double sigma = NumOps.ToDouble(_sigma);

        double logX = Math.Log(xVal);
        double z = (logX - mu) / sigma;

        double pdf = Math.Exp(-0.5 * z * z) / (xVal * sigma * Math.Sqrt(2 * Math.PI));
        return NumOps.FromDouble(pdf);
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        double xVal = NumOps.ToDouble(x);
        double mu = NumOps.ToDouble(_mu);
        double sigma = NumOps.ToDouble(_sigma);

        double logX = Math.Log(xVal);
        double z = (logX - mu) / sigma;

        double logPdf = -0.5 * Math.Log(2 * Math.PI) - Math.Log(sigma) - Math.Log(xVal) - 0.5 * z * z;
        return NumOps.FromDouble(logPdf);
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return Zero;

        double xVal = NumOps.ToDouble(x);
        double mu = NumOps.ToDouble(_mu);
        double sigma = NumOps.ToDouble(_sigma);

        double z = (Math.Log(xVal) - mu) / sigma;
        return NumOps.FromDouble(StandardNormalCdf(z));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        double mu = NumOps.ToDouble(_mu);
        double sigma = NumOps.ToDouble(_sigma);

        double z = StandardNormalInverseCdf(pVal);
        return NumOps.FromDouble(Math.Exp(mu + sigma * z));
    }

    /// <inheritdoc/>
    public override Vector<T> GradLogPdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double mu = NumOps.ToDouble(_mu);
        double sigma = NumOps.ToDouble(_sigma);

        double logX = Math.Log(xVal);
        double z = (logX - mu) / sigma;

        // d/d(mu) log(pdf) = (log(x) - μ) / σ²
        double gradMu = z / sigma;

        // d/d(sigma) log(pdf) = -1/σ + (log(x) - μ)² / σ³
        double gradSigma = -1 / sigma + z * z / sigma;

        return new Vector<T>(new[] { NumOps.FromDouble(gradMu), NumOps.FromDouble(gradSigma) });
    }

    /// <inheritdoc/>
    public override Matrix<T> FisherInformation()
    {
        double sigma = NumOps.ToDouble(_sigma);

        // Fisher Information for LogNormal(μ, σ):
        // I = [[1/σ², 0], [0, 2/σ²]]

        double iMu = 1.0 / (sigma * sigma);
        double iSigma = 2.0 / (sigma * sigma);

        return new Matrix<T>(new T[,]
        {
            { NumOps.FromDouble(iMu), Zero },
            { Zero, NumOps.FromDouble(iSigma) }
        });
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new LogNormalDistribution<T>(_mu, _sigma);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double mu = NumOps.ToDouble(_mu);
        double sigma = NumOps.ToDouble(_sigma);

        double z = BoxMullerNormal(random);
        return NumOps.FromDouble(Math.Exp(mu + sigma * z));
    }

    /// <summary>
    /// Creates a Log-Normal distribution from the desired mean and variance of X (not log(X)).
    /// </summary>
    public static LogNormalDistribution<T> FromMeanVariance(T mean, T variance)
    {
        double m = NumOps.ToDouble(mean);
        double v = NumOps.ToDouble(variance);

        double mu = Math.Log(m * m / Math.Sqrt(v + m * m));
        double sigma = Math.Sqrt(Math.Log(1 + v / (m * m)));

        return new LogNormalDistribution<T>(NumOps.FromDouble(mu), NumOps.FromDouble(sigma));
    }
}
