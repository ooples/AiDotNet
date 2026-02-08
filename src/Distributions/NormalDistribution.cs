namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Normal (Gaussian) probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Normal distribution is the most common probability distribution, characterized
/// by its bell-shaped curve. It is defined by two parameters: mean (μ) and variance (σ²).
/// </para>
/// <para>
/// <b>For Beginners:</b> The Normal distribution describes many natural phenomena
/// like heights, test scores, and measurement errors. Most values cluster around the
/// mean, with fewer values as you move away from the center. About 68% of values
/// fall within one standard deviation of the mean, 95% within two, and 99.7% within three.
/// </para>
/// <para>
/// PDF: f(x) = (1 / √(2πσ²)) * exp(-(x - μ)² / (2σ²))
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NormalDistribution<T> : DistributionBase<T>
{
    private T _mean;
    private T _variance;

    /// <summary>
    /// Initializes a new Normal distribution with specified mean and variance.
    /// </summary>
    /// <param name="mean">The mean (location parameter).</param>
    /// <param name="variance">The variance (must be positive).</param>
    public NormalDistribution(T mean, T variance)
    {
        if (NumOps.Compare(variance, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(variance), "Variance must be positive.");

        _mean = mean;
        _variance = variance;
    }

    /// <summary>
    /// Initializes a standard Normal distribution (mean=0, variance=1).
    /// </summary>
    public NormalDistribution() : this(NumOps.Zero, NumOps.One)
    {
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override Vector<T> Parameters
    {
        get => new Vector<T>(new[] { _mean, _variance });
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Normal distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Variance must be positive.");

            _mean = value[0];
            _variance = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["mean", "variance"];

    /// <inheritdoc/>
    public override T Mean => _mean;

    /// <inheritdoc/>
    public override T Variance => _variance;

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        T diff = NumOps.Subtract(x, _mean);
        T diffSquared = NumOps.Multiply(diff, diff);
        T twoVariance = NumOps.Multiply(Two, _variance);
        T exponent = NumOps.Negate(NumOps.Divide(diffSquared, twoVariance));
        T normalizer = NumOps.Multiply(SqrtTwoPi, NumOps.Sqrt(_variance));
        return NumOps.Divide(NumOps.Exp(exponent), normalizer);
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        T diff = NumOps.Subtract(x, _mean);
        T diffSquared = NumOps.Multiply(diff, diff);
        T twoVariance = NumOps.Multiply(Two, _variance);

        // log(pdf) = -0.5 * log(2π) - 0.5 * log(σ²) - (x-μ)²/(2σ²)
        T logNormalizer = NumOps.Add(
            NumOps.Multiply(Half, LogTwoPi),
            NumOps.Multiply(Half, NumOps.Log(_variance)));
        T quadraticTerm = NumOps.Divide(diffSquared, twoVariance);

        return NumOps.Negate(NumOps.Add(logNormalizer, quadraticTerm));
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double meanVal = NumOps.ToDouble(_mean);
        double stdVal = Math.Sqrt(NumOps.ToDouble(_variance));

        double z = (xVal - meanVal) / stdVal;
        return NumOps.FromDouble(StandardNormalCdf(z));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        double z = StandardNormalInverseCdf(pVal);
        double stdVal = Math.Sqrt(NumOps.ToDouble(_variance));
        double meanVal = NumOps.ToDouble(_mean);

        return NumOps.FromDouble(meanVal + stdVal * z);
    }

    /// <inheritdoc/>
    public override Vector<T> GradLogPdf(T x)
    {
        // d/d(mean) log(pdf) = (x - mean) / variance
        // d/d(variance) log(pdf) = -1/(2*variance) + (x-mean)²/(2*variance²)

        T diff = NumOps.Subtract(x, _mean);
        T diffSquared = NumOps.Multiply(diff, diff);
        T varianceSquared = NumOps.Multiply(_variance, _variance);

        T gradMean = NumOps.Divide(diff, _variance);
        T gradVariance = NumOps.Subtract(
            NumOps.Divide(diffSquared, NumOps.Multiply(Two, varianceSquared)),
            NumOps.Divide(One, NumOps.Multiply(Two, _variance)));

        return new Vector<T>(new[] { gradMean, gradVariance });
    }

    /// <inheritdoc/>
    public override Matrix<T> FisherInformation()
    {
        // Fisher Information Matrix for Normal(μ, σ²):
        // I = [[1/σ², 0], [0, 1/(2σ⁴)]]

        T oneOverVariance = NumOps.Divide(One, _variance);
        T varianceSquared = NumOps.Multiply(_variance, _variance);
        T oneOverTwoVarianceSquared = NumOps.Divide(One, NumOps.Multiply(Two, varianceSquared));

        return new Matrix<T>(new T[,]
        {
            { oneOverVariance, Zero },
            { Zero, oneOverTwoVarianceSquared }
        });
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new NormalDistribution<T>(_mean, _variance);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double z = BoxMullerNormal(random);
        double stdVal = Math.Sqrt(NumOps.ToDouble(_variance));
        double meanVal = NumOps.ToDouble(_mean);
        return NumOps.FromDouble(meanVal + stdVal * z);
    }

    /// <summary>
    /// Creates a Normal distribution from mean and standard deviation.
    /// </summary>
    public static NormalDistribution<T> FromMeanStdDev(T mean, T stdDev)
    {
        T variance = NumOps.Multiply(stdDev, stdDev);
        return new NormalDistribution<T>(mean, variance);
    }
}
