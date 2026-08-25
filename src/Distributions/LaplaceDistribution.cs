namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Laplace (double exponential) probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Laplace distribution is characterized by a sharp peak at its location parameter
/// and heavier tails than the Normal distribution. It's useful for modeling data with
/// outliers or for robust regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Laplace distribution looks like two exponential distributions
/// joined back-to-back. It has a pointy peak (unlike the Normal's rounded peak) and
/// fatter tails, making it more suitable when you expect more extreme values. It's
/// commonly used in robust statistics because it's less sensitive to outliers.
/// </para>
/// <para>
/// PDF: f(x) = (1 / 2b) * exp(-|x - μ| / b)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LaplaceDistribution<T> : DistributionBase<T>
{
    private T _location;  // μ
    private T _scale;     // b

    /// <summary>
    /// Initializes a new Laplace distribution with specified location and scale.
    /// </summary>
    /// <param name="location">The location parameter (μ).</param>
    /// <param name="scale">The scale parameter (b, must be positive).</param>
    public LaplaceDistribution(T location, T scale)
    {
        if (NumOps.Compare(scale, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be positive.");

        _location = location;
        _scale = scale;
    }

    /// <summary>
    /// Initializes a standard Laplace distribution (location=0, scale=1).
    /// </summary>
    public LaplaceDistribution() : this(NumOps.Zero, NumOps.One)
    {
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override Vector<T> Parameters
    {
        get => new Vector<T>(new[] { _location, _scale });
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Laplace distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Scale must be positive.");

            _location = value[0];
            _scale = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["location", "scale"];

    /// <summary>
    /// Gets the location parameter (μ).
    /// </summary>
    public T Location => _location;

    /// <summary>
    /// Gets the scale parameter (b).
    /// </summary>
    public T Scale => _scale;

    /// <inheritdoc/>
    public override T Mean => _location;

    /// <inheritdoc/>
    public override T Variance => NumOps.Multiply(Two, NumOps.Multiply(_scale, _scale));

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        T diff = NumOps.Subtract(x, _location);
        T absDiff = NumOps.Abs(diff);
        T exponent = NumOps.Negate(NumOps.Divide(absDiff, _scale));
        T normalizer = NumOps.Multiply(Two, _scale);
        return NumOps.Divide(NumOps.Exp(exponent), normalizer);
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        T diff = NumOps.Subtract(x, _location);
        T absDiff = NumOps.Abs(diff);

        // log(pdf) = -log(2b) - |x - μ| / b
        T logNormalizer = NumOps.Log(NumOps.Multiply(Two, _scale));
        T scaledAbsDiff = NumOps.Divide(absDiff, _scale);

        return NumOps.Negate(NumOps.Add(logNormalizer, scaledAbsDiff));
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double locVal = NumOps.ToDouble(_location);
        double scaleVal = NumOps.ToDouble(_scale);

        double diff = xVal - locVal;
        if (diff < 0)
        {
            return NumOps.FromDouble(0.5 * Math.Exp(diff / scaleVal));
        }
        else
        {
            return NumOps.FromDouble(1.0 - 0.5 * Math.Exp(-diff / scaleVal));
        }
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        double locVal = NumOps.ToDouble(_location);
        double scaleVal = NumOps.ToDouble(_scale);

        if (pVal < 0.5)
        {
            return NumOps.FromDouble(locVal + scaleVal * Math.Log(2 * pVal));
        }
        else
        {
            return NumOps.FromDouble(locVal - scaleVal * Math.Log(2 * (1 - pVal)));
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GradLogPdf(T x)
    {
        // d/d(location) log(pdf) = sign(x - location) / scale
        // d/d(scale) log(pdf) = -1/scale + |x - location| / scale²

        T diff = NumOps.Subtract(x, _location);
        double diffVal = NumOps.ToDouble(diff);
        T sign = NumOps.FromDouble(Math.Sign(diffVal));
        T absDiff = NumOps.Abs(diff);
        T scaleSquared = NumOps.Multiply(_scale, _scale);

        T gradLocation = NumOps.Divide(sign, _scale);
        T gradScale = NumOps.Subtract(
            NumOps.Divide(absDiff, scaleSquared),
            NumOps.Divide(One, _scale));

        return new Vector<T>(new[] { gradLocation, gradScale });
    }

    /// <inheritdoc/>
    public override Matrix<T> FisherInformation()
    {
        // Fisher Information Matrix for Laplace(μ, b):
        // I = [[1/b², 0], [0, 1/b²]]

        T scaleSquared = NumOps.Multiply(_scale, _scale);
        T oneOverScaleSquared = NumOps.Divide(One, scaleSquared);

        return new Matrix<T>(new T[,]
        {
            { oneOverScaleSquared, Zero },
            { Zero, oneOverScaleSquared }
        });
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new LaplaceDistribution<T>(_location, _scale);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double u = random.NextDouble() - 0.5;
        double scaleVal = NumOps.ToDouble(_scale);
        double locVal = NumOps.ToDouble(_location);

        // Inverse CDF sampling
        double sample = locVal - scaleVal * Math.Sign(u) * Math.Log(1 - 2 * Math.Abs(u));
        return NumOps.FromDouble(sample);
    }
}
