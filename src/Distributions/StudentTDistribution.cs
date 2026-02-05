namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Student's t probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Student's t distribution is used when estimating the mean of a normally distributed
/// population when the sample size is small and the population standard deviation is unknown.
/// It has heavier tails than the Normal distribution, approaching Normal as degrees of freedom increase.
/// </para>
/// <para>
/// <b>For Beginners:</b> The t-distribution is like a "wider" version of the Normal distribution.
/// It's more spread out, especially in the tails, which makes it useful when you're less certain
/// about your data (like with small samples). As you get more data, it becomes more like a Normal.
/// With 30+ degrees of freedom, it's practically identical to Normal.
/// </para>
/// <para>
/// PDF: f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + (x-μ)²/(νσ²))^(-(ν+1)/2)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class StudentTDistribution<T> : DistributionBase<T>
{
    private T _location;      // μ
    private T _scale;         // σ
    private T _degreesOfFreedom;  // ν

    /// <summary>
    /// Initializes a new Student's t distribution.
    /// </summary>
    /// <param name="location">The location parameter (μ).</param>
    /// <param name="scale">The scale parameter (σ, must be positive).</param>
    /// <param name="degreesOfFreedom">Degrees of freedom (ν, must be positive).</param>
    public StudentTDistribution(T location, T scale, T degreesOfFreedom)
    {
        if (NumOps.Compare(scale, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be positive.");
        if (NumOps.Compare(degreesOfFreedom, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), "Degrees of freedom must be positive.");

        _location = location;
        _scale = scale;
        _degreesOfFreedom = degreesOfFreedom;
    }

    /// <summary>
    /// Initializes a standard Student's t distribution with given degrees of freedom.
    /// </summary>
    public StudentTDistribution(T degreesOfFreedom)
        : this(NumOps.Zero, NumOps.One, degreesOfFreedom)
    {
    }

    /// <inheritdoc/>
    public override int NumParameters => 3;

    /// <inheritdoc/>
    public override T[] Parameters
    {
        get => [_location, _scale, _degreesOfFreedom];
        set
        {
            if (value.Length != 3)
                throw new ArgumentException("Student's t distribution requires exactly 3 parameters.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Scale must be positive.");
            if (NumOps.Compare(value[2], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Degrees of freedom must be positive.");

            _location = value[0];
            _scale = value[1];
            _degreesOfFreedom = value[2];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["location", "scale", "degreesOfFreedom"];

    /// <summary>
    /// Gets the location parameter.
    /// </summary>
    public T Location => _location;

    /// <summary>
    /// Gets the scale parameter.
    /// </summary>
    public T Scale => _scale;

    /// <summary>
    /// Gets the degrees of freedom.
    /// </summary>
    public T DegreesOfFreedom => _degreesOfFreedom;

    /// <inheritdoc/>
    public override T Mean
    {
        get
        {
            double nu = NumOps.ToDouble(_degreesOfFreedom);
            if (nu <= 1)
                throw new InvalidOperationException("Mean is undefined for ν ≤ 1.");
            return _location;
        }
    }

    /// <inheritdoc/>
    public override T Variance
    {
        get
        {
            double nu = NumOps.ToDouble(_degreesOfFreedom);
            if (nu <= 2)
                throw new InvalidOperationException("Variance is undefined for ν ≤ 2.");

            double scaleVal = NumOps.ToDouble(_scale);
            return NumOps.FromDouble(scaleVal * scaleVal * nu / (nu - 2));
        }
    }

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double locVal = NumOps.ToDouble(_location);
        double scaleVal = NumOps.ToDouble(_scale);
        double nu = NumOps.ToDouble(_degreesOfFreedom);

        double z = (xVal - locVal) / scaleVal;
        double zSquared = z * z;

        double logNumerator = LogGamma((nu + 1) / 2);
        double logDenominator = Math.Log(Math.Sqrt(nu * Math.PI)) + LogGamma(nu / 2) + Math.Log(scaleVal);
        double logBase = Math.Log(1 + zSquared / nu);

        double logPdf = logNumerator - logDenominator - ((nu + 1) / 2) * logBase;
        return NumOps.FromDouble(Math.Exp(logPdf));
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double locVal = NumOps.ToDouble(_location);
        double scaleVal = NumOps.ToDouble(_scale);
        double nu = NumOps.ToDouble(_degreesOfFreedom);

        double z = (xVal - locVal) / scaleVal;
        double zSquared = z * z;

        double logNumerator = LogGamma((nu + 1) / 2);
        double logDenominator = Math.Log(Math.Sqrt(nu * Math.PI)) + LogGamma(nu / 2) + Math.Log(scaleVal);
        double logBase = Math.Log(1 + zSquared / nu);

        return NumOps.FromDouble(logNumerator - logDenominator - ((nu + 1) / 2) * logBase);
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double locVal = NumOps.ToDouble(_location);
        double scaleVal = NumOps.ToDouble(_scale);
        double nu = NumOps.ToDouble(_degreesOfFreedom);

        double z = (xVal - locVal) / scaleVal;
        double t = nu / (nu + z * z);

        if (z < 0)
        {
            return NumOps.FromDouble(0.5 * BetaIncomplete(nu / 2, 0.5, t));
        }
        else
        {
            return NumOps.FromDouble(1 - 0.5 * BetaIncomplete(nu / 2, 0.5, t));
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

        // Newton-Raphson iteration
        double z = StandardNormalInverseCdf(pVal); // Initial guess
        double nu = NumOps.ToDouble(_degreesOfFreedom);

        for (int i = 0; i < 50; i++)
        {
            double t = nu / (nu + z * z);
            double currentCdf = z < 0
                ? 0.5 * BetaIncomplete(nu / 2, 0.5, t)
                : 1 - 0.5 * BetaIncomplete(nu / 2, 0.5, t);

            double error = currentCdf - pVal;
            if (Math.Abs(error) < 1e-10)
                break;

            // PDF for derivative
            double logNumerator = LogGamma((nu + 1) / 2);
            double logDenominator = Math.Log(Math.Sqrt(nu * Math.PI)) + LogGamma(nu / 2);
            double pdf = Math.Exp(logNumerator - logDenominator) * Math.Pow(1 + z * z / nu, -(nu + 1) / 2);

            z -= error / pdf;
        }

        return NumOps.FromDouble(locVal + scaleVal * z);
    }

    /// <inheritdoc/>
    public override T[] GradLogPdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        double locVal = NumOps.ToDouble(_location);
        double scaleVal = NumOps.ToDouble(_scale);
        double nu = NumOps.ToDouble(_degreesOfFreedom);

        double z = (xVal - locVal) / scaleVal;
        double zSquared = z * z;
        double factor = 1 + zSquared / nu;

        // d/d(location) log(pdf)
        double gradLocation = (nu + 1) * z / (nu * factor * scaleVal);

        // d/d(scale) log(pdf)
        double gradScale = -1 / scaleVal + (nu + 1) * zSquared / (nu * factor * scaleVal);

        // d/d(nu) log(pdf) - more complex involving digamma
        double gradNu = 0.5 * (Digamma((nu + 1) / 2) - Digamma(nu / 2) - 1 / nu - Math.Log(factor) + (nu + 1) * zSquared / (nu * nu * factor));

        return [NumOps.FromDouble(gradLocation), NumOps.FromDouble(gradScale), NumOps.FromDouble(gradNu)];
    }

    /// <inheritdoc/>
    public override T[,] FisherInformation()
    {
        double scaleVal = NumOps.ToDouble(_scale);
        double nu = NumOps.ToDouble(_degreesOfFreedom);

        // Approximate Fisher Information
        double iLoc = (nu + 1) / ((nu + 3) * scaleVal * scaleVal);
        double iScale = 2 * nu / ((nu + 3) * scaleVal * scaleVal);
        double iNu = 0.5 * (Trigamma((nu + 1) / 2) - Trigamma(nu / 2) - 2 * (nu + 5) / (nu * (nu + 1) * (nu + 3)));

        return new T[,]
        {
            { NumOps.FromDouble(iLoc), Zero, Zero },
            { Zero, NumOps.FromDouble(iScale), Zero },
            { Zero, Zero, NumOps.FromDouble(Math.Max(iNu, 1e-10)) }
        };
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new StudentTDistribution<T>(_location, _scale, _degreesOfFreedom);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double nu = NumOps.ToDouble(_degreesOfFreedom);
        double scaleVal = NumOps.ToDouble(_scale);
        double locVal = NumOps.ToDouble(_location);

        // Generate t-distributed sample using ratio of Normal and Chi-squared
        double z = BoxMullerNormal(random);

        // Generate Chi-squared with nu degrees of freedom
        double chi2 = 0;
        for (int i = 0; i < (int)nu; i++)
        {
            double normal = BoxMullerNormal(random);
            chi2 += normal * normal;
        }

        // For non-integer nu, use gamma distribution approximation
        if (Math.Abs(nu - Math.Round(nu)) > 1e-10)
        {
            chi2 = SampleGamma(random, nu / 2, 0.5);
        }

        double t = z / Math.Sqrt(chi2 / nu);
        return NumOps.FromDouble(locVal + scaleVal * t);
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
}
