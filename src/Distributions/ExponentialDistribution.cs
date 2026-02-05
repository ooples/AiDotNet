namespace AiDotNet.Distributions;

/// <summary>
/// Represents an Exponential probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Exponential distribution describes the time between events in a Poisson point process,
/// where events occur continuously and independently at a constant average rate.
/// It's the continuous analog of the geometric distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Exponential distribution models "waiting times" - how long until
/// something happens. For example: time until the next customer arrives, time until a light bulb
/// fails, or time until the next earthquake. It has a "memoryless" property - the probability
/// of waiting another 5 minutes is the same whether you've waited 1 minute or 1 hour already.
/// </para>
/// <para>
/// PDF: f(x) = λ * exp(-λx) for x ≥ 0
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExponentialDistribution<T> : DistributionBase<T>
{
    private T _rate;  // λ (lambda)

    /// <summary>
    /// Initializes a new Exponential distribution with specified rate.
    /// </summary>
    /// <param name="rate">The rate parameter (λ, must be positive). Events per unit time.</param>
    public ExponentialDistribution(T rate)
    {
        if (NumOps.Compare(rate, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(rate), "Rate must be positive.");

        _rate = rate;
    }

    /// <summary>
    /// Initializes a standard Exponential distribution (rate=1).
    /// </summary>
    public ExponentialDistribution() : this(NumOps.One)
    {
    }

    /// <inheritdoc/>
    public override int NumParameters => 1;

    /// <inheritdoc/>
    public override T[] Parameters
    {
        get => [_rate];
        set
        {
            if (value.Length != 1)
                throw new ArgumentException("Exponential distribution requires exactly 1 parameter.");
            if (NumOps.Compare(value[0], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Rate must be positive.");

            _rate = value[0];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["rate"];

    /// <summary>
    /// Gets the rate parameter (λ).
    /// </summary>
    public T Rate => _rate;

    /// <summary>
    /// Gets the scale parameter (1/λ), also called the mean.
    /// </summary>
    public T Scale => NumOps.Divide(One, _rate);

    /// <inheritdoc/>
    public override T Mean => NumOps.Divide(One, _rate);

    /// <inheritdoc/>
    public override T Variance => NumOps.Divide(One, NumOps.Multiply(_rate, _rate));

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        if (NumOps.Compare(x, Zero) < 0)
            return Zero;

        T exponent = NumOps.Negate(NumOps.Multiply(_rate, x));
        return NumOps.Multiply(_rate, NumOps.Exp(exponent));
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        if (NumOps.Compare(x, Zero) < 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        // log(pdf) = log(λ) - λx
        return NumOps.Subtract(NumOps.Log(_rate), NumOps.Multiply(_rate, x));
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        if (NumOps.Compare(x, Zero) < 0)
            return Zero;

        // CDF = 1 - exp(-λx)
        T exponent = NumOps.Negate(NumOps.Multiply(_rate, x));
        return NumOps.Subtract(One, NumOps.Exp(exponent));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        // x = -log(1-p) / λ
        double rate = NumOps.ToDouble(_rate);
        return NumOps.FromDouble(-Math.Log(1 - pVal) / rate);
    }

    /// <inheritdoc/>
    public override T[] GradLogPdf(T x)
    {
        // d/d(rate) log(pdf) = 1/λ - x
        T gradRate = NumOps.Subtract(NumOps.Divide(One, _rate), x);
        return [gradRate];
    }

    /// <inheritdoc/>
    public override T[,] FisherInformation()
    {
        // Fisher Information for Exponential(λ):
        // I = 1/λ²
        T rateSquared = NumOps.Multiply(_rate, _rate);
        return new T[,]
        {
            { NumOps.Divide(One, rateSquared) }
        };
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new ExponentialDistribution<T>(_rate);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double rate = NumOps.ToDouble(_rate);
        // Inverse CDF sampling
        return NumOps.FromDouble(-Math.Log(1 - random.NextDouble()) / rate);
    }

    /// <summary>
    /// Creates an Exponential distribution from the mean.
    /// </summary>
    public static ExponentialDistribution<T> FromMean(T mean)
    {
        T rate = NumOps.Divide(NumOps.One, mean);
        return new ExponentialDistribution<T>(rate);
    }
}
