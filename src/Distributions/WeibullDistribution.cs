namespace AiDotNet.Distributions;

/// <summary>
/// Represents a Weibull probability distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Weibull distribution is widely used in reliability engineering, survival analysis,
/// and extreme value theory. It can model increasing, decreasing, or constant failure rates
/// depending on the shape parameter.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Weibull distribution is extremely flexible for modeling
/// "time until failure" data:
/// - Shape &lt; 1: Failure rate decreases over time (infant mortality)
/// - Shape = 1: Constant failure rate (random failures) - same as Exponential
/// - Shape &gt; 1: Failure rate increases over time (wear-out failures)
/// This makes it ideal for predicting when machines or products will fail.
/// </para>
/// <para>
/// PDF: f(x) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k) for x ≥ 0
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class WeibullDistribution<T> : DistributionBase<T>
{
    private T _shape;  // k
    private T _scale;  // λ

    /// <summary>
    /// Initializes a new Weibull distribution.
    /// </summary>
    /// <param name="shape">The shape parameter (k, must be positive).</param>
    /// <param name="scale">The scale parameter (λ, must be positive).</param>
    public WeibullDistribution(T shape, T scale)
    {
        if (NumOps.Compare(shape, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be positive.");
        if (NumOps.Compare(scale, Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be positive.");

        _shape = shape;
        _scale = scale;
    }

    /// <inheritdoc/>
    public override int NumParameters => 2;

    /// <inheritdoc/>
    public override Vector<T> Parameters
    {
        get => new Vector<T>(new[] { _shape, _scale });
        set
        {
            if (value.Length != 2)
                throw new ArgumentException("Weibull distribution requires exactly 2 parameters.");
            if (NumOps.Compare(value[0], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Shape must be positive.");
            if (NumOps.Compare(value[1], Zero) <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Scale must be positive.");

            _shape = value[0];
            _scale = value[1];
        }
    }

    /// <inheritdoc/>
    public override string[] ParameterNames => ["shape", "scale"];

    /// <summary>
    /// Gets the shape parameter (k).
    /// </summary>
    public T Shape => _shape;

    /// <summary>
    /// Gets the scale parameter (λ).
    /// </summary>
    public T Scale => _scale;

    /// <inheritdoc/>
    public override T Mean
    {
        get
        {
            double k = NumOps.ToDouble(_shape);
            double lambda = NumOps.ToDouble(_scale);
            return NumOps.FromDouble(lambda * Gamma(1 + 1 / k));
        }
    }

    /// <inheritdoc/>
    public override T Variance
    {
        get
        {
            double k = NumOps.ToDouble(_shape);
            double lambda = NumOps.ToDouble(_scale);
            double g1 = Gamma(1 + 1 / k);
            double g2 = Gamma(1 + 2 / k);
            return NumOps.FromDouble(lambda * lambda * (g2 - g1 * g1));
        }
    }

    /// <inheritdoc/>
    public override T Pdf(T x)
    {
        if (NumOps.Compare(x, Zero) < 0)
            return Zero;
        if (NumOps.Compare(x, Zero) == 0)
        {
            double k = NumOps.ToDouble(_shape);
            // For k < 1, the theoretical PDF at x = 0 is +∞.
            if (k < 1) return NumOps.FromDouble(double.PositiveInfinity);
            if (k == 1) return NumOps.Divide(_shape, _scale);
            return Zero;
        }

        double xVal = NumOps.ToDouble(x);
        double kVal = NumOps.ToDouble(_shape);
        double lambdaVal = NumOps.ToDouble(_scale);

        double xOverLambda = xVal / lambdaVal;
        double pdf = (kVal / lambdaVal) * Math.Pow(xOverLambda, kVal - 1) * Math.Exp(-Math.Pow(xOverLambda, kVal));

        return NumOps.FromDouble(pdf);
    }

    /// <inheritdoc/>
    public override T LogPdf(T x)
    {
        if (NumOps.Compare(x, Zero) < 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        if (NumOps.Compare(x, Zero) == 0)
        {
            double k = NumOps.ToDouble(_shape);
            // For k < 1, log(PDF) at x = 0 is +∞ (since PDF is +∞).
            if (k < 1) return NumOps.FromDouble(double.PositiveInfinity);
            if (k == 1) return NumOps.FromDouble(Math.Log(NumOps.ToDouble(_shape)) - Math.Log(NumOps.ToDouble(_scale)));
            return NumOps.FromDouble(double.NegativeInfinity);
        }

        double xVal = NumOps.ToDouble(x);
        double kVal = NumOps.ToDouble(_shape);
        double lambdaVal = NumOps.ToDouble(_scale);

        double logPdf = Math.Log(kVal) - Math.Log(lambdaVal) +
                        (kVal - 1) * (Math.Log(xVal) - Math.Log(lambdaVal)) -
                        Math.Pow(xVal / lambdaVal, kVal);

        return NumOps.FromDouble(logPdf);
    }

    /// <inheritdoc/>
    public override T Cdf(T x)
    {
        if (NumOps.Compare(x, Zero) <= 0)
            return Zero;

        double xVal = NumOps.ToDouble(x);
        double kVal = NumOps.ToDouble(_shape);
        double lambdaVal = NumOps.ToDouble(_scale);

        return NumOps.FromDouble(1 - Math.Exp(-Math.Pow(xVal / lambdaVal, kVal)));
    }

    /// <inheritdoc/>
    public override T InverseCdf(T p)
    {
        double pVal = NumOps.ToDouble(p);
        if (pVal < 0 || pVal > 1)
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be in [0, 1].");

        if (pVal == 0) return Zero;
        if (pVal == 1) return NumOps.FromDouble(double.PositiveInfinity);

        double kVal = NumOps.ToDouble(_shape);
        double lambdaVal = NumOps.ToDouble(_scale);

        // x = λ * (-ln(1-p))^(1/k)
        return NumOps.FromDouble(lambdaVal * Math.Pow(-Math.Log(1 - pVal), 1 / kVal));
    }

    /// <inheritdoc/>
    public override Vector<T> GradLogPdf(T x)
    {
        double xVal = NumOps.ToDouble(x);
        if (xVal <= 0)
            return new Vector<T>(new[] { NumOps.FromDouble(double.NaN), NumOps.FromDouble(double.NaN) });

        double k = NumOps.ToDouble(_shape);
        double lambda = NumOps.ToDouble(_scale);

        double xOverLambda = xVal / lambda;
        double xOverLambdaK = Math.Pow(xOverLambda, k);

        // d/d(shape) log(pdf)
        double gradShape = 1 / k + Math.Log(xOverLambda) * (1 - xOverLambdaK);

        // d/d(scale) log(pdf)
        double gradScale = -k / lambda + k * Math.Pow(xVal, k) / Math.Pow(lambda, k + 1);

        return new Vector<T>(new[] { NumOps.FromDouble(gradShape), NumOps.FromDouble(gradScale) });
    }

    /// <inheritdoc/>
    public override Matrix<T> FisherInformation()
    {
        double k = NumOps.ToDouble(_shape);
        double lambda = NumOps.ToDouble(_scale);

        // Euler-Mascheroni constant
        const double gamma = 0.5772156649;

        double iShape = ((1 - gamma) * (1 - gamma) + Math.PI * Math.PI / 6) / (k * k);
        double iScale = k * k / (lambda * lambda);
        double iShapeScale = -(1 - gamma) / lambda;

        return new Matrix<T>(new T[,]
        {
            { NumOps.FromDouble(iShape), NumOps.FromDouble(iShapeScale) },
            { NumOps.FromDouble(iShapeScale), NumOps.FromDouble(iScale) }
        });
    }

    /// <inheritdoc/>
    public override IParametricDistribution<T> Clone()
    {
        return new WeibullDistribution<T>(_shape, _scale);
    }

    /// <inheritdoc/>
    public override T Sample(Random random)
    {
        double k = NumOps.ToDouble(_shape);
        double lambda = NumOps.ToDouble(_scale);

        // Inverse CDF sampling
        double u = random.NextDouble();
        return NumOps.FromDouble(lambda * Math.Pow(-Math.Log(1 - u), 1 / k));
    }
}
