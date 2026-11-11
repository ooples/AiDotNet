namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PCHIP interpolation creates a smooth curve that passes through all data points while
/// preserving the shape of the data, particularly maintaining monotonicity (keeping the same
/// direction of increase or decrease between points).
/// </para>
/// <para>
/// <b>For Beginners:</b> PCHIP is a method that creates a smooth curve through your data points.
/// Unlike some other methods, it avoids creating artificial "wiggles" in the curve,
/// making it particularly useful for scientific data where you want to maintain the
/// general shape and trends of your original data.
/// </para>
/// </remarks>
public partial class PchipInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the data points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The calculated slopes at each data point.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _slopes;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the PCHIP interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths or when there are fewer than 2 points.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your data points (x and y values) and prepares
    /// the interpolation algorithm. It checks that your data is valid (same number of x and y values,
    /// and at least 2 points) and then calculates the slopes needed for the interpolation.
    /// </para>
    /// </remarks>
    public PchipInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 2)
            throw new ArgumentException("PCHIP interpolation requires at least 2 points.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _slopes = CalculateSlopes();
    }

    /// <summary>
    /// Interpolates a y-value for the given x-value using PCHIP interpolation.
    /// </summary>
    /// <param name="x">The x-value for which to interpolate.</param>
    /// <returns>The interpolated y-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the y-value on the smooth curve for any x-value you provide.
    /// </para>
    /// <para>
    /// It works by:
    /// 1. Finding which segment of the curve your x-value falls into
    /// 2. Calculating where exactly in that segment the x-value is (as a percentage)
    /// 3. Using special formulas (Hermite basis functions) to blend the values and slopes
    ///    at the endpoints of that segment
    /// 4. Returning the final calculated y-value on the curve
    /// </para>
    /// </remarks>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T h = _numOps.Subtract(_x[i + 1], _x[i]);
        T t = _numOps.Divide(_numOps.Subtract(x, _x[i]), h);

        T h00 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), _numOps.Power(t, _numOps.FromDouble(3))), _numOps.Multiply(_numOps.FromDouble(-3), _numOps.Power(t, _numOps.FromDouble(2))));
        h00 = _numOps.Add(h00, _numOps.One);

        T h10 = _numOps.Add(_numOps.Power(t, _numOps.FromDouble(3)), _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Power(t, _numOps.FromDouble(2))));
        h10 = _numOps.Add(h10, t);

        T h01 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(-2), _numOps.Power(t, _numOps.FromDouble(3))), _numOps.Multiply(_numOps.FromDouble(3), _numOps.Power(t, _numOps.FromDouble(2))));

        T h11 = _numOps.Subtract(_numOps.Power(t, _numOps.FromDouble(3)), _numOps.Power(t, _numOps.FromDouble(2)));

        T result = _numOps.Add(_numOps.Multiply(h00, _y[i]), _numOps.Multiply(h10, _numOps.Multiply(h, _slopes[i])));
        result = _numOps.Add(result, _numOps.Multiply(h01, _y[i + 1]));
        result = _numOps.Add(result, _numOps.Multiply(h11, _numOps.Multiply(h, _slopes[i + 1])));

        return result;
    }

    /// <summary>
    /// Calculates the derivative at each data point the way Fritsch and Butland (1984) do, as SciPy's
    /// <c>PchipInterpolator</c> does.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how steep the curve should be at each data point.
    /// </para>
    /// <para>
    /// With <c>h_k</c> the interval widths and <c>m_k</c> the secant slopes, an interior point whose neighbouring
    /// secants have the same sign gets the weighted harmonic mean
    /// <c>(w1 + w2) / (w1 / m_(k-1) + w2 / m_k)</c>, where <c>w1 = 2 h_k + h_(k-1)</c> and <c>w2 = h_k + 2 h_(k-1)</c>.
    /// A point where the secants change sign, or either is zero, is a local extremum and gets slope 0. Each end uses
    /// the one-sided three-point estimate, set to 0 when its sign disagrees with the first (last) secant and limited
    /// to three times that secant when the data turns there. These slopes keep every segment monotone without a
    /// separate limiting pass. Two points give a straight line.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateSlopes()
    {
        var engine = AiDotNetEngine.Current;
        int n = _x.Length;
        Tensor<T> xs = _x.AsTensor();
        Tensor<T> ys = _y.AsTensor();

        Tensor<T> widths = engine.TensorSubtract(Slice(xs, 1, n - 1), Slice(xs, 0, n - 1));
        Tensor<T> secants = engine.TensorDivide(engine.TensorSubtract(Slice(ys, 1, n - 1), Slice(ys, 0, n - 1)), widths);
        if (n == 2)
        {
            return engine.TensorConcatenate(new[] { secants, secants }, 0).ToVector();
        }

        Tensor<T> left = Slice(secants, 0, n - 2);
        Tensor<T> right = Slice(secants, 1, n - 2);
        Tensor<T> hLeft = Slice(widths, 0, n - 2);
        Tensor<T> hRight = Slice(widths, 1, n - 2);
        Tensor<T> w1 = engine.TensorAdd(engine.TensorMultiplyScalar(hRight, _numOps.FromDouble(2)), hLeft);
        Tensor<T> w2 = engine.TensorAdd(hRight, engine.TensorMultiplyScalar(hLeft, _numOps.FromDouble(2)));
        // Same-sign, nonzero neighbours keep the mean; a sign change or a flat neighbour is an extremum with slope 0.
        // Masked-out secants are replaced by 1 before dividing (a decimal division by a zero secant throws).
        Tensor<T> sameSign = engine.TensorGreaterThan(engine.TensorMultiply(left, right), _numOps.Zero);
        Tensor<T> ones = engine.TensorOnesLike(left);
        Tensor<T> safeLeft = engine.TensorWhere(sameSign, left, ones);
        Tensor<T> safeRight = engine.TensorWhere(sameSign, right, ones);
        Tensor<T> interior = engine.TensorDivide(
            engine.TensorAdd(w1, w2),
            engine.TensorAdd(engine.TensorDivide(w1, safeLeft), engine.TensorDivide(w2, safeRight)));
        interior = engine.TensorWhere(sameSign, interior, engine.TensorZerosLike(interior));

        var slopes = new[]
        {
            Tensor<T>.FromScalar(EndSlope(widths[0], widths[1], secants[0], secants[1])),
            interior,
            Tensor<T>.FromScalar(EndSlope(widths[n - 2], widths[n - 3], secants[n - 2], secants[n - 3]))
        };
        return engine.TensorConcatenate(slopes, 0).ToVector();
    }

    /// <summary>
    /// The one-sided three-point derivative at an end point, limited so the end segment stays monotone.
    /// </summary>
    /// <param name="h0">The width of the end interval.</param>
    /// <param name="h1">The width of the interval next to it.</param>
    /// <param name="m0">The secant slope of the end interval.</param>
    /// <param name="m1">The secant slope of the interval next to it.</param>
    private T EndSlope(T h0, T h1, T m0, T m1)
    {
        T two = _numOps.FromDouble(2);
        T three = _numOps.FromDouble(3);
        T d = _numOps.Divide(
            _numOps.Subtract(_numOps.Multiply(_numOps.Add(_numOps.Multiply(two, h0), h1), m0), _numOps.Multiply(h0, m1)),
            _numOps.Add(h0, h1));

        if (!_numOps.Equals(_numOps.SignOrZero(d), _numOps.SignOrZero(m0)))
        {
            return _numOps.Zero;
        }

        bool turns = !_numOps.Equals(_numOps.SignOrZero(m0), _numOps.SignOrZero(m1));
        return turns && _numOps.GreaterThan(_numOps.Abs(d), _numOps.Multiply(three, _numOps.Abs(m0)))
            ? _numOps.Multiply(three, m0)
            : d;
    }

    /// <summary>
    /// Finds the appropriate interval in the data points for the given x-value.
    /// </summary>
    /// <param name="x">The x-value for which to find the corresponding interval.</param>
    /// <returns>The index of the lower bound of the interval where the x-value belongs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines which segment of your data the x-value falls into.
    /// </para>
    /// <para>
    /// Imagine your data points as dots connected by line segments:
    /// 
    ///     Point 0      Point 1      Point 2      Point 3
    ///        •-----------•-----------•-----------•
    ///        |    0     |     1     |     2     |
    ///     Segments between points, labeled by index
    /// </para>
    /// <para>
    /// If you want to find a y-value for an x that's between your data points, this method
    /// tells you which segment to use. For example, if your x-value is between Point 1 and 
    /// Point 2, this method returns 1 (the index of the segment).
    /// </para>
    /// <para>
    /// If your x-value is beyond the last point, the method returns the index of the last segment,
    /// allowing for extrapolation (making predictions beyond your data).
    /// </para>
    /// </remarks>
    private int FindInterval(T x)
    {
        return InterpolationSearch.FirstSegmentReaching(_x, x, 0, _x.Length - 2);
    }

    /// <summary>
    /// Returns <paramref name="length"/> consecutive elements of a vector tensor starting at <paramref name="start"/>.
    /// </summary>
    private static Tensor<T> Slice(Tensor<T> values, int start, int length)
    {
        return AiDotNetEngine.Current.TensorSlice(values, new[] { start }, new[] { length });
    }
}
