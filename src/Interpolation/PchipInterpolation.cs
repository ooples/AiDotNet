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
public class PchipInterpolation<T> : IInterpolation<T>
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
        _slopes = new Vector<T>(x.Length);

        CalculateSlopes();
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
    /// Calculates the slopes at each data point to ensure a smooth, shape-preserving curve.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how steep the curve should be at each data point.
    /// </para>
    /// <para>
    /// The slopes are calculated in two steps:
    /// 1. Initial calculation: For points in the middle, it uses a weighted average of the slopes
    ///    of the segments on either side. For endpoints, it uses the slope of the adjacent segment.
    /// 2. Adjustment for monotonicity: The slopes are adjusted to ensure the curve doesn't create
    ///    artificial bumps or wiggles between data points.
    /// </para>
    /// <para>
    /// This is what makes PCHIP special - it preserves the shape of your data by ensuring
    /// that if your data is increasing between two points, the curve will only increase (not wiggle up and down).
    /// </para>
    /// </remarks>
    private void CalculateSlopes()
    {
        int n = _x.Length;

        for (int i = 0; i < n - 1; i++)
        {
            T dx = _numOps.Subtract(_x[i + 1], _x[i]);
            T dy = _numOps.Subtract(_y[i + 1], _y[i]);
            T slope = _numOps.Divide(dy, dx);

            if (i == 0)
            {
                _slopes[i] = slope;
            }
            else if (i == n - 2)
            {
                _slopes[n - 1] = slope;
            }
            else
            {
                T dx_prev = _numOps.Subtract(_x[i], _x[i - 1]);
                T dy_prev = _numOps.Subtract(_y[i], _y[i - 1]);
                T slope_prev = _numOps.Divide(dy_prev, dx_prev);

                _slopes[i] = WeightedHarmonicMean(slope_prev, slope);
            }
        }

        // Adjust slopes to ensure monotonicity
        for (int i = 0; i < n - 1; i++)
        {
            T dx = _numOps.Subtract(_x[i + 1], _x[i]);
            T dy = _numOps.Subtract(_y[i + 1], _y[i]);
            T slope = _numOps.Divide(dy, dx);

            if (_numOps.Equals(slope, _numOps.Zero))
            {
                _slopes[i] = _numOps.Zero;
                _slopes[i + 1] = _numOps.Zero;
            }
            else
            {
                T alpha = _numOps.Divide(_slopes[i], slope);
                T beta = _numOps.Divide(_slopes[i + 1], slope);

                if (_numOps.GreaterThan(_numOps.Add(_numOps.Power(alpha, _numOps.FromDouble(2)), _numOps.Power(beta, _numOps.FromDouble(2))), _numOps.FromDouble(9)))
                {
                    T tau = _numOps.Divide(_numOps.FromDouble(3), _numOps.Sqrt(_numOps.Add(_numOps.Power(alpha, _numOps.FromDouble(2)), _numOps.Power(beta, _numOps.FromDouble(2)))));
                    _slopes[i] = _numOps.Multiply(tau, _numOps.Multiply(alpha, slope));
                    _slopes[i + 1] = _numOps.Multiply(tau, _numOps.Multiply(beta, slope));
                }
            }
        }
    }

    /// <summary>
    /// Calculates a weighted harmonic mean of two values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The weighted harmonic mean.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method combines two slope values in a special way that gives
    /// more importance to the smaller value.
    /// </para>
    /// <para>
    /// A harmonic mean is a type of average that works well when averaging rates or slopes.
    /// The "weighted" part means that each value's importance in the calculation depends on
    /// the size of the other value.
    /// </para>
    /// <para>
    /// This helps create a smoother curve by preventing the slope at a point from being
    /// too influenced by a very steep segment nearby.
    /// </para>
    /// </remarks>
    private T WeightedHarmonicMean(T a, T b)
    {
        if (_numOps.Equals(a, _numOps.Zero) || _numOps.Equals(b, _numOps.Zero))
            return _numOps.Zero;

        T w1 = _numOps.Abs(b);
        T w2 = _numOps.Abs(a);

        return _numOps.Divide(_numOps.Add(_numOps.Multiply(w1, a), _numOps.Multiply(w2, b)),
                               _numOps.Add(w1, w2));
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
        for (int i = 0; i < _x.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
                return i;
        }

        return _x.Length - 2;
    }
}
