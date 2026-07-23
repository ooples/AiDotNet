namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Hermite interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Hermite interpolation creates a smooth curve that passes through all given data points
/// while also matching specified derivatives (slopes) at those points. This provides more
/// control over the shape of the curve compared to simpler interpolation methods.
/// 
/// <b>For Beginners:</b> This class helps you estimate values between known data points when you
/// know not only the values at certain points but also how quickly those values are changing
/// (the slopes) at those points. Imagine drawing a smooth curve through dots on a graph,
/// but also controlling which direction the curve is heading as it passes through each dot.
/// This gives you a more natural-looking curve that respects both the values and their rates
/// of change.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class HermiteInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points (independent variable).
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the data points (dependent variable).
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The slopes (derivatives) at each data point.
    /// </summary>
    private readonly Vector<T> _m;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new Hermite interpolation from the given data points and slopes.
    /// </summary>
    /// <remarks>
    /// The constructor initializes the interpolation with the provided data points and their
    /// corresponding slopes.
    /// 
    /// <b>For Beginners:</b> When you create a new HermiteInterpolation object, you need to provide
    /// three pieces of information: the x-coordinates of your points, the y-coordinates of your
    /// points, and the slopes at each point. The slope tells the interpolation how the curve
    /// should approach and leave each point. If you don't know the slopes, you might want to
    /// use a different interpolation method or estimate the slopes using another technique.
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="m">The slopes at each data point.</param>
    /// <exception cref="ArgumentException">Thrown when the input vectors have different lengths.</exception>
    public HermiteInterpolation(Vector<T> x, Vector<T> y, Vector<T> m)
    {
        if (x.Length != y.Length || x.Length != m.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        _x = x;
        _y = y;
        _m = m;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the interpolated y-value for a given x-value using Hermite interpolation.
    /// </summary>
    /// <remarks>
    /// This method finds the interval containing the x-value and evaluates the Hermite
    /// polynomial at that point.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use after creating the interpolation.
    /// Give it any x-value within your data range, and it will return the estimated y-value
    /// at that point. The estimate is based on the cubic Hermite polynomials that respect
    /// both the values and slopes at your original data points. This creates a smooth curve
    /// that feels natural and preserves the trends in your data.
    /// </remarks>
    /// <param name="x">The x-value at which to interpolate.</param>
    /// <returns>The interpolated y-value at the given x-value.</returns>
    public T Interpolate(T x)
    {
        // Find which interval contains the x-value
        int i = FindInterval(x);

        // If x is at or beyond the last point, return the last y-value
        if (i == _x.Length - 1)
        {
            return _y[i];
        }

        // Get the coordinates and slopes at the interval endpoints
        T x0 = _x[i];       // x-coordinate of left endpoint
        T x1 = _x[i + 1];   // x-coordinate of right endpoint
        T y0 = _y[i];       // y-coordinate of left endpoint
        T y1 = _y[i + 1];   // y-coordinate of right endpoint
        T m0 = _m[i];       // slope at left endpoint
        T m1 = _m[i + 1];   // slope at right endpoint

        // Calculate the width of the interval
        T h = _numOps.Subtract(x1, x0);

        // Calculate the normalized position within the interval (0 to 1)
        T t = _numOps.Divide(_numOps.Subtract(x, x0), h);

        // Calculate t² and t³ for the Hermite basis functions
        T t2 = _numOps.Multiply(t, t);
        T t3 = _numOps.Multiply(t2, t);

        // Calculate the Hermite basis functions (correct formulas):
        // h00(t) = 2t³ - 3t² + 1
        // h10(t) = t³ - 2t² + t (scaled by h in the final calculation)
        // h01(t) = -2t³ + 3t²
        // h11(t) = t³ - t² (scaled by h in the final calculation)
        T h00 = _numOps.Add(_numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), t3), _numOps.Multiply(_numOps.FromDouble(-3), t2)), _numOps.One);
        T h10 = _numOps.Multiply(h, _numOps.Add(_numOps.Add(t3, _numOps.Multiply(_numOps.FromDouble(-2), t2)), t));
        T h01 = _numOps.Add(_numOps.Multiply(_numOps.FromDouble(-2), t3), _numOps.Multiply(_numOps.FromDouble(3), t2));
        T h11 = _numOps.Multiply(h, _numOps.Add(t3, _numOps.Multiply(_numOps.FromDouble(-1), t2)));

        // Combine the basis functions with the values and slopes to get the interpolated value
        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(h00, y0),  // Contribution from left point value
                _numOps.Multiply(h10, m0)   // Contribution from left point slope
            ),
            _numOps.Add(
                _numOps.Multiply(h01, y1),  // Contribution from right point value
                _numOps.Multiply(h11, m1)   // Contribution from right point slope
            )
        );
    }

    /// <summary>
    /// Finds the index of the interval containing the given x-value.
    /// </summary>
    /// <remarks>
    /// This method uses binary search to efficiently find which pair of data points
    /// contains the given x-value.
    /// 
    /// <b>For Beginners:</b> This helper method finds which two original data points the new
    /// x-value falls between. It uses a technique called binary search, which is much
    /// faster than checking each interval one by one, especially when you have many
    /// data points.
    /// </remarks>
    /// <param name="x">The x-value to locate.</param>
    /// <returns>The index of the left endpoint of the containing interval.</returns>
    private int FindInterval(T x)
    {
        // Handle edge cases: if x is at or before the first point
        if (_numOps.LessThanOrEquals(x, _x[0]))
            return 0;

        // Handle edge cases: if x is at or after the last point
        if (_numOps.GreaterThanOrEquals(x, _x[_x.Length - 1]))
            return _x.Length - 1;

        // Use binary search to find the interval
        int low = 0;
        int high = _x.Length - 1;

        while (low < high - 1)
        {
            int mid = (low + high) / 2;
            if (_numOps.LessThanOrEquals(_x[mid], x))
                low = mid;
            else
                high = mid;
        }

        return low;
    }
}
