namespace AiDotNet.Interpolation;

/// <summary>
/// Implements linear interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Linear interpolation is the simplest form of interpolation, connecting data points with straight lines.
/// It estimates values between known data points by assuming a straight line between them.
/// 
/// <b>For Beginners:</b> Linear interpolation is like drawing straight lines between dots on a graph.
/// If you have two points (like one at x=1, y=10 and another at x=3, y=20), and you want to know
/// what the y-value would be at x=2, linear interpolation would give you y=15 because it's exactly
/// halfway between the two known points. It's the simplest and most intuitive way to estimate values
/// between known points.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LinearInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates (values) of the known data points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance of the linear interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the interpolator with your data points.
    /// 
    /// <b>For Beginners:</b> When you create a linear interpolator, you provide the x-coordinates
    /// and corresponding y-values of your known data points. The interpolator will then be
    /// ready to estimate y-values for any x-coordinate you specify later by drawing straight
    /// lines between your known points.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-values of the known data points.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths.</exception>
    public LinearInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Interpolates the y-value at a given x-coordinate using linear interpolation.
    /// </summary>
    /// <remarks>
    /// This method calculates the y-value at any x-coordinate by finding the two nearest known points
    /// and drawing a straight line between them.
    /// 
    /// <b>For Beginners:</b> Once you've set up the interpolator with your known points, this method
    /// lets you ask "What would the y-value be at this specific x-coordinate?" It finds the two
    /// known points that are closest to your target x-coordinate (one on each side), draws a straight
    /// line between them, and then determines where on that line your target x-coordinate falls.
    /// 
    /// The formula used is: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    /// where (x0, y0) and (x1, y1) are the two nearest known points.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        if (i == _x.Length - 1)
        {
            return _y[i];
        }

        T x0 = _x[i];
        T x1 = _x[i + 1];
        T y0 = _y[i];
        T y1 = _y[i + 1];

        T t = _numOps.Divide(_numOps.Subtract(x, x0), _numOps.Subtract(x1, x0));
        return _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, t), y0), _numOps.Multiply(t, y1));
    }

    /// <summary>
    /// Finds the interval in the x-coordinates array that contains the given x-value.
    /// </summary>
    /// <remarks>
    /// This method uses binary search to efficiently find which pair of known points
    /// the target x-coordinate falls between.
    /// 
    /// <b>For Beginners:</b> Before we can interpolate, we need to know which two known points
    /// to draw a line between. This method efficiently finds the right pair of points by
    /// using a technique called "binary search" (like when you search for a word in a dictionary
    /// by repeatedly dividing the book in half). It returns the index of the known point
    /// that comes just before your target x-coordinate.
    /// 
    /// If your target x is smaller than all known x-coordinates, it returns 0.
    /// If your target x is larger than all known x-coordinates, it returns the index of the last point.
    /// </remarks>
    /// <param name="x">The x-coordinate to find the interval for.</param>
    /// <returns>The index of the lower bound of the interval containing x.</returns>
    private int FindInterval(T x)
    {
        if (_numOps.LessThanOrEquals(x, _x[0]))
            return 0;
        if (_numOps.GreaterThanOrEquals(x, _x[_x.Length - 1]))
            return _x.Length - 1;

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
