namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Catmull-Rom spline interpolation for smooth curve fitting through a series of points.
/// </summary>
/// <remarks>
/// Catmull-Rom splines create smooth curves that pass through all the provided data points,
/// making them useful for animation paths, curve drawing, and data visualization.
/// 
/// <b>For Beginners:</b> Think of this as drawing a smooth curve through a set of dots. Unlike simpler
/// methods that might create sharp corners or jagged lines, Catmull-Rom splines create naturally
/// flowing curves that pass exactly through each point while maintaining smoothness.
/// 
/// Imagine connecting dots to draw the outline of a mountain range or a river - you want the
/// line to flow naturally between points rather than making sharp turns. That's what this
/// interpolation method provides.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class CatmullRomSplineInterpolation<T> : IInterpolation<T>
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
    /// The tension parameter that controls the curvature of the spline.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of tension as controlling how "tight" or "loose" the curve is.
    /// Lower values (closer to 0) create looser curves that may swing wide between points.
    /// Higher values create tighter curves that stay closer to straight lines between points.
    /// The default value of 0.5 provides a balanced curve for most applications.
    /// </remarks>
    private readonly T _tension;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a utility that helps perform math operations (like addition
    /// and multiplication) on different types of numbers. You don't need to interact with
    /// this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the CatmullRomSplineInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor validates the input data and initializes the necessary components
    /// for performing Catmull-Rom spline interpolation.
    /// 
    /// <b>For Beginners:</b> This sets up everything needed to create smooth curves through your points:
    /// 1. It checks that you have enough points (at least 4)
    /// 2. It stores your x and y coordinates
    /// 3. It sets up the tension parameter that controls how curvy the result will be
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="tension">
    /// Controls the "tightness" of the curve. Default is 0.5.
    /// Lower values (closer to 0) create looser curves, higher values create tighter curves.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors don't have the same length or when there are fewer than 4 points.
    /// </exception>
    public CatmullRomSplineInterpolation(Vector<T> x, Vector<T> y, double tension = 0.5)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("At least 4 points are required for Catmull-Rom spline interpolation.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _tension = _numOps.FromDouble(tension);
    }

    /// <summary>
    /// Interpolates a y-value at the specified x-coordinate using Catmull-Rom spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method finds the segment containing the target x-coordinate and calculates
    /// the corresponding y-value using Catmull-Rom spline interpolation.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an x-value, and it will:
    /// 1. Find which segment of your data contains this x-value
    /// 2. Use the four points surrounding this segment (two on each side)
    /// 3. Calculate a smooth curve through these points
    /// 4. Return the y-value on that curve at your requested x-position
    /// 
    /// For example, if you have data points at x = [0, 10, 20, 30] and you want to know
    /// the y-value at x = 15, this method will create a smooth curve through all your points
    /// and tell you the height of that curve at position 15.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        // Check if x exactly matches a known point
        for (int k = 0; k < _x.Length; k++)
        {
            if (_numOps.Equals(x, _x[k]))
            {
                return _y[k];
            }
        }

        int i = FindInterval(x);

        // Get the four points needed for the spline calculation
        T x0 = _x[i - 1];  // Point before the interval
        T x1 = _x[i];      // Start of the interval
        T x2 = _x[i + 1];  // End of the interval
        T x3 = _x[i + 2];  // Point after the interval

        T y0 = _y[i - 1];  // y-value before the interval
        T y1 = _y[i];      // y-value at start of the interval
        T y2 = _y[i + 1];  // y-value at end of the interval
        T y3 = _y[i + 2];  // y-value after the interval

        // Calculate the parameter t (0 to 1) representing the position within the interval
        T t = _numOps.Divide(_numOps.Subtract(x, x1), _numOps.Subtract(x2, x1));

        return CatmullRomSpline(y0, y1, y2, y3, t);
    }

    /// <summary>
    /// Finds the index of the interval containing the specified x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method locates which segment of the data contains the target x-coordinate.
    /// 
    /// <b>For Beginners:</b> This helper method finds which segment of your data contains the x-value
    /// you're interested in. For example, if you have points at x = [0, 10, 20, 30] and you
    /// want to interpolate at x = 15, this method will tell you that 15 is in the segment
    /// between index 1 (value 10) and index 2 (value 20).
    /// </remarks>
    /// <param name="x">The x-coordinate to locate.</param>
    /// <returns>The index of the lower bound of the interval containing the x-coordinate.</returns>
    private int FindInterval(T x)
    {
        for (int i = 1; i < _x.Length - 2; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
            {
                return i;
            }
        }
        return _x.Length - 3;
    }

    /// <summary>
    /// Calculates the Catmull-Rom spline value for a given parameter t and four control points.
    /// </summary>
    /// <remarks>
    /// This method implements the core Catmull-Rom spline formula to calculate a point on the curve.
    /// 
    /// <b>For Beginners:</b> This is where the mathematical "magic" happens to create a smooth curve.
    /// The method takes four points and a position parameter (t) between 0 and 1, and calculates
    /// the exact y-value on the smooth curve at that position.
    /// 
    /// The formula uses cubic polynomials (equations with terms up to t³) to create a curve
    /// that not only passes through the middle two points but also maintains a smooth direction
    /// as it passes through them.
    /// </remarks>
    /// <param name="y0">The y-value of the first control point (before the current segment).</param>
    /// <param name="y1">The y-value of the second control point (start of the current segment).</param>
    /// <param name="y2">The y-value of the third control point (end of the current segment).</param>
    /// <param name="y3">The y-value of the fourth control point (after the current segment).</param>
    /// <param name="t">The parameter value (0 to 1) representing the position within the segment.</param>
    /// <returns>The interpolated y-value at parameter t.</returns>
    private T CatmullRomSpline(T y0, T y1, T y2, T y3, T t)
    {
        // Calculate powers of t for the cubic polynomial
        T t2 = _numOps.Multiply(t, t);      // t²
        T t3 = _numOps.Multiply(t2, t);     // t³

        // Calculate tangents at y1 and y2 using Catmull-Rom formula:
        // m1 = tension * (y2 - y0)
        // m2 = tension * (y3 - y1)
        T m1 = _numOps.Multiply(_tension, _numOps.Subtract(y2, y0));
        T m2 = _numOps.Multiply(_tension, _numOps.Subtract(y3, y1));

        // Hermite basis functions:
        // h00(t) = 2t³ - 3t² + 1
        // h10(t) = t³ - 2t² + t
        // h01(t) = -2t³ + 3t²
        // h11(t) = t³ - t²
        T h00 = _numOps.Add(
            _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(2), t3), _numOps.Multiply(_numOps.FromDouble(3), t2)),
            _numOps.One
        );
        T h10 = _numOps.Add(
            _numOps.Subtract(t3, _numOps.Multiply(_numOps.FromDouble(2), t2)),
            t
        );
        T h01 = _numOps.Add(
            _numOps.Multiply(_numOps.FromDouble(-2), t3),
            _numOps.Multiply(_numOps.FromDouble(3), t2)
        );
        T h11 = _numOps.Subtract(t3, t2);

        // P(t) = h00*y1 + h10*m1 + h01*y2 + h11*m2
        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(h00, y1),
                _numOps.Multiply(h10, m1)
            ),
            _numOps.Add(
                _numOps.Multiply(h01, y2),
                _numOps.Multiply(h11, m2)
            )
        );
    }
}
