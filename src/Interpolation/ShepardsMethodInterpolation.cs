namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Shepard's Method for interpolating scattered data points in 2D space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Shepard's Method is a form of inverse distance weighting interpolation that creates
/// a smooth surface passing through all the provided data points.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you have several points with known heights (like hills on a landscape).
/// Shepard's Method helps you estimate the height at any other location by considering all known points,
/// but giving more importance to the closest ones. It's like saying "this unknown point is probably
/// more similar to nearby points than to faraway points." The power parameter controls how quickly
/// the influence of distant points diminishes.
/// </para>
/// </remarks>
public class ShepardsMethodInterpolation<T> : I2DInterpolation<T>
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
    /// The z-values (heights) at each data point.
    /// </summary>
    private readonly Vector<T> _z;

    /// <summary>
    /// The power parameter that controls how quickly the influence of points decreases with distance.
    /// </summary>
    /// <remarks>
    /// Higher values make distant points have less influence on the interpolated value.
    /// </remarks>
    private readonly T _power;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of Shepard's Method interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="z">The z-values (heights) at each data point.</param>
    /// <param name="power">
    /// The power parameter that controls how quickly the influence of points decreases with distance.
    /// Default is 2.0 (inverse-square weighting).
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your 3D data points (x, y, and z values) and prepares
    /// the interpolation algorithm. It checks that your data is valid (same number of x, y, and z values)
    /// and sets up the power parameter.
    /// </para>
    /// <para>
    /// The power parameter is important:
    /// - A value of 1.0 means influence decreases linearly with distance
    /// - A value of 2.0 (default) means influence decreases with the square of distance
    /// - Higher values (like 3.0 or 4.0) make distant points have even less influence
    /// </para>
    /// <para>
    /// Common values are between 1.0 and 3.0. Higher values create more "peaked" surfaces around data points.
    /// </para>
    /// </remarks>
    public ShepardsMethodInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, double power = 2.0)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _power = _numOps.FromDouble(power);
    }

    /// <summary>
    /// Interpolates a z-value for the given x and y coordinates using Shepard's Method.
    /// </summary>
    /// <param name="x">The x-coordinate for which to interpolate.</param>
    /// <param name="y">The y-coordinate for which to interpolate.</param>
    /// <returns>The interpolated z-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the height (z-value) at any location (x,y) you specify,
    /// even if it's between or outside your original data points.
    /// </para>
    /// <para>
    /// It works by:
    /// 1. Calculating the distance from your query point to each of the original data points
    /// 2. If your query point exactly matches one of the original points, it returns the exact height
    /// 3. Otherwise, it calculates a weighted average of all data points, where closer points have more influence
    /// </para>
    /// <para>
    /// The formula used is: z = S(z_i * w_i) / S(w_i), where w_i = 1/distance^power
    /// </para>
    /// <para>
    /// This creates a smooth surface that passes exactly through all your original data points.
    /// </para>
    /// </remarks>
    public T Interpolate(T x, T y)
    {
        T numerator = _numOps.Zero;
        T denominator = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T distance = CalculateDistance(x, y, _x[i], _y[i]);

            if (_numOps.Equals(distance, _numOps.Zero))
            {
                return _z[i]; // Return exact value if the point coincides with a known point
            }

            T weight = _numOps.Power(MathHelper.Reciprocal(distance), _power);
            numerator = _numOps.Add(numerator, _numOps.Multiply(weight, _z[i]));
            denominator = _numOps.Add(denominator, weight);
        }

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the Euclidean distance between two points in 2D space.
    /// </summary>
    /// <param name="x1">The x-coordinate of the first point.</param>
    /// <param name="y1">The y-coordinate of the first point.</param>
    /// <param name="x2">The x-coordinate of the second point.</param>
    /// <param name="y2">The y-coordinate of the second point.</param>
    /// <returns>The distance between the two points.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the straight-line distance between two points
    /// using the Pythagorean theorem (a² + b² = c²).
    /// </para>
    /// <para>
    /// In Shepard's Method, this distance is used to determine how much influence each known
    /// data point has on the interpolated value. Points that are closer (smaller distance)
    /// have more influence than points that are farther away.
    /// </para>
    /// </remarks>
    private T CalculateDistance(T x1, T y1, T x2, T y2)
    {
        T dx = _numOps.Subtract(x1, x2);
        T dy = _numOps.Subtract(y1, y2);

        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
    }
}
