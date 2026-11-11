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
public partial class ShepardsMethodInterpolation<T> : I2DInterpolation<T>
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
    [AiDotNet.Attributes.TrainableParameter]
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
        if (_x.Length == 0)
        {
            return _numOps.Divide(_numOps.Zero, _numOps.Zero);
        }

        Vector<T> distances = CalculateDistances(x, y);

        // Return the exact value if the point coincides with a known point
        int coincident = InterpolationSearch.IndexOfFirstEqual(distances, _numOps.Zero);
        if (coincident >= 0)
        {
            return _z[coincident];
        }

        var engine = AiDotNetEngine.Current;
        Tensor<T> weights = engine.TensorPower(engine.TensorReciprocal(distances.AsTensor()), _power);
        T numerator = engine.TensorSum(engine.TensorMultiply(weights, _z.AsTensor()));
        T denominator = engine.TensorSum(weights);

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the Euclidean distance from a query point to every known point in 2D space.
    /// </summary>
    /// <param name="x">The x-coordinate of the query point.</param>
    /// <param name="y">The y-coordinate of the query point.</param>
    /// <returns>The distance from the query point to every known point.</returns>
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
    private Vector<T> CalculateDistances(T x, T y)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> dx = engine.ScalarMinusTensor(x, _x.AsTensor());
        Tensor<T> dy = engine.ScalarMinusTensor(y, _y.AsTensor());

        return engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy))).ToVector();
    }
}
