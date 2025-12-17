namespace AiDotNet.Interpolation;

/// <summary>
/// Implements nearest neighbor interpolation, a simple method that finds the closest known data point
/// and returns its corresponding value.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Interpolation is a way to estimate values between known data points. Imagine you have
/// a set of (x,y) points on a graph, and you want to find the y-value for an x that isn't in your original data.
/// Nearest neighbor interpolation simply finds the closest x-value in your data and returns its corresponding y-value.
/// </para>
/// <para>
/// This is the simplest form of interpolation and works like a "staircase" function rather than a smooth curve.
/// </para>
/// </remarks>
public class NearestNeighborInterpolation<T> : IInterpolation<T>
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
    /// Operations for performing numeric calculations with generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="NearestNeighborInterpolation{T}"/> class.
    /// </summary>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates (values) of the known data points.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes two vectors (arrays) of equal length:
    /// - The x vector contains the input values (like time points, positions, etc.)
    /// - The y vector contains the corresponding output values
    /// 
    /// For example, if you're tracking temperature over time, x might be the hours [1,2,3,4]
    /// and y might be the temperatures [68,72,75,73].
    /// </remarks>
    public NearestNeighborInterpolation(Vector<T> x, Vector<T> y)
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
    /// Performs nearest neighbor interpolation to estimate a y-value for the given x-value.
    /// </summary>
    /// <param name="x">The x-value for which to estimate the corresponding y-value.</param>
    /// <returns>The y-value of the nearest known data point.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes an x-value and returns the y-value of the closest point in our data.
    /// For example, if we have data points at x = [1, 3, 5] with values y = [10, 30, 50],
    /// and we ask for x = 2.7, the method will return 30 because 3 is the closest x-value to 2.7.
    /// </remarks>
    public T Interpolate(T x)
    {
        int nearestIndex = FindNearestIndex(x);
        return _y[nearestIndex];
    }

    /// <summary>
    /// Finds the index of the data point whose x-value is closest to the given value.
    /// </summary>
    /// <param name="x">The x-value to find the nearest neighbor for.</param>
    /// <returns>The index of the nearest data point in the internal arrays.</returns>
    /// <remarks>
    /// This method calculates the absolute distance between the given x-value and each known x-value,
    /// then returns the index of the data point with the smallest distance.
    /// </remarks>
    private int FindNearestIndex(T x)
    {
        int nearestIndex = 0;
        T minDistance = _numOps.Abs(_numOps.Subtract(x, _x[0]));

        for (int i = 1; i < _x.Length; i++)
        {
            T distance = _numOps.Abs(_numOps.Subtract(x, _x[i]));
            if (_numOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                nearestIndex = i;
            }
        }

        return nearestIndex;
    }
}
