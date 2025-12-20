namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Lagrange polynomial interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Lagrange polynomial interpolation creates a smooth curve that passes exactly through 
/// all provided data points. It's particularly useful for estimating values between known points.
/// 
/// <b>For Beginners:</b> Think of this like connecting dots with a smooth curve. If you have several
/// points on a graph, Lagrange interpolation draws a smooth line through all of them, allowing
/// you to estimate values between your known points. Unlike simpler methods like linear interpolation
/// (which just draws straight lines between points), this creates a natural-looking curve.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LagrangePolynomialInterpolation<T> : IInterpolation<T>
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
    /// Creates a new instance of the Lagrange polynomial interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the interpolator with your data points.
    /// 
    /// <b>For Beginners:</b> When you create a Lagrange interpolator, you provide the x-coordinates
    /// and corresponding y-values of your known data points. The interpolator will then be
    /// ready to estimate y-values for any x-coordinate you specify later.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-values of the known data points.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths or fewer than 2 points are provided.</exception>
    public LagrangePolynomialInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Lagrange polynomial interpolation requires at least 2 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Interpolates the y-value at a given x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method calculates the y-value at any x-coordinate using the Lagrange polynomial formula.
    /// 
    /// <b>For Beginners:</b> Once you've set up the interpolator with your known points, this method
    /// lets you ask "What would the y-value be at this specific x-coordinate?" It uses a special
    /// mathematical formula to give you the best estimate based on your known points. The result
    /// will be a point that lies on a smooth curve passing through all your original points.
    /// 
    /// The Lagrange formula works by creating a weighted sum of your known y-values, where the
    /// weights are calculated based on how far your target x-coordinate is from each known x-coordinate.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T term = _y[i];
            for (int j = 0; j < _x.Length; j++)
            {
                if (i != j)
                {
                    term = _numOps.Multiply(term,
                        _numOps.Divide(
                            _numOps.Subtract(x, _x[j]),
                            _numOps.Subtract(_x[i], _x[j])
                        )
                    );
                }
            }
            result = _numOps.Add(result, term);
        }

        return result;
    }
}
