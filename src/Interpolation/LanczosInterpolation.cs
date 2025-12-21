namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Lanczos interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Lanczos interpolation is a high-quality resampling technique that uses a windowed sinc function
/// to create smooth interpolations between data points. It's commonly used in image and signal processing.
/// 
/// <b>For Beginners:</b> Lanczos interpolation is like a sophisticated way of estimating values between known points.
/// Imagine you have several dots on a graph and want to draw a smooth curve through them. Lanczos uses a special
/// mathematical approach that creates a natural-looking curve while preserving important details in your data.
/// It's particularly good at maintaining sharp edges while still creating smooth transitions, which is why
/// it's popular for resizing images and processing signals.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LanczosInterpolation<T> : IInterpolation<T>
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
    /// The 'a' parameter that controls the size of the Lanczos window.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This parameter determines how many neighboring points influence each interpolated value.
    /// A larger value creates smoother results but may lose some detail, while a smaller value preserves
    /// more detail but might create less smooth transitions.
    /// </remarks>
    private readonly int _a;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance of the Lanczos interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the interpolator with your data points and the Lanczos window size.
    /// 
    /// <b>For Beginners:</b> When you create a Lanczos interpolator, you provide the x-coordinates and
    /// corresponding y-values of your known data points. You can also specify the 'a' parameter,
    /// which controls how smooth versus detailed your interpolation will be. The default value of 3
    /// works well for most purposes, but you can adjust it based on your needs.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-values of the known data points.</param>
    /// <param name="a">The Lanczos window size parameter (default is 3).</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths or parameter 'a' is less than 1.</exception>
    public LanczosInterpolation(Vector<T> x, Vector<T> y, int a = 3)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (a < 1)
            throw new ArgumentException("Parameter 'a' must be a positive integer.");

        _x = x;
        _y = y;
        _a = a;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Interpolates the y-value at a given x-coordinate using Lanczos interpolation.
    /// </summary>
    /// <remarks>
    /// This method calculates the y-value at any x-coordinate using a weighted sum of nearby known points,
    /// where the weights are determined by the Lanczos kernel function.
    /// 
    /// <b>For Beginners:</b> Once you've set up the interpolator with your known points, this method
    /// lets you estimate the y-value at any x-coordinate. It works by looking at all your known points,
    /// giving more importance to points that are closer to your target x-coordinate, and less importance
    /// to points that are farther away. The special Lanczos formula used for weighting creates a smooth
    /// and accurate interpolation.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        T result = _numOps.Zero;
        T sum = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T diff = _numOps.Subtract(x, _x[i]);
            T lanczosValue = LanczosKernel(diff);
            result = _numOps.Add(result, _numOps.Multiply(_y[i], lanczosValue));
            sum = _numOps.Add(sum, lanczosValue);
        }

        // Normalize the result
        return _numOps.Divide(result, sum);
    }

    /// <summary>
    /// Calculates the Lanczos kernel value for a given distance.
    /// </summary>
    /// <remarks>
    /// The Lanczos kernel is a mathematical function that determines how much influence
    /// a known point has on an interpolated value based on the distance between them.
    /// 
    /// <b>For Beginners:</b> This function calculates a weight or importance value for each known point
    /// based on how far it is from the point you're trying to estimate. Points that are very close
    /// get a high weight (close to 1), while points that are far away get a weight of 0 (no influence).
    /// The Lanczos kernel creates a special pattern of weights that produces smooth, high-quality
    /// interpolation results.
    /// </remarks>
    /// <param name="x">The distance between the interpolation point and a known data point.</param>
    /// <returns>The Lanczos kernel value for the given distance.</returns>
    private T LanczosKernel(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
        {
            return _numOps.One;
        }

        if (_numOps.GreaterThanOrEquals(_numOps.Abs(x), _numOps.FromDouble(_a)))
        {
            return _numOps.Zero;
        }

        T piX = _numOps.Multiply(MathHelper.Pi<T>(), x);
        T sinc = _numOps.Divide(MathHelper.Sin(piX), piX);
        T sinc2 = _numOps.Divide(MathHelper.Sin(_numOps.Divide(piX, _numOps.FromDouble(_a))), _numOps.Divide(piX, _numOps.FromDouble(_a)));

        return _numOps.Multiply(sinc, sinc2);
    }
}
