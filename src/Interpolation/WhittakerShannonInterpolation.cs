namespace AiDotNet.Interpolation;

/// <summary>
/// Implements the Whittaker-Shannon interpolation method, also known as sinc interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Whittaker-Shannon interpolation is a technique based on the sampling theorem, which states that
/// a band-limited function can be perfectly reconstructed from its samples if the sampling rate
/// is at least twice the highest frequency in the function.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this interpolation like recreating a smooth curve from a series of dots.
/// Imagine you have a photograph that's been converted to a grid of pixels. This method helps you
/// "zoom in" between those pixels to create a smoother, higher-resolution image. It works best when
/// your data points are evenly spaced (like pixels in a digital image) and when the underlying pattern
/// doesn't contain frequencies that are too high (meaning the data doesn't wiggle up and down too rapidly).
/// </para>
/// <para>
/// This method is particularly useful for signal processing applications, such as audio or image processing,
/// where you need to reconstruct continuous signals from discrete samples.
/// </para>
/// </remarks>
public class WhittakerShannonInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates (values) of the data points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of Whittaker-Shannon interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates (values) of the data points.</param>
    /// <exception cref="ArgumentException">Thrown when the input vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your data points (x and y coordinates) and prepares the
    /// interpolation algorithm. The x-values represent positions (like time points in a signal) and
    /// the y-values represent measurements at those positions (like amplitude values in a sound wave).
    /// </para>
    /// <para>
    /// This method works best when your data points are evenly spaced along the x-axis. If they're not,
    /// you'll see a warning message, and the results might not be as accurate. Evenly spaced means that
    /// the distance between consecutive x-values is constant (like x = 1, 2, 3, 4... or x = 0.5, 1.0, 1.5, 2.0...).
    /// </para>
    /// </remarks>
    public WhittakerShannonInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        if (!IsUniformlySampled())
        {
            Console.WriteLine("Warning: Input data is not uniformly sampled. Interpolation results may be less accurate.");
        }
    }

    /// <summary>
    /// Interpolates a y-value for the given x coordinate using Whittaker-Shannon interpolation.
    /// </summary>
    /// <param name="x">The x-coordinate for which to interpolate.</param>
    /// <returns>The interpolated y-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the y-value at any x-position you specify, even if it's
    /// between your original data points. It uses a special mathematical function called the "sinc function"
    /// to create a smooth curve that passes through all your original data points.
    /// </para>
    /// <para>
    /// The sinc function looks like a wave that peaks at one point and gradually diminishes as you move
    /// away from that peak. By combining multiple sinc functions (one centered at each of your original
    /// data points), this method creates a smooth curve that represents your data.
    /// </para>
    /// <para>
    /// This is similar to how digital-to-analog conversion works in audio equipment, where discrete
    /// digital samples are converted back into a continuous sound wave.
    /// </para>
    /// </remarks>
    public T Interpolate(T x)
    {
        if (_x.Length == 0)
        {
            return _numOps.Zero;
        }

        // sum over i of y[i] * sinc(x - x[i])
        var engine = AiDotNetEngine.Current;
        Tensor<T> sincValues = Sinc(engine.ScalarMinusTensor(x, _x.AsTensor()));

        return engine.TensorSum(engine.TensorMultiply(_y.AsTensor(), sincValues));
    }

    /// <summary>
    /// Calculates the sinc function value for a given input.
    /// </summary>
    /// <param name="x">The input values.</param>
    /// <returns>The sinc function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sinc function is a special mathematical function that looks like a wave
    /// with a peak at x=0 and diminishing oscillations as you move away from the center.
    /// </para>
    /// <para>
    /// Mathematically, it's defined as sin(px)/(px) for x?0 and 1 for x=0.
    /// </para>
    /// <para>
    /// This function is fundamental to signal processing because it represents the ideal way to
    /// reconstruct a continuous signal from discrete samples. Each data point contributes to the
    /// final result through its own sinc function.
    /// </para>
    /// </remarks>
    private Tensor<T> Sinc(Tensor<T> x)
    {
        // sin(pi x) / (pi x), with the removable singularity at x = 0 filled by 1
        var engine = AiDotNetEngine.Current;
        Tensor<T> atZero = engine.TensorEquals(x, _numOps.Zero);
        Tensor<T> piX = engine.TensorMultiplyScalar(x, MathHelper.Pi<T>());

        // piX is exactly 0 wherever atZero is 1, so adding the mask only replaces the zero denominators
        // (a decimal 0 / 0 throws); the mask overrides those entries below
        Tensor<T> ratio = engine.TensorDivide(engine.TensorSin(piX), engine.TensorAdd(piX, atZero));

        return engine.TensorWhere(
            atZero,
            Tensor<T>.CreateDefault(new[] { x.Length }, _numOps.One),
            ratio);
    }

    /// <summary>
    /// Checks if the input data points are uniformly sampled (evenly spaced).
    /// </summary>
    /// <returns>True if the data points are uniformly sampled; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks if your x-values are evenly spaced, which is important for
    /// this interpolation method to work correctly. "Evenly spaced" means that the distance between
    /// consecutive x-values is constant.
    /// </para>
    /// <para>
    /// For example, the sequence [1, 2, 3, 4] is evenly spaced because the difference between
    /// consecutive values is always 1. Similarly, [0.5, 1.0, 1.5, 2.0] is evenly spaced with a
    /// difference of 0.5.
    /// </para>
    /// <para>
    /// However, [1, 2, 4, 8] is not evenly spaced because the differences are 1, 2, and 4.
    /// </para>
    /// <para>
    /// The method includes a small tolerance to account for floating-point precision issues,
    /// so tiny variations in spacing due to rounding errors won't trigger a false negative.
    /// </para>
    /// </remarks>
    private bool IsUniformlySampled()
    {
        if (_x.Length <= 2) return true;

        T interval = _numOps.Subtract(_x[1], _x[0]);
        T tolerance = _numOps.Multiply(interval, _numOps.FromDouble(1e-6));

        // Every later spacing x[i] - x[i-1] (i >= 2) must stay within the tolerance of the first one
        var engine = AiDotNetEngine.Current;
        Tensor<T> xs = _x.AsTensor();
        int count = _x.Length - 2;
        Tensor<T> spacings = engine.TensorSubtract(
            engine.TensorSlice(xs, new[] { 2 }, new[] { count }),
            engine.TensorSlice(xs, new[] { 1 }, new[] { count }));
        Tensor<T> deviations = engine.TensorAbs(engine.TensorSubtractScalar(spacings, interval));
        Tensor<T> outside = engine.TensorGreaterThan(deviations, tolerance);

        return !_numOps.GreaterThan(engine.TensorSum(outside), _numOps.Zero);
    }
}
