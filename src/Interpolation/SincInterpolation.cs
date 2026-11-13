namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Sinc interpolation for 1D data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sinc interpolation is a technique based on the Whittaker–Shannon interpolation formula,
/// which is theoretically perfect for band-limited signals.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sinc interpolation is like creating a smooth curve through your data points
/// that preserves the frequency characteristics of your original data. It's particularly good
/// for signals like audio or sensor data where you want to maintain the original frequencies
/// when filling in gaps between known points. Think of it as drawing a curve that not only passes
/// through your points but also maintains the "rhythm" or "pattern" of your data.
/// </para>
/// </remarks>
public class SincInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-values at each data point.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The cutoff frequency that controls the bandwidth of the interpolation.
    /// </summary>
    /// <remarks>
    /// Higher values allow for more rapid changes in the interpolated curve.
    /// </remarks>
    private readonly T _cutoffFrequency;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of Sinc interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-values at each data point.</param>
    /// <param name="cutoffFrequency">
    /// The cutoff frequency that controls the bandwidth of the interpolation.
    /// The two-argument constructor uses 1.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input collections have different lengths.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your data points (x and y values) and prepares
    /// the interpolation algorithm. It checks that your data is valid (same number of x and y values)
    /// and sets up the cutoff frequency.
    /// </para>
    /// <para>
    /// The cutoff frequency is like a "detail control" for your interpolation:
    /// - Lower values (like 0.5) create smoother curves with less detail
    /// - Higher values (like 2.0) allow for more detail but might introduce oscillations
    /// - The default value (1.0) is a good balance for many applications
    /// </para>
    /// <para>
    /// If your interpolated curve looks too "wiggly", try reducing the cutoff frequency.
    /// If it's too smooth and misses important details, try increasing it.
    /// </para>
    /// </remarks>
    public SincInterpolation(Vector<T> x, Vector<T> y, T cutoffFrequency)
    {
        Guard.NotNull(x);
        Guard.NotNull(y);
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _numOps = MathHelper.GetNumericOperations<T>();
        // Snapshot the inputs so later changes to the caller's vectors do not alter this interpolator
        _x = x.Clone();
        _y = y.Clone();
        _cutoffFrequency = cutoffFrequency;
    }

    /// <summary>
    /// Creates a Sinc interpolation with a cutoff frequency of 1.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates (values) of the data points.</param>
    /// <exception cref="ArgumentException">Thrown when the input vectors have different lengths.</exception>
    public SincInterpolation(Vector<T> x, Vector<T> y)
        : this(x, y, MathHelper.GetNumericOperations<T>().One)
    {
    }

    /// <summary>
    /// Interpolates a y-value for the given x coordinate using Sinc interpolation.
    /// </summary>
    /// <param name="x">The x-coordinate for which to interpolate.</param>
    /// <returns>The interpolated y-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the y-value at any x-location you specify,
    /// even if it's between or outside your original data points.
    /// </para>
    /// <para>
    /// It works by:
    /// 1. Calculating a weighted sum of all your original data points
    /// 2. The weight for each point depends on how far it is from your query point
    /// 3. The sinc function (sin(x)/x) determines these weights
    /// </para>
    /// <para>
    /// Unlike simpler methods like linear interpolation, Sinc interpolation considers
    /// ALL your data points when calculating each new value, not just the nearest ones.
    /// This gives a more accurate result for many types of data, especially when your
    /// data represents a continuous signal like audio or sensor readings.
    /// </para>
    /// </remarks>
    public T Interpolate(T x)
    {
        if (_x.Length == 0)
        {
            return _numOps.Zero;
        }

        // sum over i of y[i] * sinc(cutoff * (x - x[i]))
        var engine = AiDotNetEngine.Current;
        Tensor<T> diff = engine.ScalarMinusTensor(x, _x.AsTensor());
        Tensor<T> sincValues = Sinc(engine.TensorMultiplyScalar(diff, _cutoffFrequency));

        return engine.TensorSum(engine.TensorMultiply(_y.AsTensor(), sincValues));
    }

    /// <summary>
    /// Calculates the Sinc function value: sin(px)/(px).
    /// </summary>
    /// <param name="x">The input values.</param>
    /// <returns>The Sinc function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Sinc function is a special mathematical function that looks like
    /// a wave with a peak at zero and smaller oscillations that diminish as you move away from zero.
    /// </para>
    /// <para>
    /// It's defined as sin(px)/(px) when x is not zero, and 1 when x is zero.
    /// </para>
    /// <para>
    /// In Sinc interpolation, this function acts as a "weight function" that determines
    /// how much each known data point contributes to the interpolated value. Points closer
    /// to your query point have more influence than points farther away.
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
}
