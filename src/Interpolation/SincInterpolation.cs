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
    /// Default is 1.0.
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
    public SincInterpolation(IEnumerable<double> x, IEnumerable<double> y, double cutoffFrequency = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        var xList = x.ToList();
        var yList = y.ToList();

        if (xList.Count != yList.Count)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = new Vector<T>(xList.Select(val => _numOps.FromDouble(val)));
        _y = new Vector<T>(yList.Select(val => _numOps.FromDouble(val)));
        _cutoffFrequency = _numOps.FromDouble(cutoffFrequency);
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
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T diff = _numOps.Subtract(x, _x[i]);
            T sincValue = Sinc(_numOps.Multiply(_cutoffFrequency, diff));
            result = _numOps.Add(result, _numOps.Multiply(_y[i], sincValue));
        }

        return result;
    }

    /// <summary>
    /// Calculates the Sinc function value: sin(px)/(px).
    /// </summary>
    /// <param name="x">The input value.</param>
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
    private T Sinc(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
        {
            return _numOps.One;
        }

        T piX = _numOps.Multiply(MathHelper.Pi<T>(), x);
        return _numOps.Divide(MathHelper.Sin(piX), piX);
    }
}
