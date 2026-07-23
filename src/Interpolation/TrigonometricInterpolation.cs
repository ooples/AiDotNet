namespace AiDotNet.Interpolation;

/// <summary>
/// Implements trigonometric interpolation for periodic data using Fourier series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Trigonometric interpolation is a method for fitting a trigonometric polynomial (a sum of sines and cosines)
/// to a set of data points. It is particularly effective for periodic data, such as seasonal patterns,
/// wave forms, or any data that repeats over a fixed interval.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of trigonometric interpolation like finding a musical chord that matches a set of notes.
/// Just as a complex sound can be broken down into simple sine waves of different frequencies (harmonics),
/// this method breaks down your data into simple wave patterns. It works best when your data has a repeating
/// pattern, like daily temperature cycles, seasonal sales data, or sound waves. The interpolation creates
/// a smooth curve that passes through all your data points and can predict values between them.
/// </para>
/// </remarks>
public class TrigonometricInterpolation<T> : IInterpolation<T>
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
    /// The coefficients for the cosine terms in the Fourier series.
    /// </summary>
    private readonly Vector<T> _a;

    /// <summary>
    /// The coefficients for the sine terms in the Fourier series.
    /// </summary>
    private readonly Vector<T> _b;

    /// <summary>
    /// The period of the data (the interval after which the pattern repeats).
    /// </summary>
    private readonly T _period;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of trigonometric interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates (values) of the data points.</param>
    /// <param name="customPeriod">Optional custom period for the data. If not provided, it will be calculated as max(x) - min(x).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths or when the number of points is even.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your data points (x and y coordinates) and prepares the
    /// interpolation algorithm. The x-values represent positions (like time points) and the y-values
    /// represent measurements at those positions (like temperature readings).
    /// </para>
    /// <para>
    /// The "period" is how long it takes for your data pattern to repeat. For example, if you're tracking
    /// daily temperatures over a year, the period would be 365 days. You can either:
    /// - Let the algorithm calculate this automatically (by finding the range of your x-values)
    /// - Provide your own period if you know exactly what it should be
    /// </para>
    /// <para>
    /// Note: This method requires an odd number of data points for mathematical reasons. If you have an
    /// even number of points, you might need to add or remove a point.
    /// </para>
    /// </remarks>
    public TrigonometricInterpolation(IEnumerable<double> x, IEnumerable<double> y, double? customPeriod = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        var xList = x.ToList();
        var yList = y.ToList();

        if (xList.Count != yList.Count)
            throw new ArgumentException("Input vectors must have the same length.");
        if (xList.Count % 2 == 0)
            throw new ArgumentException("Number of points must be odd for trigonometric interpolation.");

        _x = new Vector<T>(xList.Select(val => _numOps.FromDouble(val)));
        _y = new Vector<T>(yList.Select(val => _numOps.FromDouble(val)));

        // Calculate or set the period
        if (customPeriod.HasValue)
        {
            _period = _numOps.FromDouble(customPeriod.Value);
        }
        else
        {
            double calculatedPeriod = xList.Max() - xList.Min();
            _period = _numOps.FromDouble(calculatedPeriod);
        }

        int n = (xList.Count - 1) / 2;
        _a = new Vector<T>(n + 1);
        _b = new Vector<T>(n);

        CalculateCoefficients();
    }

    /// <summary>
    /// Interpolates a y-value for the given x coordinate using trigonometric interpolation.
    /// </summary>
    /// <param name="x">The x-coordinate for which to interpolate.</param>
    /// <returns>The interpolated y-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the y-value at any x-position you specify, even if it's
    /// between your original data points. It uses the trigonometric formula (a sum of sine and
    /// cosine waves) that was calculated to fit your data.
    /// </para>
    /// <para>
    /// The formula looks like: y = a0 + a1cos(2px/P) + b1sin(2px/P) + a2cos(4px/P) + b2sin(4px/P) + ...
    /// where P is the period, and a0, a1, b1, etc. are the coefficients calculated from your data.
    /// </para>
    /// <para>
    /// Since this is a periodic function, you can even ask for x-values outside your original data range,
    /// and it will give you predictions based on the repeating pattern it found.
    /// </para>
    /// </remarks>
    public T Interpolate(T x)
    {
        T result = _a[0];
        int n = _a.Length - 1;

        for (int k = 1; k <= n; k++)
        {
            T angle = _numOps.Multiply(_numOps.Divide(_numOps.FromDouble(2 * k * Math.PI), _period), x);
            result = _numOps.Add(result, _numOps.Multiply(_a[k], MathHelper.Cos(angle)));

            if (k < n)
            {
                result = _numOps.Add(result, _numOps.Multiply(_b[k - 1], MathHelper.Sin(angle)));
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the Fourier coefficients for the trigonometric interpolation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting to find the "recipe" for
    /// recreating your data using sine and cosine waves. It calculates how much of each wave
    /// frequency to include in the final mix.
    /// </para>
    /// <para>
    /// The coefficients it calculates (stored in _a and _b) represent:
    /// - _a[0]: The average value of your data (like a baseline)
    /// - _a[1], _a[2], etc.: How much of each cosine wave to include
    /// - _b[0], _b[1], etc.: How much of each sine wave to include
    /// </para>
    /// <para>
    /// These calculations use the Discrete Fourier Transform (DFT) principle, which finds the
    /// strength of different frequencies in your data. The more data points you have, the more
    /// accurate these coefficients will be.
    /// </para>
    /// </remarks>
    private void CalculateCoefficients()
    {
        int n = (_x.Length - 1) / 2;
        T twoOverN = _numOps.Divide(_numOps.FromDouble(2), _numOps.FromDouble(_x.Length));

        for (int k = 0; k <= n; k++)
        {
            _a[k] = _numOps.Zero;
            for (int j = 0; j < _x.Length; j++)
            {
                T angle = _numOps.Multiply(_numOps.Divide(_numOps.FromDouble(2 * k * Math.PI), _period), _x[j]);
                _a[k] = _numOps.Add(_a[k], _numOps.Multiply(_y[j], MathHelper.Cos(angle)));
            }

            _a[k] = _numOps.Multiply(_a[k], twoOverN);
        }

        for (int k = 1; k < n; k++)
        {
            _b[k - 1] = _numOps.Zero;
            for (int j = 0; j < _x.Length; j++)
            {
                T angle = _numOps.Multiply(_numOps.Divide(_numOps.FromDouble(2 * k * Math.PI), _period), _x[j]);
                _b[k - 1] = _numOps.Add(_b[k - 1], _numOps.Multiply(_y[j], MathHelper.Sin(angle)));
            }

            _b[k - 1] = _numOps.Multiply(_b[k - 1], twoOverN);
        }
    }
}
