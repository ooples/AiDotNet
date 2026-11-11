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
public partial class TrigonometricInterpolation<T> : IInterpolation<T>
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
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _a;

    /// <summary>
    /// The coefficients for the sine terms in the Fourier series.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _b;

    /// <summary>
    /// The period of the data (the interval after which the pattern repeats).
    /// </summary>
    private readonly T _period;

    /// <summary>
    /// The angular frequency 2 pi k / period of each Fourier term, for k = 0..n.
    /// </summary>
    private readonly Tensor<T> _frequencies;

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

        _x = new Vector<double>(xList).AsTensor().Cast<T>().ToVector();
        _y = new Vector<double>(yList).AsTensor().Cast<T>().ToVector();

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

        // Angular frequency 2 pi k / period of every Fourier term k = 0..n
        int n = (xList.Count - 1) / 2;
        _frequencies = AiDotNetEngine.Current.TensorDivideScalar(
            AiDotNetEngine.Current.TensorMultiplyScalar(Vector<T>.Range(0, n + 1).AsTensor(), _numOps.FromDouble(2 * Math.PI)),
            _period);

        (_a, _b) = CalculateCoefficients();
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
        if (n < 1)
        {
            return result;
        }

        // a[0] + sum_{k=1..n} a[k] cos(f_k x) + sum_{k=1..n-1} b[k - 1] sin(f_k x)
        var engine = AiDotNetEngine.Current;
        Tensor<T> angles = engine.TensorMultiplyScalar(engine.TensorSlice(_frequencies, new[] { 1 }, new[] { n }), x);
        Tensor<T> cosineCoefficients = engine.TensorSlice(_a.AsTensor(), new[] { 1 }, new[] { n });
        result = _numOps.Add(result, engine.TensorSum(engine.TensorMultiply(cosineCoefficients, engine.TensorCos(angles))));

        if (n > 1)
        {
            Tensor<T> sineCoefficients = engine.TensorSlice(_b.AsTensor(), new[] { 0 }, new[] { n - 1 });
            Tensor<T> sineAngles = engine.TensorSlice(angles, new[] { 0 }, new[] { n - 1 });
            result = _numOps.Add(result, engine.TensorSum(engine.TensorMultiply(sineCoefficients, engine.TensorSin(sineAngles))));
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
    private (Vector<T> CosineCoefficients, Vector<T> SineCoefficients) CalculateCoefficients()
    {
        var engine = AiDotNetEngine.Current;
        int count = _x.Length;
        int n = (count - 1) / 2;
        T twoOverN = _numOps.Divide(_numOps.FromDouble(2), _numOps.FromDouble(count));
        Tensor<T> ys = _y.AsTensor().Reshape(count, 1);

        // a[k] = 2/N * sum_j y[j] cos(f_k x[j]) for k = 0..n, one row of angles per frequency
        Tensor<T> cosines = engine.TensorCos(engine.TensorOuter(_frequencies, _x.AsTensor()));
        Vector<T> a = engine.TensorMultiplyScalar(engine.TensorMatMul(cosines, ys), twoOverN).Reshape(n + 1).ToVector();

        // b[k - 1] = 2/N * sum_j y[j] sin(f_k x[j]) for k = 1..n-1; the last sine coefficient stays zero
        if (n <= 1)
        {
            return (a, new Vector<T>(n));
        }

        Tensor<T> sines = engine.TensorSin(engine.TensorOuter(
            engine.TensorSlice(_frequencies, new[] { 1 }, new[] { n - 1 }),
            _x.AsTensor()));
        Tensor<T> b = engine.TensorMultiplyScalar(engine.TensorMatMul(sines, ys), twoOverN).Reshape(n - 1);
        Vector<T> sineCoefficients = engine.TensorConcatenate(new[] { b, Tensor<T>.FromScalar(_numOps.Zero) }, 0).ToVector();

        return (a, sineCoefficients);
    }
}
