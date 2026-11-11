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
    /// <param name="customPeriod">The period of the data. The two-argument constructor uses max(x) - min(x).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths, when the number of points is even, or when there are
    /// fewer than three points.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the period is not positive and finite.</exception>
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
    public TrigonometricInterpolation(Vector<T> x, Vector<T> y, T customPeriod)
    {
        Guard.NotNull(x);
        Guard.NotNull(y);
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length % 2 == 0)
            throw new ArgumentException("Number of points must be odd for trigonometric interpolation.");
        if (x.Length < 3)
            throw new ArgumentException("At least three points are required for trigonometric interpolation.", nameof(x));

        _numOps = MathHelper.GetNumericOperations<T>();
        if (!_numOps.GreaterThan(customPeriod, _numOps.Zero) || _numOps.IsInfinity(customPeriod))
            throw new ArgumentOutOfRangeException(nameof(customPeriod), "The period must be positive and finite.");
        _x = x;
        _y = y;
        _period = customPeriod;

        // Angular frequency 2 pi k / period of every Fourier term k = 0..n
        int n = (x.Length - 1) / 2;
        _frequencies = AiDotNetEngine.Current.TensorDivideScalar(
            AiDotNetEngine.Current.TensorMultiplyScalar(Vector<T>.Range(0, n + 1).AsTensor(), _numOps.FromDouble(2 * Math.PI)),
            _period);

        (_a, _b) = CalculateCoefficients();
    }

    /// <summary>
    /// Creates a trigonometric interpolation whose period is the span of the x-values, max(x) - min(x).
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates (values) of the data points.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths, when the number of points is even, or when there are no points.
    /// </exception>
    public TrigonometricInterpolation(Vector<T> x, Vector<T> y)
        : this(x, y, Span(x))
    {
    }

    /// <summary>
    /// The span of <paramref name="x"/>, max(x) - min(x), used as the default period.
    /// </summary>
    private static T Span(Vector<T> x)
    {
        Guard.NotNull(x);
        if (x.Length == 0)
            throw new ArgumentException("At least one point is required for trigonometric interpolation.", nameof(x));

        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Subtract(TensorPrimitivesHelper<T>.Max(x), TensorPrimitivesHelper<T>.Min(x));
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

        // a[0] (already halved to the mean) + sum_{k=1..n} a[k] cos(f_k x) + sum_{k=1..n-1} b[k - 1] sin(f_k x)
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

        // a[k] = 2/N * sum_j y[j] cos(f_k x[j]) for k = 0..n, one row of angles per frequency. The interpolant is
        // a[0] / 2 + sum_k (a[k] cos + b[k] sin), so the stored constant term is halved to the data mean:
        // scale = [1/2, 1, ..., 1] applied in one engine multiply.
        Tensor<T> cosineProjections = engine.TensorMultiplyScalar(
            ProjectOnto(engine, _frequencies, ys, cosine: true), twoOverN).Reshape(n + 1);
        Tensor<T> scale = engine.TensorConcatenate(new[]
        {
            Tensor<T>.CreateDefault(new[] { 1 }, _numOps.FromDouble(0.5)),
            Tensor<T>.CreateDefault(new[] { n }, _numOps.One)
        }, 0);
        Vector<T> a = engine.TensorMultiply(cosineProjections, scale).ToVector();

        // b[k - 1] = 2/N * sum_j y[j] sin(f_k x[j]) for k = 1..n-1; the last sine coefficient stays zero
        if (n <= 1)
        {
            return (a, new Vector<T>(n));
        }

        Tensor<T> sineFrequencies = engine.TensorSlice(_frequencies, new[] { 1 }, new[] { n - 1 });
        Tensor<T> b = engine.TensorMultiplyScalar(ProjectOnto(engine, sineFrequencies, ys, cosine: false), twoOverN)
            .Reshape(n - 1);
        Vector<T> sineCoefficients = engine.TensorConcatenate(new[] { b, Tensor<T>.FromScalar(_numOps.Zero) }, 0).ToVector();

        return (a, sineCoefficients);
    }

    /// <summary>
    /// Upper bound on the elements of one [frequencies, points] angle block in <see cref="ProjectOnto"/>, so the
    /// coefficient calculation never holds the full [n + 1, N] angle matrix.
    /// </summary>
    private const int CoefficientBlockElements = 1 << 20;

    /// <summary>
    /// sum_j ys[j] * cos(f_k x[j]) (or sin) for every frequency f_k, as a [frequencies, 1] tensor, computed over
    /// bounded blocks of frequency rows.
    /// </summary>
    private Tensor<T> ProjectOnto(IEngine engine, Tensor<T> frequencies, Tensor<T> ys, bool cosine)
    {
        int rows = frequencies.Length;
        int count = _x.Length;
        int block = Math.Max(1, Math.Min(rows, CoefficientBlockElements / Math.Max(1, count)));
        var parts = new List<Tensor<T>>();
        for (int start = 0; start < rows; start += block)
        {
            int size = Math.Min(block, rows - start);
            Tensor<T> angles = engine.TensorOuter(engine.TensorNarrow(frequencies, 0, start, size).Contiguous(), _x.AsTensor());
            Tensor<T> basis = cosine ? engine.TensorCos(angles) : engine.TensorSin(angles);
            parts.Add(engine.TensorMatMul(basis, ys));
        }

        return parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0);
    }
}
