namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Newton's divided difference interpolation method, which creates a polynomial that passes through all given data points.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Interpolation is a way to estimate values between known data points. Newton's divided difference
/// interpolation creates a smooth curve (a polynomial) that passes exactly through all your known data points.
/// </para>
/// <para>
/// Unlike simpler methods like nearest neighbor, this method creates a continuous curve that can provide more
/// accurate estimates between your data points. It's especially useful when you need a smooth function that
/// exactly matches your known data.
/// </para>
/// <para>
/// Think of it like connecting dots with a smooth curve instead of straight lines or steps.
/// </para>
/// </remarks>
public partial class NewtonDividedDifferenceInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The coefficients of the Newton polynomial, calculated from the input data.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _coefficients;

    /// <summary>
    /// Operations for performing numeric calculations with generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="NewtonDividedDifferenceInterpolation{T}"/> class.
    /// </summary>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates (values) of the known data points.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input vectors have different lengths or when fewer than 2 points are provided.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes two vectors (arrays) of equal length:
    /// - The x vector contains the input values (like time points, positions, etc.)
    /// - The y vector contains the corresponding output values
    /// </para>
    /// <para>
    /// For example, if you're tracking temperature over time, x might be the hours [1,2,3,4]
    /// and y might be the temperatures [68,72,75,73].
    /// </para>
    /// <para>
    /// When you create this object, it automatically calculates the mathematical formula
    /// (polynomial coefficients) needed to create a smooth curve through all your points.
    /// </para>
    /// </remarks>
    public NewtonDividedDifferenceInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Newton's divided difference interpolation requires at least 2 points.");
        }

        _x = x;
        _numOps = MathHelper.GetNumericOperations<T>();
        _coefficients = CalculateCoefficients(x, y);
    }

    /// <summary>
    /// Performs Newton's divided difference interpolation to estimate a y-value for the given x-value.
    /// </summary>
    /// <param name="x">The x-value for which to estimate the corresponding y-value.</param>
    /// <returns>The interpolated y-value at the specified x-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes an x-value and returns the estimated y-value based on the smooth curve
    /// that was created from your data points.
    /// </para>
    /// <para>
    /// For example, if we have data points at x = [1, 3, 5] with values y = [10, 30, 50],
    /// and we ask for x = 2, the method will return a value around 20, which lies on the smooth curve
    /// passing through all three original points.
    /// </para>
    /// <para>
    /// Unlike simpler methods, this will give you a smooth transition between points rather than jumps or steps.
    /// </para>
    /// </remarks>
    public T Interpolate(T x)
    {
        int n = _coefficients.Length;
        if (n == 1)
        {
            return _coefficients[0];
        }

        // term[i] = (x - x[0]) * ... * (x - x[i - 1]) as a running product, then c[0] + sum c[i] * term[i]
        var engine = AiDotNetEngine.Current;
        Tensor<T> factors = engine.ScalarMinusTensor(x, engine.TensorSlice(_x.AsTensor(), new[] { 0 }, new[] { n - 1 }));
        Tensor<T> terms = engine.TensorCumProd(factors, 0);
        Tensor<T> higherCoefficients = engine.TensorSlice(_coefficients.AsTensor(), new[] { 1 }, new[] { n - 1 });

        return _numOps.Add(_coefficients[0], engine.TensorSum(engine.TensorMultiply(higherCoefficients, terms)));
    }

    /// <summary>
    /// Calculates the coefficients for Newton's divided difference polynomial.
    /// </summary>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates (values) of the known data points.</param>
    /// <returns>A vector of coefficients for the Newton polynomial.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the divided difference algorithm to find the coefficients of the Newton form
    /// of the interpolation polynomial.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the "behind the scenes" math that makes the interpolation work. It calculates
    /// the formula for the smooth curve that will pass through all your data points. The math involves
    /// calculating how the rate of change varies across your data points.
    /// </para>
    /// <para>
    /// The divided differences table is a systematic way to calculate these rates of change at different levels,
    /// which then become the coefficients for our polynomial formula.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateCoefficients(Vector<T> x, Vector<T> y)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Length;
        Tensor<T> xs = x.AsTensor();

        // Column j of the divided-difference table is (column[i + 1] - column[i]) / (x[i + j] - x[i]),
        // and the Newton coefficients are the first entry of every column
        Tensor<T> column = y.AsTensor();
        var leadingEntries = new Tensor<T>[n];
        leadingEntries[0] = engine.TensorSlice(column, new[] { 0 }, new[] { 1 });
        for (int j = 1; j < n; j++)
        {
            int length = n - j;
            column = engine.TensorDivide(
                engine.TensorSubtract(
                    engine.TensorSlice(column, new[] { 1 }, new[] { length }),
                    engine.TensorSlice(column, new[] { 0 }, new[] { length })),
                engine.TensorSubtract(
                    engine.TensorSlice(xs, new[] { j }, new[] { length }),
                    engine.TensorSlice(xs, new[] { 0 }, new[] { length })));
            leadingEntries[j] = engine.TensorSlice(column, new[] { 0 }, new[] { 1 });
        }

        return engine.TensorConcatenate(leadingEntries, 0).ToVector();
    }
}
