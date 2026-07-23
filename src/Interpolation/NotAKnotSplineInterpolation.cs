namespace AiDotNet.Interpolation;

/// <summary>
/// Implements the Not-a-Knot cubic spline interpolation method, which creates a smooth curve through a set of data points.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Spline interpolation creates a smooth curve that passes through all your data points.
/// Unlike simpler methods, it ensures the curve is not just continuous but also has continuous first and 
/// second derivatives, making it appear very smooth and natural.
/// </para>
/// <para>
/// The "Not-a-Knot" condition is a specific way to handle the endpoints of the curve. In simple terms,
/// it makes the curve extra smooth at the first and last interior points by ensuring the third derivative
/// is continuous there.
/// </para>
/// <para>
/// Think of it like drawing a smooth curve through points with a flexible ruler (spline) that has special
/// properties at the ends to make the transition particularly smooth.
/// </para>
/// <para>
/// This method is excellent for creating natural-looking curves through data points, such as in animation,
/// graphics, or when modeling physical phenomena.
/// </para>
/// </remarks>
public class NotAKnotSplineInterpolation<T> : IInterpolation<T>
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
    /// Optional matrix decomposition method used to solve the linear system of equations.
    /// </summary>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The coefficients of the cubic spline polynomials for each segment.
    /// </summary>
    /// <remarks>
    /// This array contains 4 vectors, each representing a set of coefficients:
    /// - _coefficients[0]: Constant terms (a)
    /// - _coefficients[1]: Linear terms (b)
    /// - _coefficients[2]: Quadratic terms (c)
    /// - _coefficients[3]: Cubic terms (d)
    /// </remarks>
    private readonly Vector<T>[] _coefficients;

    /// <summary>
    /// Operations for performing numeric calculations with generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="NotAKnotSplineInterpolation{T}"/> class.
    /// </summary>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates (values) of the known data points.</param>
    /// <param name="decomposition">Optional matrix decomposition method to use for solving the system of equations.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input vectors have different lengths or when fewer than 4 points are provided.
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
    /// needed to create a smooth curve through all your points.
    /// </para>
    /// <para>
    /// The optional decomposition parameter is for advanced users who want to specify a particular
    /// method for solving the internal mathematical equations.
    /// </para>
    /// </remarks>
    public NotAKnotSplineInterpolation(Vector<T> x, Vector<T> y, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("Not-a-knot spline interpolation requires at least 4 points.");

        _x = x;
        _y = y;
        _decomposition = decomposition;
        _numOps = MathHelper.GetNumericOperations<T>();

        _coefficients = new Vector<T>[4];
        for (int i = 0; i < 4; i++)
        {
            _coefficients[i] = new Vector<T>(x.Length - 1);
        }

        CalculateCoefficients();
    }

    /// <summary>
    /// Performs cubic spline interpolation to estimate a y-value for the given x-value.
    /// </summary>
    /// <param name="x">The x-value for which to estimate the corresponding y-value.</param>
    /// <returns>The interpolated y-value at the specified x-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes an x-value and returns the estimated y-value based on the smooth curve
    /// that was created from your data points.
    /// </para>
    /// <para>
    /// For example, if we have data points at x = [1, 3, 5, 7] with values y = [10, 30, 50, 40],
    /// and we ask for x = 4, the method will return a value around 40, which lies on the smooth curve
    /// passing through all four original points.
    /// </para>
    /// <para>
    /// The cubic spline creates a very smooth and natural-looking curve between your points, which is
    /// often more realistic than simpler interpolation methods.
    /// </para>
    /// </remarks>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T dx = _numOps.Subtract(x, _x[i]);
        T result = _y[i];

        for (int j = 1; j < 4; j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[j][i], Power(dx, j)));
        }

        return result;
    }

    /// <summary>
    /// Calculates the coefficients for the cubic spline polynomials.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up and solves a system of equations to find the coefficients for the cubic spline.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the "behind the scenes" math that makes the interpolation work. It calculates
    /// the formula for the smooth curve that will pass through all your data points.
    /// </para>
    /// <para>
    /// The method creates a set of cubic polynomials (one for each segment between data points) that together
    /// form a smooth curve. The "Not-a-Knot" condition is a special way to handle the endpoints to make the
    /// curve extra smooth.
    /// </para>
    /// <para>
    /// Each cubic polynomial has the form: a + b(x-xi) + c(x-xi)² + d(x-xi)³, where a, b, c, and d are the
    /// coefficients stored in the _coefficients array.
    /// </para>
    /// </remarks>
    private void CalculateCoefficients()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);
        Vector<T> b = new Vector<T>(n);

        // Set up the system of equations
        for (int i = 1; i < n - 1; i++)
        {
            T h_prev = _numOps.Subtract(_x[i], _x[i - 1]);
            T h_next = _numOps.Subtract(_x[i + 1], _x[i]);

            A[i, i - 1] = h_prev;
            A[i, i] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Add(h_prev, h_next));
            A[i, i + 1] = h_next;

            T dy_prev = _numOps.Divide(_numOps.Subtract(_y[i], _y[i - 1]), h_prev);
            T dy_next = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h_next);
            b[i] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(dy_next, dy_prev));
        }

        // Apply not-a-knot conditions
        T h0 = _numOps.Subtract(_x[1], _x[0]);
        T h1 = _numOps.Subtract(_x[2], _x[1]);
        T hn_2 = _numOps.Subtract(_x[n - 2], _x[n - 3]);
        T hn_1 = _numOps.Subtract(_x[n - 1], _x[n - 2]);

        A[0, 0] = _numOps.Multiply(h1, _numOps.Add(h0, h1));
        A[0, 1] = _numOps.Multiply(h0, _numOps.Negate(_numOps.Add(h0, h1)));
        A[0, 2] = _numOps.Multiply(h0, h0);

        A[n - 1, n - 3] = _numOps.Multiply(hn_1, hn_1);
        A[n - 1, n - 2] = _numOps.Multiply(hn_2, _numOps.Negate(_numOps.Add(hn_2, hn_1)));
        A[n - 1, n - 1] = _numOps.Multiply(hn_2, _numOps.Add(hn_2, hn_1));

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        Vector<T> m = MatrixSolutionHelper.SolveLinearSystem(b, decomposition);

        // Calculate the coefficients
        for (int i = 0; i < n - 1; i++)
        {
            T h = _numOps.Subtract(_x[i + 1], _x[i]);
            _coefficients[0][i] = _y[i];
            _coefficients[1][i] = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h);
            _coefficients[1][i] = _numOps.Subtract(_coefficients[1][i], _numOps.Multiply(_numOps.Divide(h, _numOps.FromDouble(6)), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), m[i]), m[i + 1])));
            _coefficients[2][i] = _numOps.Divide(m[i], _numOps.FromDouble(2));
            _coefficients[3][i] = _numOps.Divide(_numOps.Subtract(m[i + 1], m[i]), _numOps.Multiply(_numOps.FromDouble(6), h));
        }
    }

    /// <summary>
    /// Finds the appropriate interval in the data points for the given x-value.
    /// </summary>
    /// <param name="x">The x-value for which to find the corresponding interval.</param>
    /// <returns>The index of the interval where the x-value belongs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines which segment of the curve should be used for interpolation.
    /// </para>
    /// <para>
    /// Think of it like this: if you have data points at x = [1, 3, 5, 7], and you want to find the y-value
    /// at x = 4, this method will tell you that x = 4 falls between the second and third points (index 1).
    /// </para>
    /// <para>
    /// The method searches through the x-coordinates to find where your input value fits. If the input value
    /// is beyond the last data point, it uses the last interval for extrapolation.
    /// </para>
    /// </remarks>
    private int FindInterval(T x)
    {
        for (int i = 0; i < _x.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
                return i;
        }

        return _x.Length - 2;
    }

    /// <summary>
    /// Calculates the power of a value (x raised to the specified power).
    /// </summary>
    /// <param name="x">The base value.</param>
    /// <param name="power">The exponent (power) to raise the base value to.</param>
    /// <returns>The result of x raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates x raised to a power. For example:
    /// - Power(2, 3) = 2³ = 2 × 2 × 2 = 8
    /// - Power(5, 2) = 5² = 5 × 5 = 25
    /// </para>
    /// <para>
    /// The method works by multiplying the base value by itself the specified number of times.
    /// This is needed for calculating the polynomial terms in the cubic spline formula.
    /// </para>
    /// </remarks>
    private T Power(T x, int power)
    {
        T result = _numOps.One;
        for (int i = 0; i < power; i++)
        {
            result = _numOps.Multiply(result, x);
        }

        return result;
    }
}
