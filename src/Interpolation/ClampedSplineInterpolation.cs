namespace AiDotNet.Interpolation;

/// <summary>
/// Implements clamped cubic spline interpolation for smooth curve fitting with controlled endpoints.
/// </summary>
/// <remarks>
/// Clamped cubic splines create smooth curves that pass through all provided data points
/// while allowing you to specify the slope at the endpoints of the curve.
/// 
/// <b>For Beginners:</b> Think of this as drawing a smooth curve through a set of points where
/// you can control the "direction" the curve enters and exits the first and last points.
/// This is useful when you need the curve to approach the endpoints from specific angles,
/// such as when connecting to other curves or when modeling physical phenomena with known
/// behavior at the boundaries.
/// 
/// Unlike other interpolation methods, clamped splines give you control over how the curve
/// behaves at its edges, making them ideal for scenarios where the slope at the endpoints
/// is known or important.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class ClampedSplineInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the data points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The slope of the curve at the first data point.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls the "direction" or "steepness" of the curve as it
    /// leaves the first point. A positive value means the curve goes upward, a negative
    /// value means it goes downward, and zero means it's flat (horizontal).
    /// </remarks>
    private readonly T _startSlope;

    /// <summary>
    /// The slope of the curve at the last data point.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls the "direction" or "steepness" of the curve as it
    /// approaches the last point. A positive value means the curve comes from below,
    /// a negative value means it comes from above, and zero means it approaches horizontally.
    /// </remarks>
    private readonly T _endSlope;

    /// <summary>
    /// The matrix decomposition method used to solve the spline equations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an advanced mathematical tool that helps solve the equations
    /// needed to create the smooth curve. You typically don't need to worry about this
    /// unless you have specific performance requirements.
    /// </remarks>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The coefficients of the cubic polynomials that define the spline segments.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the mathematical values that define the shape of the curve
    /// between each pair of points. They're calculated automatically when you create the spline.
    /// </remarks>
    private readonly Vector<T>[] _coefficients;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a utility that helps perform math operations (like addition
    /// and multiplication) on different types of numbers. You don't need to interact with
    /// this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the ClampedSplineInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor validates the input data, sets up the spline parameters, and calculates
    /// the coefficients needed for interpolation.
    /// 
    /// <b>For Beginners:</b> This sets up everything needed to create a smooth curve through your points:
    /// 1. It checks that you have enough points (at least 2)
    /// 2. It stores your x and y coordinates
    /// 3. It sets up the slopes at the endpoints
    /// 4. It calculates all the mathematical values needed to create the smooth curve
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="startSlope">
    /// The slope at the first point. Default is 0.0 (horizontal).
    /// </param>
    /// <param name="endSlope">
    /// The slope at the last point. Default is 0.0 (horizontal).
    /// </param>
    /// <param name="decomposition">
    /// Optional matrix decomposition method for solving the spline equations.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors don't have the same length or when there are fewer than 2 points.
    /// </exception>
    public ClampedSplineInterpolation(Vector<T> x, Vector<T> y, double startSlope = 0.0, double endSlope = 0.0, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 2)
            throw new ArgumentException("Clamped spline interpolation requires at least 2 points.");

        _x = x;
        _y = y;
        _decomposition = decomposition;
        _numOps = MathHelper.GetNumericOperations<T>();

        _startSlope = _numOps.FromDouble(startSlope);
        _endSlope = _numOps.FromDouble(endSlope);

        _coefficients = new Vector<T>[4];

        for (int i = 0; i < 4; i++)
        {
            _coefficients[i] = new Vector<T>(x.Length - 1);
        }

        CalculateCoefficients();
    }

    /// <summary>
    /// Interpolates a y-value at the specified x-coordinate using clamped cubic spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method finds the segment containing the target x-coordinate and calculates
    /// the corresponding y-value using the cubic polynomial for that segment.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an x-value, and it will:
    /// 1. Find which segment of your data contains this x-value
    /// 2. Use the cubic polynomial formula for that segment
    /// 3. Calculate the y-value on the smooth curve at your requested x-position
    /// 
    /// For example, if you have data points at x = [0, 10, 20, 30] and you want to know
    /// the y-value at x = 15, this method will find that 15 is between 10 and 20, and then
    /// use the appropriate formula to calculate the exact y-value on the smooth curve at x = 15.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
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
    /// Calculates the coefficients needed for the cubic spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method sets up and solves a system of equations to determine the coefficients
    /// that define the cubic polynomial for each segment of the spline.
    /// 
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting to create the smooth curve.
    /// It works by:
    /// 1. Setting up a system of equations based on your data points
    /// 2. Applying the special "clamped" conditions at the endpoints using your specified slopes
    /// 3. Solving these equations to find values called "moments" at each point
    /// 4. Using these moments to calculate the final coefficients that define the curve
    /// 
    /// You don't need to call this method directly - it's automatically called when you create
    /// a new ClampedSplineInterpolation object.
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

        // Apply clamped spline conditions
        T h_first = _numOps.Subtract(_x[1], _x[0]);
        T h_last = _numOps.Subtract(_x[n - 1], _x[n - 2]);

        A[0, 0] = _numOps.Multiply(_numOps.FromDouble(2), h_first);
        A[0, 1] = h_first;
        b[0] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(_numOps.Divide(_numOps.Subtract(_y[1], _y[0]), h_first), _startSlope));

        A[n - 1, n - 2] = h_last;
        A[n - 1, n - 1] = _numOps.Multiply(_numOps.FromDouble(2), h_last);
        b[n - 1] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(_endSlope, _numOps.Divide(_numOps.Subtract(_y[n - 1], _y[n - 2]), h_last)));

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
    /// Finds the interval in the data that contains the given x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method searches through the x-coordinates to find which segment
    /// contains the target x-value.
    /// 
    /// <b>For Beginners:</b> Think of this as finding which "piece" of the curve contains
    /// your x-value. For example, if your data points are at x = [0, 10, 20, 30]
    /// and you want to find where x = 15 belongs, this method will tell you it's
    /// in the segment between 10 and 20 (which is segment index 1).
    /// 
    /// If the x-value is beyond the last data point, it returns the index of the
    /// last segment.
    /// </remarks>
    /// <param name="x">The x-coordinate to locate within the data.</param>
    /// <returns>The index of the interval containing the x-coordinate.</returns>
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
    /// Raises a value to the specified power through repeated multiplication.
    /// </summary>
    /// <remarks>
    /// This method calculates x raised to the power of 'power' by multiplying
    /// x by itself 'power' times.
    /// 
    /// <b>For Beginners:</b> This is a simple way to calculate powers. For example:
    /// - Power(x, 2) means x² (x squared)
    /// - Power(x, 3) means x³ (x cubed)
    /// 
    /// This is needed because cubic splines use polynomial equations that include
    /// terms like x², x³, etc.
    /// </remarks>
    /// <param name="x">The base value.</param>
    /// <param name="power">The exponent (power) to raise the base to.</param>
    /// <returns>The result of x raised to the specified power.</returns>
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
