namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Natural Cubic Spline interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Natural spline interpolation creates a smooth curve that passes through all given data points.
/// It ensures that the curve has continuous first and second derivatives throughout, resulting
/// in a visually pleasing and mathematically well-behaved interpolation.
/// 
/// <b>For Beginners:</b> Think of natural spline interpolation like drawing a smooth curve through a set of dots.
/// Unlike simpler methods that might connect dots with straight lines, spline interpolation creates
/// gentle curves that flow naturally through each point. It's similar to how artists use flexible rulers
/// (called splines) to draw smooth curves through a set of fixed points. This method is particularly
/// useful when you need a smooth representation of your data without abrupt changes in direction.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NaturalSplineInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _x = default!;
    
    /// <summary>
    /// The y-coordinates (values) of the known data points.
    /// </summary>
    private readonly Vector<T> _y = default!;
    
    /// <summary>
    /// The degree of the spline polynomial. Default is 3 for cubic splines.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree determines how "flexible" the curve can be. A higher degree allows
    /// for more complex curves but might introduce unwanted oscillations. Cubic splines (degree=3)
    /// are the most commonly used as they provide a good balance between smoothness and simplicity.
    /// </remarks>
    private readonly int _degree;
    
    /// <summary>
    /// Optional matrix decomposition method for solving the spline system.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an advanced option that determines how the mathematical equations
    /// are solved internally. Most users can leave this as the default (null).
    /// </remarks>
    private readonly IMatrixDecomposition<T>? _decomposition;
    
    /// <summary>
    /// The calculated coefficients for the spline polynomials.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the values that define the shape of the curve between each pair
    /// of data points. They're calculated automatically when you create the interpolator.
    /// </remarks>
    private readonly Vector<T>[] _coefficients;
    
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Creates a new instance of the Natural Spline interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the interpolator with your data points and configuration options,
    /// and automatically calculates the necessary coefficients for the spline curves.
    /// 
    /// <b>For Beginners:</b> When you create a Natural Spline interpolator, you provide:
    /// 1. Your known data points (x and y coordinates)
    /// 2. The degree of the polynomial (usually 3 for cubic splines)
    /// 
    /// The interpolator will then be ready to estimate y-values at any x location between your data points.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates (values) of the known data points.</param>
    /// <param name="degree">The degree of the spline polynomial. Default is 3 for cubic splines.</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving the system.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths, there are fewer than 2 points, or the degree is less than 2.</exception>
    public NaturalSplineInterpolation(Vector<T> x, Vector<T> y, int degree = 3, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 2)
            throw new ArgumentException("Natural spline interpolation requires at least 2 points.");
        if (degree < 2)
            throw new ArgumentException("Degree must be at least 2 for natural spline interpolation.");

        _x = x;
        _y = y;
        _degree = degree;
        _decomposition = decomposition;
        _numOps = MathHelper.GetNumericOperations<T>();
        _coefficients = new Vector<T>[_degree];

        for (int i = 0; i < _degree; i++)
        {
            _coefficients[i] = new Vector<T>(x.Length - 1);
        }

        CalculateCoefficients();
    }

    /// <summary>
    /// Interpolates the y-value at a given x-coordinate using Natural Spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method calculates the y-value at any x-coordinate by evaluating the appropriate
    /// spline polynomial for the interval containing the x-coordinate.
    /// 
    /// <b>For Beginners:</b> Once you've set up the interpolator with your known points, this method
    /// lets you ask "What would the y-value be at this specific x location?" The algorithm:
    /// 1. Finds which interval (between which two known points) your x-value falls into
    /// 2. Uses the pre-calculated coefficients to evaluate the spline polynomial at that point
    /// 3. Returns the resulting y-value
    /// 
    /// This gives you a smooth estimate that respects the natural curvature suggested by your data points.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        T dx = _numOps.Subtract(x, _x[i]);
        T result = _y[i];

        for (int j = 1; j < _degree; j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[j][i], Power(dx, j)));
        }

        return result;
    }

    /// <summary>
    /// Calculates the coefficients needed for the natural spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method sets up and solves the linear system to find the coefficients for each spline segment.
    /// 
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting to prepare the interpolator.
    /// It calculates how each segment of the curve should be shaped to ensure a smooth transition
    /// between points. This happens automatically when you create the interpolator, so you
    /// don't need to call this method directly.
    /// 
    /// The "natural" in natural spline refers to the boundary conditions that make the second
    /// derivative zero at the endpoints, which creates a more natural-looking curve.
    /// </remarks>
    private void CalculateCoefficients()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);
        Vector<T> b = new Vector<T>(n);

        // Set up the system of equations
        for (int i = 0; i < n - 1; i++)
        {
            T h = _numOps.Subtract(_x[i + 1], _x[i]);
            A[i, i] = h;
            if (i < n - 2)
                A[i, i + 1] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Add(h, _numOps.Subtract(_x[i + 2], _x[i + 1])));
            if (i > 0)
                A[i, i - 1] = h;

            if (i < n - 2)
            {
                T dy1 = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h);
                T dy2 = _numOps.Divide(_numOps.Subtract(_y[i + 2], _y[i + 1]), _numOps.Subtract(_x[i + 2], _x[i + 1]));
                b[i] = _numOps.Multiply(_numOps.FromDouble(6), _numOps.Subtract(dy2, dy1));
            }
        }

        // Apply natural spline conditions
        A[0, 0] = _numOps.One;
        A[n - 1, n - 1] = _numOps.One;
        b[0] = _numOps.Zero;
        b[n - 1] = _numOps.Zero;

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
    /// Finds the interval index in which the given x-coordinate falls.
    /// </summary>
    /// <remarks>
    /// This method determines which pair of known data points the given x-coordinate falls between.
    /// 
    /// <b>For Beginners:</b> Before we can calculate an interpolated value, we need to know which segment
    /// of the curve to use. This method finds the right segment by determining which pair of your
    /// original data points the new x-value falls between. If the x-value is outside the range of
    /// your data, it uses the first or last segment as appropriate.
    /// </remarks>
    /// <param name="x">The x-coordinate to locate.</param>
    /// <returns>The index of the lower bound of the interval containing x.</returns>
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
    /// Calculates the value of x raised to the specified power.
    /// </summary>
    /// <remarks>
    /// This helper method computes x^power by repeatedly multiplying x by itself.
    /// 
    /// <b>For Beginners:</b> This method calculates what happens when you multiply a number by itself
    /// multiple times. For example:
    /// - x^1 = x
    /// - x^2 = x × x
    /// - x^3 = x × x × x
    /// 
    /// In the context of spline interpolation, we need these calculations to evaluate
    /// polynomial terms like x, x², x³, etc., which are used to create the smooth curve.
    /// </remarks>
    /// <param name="x">The base value to be raised to a power.</param>
    /// <param name="power">The exponent (how many times to multiply x by itself).</param>
    /// <returns>The value of x raised to the specified power.</returns>
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