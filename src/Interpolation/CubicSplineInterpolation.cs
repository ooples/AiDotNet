namespace AiDotNet.Interpolation;

/// <summary>
/// Implements cubic spline interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Cubic spline interpolation creates a smooth curve that passes through all given data points.
/// The curve consists of piecewise cubic polynomials with continuous first and second derivatives.
/// 
/// <b>For Beginners:</b> This class helps you estimate values between known data points.
/// Imagine you have measurements at specific times (like temperature readings every hour),
/// and you want to estimate what happened between those measurements. Cubic spline
/// interpolation creates a smooth curve that passes through all your known points and
/// provides natural-looking estimates for the points in between. It's like connecting
/// dots with a flexible curve rather than straight lines.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class CubicSplineInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points (independent variable).
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the data points (dependent variable).
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The constant coefficients of the cubic polynomials (equal to the y values).
    /// </summary>
    private readonly Vector<T> _a;

    /// <summary>
    /// The coefficients of the linear terms in the cubic polynomials.
    /// </summary>
    private readonly Vector<T> _b;

    /// <summary>
    /// The coefficients of the quadratic terms in the cubic polynomials.
    /// </summary>
    private readonly Vector<T> _c;

    /// <summary>
    /// The coefficients of the cubic terms in the cubic polynomials.
    /// </summary>
    private readonly Vector<T> _d;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new cubic spline interpolation from the given data points.
    /// </summary>
    /// <remarks>
    /// The constructor initializes the spline by calculating all the coefficients
    /// needed for interpolation.
    /// 
    /// <b>For Beginners:</b> When you create a new CubicSplineInterpolation object with your data points,
    /// it automatically does all the complex math needed to set up the smooth curve.
    /// You just need to provide the x and y coordinates of your known points, and the
    /// constructor handles the rest.
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points (must be in ascending order).</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    public CubicSplineInterpolation(Vector<T> x, Vector<T> y)
    {
        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        int n = x.Length - 1;
        _a = y.Clone();
        _b = new Vector<T>(n);
        _c = new Vector<T>(n + 1);
        _d = new Vector<T>(n);

        CalculateCoefficients();
    }

    /// <summary>
    /// Calculates the interpolated y-value for a given x-value using cubic spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method finds the interval containing the x-value and evaluates the corresponding
    /// cubic polynomial at that point.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use after creating the interpolation.
    /// Give it any x-value within your data range, and it will return the estimated y-value
    /// at that point. It's like asking "if I have this x-value, what would the y-value be
    /// on the smooth curve that passes through all my known points?"
    /// </remarks>
    /// <param name="x">The x-value at which to interpolate.</param>
    /// <returns>The interpolated y-value at the given x-value.</returns>
    public T Interpolate(T x)
    {
        // Find which interval contains the x-value
        int i = FindInterval(x);

        // Calculate the distance from the left endpoint of the interval
        T dx = _numOps.Subtract(x, _x[i]);

        // Evaluate the cubic polynomial: a + b*dx + c*dx² + d*dx³
        return _numOps.Add(
            _numOps.Add(
                _numOps.Add(
                    _a[i],
                    _numOps.Multiply(_b[i], dx)
                ),
                _numOps.Multiply(_c[i], _numOps.Multiply(dx, dx))
            ),
            _numOps.Multiply(_d[i], _numOps.Multiply(_numOps.Multiply(dx, dx), dx))
        );
    }

    /// <summary>
    /// Calculates the coefficients of the cubic spline polynomials.
    /// </summary>
    /// <remarks>
    /// This method implements the tridiagonal algorithm to solve for the coefficients
    /// of the cubic spline. It ensures that the resulting curve is continuous and has
    /// continuous first and second derivatives at all data points.
    /// 
    /// <b>For Beginners:</b> This method does the complex math needed to create a smooth curve.
    /// It calculates the coefficients (a, b, c, d) that define the shape of the curve
    /// between each pair of data points. These coefficients ensure that the curve not only
    /// passes through all points but also transitions smoothly between segments.
    /// </remarks>
    private void CalculateCoefficients()
    {
        int n = _x.Length - 1;

        // Calculate the width of each interval
        Vector<T> h = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            h[i] = _numOps.Subtract(_x[i + 1], _x[i]);
        }

        // Calculate the right-hand side of the tridiagonal system
        Vector<T> alpha = new Vector<T>(n);
        for (int i = 1; i < n; i++)
        {
            alpha[i] = _numOps.Multiply(
                _numOps.FromDouble(3),
                _numOps.Subtract(
                    _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h[i]),
                    _numOps.Divide(_numOps.Subtract(_y[i], _y[i - 1]), h[i - 1])
                )
            );
        }

        // Solve the tridiagonal system using the Thomas algorithm
        Vector<T> l = new Vector<T>(n + 1);
        Vector<T> mu = new Vector<T>(n + 1);
        Vector<T> z = new Vector<T>(n + 1);

        // Initialize the first row
        l[0] = _numOps.One;
        mu[0] = _numOps.Zero;
        z[0] = _numOps.Zero;

        // Forward elimination
        for (int i = 1; i < n; i++)
        {
            l[i] = _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(2), _numOps.Add(_x[i + 1], _x[i - 1])),
                _numOps.Multiply(mu[i - 1], h[i - 1])
            );
            mu[i] = _numOps.Divide(h[i], l[i]);
            z[i] = _numOps.Divide(
                _numOps.Subtract(alpha[i], _numOps.Multiply(z[i - 1], h[i - 1])),
                l[i]
            );
        }

        // Set the boundary conditions for natural spline
        l[n] = _numOps.One;
        z[n] = _numOps.Zero;
        _c[n] = _numOps.Zero;

        // Back substitution to find the c coefficients
        for (int j = n - 1; j >= 0; j--)
        {
            _c[j] = _numOps.Subtract(z[j], _numOps.Multiply(mu[j], _c[j + 1]));

            // Calculate b and d coefficients from c coefficients
            _b[j] = _numOps.Divide(
                _numOps.Subtract(_numOps.Subtract(_y[j + 1], _y[j]), _numOps.Multiply(h[j], _numOps.Add(_c[j + 1], _numOps.Multiply(_numOps.FromDouble(2), _c[j])))),
                h[j]
            );
            _d[j] = _numOps.Divide(_numOps.Subtract(_c[j + 1], _c[j]), h[j]);
        }
    }

    /// <summary>
    /// Finds the index of the interval in the x-array that contains the given x-value.
    /// </summary>
    /// <remarks>
    /// This method uses binary search to efficiently find which interval contains the x-value.
    /// 
    /// <b>For Beginners:</b> This method finds which segment of your data the x-value falls into.
    /// It's like finding which chapter of a book contains a specific page number.
    /// Binary search makes this process very efficient, even with large datasets.
    /// </remarks>
    /// <param name="x">The x-value to locate within the intervals.</param>
    /// <returns>The index of the interval containing the x-value.</returns>
    private int FindInterval(T x)
    {
        int low = 0;
        int high = _x.Length - 1;

        // Binary search to find the interval
        while (low < high - 1)
        {
            int mid = (low + high) / 2;
            if (_numOps.LessThanOrEquals(_x[mid], x))
                low = mid;
            else
                high = mid;
        }

        return low;
    }
}
