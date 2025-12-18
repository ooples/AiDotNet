namespace AiDotNet.Interpolation;

/// <summary>
/// Implements cubic B-spline interpolation for smooth curve fitting through data points.
/// </summary>
/// <remarks>
/// Cubic B-spline interpolation creates a smooth curve that passes through or near all provided data points.
/// It's particularly useful for creating natural-looking curves with continuous first and second derivatives.
/// 
/// <b>For Beginners:</b> B-splines are a special type of smooth curve used in computer graphics and data analysis.
/// Unlike simpler interpolation methods, B-splines create exceptionally smooth curves that don't have
/// sudden changes in direction. Think of them as drawing a smooth line through your points using a
/// flexible ruler that naturally creates gentle curves.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public class CubicBSplineInterpolation<T> : IInterpolation<T>
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
    /// The knot vector that defines the B-spline curve.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Knots are special points that help define how the curve behaves.
    /// They're like invisible control points that determine where the curve bends.
    /// </remarks>
    private readonly Vector<T> _knots;

    /// <summary>
    /// The calculated coefficients that define the B-spline curve.
    /// </summary>
    private readonly Vector<T> _coefficients;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The degree of the B-spline curve (default is 3 for cubic).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The degree determines how smooth the curve is. A degree of 3 (cubic)
    /// creates a curve that's smooth in both direction and curvature.
    /// </remarks>
    private readonly int _degree;

    /// <summary>
    /// The type of matrix decomposition used for solving the linear system.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// Creates a new cubic B-spline interpolation from the given data points.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your data points (x and y values) and sets up
    /// everything needed to create a smooth curve through those points. Once created,
    /// you can use the Interpolate method to find y-values for any x-value along the curve.
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="degree">The degree of the B-spline (default is 3 for cubic).</param>
    /// <param name="decompositionType">The method used to solve the system of equations (default is LU decomposition).</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths or fewer than 4 points are provided.</exception>
    public CubicBSplineInterpolation(Vector<T> x, Vector<T> y, int degree = 3, MatrixDecompositionType decompositionType = MatrixDecompositionType.Lu)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("Cubic B-Spline interpolation requires at least 4 points.");

        _x = x;
        _y = y;
        _degree = degree;
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();
        _knots = GenerateKnots();
        _coefficients = CalculateCoefficients();
    }

    /// <summary>
    /// Calculates the interpolated y-value for a given x-value using the B-spline curve.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main method you'll use after creating the interpolation.
    /// Give it any x-value within your data range, and it will return the corresponding
    /// y-value on the smooth curve. It's like asking "if I have this x-value, what would
    /// the y-value be on the smooth curve that passes through my data points?"
    /// </remarks>
    /// <param name="x">The x-value for which to calculate the interpolated y-value.</param>
    /// <returns>The interpolated y-value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the x-value is outside the range of the input data.</exception>
    public T Interpolate(T x)
    {
        if (_numOps.LessThan(x, _x[0]) || _numOps.GreaterThan(x, _x[_x.Length - 1]))
            throw new ArgumentOutOfRangeException(nameof(x), "Interpolation point is outside the data range.");

        int i = FindSpan(x);
        Vector<T> basis = CalculateBasisFunctions(x, i);

        T result = _numOps.Zero;
        for (int j = 0; j < 4; j++)
        {
            result = _numOps.Add(result, _numOps.Multiply(basis[j], _coefficients[i - 3 + j]));
        }

        return result;
    }

    /// <summary>
    /// Generates the knot vector for the B-spline curve.
    /// </summary>
    /// <remarks>
    /// This method creates the knot vector that defines how the B-spline curve is constructed.
    /// It uses a "not-a-knot" condition at the endpoints for natural boundary behavior.
    /// 
    /// <b>For Beginners:</b> Knots are special points that help control the shape of the curve.
    /// This method creates extra knots beyond your data points to ensure the curve behaves
    /// naturally at the edges. Think of it as setting up the "rules" for how the curve
    /// should flow through your points.
    /// </remarks>
    /// <returns>A vector containing the knot values.</returns>
    private Vector<T> GenerateKnots()
    {
        int n = _x.Length;
        Vector<T> knots = new Vector<T>(n + 6);

        // Not-a-knot condition
        knots[0] = _numOps.Subtract(_x[0], _numOps.Multiply(_numOps.FromDouble(3), _numOps.Subtract(_x[1], _x[0])));
        knots[1] = _numOps.Subtract(_x[0], _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(_x[1], _x[0])));
        knots[2] = _numOps.Subtract(_x[0], _numOps.Subtract(_x[1], _x[0]));

        for (int i = 0; i < n; i++)
        {
            knots[i + 3] = _x[i];
        }

        // Not-a-knot condition
        knots[n + 3] = _numOps.Add(_x[n - 1], _numOps.Subtract(_x[n - 1], _x[n - 2]));
        knots[n + 4] = _numOps.Add(_x[n - 1], _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(_x[n - 1], _x[n - 2])));
        knots[n + 5] = _numOps.Add(_x[n - 1], _numOps.Multiply(_numOps.FromDouble(3), _numOps.Subtract(_x[n - 1], _x[n - 2])));

        return knots;
    }

    /// <summary>
    /// Calculates the coefficients that define the B-spline curve.
    /// </summary>
    /// <remarks>
    /// This method sets up and solves a system of equations to find the coefficients
    /// that make the B-spline curve pass through the provided data points.
    /// 
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting to create the curve.
    /// It solves a set of equations to find the values needed to make the curve pass through
    /// your data points. These coefficients are like the "recipe" for creating the curve.
    /// </remarks>
    /// <returns>A vector containing the calculated coefficients.</returns>
    private Vector<T> CalculateCoefficients()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);
        Vector<T> b = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> basis = CalculateBasisFunctions(_x[i], FindSpan(_x[i]));
            for (int j = 0; j < 4; j++)
            {
                if (i - 3 + j >= 0 && i - 3 + j < n)
                {
                    A[i, i - 3 + j] = basis[j];
                }
            }
            b[i] = _y[i];
        }

        return MatrixSolutionHelper.SolveLinearSystem(A, b, _decompositionType);
    }

    /// <summary>
    /// Finds the knot span index that contains the given x-value.
    /// </summary>
    /// <remarks>
    /// This method uses binary search to efficiently find which segment of the
    /// B-spline curve contains the given x-value.
    /// 
    /// <b>For Beginners:</b> A B-spline curve is made up of multiple segments. This method
    /// figures out which segment contains your x-value. It's like finding which
    /// chapter of a book contains a specific page number, but using an efficient
    /// search method that quickly narrows down the location.
    /// </remarks>
    /// <param name="x">The x-value to locate within the knot vector.</param>
    /// <returns>The index of the knot span containing the x-value.</returns>
    private int FindSpan(T x)
    {
        if (_numOps.GreaterThanOrEquals(x, _knots[_knots.Length - 4]))
            return _knots.Length - 5;

        int low = 3;
        int high = _knots.Length - 4;

        while (low < high)
        {
            int mid = (low + high) / 2;
            if (_numOps.LessThan(x, _knots[mid]))
                high = mid;
            else
                low = mid + 1;
        }

        return low - 1;
    }

    /// <summary>
    /// Calculates the basis functions for a B-spline at a specific point.
    /// </summary>
    /// <remarks>
    /// This method implements the De Boor algorithm to calculate the B-spline basis functions
    /// at a given point x within the knot span i.
    /// 
    /// <b>For Beginners:</b> Basis functions are like building blocks that determine the shape of the curve
    /// at each point. Think of them as "influence weights" that control how much each control point
    /// affects the final curve at a specific location. This method calculates these weights using
    /// a special algorithm called De Boor's algorithm, which is a standard approach for B-splines.
    /// The math looks complex, but it's essentially determining how much influence each nearby
    /// control point has on the curve at your chosen x-value.
    /// </remarks>
    /// <param name="x">The x-value at which to calculate the basis functions.</param>
    /// <param name="i">The knot span index containing the x-value.</param>
    /// <returns>A vector containing the four basis function values at point x.</returns>
    private Vector<T> CalculateBasisFunctions(T x, int i)
    {
        // Create vectors to store the basis functions and temporary values
        Vector<T> basis = new Vector<T>(4);
        Vector<T> left = new Vector<T>(4);
        Vector<T> right = new Vector<T>(4);

        // Initialize the first basis function to 1
        basis[0] = _numOps.One;

        // Calculate the basis functions using the Cox-de Boor recursion formula
        for (int j = 1; j <= 3; j++)
        {
            // Calculate distances from x to the knots on the left and right
            left[j] = _numOps.Subtract(x, _knots[i + 1 - j]);
            right[j] = _numOps.Subtract(_knots[i + j], x);
            T saved = _numOps.Zero;

            // Apply the recursive formula to build higher-degree basis functions
            for (int r = 0; r < j; r++)
            {
                // Calculate the denominator (knot distance)
                T temp = _numOps.Divide(basis[r], _numOps.Add(right[r + 1], left[j - r]));

                // Update the current basis function
                basis[r] = _numOps.Add(saved, _numOps.Multiply(right[r + 1], temp));

                // Save the left term for the next basis function
                saved = _numOps.Multiply(left[j - r], temp);
            }

            // Store the last basis function of this degree
            basis[j] = saved;
        }

        return basis;
    }
}
