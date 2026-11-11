using AiDotNet.DecompositionMethods.MatrixDecomposition;

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
public partial class CubicBSplineInterpolation<T> : IInterpolation<T>
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
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _knots;

    /// <summary>
    /// The calculated coefficients that define the B-spline curve.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
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

        var engine = AiDotNetEngine.Current;
        int i = FindSpan(x);
        Tensor<T>[] basis = CalculateBasisFunctions(Tensor<T>.FromScalar(x), new Tensor<int>(new[] { i }, new[] { 1 }));

        // Basis j weights coefficient i - 3 + j. FindSpan clamps at the right-hand end, which can name a coefficient
        // one past the last; its basis value there is zero, so only the in-range coefficients are summed.
        int first = Math.Max(0, i - 3);
        int last = Math.Min(_coefficients.Length - 1, i);
        if (last < first)
        {
            return _numOps.Zero;
        }

        int count = last - first + 1;
        Tensor<T> weights = engine.TensorSlice(engine.TensorConcatenate(basis, 0), new[] { first - (i - 3) }, new[] { count });
        Tensor<T> coefficients = engine.TensorSlice(_coefficients.AsTensor(), new[] { first }, new[] { count });
        return engine.TensorSum(engine.TensorMultiply(weights, coefficients));
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
        var engine = AiDotNetEngine.Current;
        int n = _x.Length;

        // Not-a-knot condition: three knots before the data, spaced like the first interval
        T firstStep = _numOps.Subtract(_x[1], _x[0]);
        T k0 = _numOps.Subtract(_x[0], _numOps.Multiply(_numOps.FromDouble(3), firstStep));
        T k1 = _numOps.Subtract(_x[0], _numOps.Multiply(_numOps.FromDouble(2), firstStep));
        T k2 = _numOps.Subtract(_x[0], firstStep);

        // Not-a-knot condition: three knots after the data, spaced like the last interval
        T lastStep = _numOps.Subtract(_x[n - 1], _x[n - 2]);
        T kN3 = _numOps.Add(_x[n - 1], lastStep);
        T kN4 = _numOps.Add(_x[n - 1], _numOps.Multiply(_numOps.FromDouble(2), lastStep));
        T kN5 = _numOps.Add(_x[n - 1], _numOps.Multiply(_numOps.FromDouble(3), lastStep));

        Tensor<T> knots = engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { k0, k1, k2 }, new[] { 3 }),
            _x.AsTensor(),
            new Tensor<T>(new[] { kN3, kN4, kN5 }, new[] { 3 })
        }, 0);

        return knots.ToVector();
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

        // A cubic B-spline over this knot vector has (knots - degree - 1) = n + 2 basis functions,
        // not n. The system was previously built square, which made it singular: LU reported a
        // zero pivot rather than an answer.
        int coefficientCount = _knots.Length - 4;

        var engine = AiDotNetEngine.Current;

        // The columns are indexed from the SPAN, not from the data index. Interpolate reads
        // _coefficients[span - 3 + j], and the two have to agree; with this knot vector the
        // span of x[i] is i + 3, so writing to i - 3 + j put every basis value three columns
        // to the left of where it is later read.
        Tensor<T> points = _x.AsTensor();
        Tensor<int> spans = FindSpans(points);
        Tensor<T>[] basis = CalculateBasisFunctions(points, spans);

        // A[i, span[i] - 3 + j] = basis[j][i]. One-hot rows place each basis value in its column; a column outside
        // [0, coefficientCount) has an all-zero one-hot row, so that value is left out as before.
        Tensor<T> system = new Tensor<T>(new[] { n, coefficientCount });
        for (int j = 0; j < 4; j++)
        {
            Tensor<T> placement = engine.TensorOneHot<T>(engine.TensorAddScalar(spans, j - 3), coefficientCount);
            system = engine.TensorAdd(system, engine.TensorMultiply(placement, basis[j].Reshape(n, 1)));
        }

        Matrix<T> A = system.ToMatrix();
        Vector<T> b = _y.Clone();

        // n equations in n + 2 unknowns: every spline through the data is a solution, and the
        // pseudoinverse returns the one of smallest coefficient norm. That is a deliberate choice
        // of end condition - the flattest interpolant available - and it is always defined, which
        // a square solve of an underdetermined system is not.
        return new SvdDecomposition<T>(A).Solve(b);
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
        // The span is the last knot at or below x among knots 3 .. Length - 5; the clamp also sends
        // x >= knots[Length - 4] to the last span, Length - 5.
        return InterpolationSearch.LastKnotAtOrBelow(_knots, x, 2, _knots.Length - 5);
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
    /// <param name="points">The x-values at which to calculate the basis functions.</param>
    /// <param name="spans">The knot span index containing each x-value.</param>
    /// <returns>The four basis function values; entry j holds basis j at every point.</returns>
    private Tensor<T>[] CalculateBasisFunctions(Tensor<T> points, Tensor<int> spans)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> knots = _knots.AsTensor();

        // Hold the basis functions and temporary values, one tensor across all points per entry
        var basis = new Tensor<T>[4];
        var left = new Tensor<T>[4];
        var right = new Tensor<T>[4];
        Tensor<T> zeros = new Tensor<T>(new[] { points.Length });

        // Initialize the first basis function to 1
        basis[0] = new Tensor<T>(new[] { points.Length });
        engine.TensorFill(basis[0], _numOps.One);

        // Calculate the basis functions using the Cox-de Boor recursion formula
        for (int j = 1; j <= 3; j++)
        {
            // Calculate distances from x to the knots on the left and right
            left[j] = engine.TensorSubtract(points, engine.TensorGather(knots, engine.TensorAddScalar(spans, 1 - j), 0));
            right[j] = engine.TensorSubtract(engine.TensorGather(knots, engine.TensorAddScalar(spans, j), 0), points);
            Tensor<T> saved = zeros;

            // Apply the recursive formula to build higher-degree basis functions
            for (int r = 0; r < j; r++)
            {
                // Calculate the denominator (knot distance)
                Tensor<T> temp = engine.TensorDivide(basis[r], engine.TensorAdd(right[r + 1], left[j - r]));

                // Update the current basis function
                basis[r] = engine.TensorAdd(saved, engine.TensorMultiply(right[r + 1], temp));

                // Save the left term for the next basis function
                saved = engine.TensorMultiply(left[j - r], temp);
            }

            // Store the last basis function of this degree
            basis[j] = saved;
        }

        return basis;
    }

    /// <summary>
    /// Finds the knot span index of every x-value at once, matching <see cref="FindSpan"/> for each.
    /// </summary>
    /// <param name="points">The x-values to locate within the knot vector.</param>
    /// <returns>The knot span index of each x-value.</returns>
    private Tensor<int> FindSpans(Tensor<T> points)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<int> insertion = engine.TensorSearchSorted(_knots.AsTensor(), points, right: true);
        return engine.TensorSubtractScalar(engine.TensorClamp(insertion, 3, _knots.Length - 4), 1);
    }
}
