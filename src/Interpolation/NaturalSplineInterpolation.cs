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
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates (values) of the known data points.
    /// </summary>
    private readonly Vector<T> _y;

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
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The coefficients of each segment's (x - x[i])^1..(x - x[i])^degree terms, one row per segment.
    /// </summary>
    private readonly Tensor<T> _segmentTerms;

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
        _coefficients = new Vector<T>[_degree + 1];

        for (int i = 0; i <= _degree; i++)
        {
            _coefficients[i] = new Vector<T>(x.Length - 1);
        }

        CalculateCoefficients();

        // Row i holds coefficients[1..degree][i], so evaluation reads one segment's terms with a single slice
        _segmentTerms = AiDotNetEngine.Current.TensorStack(
            Array.ConvertAll(_coefficients, coefficient => coefficient.AsTensor()).Skip(1).ToArray(), 1);
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

        // sum over j = 1..degree of coefficients[j][i] * dx^j, with dx^j built as a running product
        var engine = AiDotNetEngine.Current;
        Tensor<T> powers = engine.TensorCumProd(Tensor<T>.CreateDefault(new[] { _degree }, dx), 0);
        Tensor<T> terms = engine.TensorSlice(_segmentTerms, new[] { i, 0 }, new[] { 1, _degree }).Reshape(_degree);

        return _numOps.Add(_y[i], engine.TensorSum(engine.TensorMultiply(terms, powers)));
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
        var engine = AiDotNetEngine.Current;
        int n = _x.Length;
        Tensor<T> xs = _x.AsTensor();
        Tensor<T> ys = _y.AsTensor();
        T two = _numOps.FromDouble(2);
        T six = _numOps.FromDouble(6);

        // Interval widths h[i] = x[i+1] - x[i] and secant slopes (y[i+1] - y[i]) / h[i]
        Tensor<T> h = engine.TensorSubtract(Slice(xs, 1, n - 1), Slice(xs, 0, n - 1));
        Tensor<T> slope = engine.TensorDivide(engine.TensorSubtract(Slice(ys, 1, n - 1), Slice(ys, 0, n - 1)), h);

        // Rows 1..n-2 are the interior equations; rows 0 and n-1 are the natural boundary rows M[0] = M[n-1] = 0
        Tensor<T> interior = Tensor<T>.CreateDefault(new[] { n }, _numOps.One);
        interior[0] = _numOps.Zero;
        interior[n - 1] = _numOps.Zero;
        Tensor<T> zero = Tensor<T>.FromScalar(_numOps.Zero);
        Tensor<T> zerosN = new Tensor<T>(new[] { n });
        Tensor<T> zerosSegments = new Tensor<T>(new[] { n - 1 });

        // Interior row i: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*(slope[i] - slope[i-1])
        Tensor<T> hPrev = engine.TensorConcatenate(new[] { zero, h }, 0);
        Tensor<T> hNext = engine.TensorConcatenate(new[] { h, zero }, 0);
        Tensor<T> diagonal = engine.TensorWhere(
            interior,
            engine.TensorMultiplyScalar(engine.TensorAdd(hPrev, hNext), two),
            Tensor<T>.CreateDefault(new[] { n }, _numOps.One));
        Tensor<T> lower = engine.TensorWhere(Slice(interior, 1, n - 1), h, zerosSegments);
        Tensor<T> upper = engine.TensorWhere(Slice(interior, 0, n - 1), h, zerosSegments);
        Matrix<T> A = engine.TensorAdd(
            engine.TensorAdd(engine.TensorDiagEmbed(lower, -1), engine.TensorDiagEmbed(diagonal, 0)),
            engine.TensorDiagEmbed(upper, 1)).ToMatrix();

        Tensor<T> slopePrev = engine.TensorConcatenate(new[] { zero, slope }, 0);
        Tensor<T> slopeNext = engine.TensorConcatenate(new[] { slope, zero }, 0);
        Vector<T> b = engine.TensorWhere(
            interior,
            engine.TensorMultiplyScalar(engine.TensorSubtract(slopeNext, slopePrev), six),
            zerosN).ToVector();

        // Solve the system for second derivatives M
        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        Tensor<T> M = MatrixSolutionHelper.SolveLinearSystem(b, decomposition).AsTensor();
        Tensor<T> mLeft = Slice(M, 0, n - 1);
        Tensor<T> mRight = Slice(M, 1, n - 1);

        // S(x) = a + b*(x-x[i]) + c*(x-x[i])^2 + d*(x-x[i])^3 on each segment, with
        // a = y[i], b = slope[i] - h[i]*(2*M[i] + M[i+1])/6, c = M[i]/2, d = (M[i+1] - M[i])/(6*h[i])
        _coefficients[0] = Slice(ys, 0, n - 1).ToVector();
        _coefficients[1] = engine.TensorSubtract(
            slope,
            engine.TensorDivideScalar(
                engine.TensorMultiply(h, engine.TensorAdd(engine.TensorMultiplyScalar(mLeft, two), mRight)),
                six)).ToVector();
        _coefficients[2] = engine.TensorDivideScalar(mLeft, two).ToVector();
        _coefficients[3] = engine.TensorDivide(
            engine.TensorSubtract(mRight, mLeft),
            engine.TensorMultiplyScalar(h, six)).ToVector();
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
        return InterpolationSearch.FirstSegmentReaching(_x, x, 0, _x.Length - 2);
    }

    /// <summary>
    /// Returns <paramref name="length"/> consecutive elements of a vector tensor starting at <paramref name="start"/>.
    /// </summary>
    private static Tensor<T> Slice(Tensor<T> values, int start, int length)
    {
        return AiDotNetEngine.Current.TensorSlice(values, new[] { start }, new[] { length });
    }
}
