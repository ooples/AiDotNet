namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Moving Least Squares interpolation for two-dimensional data points.
/// </summary>
/// <remarks>
/// Moving Least Squares (MLS) is a method for smoothly interpolating scattered data points
/// in two dimensions. It creates a continuous surface that passes through or near the original data points.
/// 
/// <b>For Beginners:</b> Imagine you have several points with known heights (like mountains on a map),
/// and you want to estimate the height at any location between these points. Moving Least Squares
/// creates a smooth surface that respects your known points while providing reasonable estimates
/// for all the areas in between. It's like creating a smooth terrain from a set of elevation measurements.
/// 
/// Unlike simpler methods, MLS adapts to the local density and arrangement of your data points,
/// giving more weight to nearby points when estimating a value at a specific location.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public partial class MovingLeastSquaresInterpolation<T> : I2DInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the known data points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The z-values (heights) of the known data points.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _z;

    /// <summary>
    /// Controls how far the influence of each data point extends.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The smoothing length determines how far each known point influences the
    /// interpolated surface. A larger value creates a smoother surface but might not follow your
    /// data points as closely. A smaller value creates a surface that more closely follows your
    /// data points but might be less smooth overall.
    /// </remarks>
    private readonly T _smoothingLength;

    /// <summary>
    /// The degree of the polynomial used for local approximation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines the complexity of the mathematical function used to
    /// approximate your data. A higher degree (like 3 or 4) can capture more complex patterns
    /// but might overfit to noise in your data. A lower degree (like 1 or 2) creates simpler,
    /// smoother approximations but might miss some important patterns in your data.
    /// 
    /// Degree 1 = linear (flat planes)
    /// Degree 2 = quadratic (can form simple curves and hills)
    /// Degree 3 = cubic (can form more complex shapes)
    /// </remarks>
    private readonly int _polynomialDegree;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Optional matrix decomposition method for solving the least squares system.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an advanced option that determines how the mathematical equations
    /// are solved internally. Most users can leave this as the default (null).
    /// </remarks>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The power of x in each polynomial basis term, in the order the terms are solved for.
    /// </summary>
    /// <remarks>
    /// Term k is x^_xExponents[k] * y^_yExponents[k]; the terms run over total degree p = 0.._polynomialDegree and,
    /// within each p, over q = 0..p as x^(p - q) * y^q.
    /// </remarks>
    private readonly Tensor<T> _xExponents;

    /// <summary>
    /// The power of y in each polynomial basis term, aligned with <see cref="_xExponents"/>.
    /// </summary>
    private readonly Tensor<T> _yExponents;

    /// <summary>
    /// Creates a new instance of the Moving Least Squares interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the interpolator with your data points and configuration options.
    /// 
    /// <b>For Beginners:</b> When you create a Moving Least Squares interpolator, you provide:
    /// 1. Your known data points (x, y coordinates and z values)
    /// 2. A smoothing length that controls how far each point's influence extends
    /// 3. A polynomial degree that determines how complex the local approximations can be
    /// 
    /// The interpolator will then be ready to estimate z-values at any (x,y) location.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates of the known data points.</param>
    /// <param name="z">The z-values (heights) of the known data points.</param>
    /// <param name="smoothingLength">Controls how far the influence of each data point extends. Default is 1.0.</param>
    /// <param name="polynomialDegree">The degree of the polynomial used for local approximation. Default is 2.</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving the least squares system.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths.</exception>
    public MovingLeastSquaresInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, double smoothingLength = 1.0, int polynomialDegree = 2, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _smoothingLength = _numOps.FromDouble(smoothingLength);
        _polynomialDegree = polynomialDegree;
        _decomposition = decomposition;

        int basisSize = (_polynomialDegree + 1) * (_polynomialDegree + 2) / 2;
        var xExponents = new int[basisSize];
        var yExponents = new int[basisSize];
        int index = 0;
        for (int p = 0; p <= _polynomialDegree; p++)
        {
            for (int q = 0; q <= p; q++)
            {
                xExponents[index] = p - q;
                yExponents[index] = q;
                index++;
            }
        }

        _xExponents = new Tensor<int>(xExponents, new[] { basisSize }).Cast<T>();
        _yExponents = new Tensor<int>(yExponents, new[] { basisSize }).Cast<T>();
    }

    /// <summary>
    /// Interpolates the z-value at a given (x,y) coordinate using Moving Least Squares.
    /// </summary>
    /// <remarks>
    /// This method calculates the z-value at any (x,y) coordinate by fitting a weighted
    /// polynomial to the nearby data points.
    /// 
    /// <b>For Beginners:</b> Once you've set up the interpolator with your known points, this method
    /// lets you ask "What would the z-value (height) be at this specific (x,y) location?"
    /// 
    /// The algorithm works by:
    /// 1. Calculating how far the target point is from each known data point
    /// 2. Assigning weights to each known point based on these distances (closer points get higher weights)
    /// 3. Fitting a polynomial surface to the known points, with each point's influence determined by its weight
    /// 4. Evaluating this polynomial at the target location to get the estimated z-value
    /// 
    /// This creates a smooth surface that adapts to the local arrangement of your data points.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <param name="y">The y-coordinate at which to interpolate.</param>
    /// <returns>The interpolated z-value at the specified (x,y) coordinate.</returns>
    public T Interpolate(T x, T y)
    {
        var engine = AiDotNetEngine.Current;
        int n = _x.Length;
        int basisSize = _xExponents.Length;
        Tensor<T> xs = _x.AsTensor();
        Tensor<T> ys = _y.AsTensor();

        Tensor<T> dx = engine.ScalarMinusTensor(x, xs);
        Tensor<T> dy = engine.ScalarMinusTensor(y, ys);
        Tensor<T> distance = engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy)));
        Tensor<T> weight = CalculateWeight(distance);

        var W = engine.TensorDiag(weight).ToMatrix();
        var b = engine.TensorMultiply(_z.AsTensor(), weight).ToVector();

        // A[i, k] = weight[i] * (x[i]^xExponent[k] * y[i]^yExponent[k])
        int[] termShape = { n, basisSize };
        Tensor<T> xPowers = engine.TensorPower(
            engine.TensorBroadcastTo(xs.Reshape(n, 1), termShape),
            engine.TensorBroadcastTo(_xExponents.Reshape(1, basisSize), termShape));
        Tensor<T> yPowers = engine.TensorPower(
            engine.TensorBroadcastTo(ys.Reshape(n, 1), termShape),
            engine.TensorBroadcastTo(_yExponents.Reshape(1, basisSize), termShape));
        var A = engine.TensorMultiply(weight.Reshape(n, 1), engine.TensorMultiply(xPowers, yPowers)).ToMatrix();

        var AtW = A.Transpose().Multiply(W);
        var AtWA = AtW.Multiply(A);
        var AtWb = AtW.Multiply(b);

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(AtWA);
        var coefficients = MatrixSolutionHelper.SolveLinearSystem(AtWb, decomposition);

        Tensor<T> terms = engine.TensorMultiply(
            engine.TensorPower(engine.Fill(basisSize, x).AsTensor(), _xExponents),
            engine.TensorPower(engine.Fill(basisSize, y).AsTensor(), _yExponents));

        return engine.TensorSum(engine.TensorMultiply(coefficients.AsTensor(), terms));
    }

    /// <summary>
    /// Calculates the weight for a data point based on its distance from the target point.
    /// </summary>
    /// <remarks>
    /// This method implements a quadratic weight function that decreases as distance increases,
    /// reaching zero at the smoothing length.
    /// 
    /// <b>For Beginners:</b> This function determines how much influence each known point has on the
    /// interpolated value, based on how far away it is from the target location. Points that
    /// are closer have more influence (higher weight), while points that are farther away have
    /// less influence (lower weight). Points beyond the smoothing length have no influence at all.
    /// 
    /// The specific weight function used here is w = (1 - (d/h)²) for d &lt; h, and w = 0 for d = h,
    /// where d is the distance and h is the smoothing length.
    /// </remarks>
    /// <param name="distance">The distance from the target point to each data point.</param>
    /// <returns>The weight assigned to each data point.</returns>
    private Tensor<T> CalculateWeight(Tensor<T> distance)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> ratio = engine.TensorDivideScalar(distance, _smoothingLength);
        Tensor<T> weight = engine.ScalarMinusTensor(_numOps.One, engine.TensorMultiply(ratio, ratio));

        // Points at or beyond the smoothing length have no influence
        Tensor<T> beyond = engine.TensorAdd(
            engine.TensorGreaterThan(distance, _smoothingLength),
            engine.TensorEquals(distance, _smoothingLength));
        return engine.TensorMaskedFill(weight, engine.TensorEqScalar(beyond, _numOps.One), _numOps.Zero);
    }
}
