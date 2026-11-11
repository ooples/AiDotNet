namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Thin Plate Spline interpolation for 2D scattered data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Thin Plate Spline (TPS) is a technique for interpolating smooth surfaces through scattered data points
/// in two dimensions. It minimizes the "bending energy" of the surface, creating a result that is
/// analogous to the shape a thin metal plate would take if forced to pass through all the data points.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of Thin Plate Spline interpolation like placing a flexible sheet of metal
/// over a set of pins (your data points) at different heights. The metal bends to touch all pins
/// while maintaining the smoothest possible surface between them. This method is excellent for
/// creating smooth surfaces from scattered measurements, such as elevation data or temperature
/// readings taken at irregular locations.
/// </para>
/// </remarks>
public partial class ThinPlateSplineInterpolation<T> : I2DInterpolation<T>
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
    /// The z-values (heights) at each data point.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _z;

    /// <summary>
    /// The weights calculated for each data point used in the interpolation.
    /// </summary>
    private Vector<T> _weights;

    /// <summary>
    /// The constant term in the polynomial part of the TPS equation.
    /// </summary>
    private T _a0;

    /// <summary>
    /// The coefficient for x in the polynomial part of the TPS equation.
    /// </summary>
    private T _ax;

    /// <summary>
    /// The coefficient for y in the polynomial part of the TPS equation.
    /// </summary>
    private T _ay;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The matrix decomposition method used to solve the linear system.
    /// </summary>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// Initializes a new instance of Thin Plate Spline interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="z">The z-values (heights) at each data point.</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving the linear system.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths or when fewer than 3 points are provided.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your 3D data points (x, y, and z coordinates) and
    /// prepares the interpolation algorithm. It checks that your data is valid (same number of
    /// x, y, and z values, and at least 3 points) and then calculates the necessary weights
    /// to create a smooth surface through all your points.
    /// </para>
    /// <para>
    /// The optional decomposition parameter is for advanced users who want to specify a particular
    /// method for solving the mathematical equations. If you're just getting started, you can
    /// ignore this parameter as a default method will be used automatically.
    /// </para>
    /// </remarks>
    public ThinPlateSplineInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 3)
            throw new ArgumentException("At least 3 points are required for Thin Plate Spline interpolation.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _a0 = _numOps.Zero;
        _ax = _numOps.Zero;
        _ay = _numOps.Zero;
        _weights = Vector<T>.Empty();
        _decomposition = decomposition;

        CalculateWeights();
    }

    /// <summary>
    /// Interpolates a z-value for the given x and y coordinates using Thin Plate Spline interpolation.
    /// </summary>
    /// <param name="x">The x-coordinate for which to interpolate.</param>
    /// <param name="y">The y-coordinate for which to interpolate.</param>
    /// <returns>The interpolated z-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the height (z-value) at any location (x,y) you specify,
    /// even if it's between or outside your original data points. It uses the smooth surface
    /// created by the Thin Plate Spline algorithm to determine this height.
    /// </para>
    /// <para>
    /// The interpolated value is calculated using a combination of:
    /// 1. A simple linear function (like a tilted plane)
    /// 2. A weighted sum of special functions centered at each of your original data points
    /// </para>
    /// <para>
    /// This combination creates a surface that passes exactly through all your original points
    /// while maintaining smoothness between them.
    /// </para>
    /// </remarks>
    public T Interpolate(T x, T y)
    {
        T result = _numOps.Add(
            _numOps.Add(_a0, _numOps.Multiply(_ax, x)),
            _numOps.Multiply(_ay, y)
        );

        var engine = AiDotNetEngine.Current;
        Tensor<T> basis = DistanceBasis(
            engine.ScalarMinusTensor(x, _x.AsTensor()),
            engine.ScalarMinusTensor(y, _y.AsTensor()));

        return _numOps.Add(result, engine.TensorSum(engine.TensorMultiply(_weights.AsTensor(), basis)));
    }

    /// <summary>
    /// Calculates the weights and coefficients needed for the interpolation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves a system of mathematical equations to find the
    /// optimal weights for creating a smooth surface through all your data points.
    /// </para>
    /// <para>
    /// It works by:
    /// 1. Creating a special matrix (called L) that represents the relationships between all your data points
    /// 2. Solving a system of equations using this matrix
    /// 3. Extracting the weights and coefficients from the solution
    /// </para>
    /// <para>
    /// This is the "setup" phase of the interpolation that only needs to be done once,
    /// after which you can quickly interpolate values at any location.
    /// </para>
    /// </remarks>
    private void CalculateWeights()
    {
        var engine = AiDotNetEngine.Current;
        int n = _x.Length;
        Tensor<T> xs = _x.AsTensor();
        Tensor<T> ys = _y.AsTensor();

        // L = [[K, P], [P^T, 0]] with K[i, j] = phi(|p_i - p_j|) and P = [1, x, y]
        Tensor<T> kernel = DistanceBasis(
            engine.TensorSubtract(xs.Reshape(n, 1), xs.Reshape(1, n)),
            engine.TensorSubtract(ys.Reshape(n, 1), ys.Reshape(1, n)));
        Tensor<T> polynomial = engine.TensorConcatenate(
            new[] { Tensor<T>.CreateDefault(new[] { n, 1 }, _numOps.One), xs.Reshape(n, 1), ys.Reshape(n, 1) }, 1);
        Tensor<T> top = engine.TensorConcatenate(new[] { kernel, polynomial }, 1);
        Tensor<T> bottom = engine.TensorConcatenate(
            new[] { engine.TensorTranspose(polynomial), new Tensor<T>(new[] { 3, 3 }) }, 1);
        Matrix<T> L = engine.TensorConcatenate(new[] { top, bottom }, 0).ToMatrix();
        Vector<T> rhs = engine.TensorConcatenate(new[] { _z.AsTensor(), new Tensor<T>(new[] { 3 }) }, 0).ToVector();

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(L);
        Vector<T> solution = MatrixSolutionHelper.SolveLinearSystem(rhs, decomposition);

        // Extract weights and coefficients
        _weights = solution.GetSubVector(0, n);
        _a0 = solution[n];
        _ax = solution[n + 1];
        _ay = solution[n + 2];
    }
    /// <summary>
    /// Calculates the thin plate basis for the Euclidean distances given by coordinate differences in 2D space.
    /// </summary>
    /// <param name="dx">The x-coordinate differences.</param>
    /// <param name="dy">The y-coordinate differences.</param>
    /// <returns>The radial basis function of each distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the straight-line distance for each pair of points
    /// using the Pythagorean theorem (the square root of the sum of squared differences
    /// in x and y coordinates), then applies the radial basis function to it.
    /// </para>
    /// <para>
    /// For example, the distance between points (0,0) and (3,4) would be 5 units.
    /// </para>
    /// </remarks>
    private Tensor<T> DistanceBasis(Tensor<T> dx, Tensor<T> dy)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> r = engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy)));

        return RadialBasisFunction(r);
    }

    /// <summary>
    /// Calculates the radial basis function used in Thin Plate Spline interpolation.
    /// </summary>
    /// <param name="r">The distance values.</param>
    /// <returns>The result of the radial basis function for each distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function is a mathematical formula that determines how the
    /// influence of each data point spreads out in space. For Thin Plate Splines,
    /// this function is r^2 * log(r), where r is the distance from a point.
    /// </para>
    /// <para>
    /// Think of it as defining how the "bending" of our imaginary metal plate works
    /// at different distances from each pin (data point). This particular function
    /// creates a smooth surface that minimizes the total amount of bending.
    /// </para>
    /// <para>
    /// When the distance is zero (exactly at a data point), the function returns zero
    /// to avoid mathematical problems with the logarithm of zero.
    /// </para>
    /// </remarks>
    private Tensor<T> RadialBasisFunction(Tensor<T> r)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> value = engine.TensorMultiply(engine.TensorMultiply(r, r), engine.TensorLog(r));

        return engine.TensorWhere(engine.TensorEquals(r, _numOps.Zero), new Tensor<T>(r._shape), value);
    }
}
