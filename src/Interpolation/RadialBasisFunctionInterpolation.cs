global using AiDotNet.RadialBasisFunctions;

namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Radial Basis Function (RBF) interpolation for 2D data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Radial Basis Function interpolation is a powerful technique for creating smooth surfaces
/// that pass through scattered data points in 2D or higher dimensions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of RBF interpolation as creating a rubber sheet that's stretched to
/// pass through all your data points. The sheet's shape between and beyond your known points
/// is determined by special mathematical functions (called radial basis functions) that create
/// smooth transitions. This is particularly useful when your data points aren't arranged in a grid.
/// </para>
/// </remarks>
public class RadialBasisFunctionInterpolation<T> : I2DInterpolation<T>
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
    private readonly Vector<T> _z;

    /// <summary>
    /// The calculated weights for each radial basis function.
    /// </summary>
    private Vector<T> _weights;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The matrix decomposition method used to solve the linear system.
    /// </summary>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The radial basis function used for interpolation.
    /// </summary>
    private readonly IRadialBasisFunction<T> _rbf;

    /// <summary>
    /// Initializes a new instance of the RBF interpolation with the specified data points.
    /// </summary>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="z">The z-values (heights) at each data point.</param>
    /// <param name="rbf">The radial basis function to use. If null, a Gaussian RBF is used by default.</param>
    /// <param name="decomposition">The matrix decomposition method to use. If null, LU decomposition is used by default.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths or when there are fewer than 3 points.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your 3D data points (x, y, and z values) and prepares
    /// the interpolation algorithm. It checks that your data is valid (same number of x, y, and z values,
    /// and at least 3 points) and then calculates the weights needed for the interpolation.
    /// </para>
    /// <para>
    /// The optional parameters let advanced users customize how the interpolation works:
    /// - rbf: Controls the shape of the "bumps" used to create the surface (Gaussian is smooth and bell-shaped)
    /// - decomposition: Controls how the math equations are solved (most users can leave this as default)
    /// </para>
    /// </remarks>
    public RadialBasisFunctionInterpolation(Vector<T> x, Vector<T> y, Vector<T> z,
        IRadialBasisFunction<T>? rbf = null, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 3)
            throw new ArgumentException("At least 3 points are required for RBF interpolation.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = Vector<T>.Empty();
        _decomposition = decomposition;
        _rbf = rbf ?? new GaussianRBF<T>();

        CalculateWeights();
    }

    /// <summary>
    /// Interpolates a z-value for the given x and y coordinates using RBF interpolation.
    /// </summary>
    /// <param name="x">The x-coordinate for which to interpolate.</param>
    /// <param name="y">The y-coordinate for which to interpolate.</param>
    /// <returns>The interpolated z-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the height (z-value) on the interpolated surface for any
    /// x and y coordinates you provide, even if they're between or outside your original data points.
    /// </para>
    /// <para>
    /// It works by:
    /// 1. Calculating the distance from your query point (x,y) to each of the original data points
    /// 2. Applying the radial basis function to each distance
    /// 3. Multiplying each result by its corresponding weight
    /// 4. Summing all these values to get the final interpolated height
    /// </para>
    /// <para>
    /// This creates a smooth surface that passes exactly through all your original data points.
    /// </para>
    /// </remarks>
    public T Interpolate(T x, T y)
    {
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T r = CalculateDistance(x, y, _x[i], _y[i]);
            result = _numOps.Add(result, _numOps.Multiply(_weights[i], _rbf.Compute(r)));
        }

        return result;
    }

    /// <summary>
    /// Calculates the weights for each radial basis function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves a system of equations to find the right "strength" (weight)
    /// for each data point's influence on the final surface.
    /// </para>
    /// <para>
    /// Imagine each data point creates a bump on the surface, and we need to figure out how tall
    /// each bump should be so that when all bumps are combined, the surface passes exactly through
    /// all the original data points. These heights are the "weights" we're calculating.
    /// </para>
    /// <para>
    /// Technically, this method:
    /// 1. Creates a matrix of radial basis function values between all pairs of data points
    /// 2. Solves a linear system to find weights that make the interpolated surface pass through all data points
    /// </para>
    /// </remarks>
    private void CalculateWeights()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T r = CalculateDistance(_x[i], _y[i], _x[j], _y[j]);
                A[i, j] = _rbf.Compute(r);
            }
        }

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        _weights = MatrixSolutionHelper.SolveLinearSystem(_z, decomposition);
    }

    /// <summary>
    /// Calculates the Euclidean distance between two points in 2D space.
    /// </summary>
    /// <param name="x1">The x-coordinate of the first point.</param>
    /// <param name="y1">The y-coordinate of the first point.</param>
    /// <param name="x2">The x-coordinate of the second point.</param>
    /// <param name="y2">The y-coordinate of the second point.</param>
    /// <returns>The distance between the two points.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the straight-line distance between two points
    /// using the Pythagorean theorem (a² + b² = c²).
    /// </para>
    /// <para>
    /// The distance is used by the radial basis functions, which create "bumps" that depend
    /// on how far a point is from the center of the bump.
    /// </para>
    /// </remarks>
    private T CalculateDistance(T x1, T y1, T x2, T y2)
    {
        T dx = _numOps.Subtract(x1, x2);
        T dy = _numOps.Subtract(y1, y2);

        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
    }
}
