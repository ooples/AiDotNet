namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Multiquadric Radial Basis Function interpolation for two-dimensional data points.
/// </summary>
/// <remarks>
/// Multiquadric interpolation is a powerful technique for creating smooth surfaces from scattered data points.
/// It uses radial basis functions centered at each data point to construct an interpolating surface.
/// 
/// <b>For Beginners:</b> Imagine you have several points with known heights (like mountains on a map),
/// and you want to estimate the height at any location between these points. Multiquadric interpolation
/// creates a smooth surface that passes exactly through all your known points while providing reasonable
/// estimates for all the areas in between. It's particularly good at handling irregularly spaced data points
/// and can create very smooth surfaces.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultiquadricInterpolation<T> : I2DInterpolation<T>
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
    private readonly Vector<T> _z;

    /// <summary>
    /// The shape parameter that controls the smoothness of the interpolation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The epsilon parameter controls how "bumpy" or smooth your interpolated surface will be.
    /// A smaller epsilon creates steeper "hills" around your data points, while a larger epsilon creates
    /// a smoother, more gently varying surface. Finding the right value often requires some experimentation
    /// for your specific data.
    /// </remarks>
    private readonly T _epsilon;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Optional matrix decomposition method for solving the interpolation system.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an advanced option that determines how the mathematical equations
    /// are solved internally. Most users can leave this as the default (null).
    /// </remarks>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The calculated coefficients for the radial basis functions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the weights that determine how much each data point contributes
    /// to the final interpolated surface. They're calculated automatically when you create the interpolator.
    /// </remarks>
    private Vector<T>? _coefficients;

    /// <summary>
    /// Creates a new instance of the Multiquadric interpolation algorithm.
    /// </summary>
    /// <remarks>
    /// This constructor initializes the interpolator with your data points and configuration options,
    /// and automatically calculates the necessary coefficients.
    /// 
    /// <b>For Beginners:</b> When you create a Multiquadric interpolator, you provide:
    /// 1. Your known data points (x, y coordinates and z values)
    /// 2. An epsilon value that controls the smoothness of the interpolation
    /// 
    /// The interpolator will then be ready to estimate z-values at any (x,y) location.
    /// </remarks>
    /// <param name="x">The x-coordinates of the known data points.</param>
    /// <param name="y">The y-coordinates of the known data points.</param>
    /// <param name="z">The z-values (heights) of the known data points.</param>
    /// <param name="epsilon">The shape parameter that controls the smoothness. Default is 1.0.</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving the system.</param>
    /// <exception cref="ArgumentException">Thrown when input vectors have different lengths.</exception>
    public MultiquadricInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, double epsilon = 1.0, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(epsilon);
        _decomposition = decomposition;
        CalculateCoefficients();
    }

    /// <summary>
    /// Calculates the coefficients needed for the multiquadric interpolation.
    /// </summary>
    /// <remarks>
    /// This method sets up and solves the linear system to find the weights for each radial basis function.
    /// 
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting to prepare the interpolator.
    /// It figures out how much influence each of your known data points should have when estimating
    /// values at new locations. This happens automatically when you create the interpolator, so you
    /// don't need to call this method directly.
    /// </remarks>
    private void CalculateCoefficients()
    {
        int n = _x.Length;
        var A = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T dx = _numOps.Subtract(_x[i], _x[j]);
                T dy = _numOps.Subtract(_y[i], _y[j]);
                T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
                A[i, j] = MultiquadricBasis(r);
            }
        }

        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        _coefficients = MatrixSolutionHelper.SolveLinearSystem(_z, decomposition);
    }

    /// <summary>
    /// Interpolates the z-value at a given (x,y) coordinate using Multiquadric interpolation.
    /// </summary>
    /// <remarks>
    /// This method calculates the z-value at any (x,y) coordinate by combining the radial basis functions
    /// with their respective coefficients.
    /// 
    /// <b>For Beginners:</b> Once you've set up the interpolator with your known points, this method
    /// lets you ask "What would the z-value (height) be at this specific (x,y) location?"
    /// 
    /// The algorithm works by:
    /// 1. Calculating how far the target point is from each known data point
    /// 2. Applying the multiquadric function to each of these distances
    /// 3. Multiplying each result by its corresponding coefficient
    /// 4. Adding all these values together to get the final interpolated value
    /// 
    /// This creates a smooth surface that passes exactly through all your known data points.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <param name="y">The y-coordinate at which to interpolate.</param>
    /// <returns>The interpolated z-value at the specified (x,y) coordinate.</returns>
    /// <exception cref="InvalidOperationException">Thrown if coefficients have not been calculated.</exception>
    public T Interpolate(T x, T y)
    {
        if (_coefficients == null)
            throw new InvalidOperationException("Coefficients have not been calculated.");

        T result = _numOps.Zero;
        for (int i = 0; i < _x.Length; i++)
        {
            T dx = _numOps.Subtract(x, _x[i]);
            T dy = _numOps.Subtract(y, _y[i]);
            T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[i], MultiquadricBasis(r)));
        }

        return result;
    }

    /// <summary>
    /// Calculates the multiquadric radial basis function for a given distance.
    /// </summary>
    /// <remarks>
    /// This method implements the standard multiquadric function: sqrt(r² + e²).
    /// 
    /// <b>For Beginners:</b> This is the mathematical function that determines how the influence of a data point
    /// decreases with distance. The multiquadric function creates a smooth "hill" shape around each data point.
    /// The epsilon parameter controls how wide and flat these hills are.
    /// </remarks>
    /// <param name="r">The distance from the target point to a data point.</param>
    /// <returns>The value of the multiquadric function at the given distance.</returns>
    private T MultiquadricBasis(T r)
    {
        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(r, r), _numOps.Multiply(_epsilon, _epsilon)));
    }
}
