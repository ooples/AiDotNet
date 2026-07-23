namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Bicubic Interpolation for estimating values between points in a 2D grid.
/// </summary>
/// <remarks>
/// Bicubic interpolation creates a smooth surface that passes through a grid of known data points.
/// It's commonly used for image resizing, terrain modeling, and scientific data visualization.
/// 
/// <b>For Beginners:</b> Think of this as a sophisticated way to "fill in the blanks" between points in a grid.
/// Imagine you have a grid of height measurements for a landscape (like a topographic map) but you want
/// to know the height at points between your measurements. Bicubic interpolation creates a smooth surface
/// that passes through all your known points and gives reasonable estimates for the in-between areas.
/// 
/// Unlike simpler methods, bicubic interpolation considers not just the nearest points but also how the
/// surface is changing (its "slope" and "curvature"), resulting in smoother, more natural-looking results.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class BicubicInterpolation<T> : I2DInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the grid points.
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the grid points.
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// The z-values (heights) at each grid point, organized as a matrix.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is your actual data - the known values at each grid point.
    /// If you're thinking of a landscape, these would be the heights at each measured location.
    /// </remarks>
    private readonly Matrix<T> _z;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Optional matrix decomposition method used for solving the linear system.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an advanced mathematical tool that helps solve the equations
    /// needed for the interpolation. You typically don't need to worry about this parameter.
    /// </remarks>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// Initializes a new instance of the BicubicInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor validates the input data and initializes the necessary components
    /// for performing bicubic interpolation.
    /// 
    /// <b>For Beginners:</b> This sets up everything needed to perform the interpolation:
    /// 1. It checks that your data grid is valid (matching dimensions and enough points)
    /// 2. It stores your grid coordinates and values
    /// 3. It prepares the mathematical tools needed for calculations
    /// </remarks>
    /// <param name="x">The x-coordinates of the grid points.</param>
    /// <param name="y">The y-coordinates of the grid points.</param>
    /// <param name="z">The z-values (heights) at each grid point.</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving the linear system.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input dimensions don't match or when there are fewer than 4x4 grid points.
    /// </exception>
    public BicubicInterpolation(Vector<T> x, Vector<T> y, Matrix<T> z, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != z.Rows || y.Length != z.Columns)
            throw new ArgumentException("Input dimensions mismatch.");
        if (x.Length < 4 || y.Length < 4)
            throw new ArgumentException("Bicubic interpolation requires at least a 4x4 grid.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _decomposition = decomposition;
    }

    /// <summary>
    /// Interpolates a value at the specified (x,y) coordinates.
    /// </summary>
    /// <remarks>
    /// This method finds the grid cell containing the target point and uses bicubic
    /// interpolation to estimate the z-value at that point.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an (x,y) location, and it will:
    /// 1. Find which grid cell contains your point
    /// 2. Gather the necessary data from that cell and its neighbors
    /// 3. Calculate a smooth estimate of the value at your exact location
    /// 
    /// For example, if you have height measurements at grid points spaced 10 meters apart
    /// and you want to know the height at a point 3.5 meters from one grid point, this method
    /// will give you that estimate.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <param name="y">The y-coordinate at which to interpolate.</param>
    /// <returns>The interpolated z-value at the specified (x,y) coordinates.</returns>
    public T Interpolate(T x, T y)
    {
        // Check if x and y exactly match grid points - return exact value
        // Use binary search for O(log n) instead of O(n×m)
        int exactXIndex = BinarySearchExact(_x, x);
        if (exactXIndex >= 0)
        {
            int exactYIndex = BinarySearchExact(_y, y);
            if (exactYIndex >= 0)
            {
                return _z[exactXIndex, exactYIndex];
            }
        }

        int iOriginal = FindInterval(_x, x);
        int jOriginal = FindInterval(_y, y);

        // Use original intervals for dx/dy normalization (maintains correct interpolation position)
        T dx = _numOps.Divide(_numOps.Subtract(x, _x[iOriginal]), _numOps.Subtract(_x[iOriginal + 1], _x[iOriginal]));
        T dy = _numOps.Divide(_numOps.Subtract(y, _y[jOriginal]), _numOps.Subtract(_y[jOriginal + 1], _y[jOriginal]));

        // Clamp indices for 4×4 neighborhood extraction (ensures valid array access)
        int i = Math.Max(1, Math.Min(iOriginal, _x.Length - 3));
        int j = Math.Max(1, Math.Min(jOriginal, _y.Length - 3));

        // Adjust dx/dy to account for neighborhood shift when clamping occurred
        if (i != iOriginal)
        {
            // Recalculate dx relative to the clamped cell
            dx = _numOps.Divide(_numOps.Subtract(x, _x[i]), _numOps.Subtract(_x[i + 1], _x[i]));
        }
        if (j != jOriginal)
        {
            // Recalculate dy relative to the clamped cell
            dy = _numOps.Divide(_numOps.Subtract(y, _y[j]), _numOps.Subtract(_y[j + 1], _y[j]));
        }

        T[,] p = new T[4, 4];
        for (int m = -1; m <= 2; m++)
        {
            for (int n = -1; n <= 2; n++)
            {
                p[m + 1, n + 1] = _z[MathHelper.Clamp(i + m, 0, _z.Rows - 1), MathHelper.Clamp(j + n, 0, _z.Columns - 1)];
            }
        }

        return InterpolateBicubicPatch(p, dx, dy);
    }

    /// <summary>
    /// Performs the actual bicubic interpolation on a 4x4 patch of points.
    /// </summary>
    /// <remarks>
    /// This method calculates the bicubic coefficients and uses them to compute
    /// the interpolated value at the specified normalized coordinates.
    /// 
    /// <b>For Beginners:</b> This method does the actual mathematical interpolation once we've
    /// identified which grid cell contains our point of interest. It uses a mathematical
    /// formula that considers 16 surrounding points (a 4x4 grid) to create a smooth surface
    /// and then calculates the height at the exact position we want.
    /// </remarks>
    /// <param name="p">A 4x4 array of z-values around the target point.</param>
    /// <param name="x">The normalized x-coordinate within the grid cell (0 to 1).</param>
    /// <param name="y">The normalized y-coordinate within the grid cell (0 to 1).</param>
    /// <returns>The interpolated value at the specified normalized coordinates.</returns>
    private T InterpolateBicubicPatch(T[,] p, T x, T y)
    {
        T[,] a = CalculateBicubicCoefficients(p);
        T result = _numOps.Zero;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result = _numOps.Add(result, _numOps.Multiply(a[i, j],
                    _numOps.Multiply(_numOps.Power(x, _numOps.FromDouble(i)), _numOps.Power(y, _numOps.FromDouble(j)))));
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the bicubic coefficients from a 4x4 patch of points.
    /// </summary>
    /// <remarks>
    /// This method solves a 16x16 linear system to find the 16 coefficients
    /// needed for bicubic interpolation.
    /// 
    /// <b>For Beginners:</b> This is the mathematical heart of bicubic interpolation.
    /// It calculates 16 special values (coefficients) that define the shape of the
    /// smooth surface passing through our grid points. These coefficients capture
    /// not just the heights at each point but also how the surface curves between points.
    /// 
    /// This involves solving a system of 16 equations with 16 unknowns - a complex
    /// mathematical operation that's handled automatically by the library.
    /// </remarks>
    /// <param name="p">A 4x4 array of z-values around the target point.</param>
    /// <returns>A 4x4 array of bicubic coefficients.</returns>
    private T[,] CalculateBicubicCoefficients(T[,] p)
    {
        T[,] a = new T[4, 4];
        Matrix<T> coefficients = new Matrix<T>(16, 16);
        Vector<T> values = new Vector<T>(16);

        // Fill the coefficients matrix and values vector
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int index = i * 4 + j;
                values[index] = p[i, j];

                for (int k = 0; k < 4; k++)
                {
                    for (int l = 0; l < 4; l++)
                    {
                        T xTerm = _numOps.Power(_numOps.FromDouble(i - 1), _numOps.FromDouble(k));
                        T yTerm = _numOps.Power(_numOps.FromDouble(j - 1), _numOps.FromDouble(l));
                        coefficients[index, k * 4 + l] = _numOps.Multiply(xTerm, yTerm);
                    }
                }
            }
        }

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(coefficients);
        Vector<T> solution = MatrixSolutionHelper.SolveLinearSystem(values, decomposition);

        // Convert the solution vector to the 4x4 coefficient matrix
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                a[i, j] = solution[i * 4 + j];
            }
        }

        return a;
    }

    /// <summary>
    /// Finds the index of the interval containing the specified point.
    /// </summary>
    /// <remarks>
    /// This method locates which grid cell contains the target coordinate.
    /// 
    /// <b>For Beginners:</b> This helper method finds which grid cell contains our point of interest.
    /// For example, if we have grid points at x = [0, 10, 20, 30] and we want to interpolate
    /// at x = 15, this method will tell us that 15 is in the interval between index 1 (value 10)
    /// and index 2 (value 20).
    /// </remarks>
    /// <param name="values">The sorted array of coordinate values.</param>
    /// <param name="point">The coordinate value to locate.</param>
    /// <returns>The index of the lower bound of the interval containing the point.</returns>
    private int FindInterval(Vector<T> values, T point)
    {
        for (int i = 0; i < values.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(point, values[i + 1]))
                return i;
        }

        return values.Length - 2;
    }

    /// <summary>
    /// Binary search for an exact match in a sorted array.
    /// </summary>
    /// <param name="values">The sorted array to search.</param>
    /// <param name="target">The target value to find.</param>
    /// <returns>The index of the exact match, or -1 if not found.</returns>
    private int BinarySearchExact(Vector<T> values, T target)
    {
        int left = 0;
        int right = values.Length - 1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;

            if (_numOps.Equals(values[mid], target))
            {
                return mid;
            }

            if (_numOps.LessThan(values[mid], target))
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }

        return -1; // Not found
    }
}
