namespace AiDotNet.Interpolation;

/// <summary>
/// Implements cubic convolution interpolation for 2D data points.
/// </summary>
/// <remarks>
/// Cubic convolution interpolation creates smooth surfaces from a grid of data points.
/// It uses 16 neighboring points (a 4x4 grid) to calculate each interpolated value,
/// resulting in a continuous surface with smooth first derivatives.
/// 
/// <b>For Beginners:</b> This class helps you estimate values between known data points on a 2D grid.
/// Imagine having temperature readings at specific locations on a map, and you want to
/// estimate the temperature at locations between your measurement points. This interpolation
/// creates a smooth surface that passes through all your known points and provides reasonable
/// estimates for the points in between.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class CubicConvolutionInterpolation<T> : I2DInterpolation<T>
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
    /// The z-values (heights) at each (x,y) grid point.
    /// </summary>
    private readonly Matrix<T> _z;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new cubic convolution interpolation from the given 2D data points.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your grid of data points and prepares
    /// everything needed to perform interpolation. The x and y parameters represent
    /// the coordinates of your grid points, and the z matrix contains the values at
    /// each of those grid points.
    /// </remarks>
    /// <param name="x">The x-coordinates of the grid points.</param>
    /// <param name="y">The y-coordinates of the grid points.</param>
    /// <param name="z">The z-values (heights) at each (x,y) grid point.</param>
    /// <exception cref="ArgumentException">Thrown when the dimensions of the inputs don't match.</exception>
    public CubicConvolutionInterpolation(Vector<T> x, Vector<T> y, Matrix<T> z)
    {
        if (x.Length != z.Rows || y.Length != z.Columns)
            throw new ArgumentException("Input dimensions must match the z matrix dimensions.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the interpolated z-value for a given (x,y) point using cubic convolution.
    /// </summary>
    /// <remarks>
    /// This method finds the grid cell containing the point and uses cubic convolution
    /// interpolation to estimate the z-value at that point.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use after creating the interpolation.
    /// Give it any (x,y) coordinates within your data range, and it will return the estimated
    /// z-value at that point. It's like asking "if I have this location on my map, what would
    /// the value be based on the surrounding known points?"
    /// </remarks>
    /// <param name="x">The x-coordinate of the point to interpolate.</param>
    /// <param name="y">The y-coordinate of the point to interpolate.</param>
    /// <returns>The interpolated z-value at the given point.</returns>
    public T Interpolate(T x, T y)
    {
        // Find which grid cell contains the point
        int i = FindInterval(_x, x);
        int j = FindInterval(_y, y);

        // Calculate the relative position within the cell (0 to 1)
        T dx = _numOps.Divide(_numOps.Subtract(x, _x[i]), _numOps.Subtract(_x[i + 1], _x[i]));
        T dy = _numOps.Divide(_numOps.Subtract(y, _y[j]), _numOps.Subtract(_y[j + 1], _y[j]));

        // Get a 4x4 grid of points centered around the cell
        T[,] p = new T[4, 4];
        for (int m = -1; m <= 2; m++)
        {
            for (int n = -1; n <= 2; n++)
            {
                int row = MathHelper.Clamp(i + m, 0, _x.Length - 1);
                int col = MathHelper.Clamp(j + n, 0, _y.Length - 1);
                p[m + 1, n + 1] = _z[row, col];
            }
        }

        // Perform the bicubic interpolation
        return BicubicInterpolate(p, dx, dy);
    }

    /// <summary>
    /// Performs bicubic interpolation using a 4x4 grid of points.
    /// </summary>
    /// <remarks>
    /// This method applies cubic interpolation in both x and y directions to calculate
    /// the interpolated value.
    /// 
    /// <b>For Beginners:</b> Bicubic interpolation works in two steps: first, it calculates four
    /// intermediate values by interpolating along one direction (y), then it uses these
    /// four values to interpolate along the other direction (x) to get the final result.
    /// This two-step process creates a smooth surface that connects all the known points.
    /// </remarks>
    /// <param name="p">A 4x4 grid of z-values surrounding the point.</param>
    /// <param name="x">The relative x-position within the grid cell (0 to 1).</param>
    /// <param name="y">The relative y-position within the grid cell (0 to 1).</param>
    /// <returns>The interpolated value at position (x,y).</returns>
    private T BicubicInterpolate(T[,] p, T x, T y)
    {
        // First interpolate along y direction for each x column
        T[] a = new T[4];
        for (int i = 0; i < 4; i++)
        {
            a[i] = CubicInterpolate(p[i, 0], p[i, 1], p[i, 2], p[i, 3], y);
        }

        // Then interpolate these results along the x direction
        return CubicInterpolate(a[0], a[1], a[2], a[3], x);
    }

    /// <summary>
    /// Performs cubic interpolation between four points.
    /// </summary>
    /// <remarks>
    /// This method implements the cubic convolution interpolation formula to calculate
    /// a smooth curve through four points.
    /// 
    /// <b>For Beginners:</b> Cubic interpolation uses four points to create a smooth curve.
    /// Unlike simpler methods that only use two points, cubic interpolation creates
    /// a curve that not only passes through the points but also has a natural-looking
    /// shape between them. The formula looks complex, but it's essentially creating
    /// a special curve (a cubic polynomial) that gives smooth results.
    /// </remarks>
    /// <param name="p0">The first point value.</param>
    /// <param name="p1">The second point value.</param>
    /// <param name="p2">The third point value.</param>
    /// <param name="p3">The fourth point value.</param>
    /// <param name="x">The relative position for interpolation (0 to 1).</param>
    /// <returns>The interpolated value at position x.</returns>
    private T CubicInterpolate(T p0, T p1, T p2, T p3, T x)
    {
        // Calculate the coefficients of the cubic polynomial: a*x³ + b*x² + c*x + d
        T a = _numOps.Subtract(_numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(-0.5), p0), _numOps.Multiply(_numOps.FromDouble(1.5), p1)), _numOps.Multiply(_numOps.FromDouble(-1.5), p2));
        a = _numOps.Add(a, _numOps.Multiply(_numOps.FromDouble(0.5), p3));

        T b = _numOps.Add(_numOps.Add(_numOps.Multiply(p0, _numOps.FromDouble(2.5)), _numOps.Multiply(p1, _numOps.FromDouble(-4.5))), _numOps.Multiply(p2, _numOps.FromDouble(3.0)));
        b = _numOps.Subtract(b, _numOps.Multiply(p3, _numOps.FromDouble(0.5)));

        T c = _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(-0.5), p0), _numOps.Multiply(_numOps.FromDouble(0.5), p2));

        T d = p1;

        // Evaluate the polynomial at position x: a*x³ + b*x² + c*x + d
        return _numOps.Add(_numOps.Add(_numOps.Add(_numOps.Multiply(_numOps.Multiply(a, x), _numOps.Multiply(x, x)), _numOps.Multiply(_numOps.Multiply(b, x), x)), _numOps.Multiply(c, x)), d);
    }

    /// <summary>
    /// Finds the index of the interval in a sorted array that contains the given point.
    /// </summary>
    /// <remarks>
    /// This method uses binary search to efficiently find which interval contains the point.
    /// 
    /// <b>For Beginners:</b> This method finds which "box" or cell in your grid contains the point
    /// you're interested in. It's like finding which square on a map contains a specific
    /// location. The binary search approach makes this process very efficient, even with
    /// large grids.
    /// </remarks>
    /// <param name="values">The sorted array of values defining the intervals.</param>
    /// <param name="point">The point to locate within the intervals.</param>
    /// <returns>The index of the interval containing the point.</returns>
    private int FindInterval(Vector<T> values, T point)
    {
        // Use binary search to find the position
        int index = values.BinarySearch(point);
        if (index < 0)
        {
            // If point not found exactly, ~index gives the insertion point
            // We want the interval before that point
            index = ~index - 1;
        }

        // Ensure the index is within valid bounds
        return MathHelper.Clamp(index, 0, values.Length - 2);
    }
}
