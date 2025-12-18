namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Bilinear Interpolation for estimating values between points in a 2D grid.
/// </summary>
/// <remarks>
/// Bilinear interpolation creates a smooth surface that passes through a grid of known data points.
/// It's commonly used for image resizing, terrain modeling, and data visualization.
/// 
/// <b>For Beginners:</b> Think of bilinear interpolation as a way to "fill in the blanks" between points in a grid.
/// Imagine you have a grid of temperature readings taken at different locations, but you want to know
/// the temperature at a location between your measurement points. Bilinear interpolation gives you
/// an estimated value based on the four nearest known points.
/// 
/// Unlike simply picking the closest point's value, bilinear interpolation creates a smooth blend
/// between all four surrounding points, giving a more natural and accurate estimate.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class BilinearInterpolation<T> : I2DInterpolation<T>
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
    /// The z-values (data values) at each grid point, organized as a matrix.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is your actual data - the known values at each grid point.
    /// If you're thinking of a temperature map, these would be the temperature readings
    /// at each measured location.
    /// </remarks>
    private readonly Matrix<T> _z;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a utility that helps perform math operations (like addition
    /// and multiplication) on different types of numbers. You don't need to interact with
    /// this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the BilinearInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor validates the input data and initializes the necessary components
    /// for performing bilinear interpolation.
    /// 
    /// <b>For Beginners:</b> This sets up everything needed to perform the interpolation:
    /// 1. It checks that your data grid is valid (matching dimensions and enough points)
    /// 2. It stores your grid coordinates and values
    /// 3. It prepares the mathematical tools needed for calculations
    /// </remarks>
    /// <param name="x">The x-coordinates of the grid points.</param>
    /// <param name="y">The y-coordinates of the grid points.</param>
    /// <param name="z">The z-values (data values) at each grid point.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input dimensions don't match or when there are fewer than 2x2 grid points.
    /// </exception>
    public BilinearInterpolation(Vector<T> x, Vector<T> y, Matrix<T> z)
    {
        if (x.Length != z.Rows || y.Length != z.Columns)
            throw new ArgumentException("Input dimensions mismatch.");
        if (x.Length < 2 || y.Length < 2)
            throw new ArgumentException("Bilinear interpolation requires at least a 2x2 grid.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Interpolates a value at the specified (x,y) coordinates.
    /// </summary>
    /// <remarks>
    /// This method finds the grid cell containing the target point and uses bilinear
    /// interpolation to estimate the z-value at that point.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an (x,y) location, and it will:
    /// 1. Find which grid cell contains your point
    /// 2. Get the values at the four corners of that cell
    /// 3. Calculate a weighted average based on how close your point is to each corner
    /// 
    /// For example, if you have temperature readings at grid points spaced 10 miles apart
    /// and you want to know the temperature at a point 3 miles from one grid point, this method
    /// will give you that estimate by blending the values from the surrounding points.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <param name="y">The y-coordinate at which to interpolate.</param>
    /// <returns>The interpolated z-value at the specified (x,y) coordinates.</returns>
    public T Interpolate(T x, T y)
    {
        int i = FindInterval(_x, x);
        int j = FindInterval(_y, y);

        T x1 = _x[i];
        T x2 = _x[i + 1];
        T y1 = _y[j];
        T y2 = _y[j + 1];

        T q11 = _z[i, j];        // Bottom-left corner value
        T q21 = _z[i + 1, j];    // Bottom-right corner value
        T q12 = _z[i, j + 1];    // Top-left corner value
        T q22 = _z[i + 1, j + 1]; // Top-right corner value

        T dx = _numOps.Subtract(x2, x1);
        T dy = _numOps.Subtract(y2, y1);

        // Calculate normalized coordinates (0 to 1) within the grid cell
        T tx = _numOps.Divide(_numOps.Subtract(x, x1), dx);
        T ty = _numOps.Divide(_numOps.Subtract(y, y1), dy);

        // Interpolate along x-direction for both y-values
        T r1 = _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, tx), q11), _numOps.Multiply(tx, q21));
        T r2 = _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, tx), q12), _numOps.Multiply(tx, q22));

        // Interpolate along y-direction using the results from x-interpolation
        return _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, ty), r1), _numOps.Multiply(ty, r2));
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
}
