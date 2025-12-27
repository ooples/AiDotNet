namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Akima interpolation, a method for smooth curve fitting through a set of data points.
/// </summary>
/// <remarks>
/// Akima interpolation creates a smooth curve that passes through all data points while minimizing
/// unwanted oscillations that can occur with other interpolation methods.
/// 
/// <b>For Beginners:</b> Akima interpolation is like drawing a smooth line through a set of points.
/// Unlike simpler methods, it creates curves that look more natural, especially when your data
/// has sudden changes or sharp turns. It's particularly good at avoiding artificial "wiggles"
/// that other methods might create between your data points.
/// 
/// Think of it as a skilled artist drawing a smooth curve through points, rather than
/// connecting them with straight lines or creating overly wavy curves.
/// 
/// This method requires at least 5 data points to work properly.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class AkimaInterpolation<T> : IInterpolation<T>
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
    /// The first-order polynomial coefficients for each interval.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are values that help determine the slope of the curve at each point.
    /// </remarks>
    private readonly Vector<T> _b;

    /// <summary>
    /// The second-order polynomial coefficients for each interval.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These values help control how the curve bends between points.
    /// </remarks>
    private readonly Vector<T> _c;

    /// <summary>
    /// The third-order polynomial coefficients for each interval.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These values help fine-tune the shape of the curve between points.
    /// </remarks>
    private readonly Vector<T> _d;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the AkimaInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor validates the input data, initializes the necessary arrays,
    /// and calculates the coefficients needed for interpolation.
    /// 
    /// <b>For Beginners:</b> This sets up everything needed to perform Akima interpolation:
    /// 1. It checks that your data is valid (same number of x and y values, at least 5 points)
    /// 2. It stores your data points
    /// 3. It prepares the mathematical values needed to create smooth curves between your points
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points (must be in ascending order).</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths or when there are fewer than 5 data points.
    /// </exception>
    public AkimaInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 5)
        {
            throw new ArgumentException("Akima interpolation requires at least 5 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();

        int n = x.Length;
        _b = new Vector<T>(n);  // Need slopes at all n data points
        _c = new Vector<T>(n - 1);
        _d = new Vector<T>(n - 1);

        CalculateCoefficients();
    }

    /// <summary>
    /// Interpolates a value at the specified x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method finds which interval the x-coordinate falls into and then calculates
    /// the interpolated y-value using the pre-computed polynomial coefficients.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an x-value, and it will:
    /// 1. Find which segment of your data the x-value falls into
    /// 2. Use the smooth curve formula to calculate the corresponding y-value
    /// 
    /// For example, if you have data points at x = [1, 2, 3, 4, 5] and you want to know
    /// what the y-value would be at x = 2.5, this method will give you that estimate
    /// based on the smooth curve that passes through all your original points.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        T dx = _numOps.Subtract(x, _x[i]);
        T y = _y[i];
        y = _numOps.Add(y, _numOps.Multiply(_b[i], dx));
        y = _numOps.Add(y, _numOps.Multiply(_c[i], _numOps.Multiply(dx, dx)));
        y = _numOps.Add(y, _numOps.Multiply(_d[i], _numOps.Multiply(_numOps.Multiply(dx, dx), dx)));

        return y;
    }

    /// <summary>
    /// Calculates the polynomial coefficients needed for interpolation.
    /// </summary>
    /// <remarks>
    /// This method implements the Akima algorithm to calculate the coefficients that define
    /// the piecewise polynomial function used for interpolation.
    /// 
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting that makes Akima interpolation work.
    /// It calculates:
    /// 1. The slopes between adjacent points
    /// 2. Special weights that help determine how to draw the curve
    /// 3. The final values needed to create a smooth curve through all your points
    /// 
    /// You don't need to call this method directly - it's automatically called when you create
    /// a new AkimaInterpolation object.
    /// </remarks>
    private void CalculateCoefficients()
    {
        int n = _x.Length;
        Vector<T> m = new Vector<T>(n + 3);

        // Calculate slopes
        for (int i = 0; i < n - 1; i++)
        {
            m[i + 2] = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), _numOps.Subtract(_x[i + 1], _x[i]));
        }

        // Set up the first and last two slopes
        m[1] = _numOps.Add(m[2], _numOps.Subtract(m[2], m[3]));
        m[0] = _numOps.Add(m[1], _numOps.Subtract(m[1], m[2]));
        m[n + 1] = _numOps.Add(m[n], _numOps.Subtract(m[n], m[n - 1]));
        m[n + 2] = _numOps.Add(m[n + 1], _numOps.Subtract(m[n + 1], m[n]));

        // Calculate Akima weights (slopes at all n data points)
        for (int i = 0; i < n; i++)
        {
            T w1 = _numOps.Abs(_numOps.Subtract(m[i + 3], m[i + 2]));
            T w2 = _numOps.Abs(_numOps.Subtract(m[i + 1], m[i]));

            if (_numOps.Equals(w1, _numOps.Zero) && _numOps.Equals(w2, _numOps.Zero))
            {
                _b[i] = m[i + 2];
            }
            else
            {
                _b[i] = _numOps.Divide(_numOps.Add(_numOps.Multiply(w1, m[i + 1]), _numOps.Multiply(w2, m[i + 2])), _numOps.Add(w1, w2));
            }
        }

        // Calculate remaining coefficients
        for (int i = 0; i < n - 1; i++)
        {
            T h = _numOps.Subtract(_x[i + 1], _x[i]);
            T slope = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), h);
            _c[i] = _numOps.Divide(_numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(3), slope), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), _b[i]), _b[i + 1])), h);
            _d[i] = _numOps.Divide(_numOps.Subtract(_numOps.Add(_b[i], _b[i + 1]), _numOps.Multiply(_numOps.FromDouble(2), slope)), _numOps.Multiply(h, h));
        }
    }

    /// <summary>
    /// Finds the interval index that contains the specified x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method uses a binary search algorithm to efficiently find which interval
    /// the x-coordinate falls into. If the x-coordinate is outside the range of the data,
    /// it returns the first or last interval as appropriate.
    /// 
    /// <b>For Beginners:</b> This method figures out which segment of your data contains the x-value
    /// you're interested in. It uses a smart search technique (binary search) that's much faster
    /// than checking each segment one by one, especially when you have lots of data points.
    /// 
    /// For example, if your x-values are [1, 3, 5, 7, 9] and you want to find where x = 6 belongs,
    /// this method will determine that it falls between 5 and 7 (interval index 2).
    /// </remarks>
    /// <param name="x">The x-coordinate to locate.</param>
    /// <returns>The index of the interval containing the x-coordinate.</returns>
    private int FindInterval(T x)
    {
        if (_numOps.LessThanOrEquals(x, _x[0]))
            return 0;
        if (_numOps.GreaterThanOrEquals(x, _x[_x.Length - 1]))
            return _x.Length - 2;

        int low = 0;
        int high = _x.Length - 1;

        while (low < high - 1)
        {
            int mid = (low + high) / 2;
            if (_numOps.LessThanOrEquals(_x[mid], x))
                low = mid;
            else
                high = mid;
        }

        return low;
    }
}
