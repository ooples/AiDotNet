namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Barycentric Rational Interpolation, a powerful method for fitting a smooth curve through a set of data points.
/// </summary>
/// <remarks>
/// Barycentric interpolation is a stable and efficient technique that works well even with unevenly spaced data points.
/// It creates a smooth curve that passes exactly through all provided data points.
/// 
/// <b>For Beginners:</b> Think of this as a sophisticated way to "connect the dots" between your data points.
/// Unlike simpler methods that might just draw straight lines between points, this method creates a smooth
/// curve that passes exactly through each point. It's particularly good at handling data where points
/// aren't evenly spaced, and it avoids the wild oscillations that can happen with some other methods.
/// 
/// The "barycentric" part refers to a special mathematical approach that makes calculations more stable
/// and efficient, especially when dealing with many data points.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class BarycentricRationalInterpolation<T> : IInterpolation<T>
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
    /// The barycentric weights used in the interpolation formula.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These weights determine how much influence each data point has when calculating
    /// values between points. They're calculated once when you create the interpolation object and then
    /// used for all interpolation calculations.
    /// </remarks>
    private readonly Vector<T> _weights;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the BarycentricRationalInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor validates the input data, initializes the necessary arrays,
    /// and calculates the barycentric weights needed for interpolation.
    /// 
    /// <b>For Beginners:</b> This sets up everything needed to perform the interpolation:
    /// 1. It checks that your data is valid (same number of x and y values, at least 2 points)
    /// 2. It stores your data points
    /// 3. It calculates special "weights" that will be used when estimating values between your points
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input vectors have different lengths or when there are fewer than 2 data points.
    /// </exception>
    public BarycentricRationalInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Barycentric rational interpolation requires at least 2 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = CalculateWeights();
    }

    /// <summary>
    /// Interpolates a value at the specified x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method uses the barycentric interpolation formula to calculate the y-value
    /// at the given x-coordinate based on the provided data points.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an x-value, and it will:
    /// 1. Check if the x-value exactly matches one of your original data points
    ///    (if so, it returns the exact y-value from your data)
    /// 2. If not, it calculates what the y-value should be at that x-value using the smooth curve
    ///    that passes through all your original points
    /// 
    /// For example, if you have data points at x = [1, 3, 5] and you want to know
    /// what the y-value would be at x = 2, this method will give you that estimate.
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        T numerator = _numOps.Zero;
        T denominator = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            if (_numOps.Equals(x, _x[i]))
            {
                return _y[i]; // Return exact value if x matches a known point
            }

            T diff = _numOps.Subtract(x, _x[i]);
            T term = _numOps.Divide(_weights[i], diff);

            numerator = _numOps.Add(numerator, _numOps.Multiply(term, _y[i]));
            denominator = _numOps.Add(denominator, term);
        }

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the barycentric weights used in the interpolation formula.
    /// </summary>
    /// <remarks>
    /// This method computes the weights based on the x-coordinates of the data points.
    /// These weights are a key component of the barycentric interpolation formula.
    /// 
    /// <b>For Beginners:</b> This method does the mathematical heavy lifting that makes barycentric
    /// interpolation work. It calculates special values (weights) for each of your data points
    /// that will later be used to create a smooth curve through all points.
    /// 
    /// The weights are calculated based on the distances between all pairs of x-values in your data.
    /// This approach helps ensure that the resulting curve passes exactly through each of your
    /// original data points.
    /// 
    /// You don't need to call this method directly - it's automatically called when you create
    /// a new BarycentricRationalInterpolation object.
    /// </remarks>
    /// <returns>A vector containing the calculated barycentric weights.</returns>
    private Vector<T> CalculateWeights()
    {
        int n = _x.Length;
        Vector<T> weights = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T weight = _numOps.One;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    weight = _numOps.Multiply(weight, _numOps.Subtract(_x[i], _x[j]));
                }
            }
            weights[i] = _numOps.Divide(_numOps.One, weight);
        }

        return weights;
    }
}
