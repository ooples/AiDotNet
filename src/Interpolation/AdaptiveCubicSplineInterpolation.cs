namespace AiDotNet.Interpolation;

/// <summary>
/// Provides an adaptive cubic spline interpolation that automatically switches between natural and monotonic 
/// cubic splines based on data characteristics.
/// </summary>
/// <remarks>
/// This class implements an intelligent interpolation strategy that combines the smoothness of natural 
/// cubic splines with the shape-preserving properties of monotonic cubic splines.
/// 
/// <b>For Beginners:</b> Interpolation is like "connecting the dots" between data points to create a smooth curve.
/// 
/// Imagine you have several points on a graph and want to draw a smooth line through them:
/// - Natural cubic splines create very smooth curves but might overshoot or create waves where your data doesn't have them
/// - Monotonic splines preserve the "shape" of your data (keeping increasing parts increasing, etc.) but might be less smooth
/// 
/// This adaptive method gives you the best of both worlds by:
/// 1. Looking at each segment of your data
/// 2. Deciding whether to use the smoother method or the shape-preserving method for that segment
/// 3. Automatically switching between methods as needed
/// 
/// It's like having a smart drawing assistant that chooses the right tool for each part of your curve!
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class AdaptiveCubicSplineInterpolation<T> : IInterpolation<T>
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
    /// The natural cubic spline interpolation instance.
    /// </summary>
    private readonly IInterpolation<T> _naturalSpline;

    /// <summary>
    /// The monotonic cubic spline interpolation instance.
    /// </summary>
    private readonly IInterpolation<T> _monotonicSpline;

    /// <summary>
    /// Array indicating which interpolation method to use for each interval.
    /// </summary>
    /// <remarks>
    /// When _useMonotonic[i] is true, the monotonic spline is used for the interval between points i and i+1.
    /// When false, the natural spline is used.
    /// </remarks>
    private readonly bool[] _useMonotonic;

    /// <summary>
    /// Helper object for performing numeric operations on generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the AdaptiveCubicSplineInterpolation class.
    /// </summary>
    /// <remarks>
    /// This constructor creates both natural and monotonic spline interpolators and determines
    /// which one to use for each interval based on the provided threshold.
    /// 
    /// <b>For Beginners:</b> This sets up the adaptive interpolation by:
    /// 1. Storing your data points (x and y values)
    /// 2. Creating two different interpolation methods (natural and monotonic)
    /// 3. Deciding which method works best for each segment of your data
    /// 
    /// The threshold controls how aggressive the selection is - a higher threshold means
    /// the algorithm will prefer natural splines more often, while a lower threshold
    /// will favor monotonic splines.
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points (must be in ascending order).</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="threshold">The threshold value that determines when to switch between interpolation methods.</param>
    public AdaptiveCubicSplineInterpolation(Vector<T> x, Vector<T> y, T threshold)
    {
        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _naturalSpline = new CubicSplineInterpolation<T>(x, y);
        _monotonicSpline = new MonotoneCubicInterpolation<T>(x, y);
        _useMonotonic = DetermineInterpolationMethod(threshold);
    }

    /// <summary>
    /// Interpolates a value at the specified x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method finds which interval the x-coordinate falls into and then uses either
    /// the natural or monotonic spline interpolation based on what was determined to be
    /// best for that interval.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use. Give it an x-value, and it will:
    /// 1. Figure out which segment of your data the x-value falls into
    /// 2. Use the best interpolation method for that segment
    /// 3. Return the estimated y-value at that point
    /// 
    /// For example, if you have data points at x = [1, 2, 3, 4] and you ask for the value at x = 2.5,
    /// it will:
    /// 1. Determine that 2.5 is between points 2 and 3
    /// 2. Check whether natural or monotonic interpolation was chosen for that segment
    /// 3. Use the appropriate method to calculate the y-value at x = 2.5
    /// </remarks>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value at the specified x-coordinate.</returns>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);
        return _useMonotonic[i] ? _monotonicSpline.Interpolate(x) : _naturalSpline.Interpolate(x);
    }

    /// <summary>
    /// Determines which interpolation method to use for each interval based on the threshold.
    /// </summary>
    /// <remarks>
    /// This method evaluates both interpolation methods at the midpoint of each interval and
    /// compares their deviations from a linear interpolation. If the difference between these
    /// deviations exceeds the threshold, the monotonic spline is used for that interval.
    /// </remarks>
    /// <param name="threshold">The threshold value for deciding between interpolation methods.</param>
    /// <returns>An array of boolean values indicating which intervals should use monotonic interpolation.</returns>
    private bool[] DetermineInterpolationMethod(T threshold)
    {
        bool[] useMonotonic = new bool[_x.Length - 1];

        for (int i = 0; i < _x.Length - 1; i++)
        {
            T slope = _numOps.Divide(_numOps.Subtract(_y[i + 1], _y[i]), _numOps.Subtract(_x[i + 1], _x[i]));
            T naturalValue = _naturalSpline.Interpolate(_numOps.Divide(_numOps.Add(_x[i], _x[i + 1]), _numOps.FromDouble(2)));
            T monotonicValue = _monotonicSpline.Interpolate(_numOps.Divide(_numOps.Add(_x[i], _x[i + 1]), _numOps.FromDouble(2)));

            T naturalDiff = _numOps.Abs(_numOps.Subtract(naturalValue, _numOps.Divide(_numOps.Add(_y[i], _y[i + 1]), _numOps.FromDouble(2))));
            T monotonicDiff = _numOps.Abs(_numOps.Subtract(monotonicValue, _numOps.Divide(_numOps.Add(_y[i], _y[i + 1]), _numOps.FromDouble(2))));

            useMonotonic[i] = _numOps.GreaterThan(_numOps.Subtract(naturalDiff, monotonicDiff), threshold);
        }

        return useMonotonic;
    }

    /// <summary>
    /// Finds the interval index that contains the specified x-coordinate.
    /// </summary>
    /// <remarks>
    /// This method searches through the x-coordinates to find which interval the given x-value falls into.
    /// If the x-value is outside the range of the data, it returns the first or last interval as appropriate.
    /// </remarks>
    /// <param name="x">The x-coordinate to locate.</param>
    /// <returns>The index of the interval containing the x-coordinate.</returns>
    private int FindInterval(T x)
    {
        for (int i = 0; i < _x.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
                return i;
        }

        return _x.Length - 2;
    }
}
