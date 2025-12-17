namespace AiDotNet.Interpolation;

/// <summary>
/// Implements Kochanek-Bartels spline interpolation for one-dimensional data points.
/// </summary>
/// <remarks>
/// Kochanek-Bartels splines (also known as TCB splines) provide control over the shape of the curve
/// through three parameters: tension, continuity, and bias. This allows for more flexible and
/// customizable interpolation compared to simpler methods.
/// 
/// <b>For Beginners:</b> This interpolation method creates smooth curves between data points with
/// special controls that let you adjust how the curve looks. Imagine drawing a line through
/// dots on a graph, but being able to control how "tight" the curve is, how smoothly it
/// transitions between segments, and whether it tends to overshoot or undershoot. This is
/// particularly useful for animation paths or when you need precise control over the shape
/// of a curve.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class KochanekBartelsSplineInterpolation<T> : IInterpolation<T>
{
    /// <summary>
    /// The x-coordinates of the data points (independent variable).
    /// </summary>
    private readonly Vector<T> _x;

    /// <summary>
    /// The y-coordinates of the data points (dependent variable).
    /// </summary>
    private readonly Vector<T> _y;

    /// <summary>
    /// Controls how "tight" the curve is. Higher values create tighter curves.
    /// </summary>
    private readonly T _tension;

    /// <summary>
    /// Controls whether the curve tends to overshoot (negative values) or undershoot (positive values).
    /// </summary>
    private readonly T _bias;

    /// <summary>
    /// Controls the smoothness of transitions between curve segments.
    /// </summary>
    private readonly T _continuity;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new Kochanek-Bartels spline interpolation from the given data points and parameters.
    /// </summary>
    /// <remarks>
    /// The constructor initializes the interpolation with the provided data points and shape parameters.
    /// 
    /// <b>For Beginners:</b> When you create a new KochanekBartelsSplineInterpolation object, you provide
    /// your data points (x and y values) and can optionally adjust three special parameters:
    /// 
    /// - Tension (default 0): Controls how "tight" the curve is. Think of it like a rubber band:
    ///   - Positive values (0 to 1): Makes the curve tighter, with less overshoot
    ///   - Negative values (-1 to 0): Makes the curve looser, with more overshoot
    ///   - Zero: Natural curve, balanced between tight and loose
    /// 
    /// - Bias (default 0): Controls whether the curve tends to lean toward the previous point or the next point:
    ///   - Positive values (0 to 1): Curve leans toward the previous point
    ///   - Negative values (-1 to 0): Curve leans toward the next point
    ///   - Zero: Balanced curve, doesn't favor either direction
    /// 
    /// - Continuity (default 0): Controls how smoothly the curve transitions at each point:
    ///   - Positive values (0 to 1): Creates sharper corners at data points
    ///   - Negative values (-1 to 0): Creates rounder transitions at data points
    ///   - Zero: Smooth transitions at data points
    /// </remarks>
    /// <param name="x">The x-coordinates of the data points.</param>
    /// <param name="y">The y-coordinates of the data points.</param>
    /// <param name="tension">Controls the "tightness" of the curve. Default is 0.</param>
    /// <param name="bias">Controls whether the curve favors the previous or next point. Default is 0.</param>
    /// <param name="continuity">Controls the smoothness of transitions at data points. Default is 0.</param>
    /// <exception cref="ArgumentException">Thrown when the input vectors have different lengths or fewer than 4 points.</exception>
    public KochanekBartelsSplineInterpolation(Vector<T> x, Vector<T> y, double tension = 0, double bias = 0, double continuity = 0)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 4)
            throw new ArgumentException("At least 4 points are required for Kochanek–Bartels spline interpolation.");

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
        _tension = _numOps.FromDouble(tension);
        _bias = _numOps.FromDouble(bias);
        _continuity = _numOps.FromDouble(continuity);
    }

    /// <summary>
    /// Calculates the interpolated y-value for a given x-value using Kochanek-Bartels spline interpolation.
    /// </summary>
    /// <remarks>
    /// This method finds the interval containing the x-value and evaluates the spline at that point.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use after creating the interpolation.
    /// Give it any x-value within your data range, and it will return the estimated y-value
    /// using the Kochanek-Bartels spline curve that passes through your data points.
    /// </remarks>
    /// <param name="x">The x-value at which to interpolate.</param>
    /// <returns>The interpolated y-value.</returns>
    public T Interpolate(T x)
    {
        int i = FindInterval(x);

        T x0 = _x[i - 1];
        T x1 = _x[i];
        T x2 = _x[i + 1];
        T x3 = _x[i + 2];

        T y0 = _y[i - 1];
        T y1 = _y[i];
        T y2 = _y[i + 1];
        T y3 = _y[i + 2];

        T t = _numOps.Divide(_numOps.Subtract(x, x1), _numOps.Subtract(x2, x1));

        return KochanekBartelsSpline(y0, y1, y2, y3, t);
    }

    /// <summary>
    /// Finds the index of the interval containing the given x-value.
    /// </summary>
    /// <remarks>
    /// This method searches through the x-coordinates to find which segment contains the given x-value.
    /// 
    /// <b>For Beginners:</b> This helper method determines which pair of data points the x-value falls between.
    /// It's like finding which page of a book contains a specific word.
    /// </remarks>
    /// <param name="x">The x-value to locate.</param>
    /// <returns>The index of the starting point of the interval containing x.</returns>
    private int FindInterval(T x)
    {
        for (int i = 1; i < _x.Length - 2; i++)
        {
            if (_numOps.LessThanOrEquals(x, _x[i + 1]))
            {
                return i;
            }
        }
        return _x.Length - 3;
    }

    /// <summary>
    /// Calculates the value of the Kochanek-Bartels spline at a point within a segment.
    /// </summary>
    /// <remarks>
    /// This method computes the cubic polynomial that represents the spline within a segment.
    /// 
    /// <b>For Beginners:</b> Once we know which segment contains our x-value, this method calculates
    /// the actual y-value on the curve. It uses a special formula that takes into account
    /// the four nearest points and the tension, bias, and continuity parameters to create
    /// a smooth curve that passes through the data points.
    /// </remarks>
    /// <param name="y0">The y-value of the point before the segment start.</param>
    /// <param name="y1">The y-value at the segment start.</param>
    /// <param name="y2">The y-value at the segment end.</param>
    /// <param name="y3">The y-value of the point after the segment end.</param>
    /// <param name="t">The normalized position within the segment (0 at start, 1 at end).</param>
    /// <returns>The interpolated y-value.</returns>
    private T KochanekBartelsSpline(T y0, T y1, T y2, T y3, T t)
    {
        T t2 = _numOps.Multiply(t, t);
        T t3 = _numOps.Multiply(t2, t);

        T m1 = CalculateTangent(y0, y1, y2);
        T m2 = CalculateTangent(y1, y2, y3);

        T c0 = y1;
        T c1 = m1;
        T c2 = _numOps.Subtract(
            _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(3), _numOps.Subtract(y2, y1)),
                _numOps.Multiply(_numOps.FromDouble(2), m1)
            ),
            m2
        );
        T c3 = _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(2), _numOps.Subtract(y1, y2)),
                m1
            ),
            m2
        );

        return _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(c3, t3),
                _numOps.Multiply(c2, t2)
            ),
            _numOps.Add(
                _numOps.Multiply(c1, t),
                c0
            )
        );
    }

    /// <summary>
    /// Calculates the tangent at a point for the Kochanek-Bartels spline.
    /// </summary>
    /// <remarks>
    /// This method computes the tangent (slope) at a point using the Kochanek-Bartels formula,
    /// which takes into account the tension, continuity, and bias parameters.
    /// 
    /// <b>For Beginners:</b> The tangent is like the direction and speed of the curve at a specific point.
    /// Think of it as the direction a car would be heading if it was driving along the curve.
    /// This method calculates that direction based on three consecutive points and the special
    /// parameters (tension, bias, and continuity) that control the curve's shape.
    /// 
    /// The formula looks complex, but it's essentially combining the differences between points
    /// in a way that creates smooth transitions while respecting the shape controls you've set.
    /// </remarks>
    /// <param name="y0">The y-value of the previous point.</param>
    /// <param name="y1">The y-value of the current point where we're calculating the tangent.</param>
    /// <param name="y2">The y-value of the next point.</param>
    /// <returns>The calculated tangent value at the current point.</returns>
    private T CalculateTangent(T y0, T y1, T y2)
    {
        // Calculate intermediate values based on tension, continuity, and bias parameters
        // These represent mathematical transformations of the parameters to use in the formula
        T oneMinusTension = _numOps.Subtract(_numOps.One, _tension);
        T oneMinusContinuity = _numOps.Subtract(_numOps.One, _continuity);
        T onePlusContinuity = _numOps.Add(_numOps.One, _continuity);
        T oneMinusBias = _numOps.Subtract(_numOps.One, _bias);
        T onePlusBias = _numOps.Add(_numOps.One, _bias);

        // Calculate coefficient 'a' which controls the influence of the previous segment
        // This coefficient is affected by all three parameters: tension, continuity, and bias
        T a = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.Multiply(_numOps.FromDouble(0.5), oneMinusTension),
                oneMinusContinuity
            ),
            onePlusBias
        );

        // Calculate coefficient 'b' which controls the influence of the next segment
        // This coefficient is also affected by all three parameters but in a different way
        T b = _numOps.Multiply(
            _numOps.Multiply(
                _numOps.Multiply(_numOps.FromDouble(0.5), oneMinusTension),
                onePlusContinuity
            ),
            oneMinusBias
        );

        // Calculate the final tangent by combining the weighted differences between points
        // The tangent is essentially a weighted average of the slopes of adjacent segments
        return _numOps.Add(
            _numOps.Multiply(a, _numOps.Subtract(y1, y0)), // Influence from previous segment
            _numOps.Multiply(b, _numOps.Subtract(y2, y1))  // Influence from next segment
        );
    }
}
