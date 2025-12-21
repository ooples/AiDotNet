namespace AiDotNet.Interpolation;

/// <summary>
/// Adapts a two-dimensional interpolation to work as a one-dimensional interpolation by fixing one coordinate.
/// </summary>
/// <remarks>
/// This adapter allows you to use a 2D interpolation method as if it were a 1D interpolation
/// by keeping one of the coordinates (either X or Y) at a fixed value.
/// 
/// <b>For Beginners:</b> Think of this like taking a slice through a 3D surface. Imagine a landscape
/// with hills and valleys - if you cut through it in a straight line, you get a 2D profile
/// showing the heights along that line. This adapter does something similar - it takes a 2D
/// interpolation (which works with X and Y coordinates) and creates a 1D view of it (which
/// only needs one coordinate) by fixing either the X or Y value.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class Interpolation2DTo1DAdapter<T> : IInterpolation<T>
{
    /// <summary>
    /// The underlying two-dimensional interpolation method.
    /// </summary>
    private readonly I2DInterpolation<T> _interpolation2D;

    /// <summary>
    /// The value of the coordinate that remains constant.
    /// </summary>
    private readonly T _fixedCoordinate;

    /// <summary>
    /// Indicates whether the X coordinate is fixed (true) or the Y coordinate is fixed (false).
    /// </summary>
    private readonly bool _isXFixed;

    /// <summary>
    /// Gets a human-readable description of this interpolation adapter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property provides a simple text description of what this adapter is doing,
    /// showing which coordinate (X or Y) is being kept constant and what value it's set to.
    /// </remarks>
    public string Description { get; }

    /// <summary>
    /// Creates a new adapter that converts a 2D interpolation to a 1D interpolation.
    /// </summary>
    /// <remarks>
    /// This constructor sets up the adapter with the specified 2D interpolation method and
    /// configures which coordinate will be fixed and at what value.
    /// 
    /// <b>For Beginners:</b> When you create this adapter, you're telling it three things:
    /// 1. Which 2D interpolation method to use underneath
    /// 2. What value to keep constant for one of the coordinates
    /// 3. Whether it's the X or Y coordinate that should be kept constant
    /// 
    /// For example, if you have a temperature map based on latitude and longitude, and you want
    /// to see how temperature changes along the equator, you would fix the latitude coordinate
    /// at 0 and let the longitude coordinate vary.
    /// </remarks>
    /// <param name="interpolation2D">The two-dimensional interpolation method to adapt.</param>
    /// <param name="fixedCoordinate">The value to use for the fixed coordinate.</param>
    /// <param name="isXFixed">If true, the X coordinate is fixed; if false, the Y coordinate is fixed.</param>
    public Interpolation2DTo1DAdapter(I2DInterpolation<T> interpolation2D, T fixedCoordinate, bool isXFixed)
    {
        _interpolation2D = interpolation2D;
        _fixedCoordinate = fixedCoordinate;
        _isXFixed = isXFixed;
        Description = $"2D interpolation slice with {(_isXFixed ? "X" : "Y")} fixed at {_fixedCoordinate}";
    }

    /// <summary>
    /// Performs one-dimensional interpolation by calling the underlying two-dimensional interpolation
    /// with one coordinate fixed.
    /// </summary>
    /// <remarks>
    /// This method takes a single coordinate value and combines it with the fixed coordinate value
    /// to perform a 2D interpolation.
    /// 
    /// <b>For Beginners:</b> This is the main method you'll use after creating the adapter. You provide
    /// a single value (for either X or Y, depending on which one isn't fixed), and the method
    /// automatically combines it with your fixed value to get the interpolated result from the
    /// underlying 2D interpolation. It's like saying "I want to know the value at this specific
    /// point along my slice through the data."
    /// </remarks>
    /// <param name="point">The coordinate value for the non-fixed dimension.</param>
    /// <returns>The interpolated value at the specified point.</returns>
    public T Interpolate(T point)
    {
        return _isXFixed
            ? _interpolation2D.Interpolate(_fixedCoordinate, point)
            : _interpolation2D.Interpolate(point, _fixedCoordinate);
    }
}
