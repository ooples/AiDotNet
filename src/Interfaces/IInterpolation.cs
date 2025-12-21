namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for interpolation algorithms that estimate values between known data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface defines a method for "filling in the gaps" between known data points.
/// 
/// Imagine you have a few data points:
/// - You know that at 9:00 AM, the temperature was 65°F
/// - You know that at 12:00 PM, the temperature was 75°F
/// - But you don't have a measurement for 10:30 AM
/// 
/// Interpolation helps you make a reasonable guess about that missing value.
/// It's like drawing a smooth line through your known points and then reading
/// the value at any position along that line.
/// 
/// Common types of interpolation include:
/// - Linear: Draws straight lines between points (like connecting dots)
/// - Polynomial: Creates smooth curves that pass through all points
/// - Spline: Creates a series of curves that connect smoothly
/// - Nearest neighbor: Uses the value of the closest known point
/// 
/// Interpolation is used in many AI applications:
/// - Filling gaps in time series data
/// - Creating smooth transitions in animations
/// - Estimating values between training examples
/// - Generating new data points based on existing ones
/// </remarks>
public interface IInterpolation<T>
{
    /// <summary>
    /// Calculates an interpolated value at the specified point.
    /// </summary>
    /// <param name="x">The point at which to interpolate.</param>
    /// <returns>The interpolated value at point x.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method estimates a value at a specific point using surrounding known data points.
    /// 
    /// The parameter:
    /// - x: The point where you want to estimate a value
    ///   (like asking "what was the temperature at 10:30 AM?")
    /// 
    /// What this method does:
    /// 1. Takes your input point (x)
    /// 2. Looks at the known data points that were used to create this interpolator
    /// 3. Applies a mathematical formula to estimate the value at your requested point
    /// 4. Returns that estimated value
    /// 
    /// Different implementations of this interface will use different mathematical
    /// techniques to make this estimation, which affects how smooth or accurate
    /// the results are.
    /// 
    /// For example:
    /// - Linear interpolation draws straight lines between points
    /// - Cubic interpolation creates smoother curves
    /// - Spline interpolation ensures smooth transitions between segments
    /// 
    /// The type parameter T could be a simple number (like double) for 1D interpolation,
    /// or it could be a more complex type for multi-dimensional interpolation.
    /// </remarks>
    T Interpolate(T x);
}
