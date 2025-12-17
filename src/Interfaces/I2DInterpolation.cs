namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for two-dimensional interpolation algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for coordinates and interpolated values (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Interpolation is like "filling in the blanks" between known data points.
/// 
/// Imagine you have a table of temperatures measured at specific locations on a map (x,y coordinates).
/// But what if you want to know the temperature at a location where you don't have a measurement?
/// 
/// 2D interpolation helps you estimate that value based on the surrounding known values.
/// It's similar to how weather maps show smooth color gradients between weather stations.
/// 
/// This interface defines a standard way to perform this estimation for any type of
/// two-dimensional interpolation algorithm.
/// </remarks>
public interface I2DInterpolation<T>
{
    /// <summary>
    /// Calculates an interpolated value at the specified coordinates.
    /// </summary>
    /// <param name="x">The x-coordinate of the point to interpolate.</param>
    /// <param name="y">The y-coordinate of the point to interpolate.</param>
    /// <returns>The interpolated value at the specified coordinates.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a location (x,y) where you don't have data
    /// and returns an estimated value at that location based on surrounding known values.
    /// 
    /// For example, if you're working with temperature data:
    /// - x and y might represent longitude and latitude
    /// - The return value would be the estimated temperature at that location
    /// 
    /// Different implementations of this interface will use different mathematical
    /// techniques to calculate this estimate, such as linear interpolation, 
    /// bilinear interpolation, or spline interpolation.
    /// </remarks>
    T Interpolate(T x, T y);
}
