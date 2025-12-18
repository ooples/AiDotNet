namespace AiDotNet.Interfaces;

/// <summary>
/// Defines functionality for creating window functions used in signal processing and data analysis.
/// </summary>
/// <remarks>
/// Window functions are mathematical functions that are zero-valued outside of a chosen interval.
/// They are applied to signals to reduce artifacts in spectral analysis and filter design.
/// 
/// <b>For Beginners:</b> Window functions are like special "lenses" that help us focus on specific parts
/// of data while smoothly fading out the rest. 
/// 
/// Imagine you're taking a photo through a window:
/// - A rectangular window gives you a clear view of everything inside the frame, but creates a 
///   sharp cutoff at the edges (which can cause problems in signal analysis)
/// - Other window shapes (like Hamming or Gaussian) are like windows that gradually get darker 
///   toward the edges, creating a smooth transition between what's in focus and what's not
/// 
/// Window functions are commonly used for:
/// - Analyzing audio signals (like in music apps that show frequency visualizations)
/// - Processing images or video
/// - Filtering out unwanted noise or frequencies
/// - Smoothly connecting segments of data
/// 
/// Different window functions have different shapes and properties that make them suitable
/// for different applications.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IWindowFunction<T>
{
    /// <summary>
    /// Creates a window function with the specified size.
    /// </summary>
    /// <remarks>
    /// This method generates a vector containing the window function values.
    /// 
    /// <b>For Beginners:</b> This method creates an array of numbers that represent the "shape" of the window.
    /// 
    /// For example, if you request a window of size 5:
    /// - A rectangular window might return [1, 1, 1, 1, 1]
    /// - A triangular window might return [0, 0.5, 1, 0.5, 0]
    /// - A Hamming window might return [0.08, 0.54, 1, 0.54, 0.08]
    /// 
    /// These values are then typically multiplied with your data to apply the windowing effect.
    /// The larger the window size, the more data points will be included in your analysis.
    /// </remarks>
    /// <param name="windowSize">The number of points in the window function.</param>
    /// <returns>A vector containing the window function values.</returns>
    Vector<T> Create(int windowSize);

    /// <summary>
    /// Gets the type of window function being implemented.
    /// </summary>
    /// <remarks>
    /// This method returns an enumeration value that identifies which specific window function
    /// is being used (e.g., Hamming, Hanning, Blackman, etc.).
    /// 
    /// <b>For Beginners:</b> This tells you which "shape" of window is being used. Common window types include:
    /// 
    /// - Rectangular: The simplest window, like looking through a clear glass window with sharp edges
    /// - Hamming/Hanning: Bell-shaped windows that taper to near-zero at the edges
    /// - Gaussian: A window shaped like a bell curve (normal distribution)
    /// - Blackman: A window that provides good frequency resolution with low leakage
    /// - Kaiser: A flexible window where you can adjust the trade-off between main-lobe width and side-lobe level
    /// 
    /// Different window types are better for different applications:
    /// - Rectangular windows preserve the original signal amplitude but can introduce artifacts
    /// - Hamming/Hanning windows reduce artifacts but slightly modify the signal amplitude
    /// - More specialized windows offer different trade-offs between frequency resolution and spectral leakage
    /// </remarks>
    /// <returns>The type of window function.</returns>
    WindowFunctionType GetWindowFunctionType();
}
