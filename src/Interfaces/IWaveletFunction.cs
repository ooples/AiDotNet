namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the functionality for wavelet transforms used in signal processing and data analysis.
/// </summary>
/// <remarks>
/// Wavelets are mathematical functions that split data into different frequency components
/// and analyze each component with a resolution matched to its scale. They are particularly
/// useful for analyzing signals with discontinuities or sharp changes.
/// 
/// <b>For Beginners:</b> Think of wavelets as special tools for breaking down complex signals (like audio,
/// images, or any sequence of measurements) into simpler pieces that are easier to analyze.
/// 
/// Imagine you're trying to understand a song:
/// - Regular analysis might tell you which notes are played throughout the entire song
/// - Wavelet analysis tells you which notes are played at specific moments in time
/// 
/// This makes wavelets excellent for:
/// - Removing noise from signals (like cleaning up a blurry photo)
/// - Compressing data (like making image files smaller)
/// - Detecting patterns or features at different scales (like finding anomalies in heart rate data)
/// - Analyzing signals that change over time (like stock market prices)
/// 
/// Unlike simpler transforms (like Fourier), wavelets can capture both frequency and time information,
/// making them more powerful for many real-world applications.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("WaveletFunction")]
public interface IWaveletFunction<T>
{
    /// <summary>
    /// Calculates the wavelet function value at a specific point.
    /// </summary>
    /// <remarks>
    /// This method evaluates the wavelet function at a given input value.
    /// 
    /// <b>For Beginners:</b> This is like asking "what's the value of this wavelet at this specific point?"
    /// 
    /// Different wavelet functions have different shapes. Some common ones include:
    /// - Haar wavelets (simple step functions)
    /// - Daubechies wavelets (more complex, smoother functions)
    /// - Mexican hat wavelets (shaped like a Mexican sombrero)
    /// 
    /// This method gives you the y-value when you provide an x-value for the wavelet function.
    /// </remarks>
    /// <param name="x">The input value at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at the specified point.</returns>
    T Calculate(T x);

    /// <summary>
    /// Decomposes a signal into approximation and detail components using the wavelet transform.
    /// </summary>
    /// <remarks>
    /// This method applies the wavelet transform to break down the input signal into two components:
    /// an approximation (low-frequency component) and detail (high-frequency component).
    /// 
    /// <b>For Beginners:</b> This method splits your data into two parts:
    /// 
    /// 1. Approximation: The "big picture" or smooth trends in your data
    /// 2. Detail: The fine details, rapid changes, or "texture" in your data
    /// 
    /// For example, if your input is a heart rate signal:
    /// - The approximation would capture the overall rhythm and major beats
    /// - The detail would capture the smaller variations between beats
    /// 
    /// This separation is useful because you can:
    /// - Focus on just the important trends by using the approximation
    /// - Look for anomalies or specific features in the detail
    /// - Remove noise by modifying the detail component
    /// - Compress data by storing the approximation with less precision
    /// </remarks>
    /// <param name="input">The input signal (vector) to decompose.</param>
    /// <returns>A tuple containing the approximation and detail components of the signal.</returns>
    (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input);

    /// <summary>
    /// Gets the scaling coefficients associated with the wavelet function.
    /// </summary>
    /// <remarks>
    /// Scaling coefficients define the scaling function (also called father wavelet) that is used
    /// to create the approximation component in wavelet decomposition.
    /// 
    /// <b>For Beginners:</b> These coefficients are like a recipe for creating the "big picture" part
    /// of your data during wavelet analysis.
    /// 
    /// Think of them as weights that determine how neighboring data points should be combined
    /// to create a smoothed version of your signal. Different wavelet families have different
    /// sets of scaling coefficients, which affect how the smoothing is performed.
    /// 
    /// For example:
    /// - The Haar wavelet has very simple scaling coefficients [0.5, 0.5]
    /// - More complex wavelets like Daubechies have longer sets of coefficients
    /// 
    /// You typically don't need to work with these directly unless you're implementing custom
    /// wavelet transforms or need to understand the mathematical details of the wavelet being used.
    /// </remarks>
    /// <returns>A vector containing the scaling coefficients of the wavelet function.</returns>
    Vector<T> GetScalingCoefficients();

    /// <summary>
    /// Gets the wavelet coefficients associated with the wavelet function.
    /// </summary>
    /// <remarks>
    /// Wavelet coefficients define the wavelet function (also called mother wavelet) that is used
    /// to create the detail component in wavelet decomposition.
    /// 
    /// <b>For Beginners:</b> These coefficients are like a recipe for extracting the "fine details"
    /// part of your data during wavelet analysis.
    /// 
    /// While scaling coefficients help create a smoothed version of your signal, wavelet coefficients
    /// help identify changes, edges, or discontinuities in your data. Different wavelet families
    /// have different sets of wavelet coefficients, which affect how details are extracted.
    /// 
    /// For example:
    /// - The Haar wavelet has very simple wavelet coefficients [0.5, -0.5]
    /// - More complex wavelets have longer sets of coefficients designed to capture different
    ///   types of details
    /// 
    /// As with scaling coefficients, you typically don't need to work with these directly unless
    /// you're implementing custom wavelet transforms or need to understand the mathematical details.
    /// </remarks>
    /// <returns>A vector containing the wavelet coefficients of the wavelet function.</returns>
    Vector<T> GetWaveletCoefficients();
}
