namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different types of wavelet transform algorithms for signal processing.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Wavelet transforms are mathematical techniques that break down signals (like audio, 
/// images, or any data that changes over time) into different frequency components, similar to how 
/// a prism breaks light into different colors.
/// 
/// Unlike traditional Fourier transforms that only give frequency information, wavelets show both:
/// - What frequencies are present (like bass or treble in music)
/// - When these frequencies occur in time (like knowing exactly when a drum hit happens)
/// 
/// This makes wavelets extremely useful for:
/// - Analyzing signals that change over time
/// - Compressing images and audio (like JPEG2000)
/// - Removing noise from signals
/// - Detecting patterns or features in data
/// - Many scientific and engineering applications
/// 
/// Think of wavelets as special measuring tools that can zoom in on both short-lived and long-lasting 
/// patterns in your data, giving you a more complete picture than traditional methods.
/// </para>
/// </remarks>
public enum WaveletAlgorithmType
{
    /// <summary>
    /// Discrete Wavelet Transform - the standard wavelet transform algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Discrete Wavelet Transform (DWT) is the most common wavelet transform method.
    /// 
    /// Imagine having a set of different-sized magnifying glasses to examine your data:
    /// 
    /// 1. DWT first looks at your data with a "low-power" lens to capture the overall shape (low frequencies)
    /// 2. Then it uses progressively "higher-power" lenses to capture finer details (high frequencies)
    /// 3. At each step, it downsamples the data (keeps only every other point)
    /// 
    /// Key characteristics:
    /// - Very efficient computation (fast and uses little memory)
    /// - Provides a compact representation (good for compression)
    /// - The result has the same number of points as the original signal
    /// - Works best with signals whose length is a power of 2 (like 128, 256, 512)
    /// - Not shift-invariant (results change if you shift your input signal)
    /// 
    /// DWT is widely used in image compression, noise reduction, and feature extraction.
    /// </para>
    /// </remarks>
    DWT,

    /// <summary>
    /// Maximal Overlap Discrete Wavelet Transform - a redundant wavelet transform that preserves time invariance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Maximal Overlap Discrete Wavelet Transform (MODWT) is a modified version of DWT 
    /// that overcomes some of its limitations.
    /// 
    /// Unlike DWT, MODWT:
    /// 
    /// 1. Doesn't downsample the data at each step (keeps all points)
    /// 2. Produces the same number of coefficients at each scale (level of detail)
    /// 3. Is shift-invariant (shifting your input signal doesn't change the pattern of results)
    /// 
    /// Think of it like taking multiple overlapping photos of the same scene to ensure you don't miss anything:
    /// 
    /// Key characteristics:
    /// - More computationally intensive than DWT (needs more memory and processing)
    /// - Produces redundant information (more coefficients than the original signal)
    /// - Works with any signal length (not just powers of 2)
    /// - Better for analysis purposes where preserving time information is critical
    /// - Excellent for detecting patterns regardless of their position in time
    /// 
    /// MODWT is particularly useful in financial time series analysis, biomedical signal processing, 
    /// and other applications where the exact timing of events matters.
    /// </para>
    /// </remarks>
    MODWT,

    /// <summary>
    /// Stationary Wavelet Transform - another non-decimated wavelet transform similar to MODWT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Stationary Wavelet Transform (SWT) is very similar to MODWT and is sometimes 
    /// called the "undecimated wavelet transform" or "algorithme à trous" (algorithm with holes).
    /// 
    /// Like MODWT, SWT:
    /// 
    /// 1. Doesn't downsample the data
    /// 2. Is shift-invariant (results don't change if you shift your input)
    /// 3. Produces redundant information
    /// 
    /// The main difference is in the implementation details and how the filters are applied:
    /// 
    /// - SWT upsamples the filters at each level (inserts zeros between filter coefficients)
    /// - MODWT modifies the filters differently
    /// 
    /// Key characteristics:
    /// - Computationally intensive but can be implemented efficiently
    /// - Excellent for denoising applications
    /// - Particularly good for image processing tasks
    /// - Preserves the time information at all scales
    /// - Often used in applications where detecting transient signals is important
    /// 
    /// SWT is widely used in image denoising, edge detection, and feature extraction where 
    /// shift-invariance is critical.
    /// </para>
    /// </remarks>
    SWT
}
