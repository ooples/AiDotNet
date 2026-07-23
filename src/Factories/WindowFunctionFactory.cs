namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates window functions for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Window functions are mathematical functions used in signal processing to reduce 
/// unwanted effects when analyzing a finite section of a longer signal. Think of them like special lenses 
/// that help you focus on certain parts of a signal while reducing distortion around the edges.
/// </para>
/// <para>
/// This factory helps you create different types of window functions without needing to know their 
/// internal implementation details. Think of it like ordering a specific tool from a catalog - you just 
/// specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class WindowFunctionFactory
{
    /// <summary>
    /// Creates a window function of the specified type.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="type">The type of window function to create.</param>
    /// <returns>An implementation of IWindowFunction<T> for the specified window function type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported window function type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different window functions have different shapes and properties, making them 
    /// suitable for different applications. Some provide better frequency resolution, while others provide 
    /// better amplitude accuracy or reduced spectral leakage.
    /// </para>
    /// <para>
    /// Available window function types include:
    /// <list type="bullet">
    /// <item><description>Rectangular: The simplest window, equivalent to no windowing. Good frequency resolution but poor amplitude accuracy.</description></item>
    /// <item><description>Hanning: A general-purpose window with good balance between frequency resolution and amplitude accuracy.</description></item>
    /// <item><description>Hamming: Similar to Hanning but with different coefficients, providing slightly better frequency resolution.</description></item>
    /// <item><description>Blackman: Provides excellent amplitude accuracy at the cost of frequency resolution.</description></item>
    /// <item><description>Kaiser: A flexible window with a parameter that allows trading off between frequency resolution and amplitude accuracy.</description></item>
    /// <item><description>Bartlett: A triangular window, simple but effective for many applications.</description></item>
    /// <item><description>Gaussian: Based on the Gaussian function, provides good time-frequency localization.</description></item>
    /// <item><description>BartlettHann: A combination of the Bartlett and Hann windows.</description></item>
    /// <item><description>Bohman: A window with good sidelobe suppression.</description></item>
    /// <item><description>Lanczos: Based on the sinc function, good for image processing.</description></item>
    /// <item><description>Parzen: A piecewise cubic approximation of the Gaussian window.</description></item>
    /// <item><description>Poisson: An exponential window, useful for certain types of spectral analysis.</description></item>
    /// <item><description>Nuttall: A high-quality window with excellent sidelobe suppression.</description></item>
    /// <item><description>Triangular: Similar to Bartlett but with slightly different definition.</description></item>
    /// <item><description>BlackmanHarris: An improved version of the Blackman window with better sidelobe suppression.</description></item>
    /// <item><description>FlatTop: Provides excellent amplitude accuracy, ideal for calibration.</description></item>
    /// <item><description>Welch: A parabolic window, useful for power spectrum estimation.</description></item>
    /// <item><description>BlackmanNuttall: A combination of the Blackman and Nuttall windows.</description></item>
    /// <item><description>Cosine: A simple cosine-based window.</description></item>
    /// <item><description>Tukey: A rectangular window with tapered ends, good for time-domain analysis.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IWindowFunction<T> CreateWindowFunction<T>(WindowFunctionType type)
    {
        return type switch
        {
            WindowFunctionType.Rectangular => new RectangularWindow<T>(),
            WindowFunctionType.Hanning => new HanningWindow<T>(),
            WindowFunctionType.Hamming => new HammingWindow<T>(),
            WindowFunctionType.Blackman => new BlackmanWindow<T>(),
            WindowFunctionType.Kaiser => new KaiserWindow<T>(),
            WindowFunctionType.Bartlett => new BartlettWindow<T>(),
            WindowFunctionType.Gaussian => new GaussianWindow<T>(),
            WindowFunctionType.BartlettHann => new BartlettHannWindow<T>(),
            WindowFunctionType.Bohman => new BohmanWindow<T>(),
            WindowFunctionType.Lanczos => new LanczosWindow<T>(),
            WindowFunctionType.Parzen => new ParzenWindow<T>(),
            WindowFunctionType.Poisson => new PoissonWindow<T>(),
            WindowFunctionType.Nuttall => new NuttallWindow<T>(),
            WindowFunctionType.Triangular => new TriangularWindow<T>(),
            WindowFunctionType.BlackmanHarris => new BlackmanHarrisWindow<T>(),
            WindowFunctionType.FlatTop => new FlatTopWindow<T>(),
            WindowFunctionType.Welch => new WelchWindow<T>(),
            WindowFunctionType.BlackmanNuttall => new BlackmanNuttallWindow<T>(),
            WindowFunctionType.Cosine => new CosineWindow<T>(),
            WindowFunctionType.Tukey => new TukeyWindow<T>(),
            _ => throw new ArgumentException("Unsupported window function type", nameof(type)),
        };
    }
}
