using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// Window functions for audio signal processing (STFT, spectral analysis).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Window functions are used to reduce spectral leakage when performing
/// Fourier transforms on finite-length signals. They smoothly taper the
/// signal at the edges to zero.
/// </para>
/// <para>
/// <b>For Beginners:</b> When analyzing a piece of audio, we cut it into
/// small chunks called "frames". If we just cut abruptly, we get artifacts
/// at the edges. Window functions smoothly fade in and out to prevent this.
///
/// Common windows:
/// - Hann: Good general purpose, commonly used in audio
/// - Hamming: Similar to Hann but doesn't go to zero at edges
/// - Blackman: Better frequency selectivity but wider main lobe
/// - Kaiser: Adjustable trade-off between main lobe and side lobes
///
/// Usage:
/// ```csharp
/// var window = WindowFunctions&lt;float&gt;.CreateHann(2048);
/// for (int i = 0; i &lt; frame.Length; i++)
///     windowedFrame[i] = frame[i] * window[i];
/// ```
/// </para>
/// </remarks>
public static class WindowFunctions<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a Hann (Hanning) window.
    /// </summary>
    /// <param name="length">Window length in samples.</param>
    /// <param name="periodic">If true, generates a periodic window for spectral analysis.</param>
    /// <returns>Array containing the window coefficients.</returns>
    /// <remarks>
    /// <para>
    /// The Hann window is defined as:
    /// w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    ///
    /// <b>For Beginners:</b> The Hann window is one of the most commonly used windows
    /// in audio processing. It has good frequency resolution and low spectral leakage.
    /// Named after Julius von Hann, an Austrian meteorologist.
    /// </para>
    /// </remarks>
    public static T[] CreateHann(int length, bool periodic = true)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");

        var window = new T[length];
        int N = periodic ? length : length - 1;

        for (int n = 0; n < length; n++)
        {
            double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * n / N));
            window[n] = NumOps.FromDouble(w);
        }

        return window;
    }

    /// <summary>
    /// Creates a Hamming window.
    /// </summary>
    /// <param name="length">Window length in samples.</param>
    /// <param name="periodic">If true, generates a periodic window for spectral analysis.</param>
    /// <returns>Array containing the window coefficients.</returns>
    /// <remarks>
    /// <para>
    /// The Hamming window is defined as:
    /// w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
    ///
    /// <b>For Beginners:</b> The Hamming window is similar to Hann but doesn't go
    /// completely to zero at the edges. This can reduce discontinuities but at the
    /// cost of slightly higher side lobes in the frequency domain.
    /// </para>
    /// </remarks>
    public static T[] CreateHamming(int length, bool periodic = true)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");

        var window = new T[length];
        int N = periodic ? length : length - 1;

        for (int n = 0; n < length; n++)
        {
            double w = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * n / N);
            window[n] = NumOps.FromDouble(w);
        }

        return window;
    }

    /// <summary>
    /// Creates a Blackman window.
    /// </summary>
    /// <param name="length">Window length in samples.</param>
    /// <param name="periodic">If true, generates a periodic window for spectral analysis.</param>
    /// <returns>Array containing the window coefficients.</returns>
    /// <remarks>
    /// <para>
    /// The Blackman window is defined as:
    /// w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
    ///
    /// <b>For Beginners:</b> The Blackman window has lower side lobes than Hann or
    /// Hamming, making it better at isolating nearby frequencies. However, it has
    /// a wider main lobe, which reduces frequency resolution.
    /// </para>
    /// </remarks>
    public static T[] CreateBlackman(int length, bool periodic = true)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");

        var window = new T[length];
        int N = periodic ? length : length - 1;

        for (int n = 0; n < length; n++)
        {
            double w = 0.42 - 0.5 * Math.Cos(2.0 * Math.PI * n / N)
                           + 0.08 * Math.Cos(4.0 * Math.PI * n / N);
            window[n] = NumOps.FromDouble(w);
        }

        return window;
    }

    /// <summary>
    /// Creates a Kaiser window with specified beta parameter.
    /// </summary>
    /// <param name="length">Window length in samples.</param>
    /// <param name="beta">Shape parameter (higher = narrower main lobe, lower side lobes).</param>
    /// <returns>Array containing the window coefficients.</returns>
    /// <remarks>
    /// <para>
    /// The Kaiser window is defined using the zeroth-order modified Bessel function:
    /// w[n] = I0(beta * sqrt(1 - ((n - M/2) / (M/2))^2)) / I0(beta)
    ///
    /// <b>For Beginners:</b> The Kaiser window is adjustable - by changing beta you can
    /// trade off between frequency resolution (lower beta) and side lobe suppression
    /// (higher beta). Common values:
    /// - beta = 0: Rectangular window
    /// - beta = 5: Similar to Hamming
    /// - beta = 6: Similar to Hann
    /// - beta = 8.6: Similar to Blackman
    /// </para>
    /// </remarks>
    public static T[] CreateKaiser(int length, double beta = 5.0)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");

        var window = new T[length];
        double M = length - 1;
        double denominator = BesselI0(beta);

        for (int n = 0; n < length; n++)
        {
            double term = (2.0 * n / M) - 1.0;
            double arg = beta * Math.Sqrt(Math.Max(0, 1.0 - term * term));
            window[n] = NumOps.FromDouble(BesselI0(arg) / denominator);
        }

        return window;
    }

    /// <summary>
    /// Creates a rectangular (no window) function.
    /// </summary>
    /// <param name="length">Window length in samples.</param>
    /// <returns>Array containing all ones.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A rectangular window is essentially "no window" - it doesn't
    /// modify the signal at all. While this preserves signal amplitude, it causes significant
    /// spectral leakage and is generally not recommended for spectral analysis.
    /// </para>
    /// </remarks>
    public static T[] CreateRectangular(int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");

        var window = new T[length];
        for (int n = 0; n < length; n++)
        {
            window[n] = NumOps.One;
        }

        return window;
    }

    /// <summary>
    /// Creates a triangular (Bartlett) window.
    /// </summary>
    /// <param name="length">Window length in samples.</param>
    /// <returns>Array containing the window coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The triangular window linearly ramps up to the center
    /// and then linearly ramps down. It's simple but has higher side lobes than
    /// Hann or Hamming windows.
    /// </para>
    /// </remarks>
    public static T[] CreateTriangular(int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");

        var window = new T[length];
        double center = (length - 1) / 2.0;

        for (int n = 0; n < length; n++)
        {
            double w = 1.0 - Math.Abs((n - center) / center);
            window[n] = NumOps.FromDouble(w);
        }

        return window;
    }

    /// <summary>
    /// Computes the zeroth-order modified Bessel function of the first kind.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>I0(x).</returns>
    /// <remarks>
    /// Uses a polynomial approximation that is accurate for all x.
    /// </remarks>
    private static double BesselI0(double x)
    {
        double ax = Math.Abs(x);

        if (ax < 3.75)
        {
            double y = (x / 3.75) * (x / 3.75);
            return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
        }
        else
        {
            double y = 3.75 / ax;
            return (Math.Exp(ax) / Math.Sqrt(ax)) * (0.39894228 + y * (0.01328592
                + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
                + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633
                + y * 0.00392377))))))));
        }
    }

    /// <summary>
    /// Creates a window function by type.
    /// </summary>
    /// <param name="type">Type of window to create.</param>
    /// <param name="length">Window length in samples.</param>
    /// <param name="beta">Beta parameter for Kaiser window (ignored for other types).</param>
    /// <returns>Array containing the window coefficients.</returns>
    public static T[] Create(WindowType type, int length, double beta = 5.0)
    {
        return type switch
        {
            WindowType.Hann => CreateHann(length),
            WindowType.Hamming => CreateHamming(length),
            WindowType.Blackman => CreateBlackman(length),
            WindowType.Kaiser => CreateKaiser(length, beta),
            WindowType.Rectangular => CreateRectangular(length),
            WindowType.Triangular => CreateTriangular(length),
            _ => CreateHann(length)
        };
    }
}

/// <summary>
/// Types of window functions available for spectral analysis.
/// </summary>
public enum WindowType
{
    /// <summary>
    /// Hann (Hanning) window - good general purpose.
    /// </summary>
    Hann,

    /// <summary>
    /// Hamming window - similar to Hann but non-zero at edges.
    /// </summary>
    Hamming,

    /// <summary>
    /// Blackman window - lower side lobes but wider main lobe.
    /// </summary>
    Blackman,

    /// <summary>
    /// Kaiser window - adjustable shape parameter.
    /// </summary>
    Kaiser,

    /// <summary>
    /// Rectangular window - no windowing (not recommended for spectral analysis).
    /// </summary>
    Rectangular,

    /// <summary>
    /// Triangular (Bartlett) window.
    /// </summary>
    Triangular
}
