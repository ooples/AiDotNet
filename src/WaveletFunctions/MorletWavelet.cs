namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Morlet wavelet function implementation for time-frequency analysis and signal processing.
/// </summary>
/// <remarks>
/// <para>
/// The Morlet wavelet is a complex wavelet defined as a plane wave modulated by a Gaussian window.
/// It offers excellent time-frequency localization and is widely used in signal processing, geophysics,
/// audio analysis, and various other fields. This implementation supports customization of the central
/// frequency parameter, which affects the balance between time and frequency resolution.
/// </para>
/// <para><b>For Beginners:</b> The Morlet wavelet is like a tunable detector for patterns in your data.
/// 
/// Think of the Morlet wavelet like a musical note with an adjustable pitch that:
/// - Has a smooth bell-shaped envelope (like a Gaussian curve)
/// - Contains oscillations (like a cosine wave) inside this envelope
/// - Can be tuned to detect specific frequencies in your data
/// 
/// This wavelet is particularly good at analyzing signals where you need to know
/// both when something happens (time localization) and what frequencies are present
/// (frequency localization). It's commonly used for analyzing audio, brain waves (EEG),
/// vibrations, and many other types of signals.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MorletWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The central frequency parameter of the Morlet wavelet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the central frequency of the Morlet wavelet's oscillations.
    /// It affects the trade-off between time and frequency resolution. Higher values provide
    /// better frequency resolution but poorer time resolution, while lower values do the opposite.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the number of oscillations within the wavelet.
    /// 
    /// Think of omega like tuning a radio:
    /// - Higher values (more oscillations) help you distinguish between similar frequencies more clearly
    /// - Lower values (fewer oscillations) help you pinpoint exactly when something happens
    /// 
    /// The default value of 5 provides a balanced trade-off between these two goals.
    /// If you need to detect precise frequencies, use a higher value; if you need to detect
    /// exactly when something happens, use a lower value.
    /// </para>
    /// </remarks>
    private readonly T _omega;

    /// <summary>
    /// Provides Fast Fourier Transform capabilities for frequency domain analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of the Fast Fourier Transform (FFT) algorithm, which is used
    /// to convert signals between time and frequency domains. The Morlet wavelet can be applied in
    /// both domains, and FFT allows for efficient computation of the transform in the frequency domain.
    /// </para>
    /// <para><b>For Beginners:</b> This is a tool that converts data between time and frequency representations.
    /// 
    /// The Fast Fourier Transform (FFT):
    /// - Converts data from "how it changes over time" to "what frequencies it contains"
    /// - Makes certain calculations much faster and more efficient
    /// - Allows the wavelet to work in the frequency domain when that's more convenient
    /// 
    /// This is like being able to look at a piece of music either as sounds changing over time
    /// or as a collection of notes on a musical staff - two different ways of representing
    /// the same information.
    /// </para>
    /// </remarks>
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new instance of the <see cref="MorletWavelet{T}"/> class with the specified omega parameter.
    /// </summary>
    /// <param name="omega">The central frequency parameter. Defaults to 5.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Morlet wavelet with the specified central frequency parameter omega,
    /// which controls the trade-off between time and frequency resolution. It also initializes the
    /// numeric operations helper and Fast Fourier Transform implementation for the specified numeric type T.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Morlet wavelet with your chosen settings.
    /// 
    /// When creating a Morlet wavelet:
    /// - You can set the omega parameter to adjust how the wavelet analyzes your data
    /// - The default value (5) works well for many common applications
    /// - Higher values are better for frequency analysis, lower values for timing analysis
    /// 
    /// This is like choosing the right lens for a camera - you want to match the tool
    /// to the specific type of detail you're trying to capture in your data.
    /// </para>
    /// </remarks>
    public MorletWavelet(double omega = 5)
    {
        _omega = NumOps.FromDouble(omega);
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Calculates the Morlet wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated Morlet wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Morlet wavelet function at the given input point.
    /// The Morlet wavelet is defined as a cosine function modulated by a Gaussian envelope:
    /// ?(x) = cos(?·x) · exp(-x²/2), where ? is the central frequency parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the height of the Morlet wavelet at a specific point.
    /// 
    /// When you use this method:
    /// - You provide a point (x) on the horizontal axis
    /// - The method returns the height of the Morlet wavelet at that point
    /// - The result combines a wave pattern (cosine) with a bell curve (Gaussian)
    /// 
    /// This creates an oscillating pattern that's strongest in the center and fades out
    /// as you move away from the center. The number of oscillations visible depends on
    /// the omega parameter you set when creating the wavelet.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        T x2 = NumOps.Square(x);
        T cosine = MathHelper.Cos(NumOps.Multiply(_omega, x));
        T gaussian = NumOps.Exp(NumOps.Divide(NumOps.Negate(x2), NumOps.FromDouble(2)));
        return NumOps.Multiply(cosine, gaussian);
    }

    /// <summary>
    /// Decomposes an input signal using the Morlet wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Morlet wavelet transform, which decomposes the input signal into
    /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
    /// It uses the Fast Fourier Transform to convert the signal to the frequency domain, applies appropriate
    /// filters based on the Morlet wavelet, and then converts the results back to the time domain.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into low-frequency and high-frequency parts.
    /// 
    /// When decomposing a signal with the Morlet wavelet:
    /// - First, the data is converted to the frequency domain using FFT
    /// - Then, the frequencies are separated into "low" (approximation) and "high" (detail) components
    /// - Finally, these components are converted back to the time domain
    /// 
    /// This is like separating a music recording into bass (low frequencies) and treble (high frequencies).
    /// The approximation coefficients represent the slower changes in your data (like bass notes),
    /// while the detail coefficients represent the rapid changes (like treble notes).
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        // Perform FFT
        var spectrum = _fft.Forward(input);
        // Apply Morlet wavelet in frequency domain
        var scalingSpectrum = new Vector<Complex<T>>(size);
        var waveletSpectrum = new Vector<Complex<T>>(size);
        for (int i = 0; i < size; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size));
            Complex<T> psi = MorletFourierTransform(freq);

            // Low-pass (scaling) filter
            if (NumOps.LessThanOrEquals(NumOps.Abs(freq), NumOps.FromDouble(0.5)))
            {
                scalingSpectrum[i] = complexOps.Multiply(spectrum[i], psi);
            }
            // High-pass (wavelet) filter
            waveletSpectrum[i] = complexOps.Multiply(spectrum[i],
                complexOps.Subtract(new Complex<T>(NumOps.One, NumOps.Zero), psi));
        }
        // Perform inverse FFT
        var approximation = _fft.Inverse(scalingSpectrum);
        var detail = _fft.Inverse(waveletSpectrum);
        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients (low-pass filter) used in the Morlet wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients used in the Morlet wavelet transform, which
    /// represent the low-pass filter in the frequency domain. These coefficients are derived from
    /// the Fourier transform of the Morlet wavelet and are designed to extract low-frequency components
    /// from the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern used to extract low frequencies.
    /// 
    /// The scaling coefficients in the Morlet wavelet:
    /// - Act as a low-pass filter that keeps frequencies below a certain threshold
    /// - Are based on the Fourier transform of the Morlet wavelet
    /// - Have stronger values for low frequencies and weaker values for high frequencies
    /// 
    /// This is like an audio equalizer setting that emphasizes the bass notes while
    /// reducing the treble. When applied to your data, these coefficients help extract
    /// the overall trends and slower changes.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coeffs = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size));
            coeffs[i] = NumOps.LessThanOrEquals(NumOps.Abs(freq), NumOps.FromDouble(0.5))
                ? MorletFourierTransform(freq).Magnitude
                : NumOps.Zero;
        }

        return coeffs;
    }

    /// <summary>
    /// Gets the wavelet coefficients (high-pass filter) used in the Morlet wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients used in the Morlet wavelet transform, which
    /// represent the high-pass filter in the frequency domain. These coefficients are designed
    /// to complement the scaling coefficients, extracting high-frequency components from the
    /// signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern used to extract high frequencies.
    /// 
    /// The wavelet coefficients in the Morlet wavelet:
    /// - Act as a high-pass filter that keeps frequencies above a certain threshold
    /// - Are designed to complement the scaling coefficients
    /// - Have stronger values for high frequencies and weaker values for low frequencies
    /// 
    /// This is like an audio equalizer setting that emphasizes the treble notes while
    /// reducing the bass. When applied to your data, these coefficients help extract
    /// the fine details and rapid changes.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coeffs = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size));
            coeffs[i] = NumOps.Subtract(NumOps.One, MorletFourierTransform(freq).Magnitude);
        }

        return coeffs;
    }

    /// <summary>
    /// Calculates the Fourier transform of the Morlet wavelet at the specified frequency.
    /// </summary>
    /// <param name="omega">The frequency at which to calculate the Fourier transform.</param>
    /// <returns>The complex-valued Fourier transform of the Morlet wavelet.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Fourier transform of the Morlet wavelet at the given frequency.
    /// The transform is used in the frequency domain implementation of the wavelet transform,
    /// providing a more efficient way to apply the wavelet to signals in some cases.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method calculates how the Morlet wavelet behaves in the frequency domain.
    /// 
    /// The Fourier transform calculation:
    /// - Converts the Morlet wavelet from the time domain to the frequency domain
    /// - Shows how strongly the wavelet responds to different frequencies
    /// - Is used when applying the wavelet transform efficiently using FFT
    /// 
    /// While the Morlet wavelet is defined as an oscillation within a Gaussian envelope in the time domain,
    /// this method shows its equivalent representation in the frequency domain. This dual representation
    /// allows the wavelet to be applied efficiently in either domain, depending on the application.
    /// </para>
    /// </remarks>
    private Complex<T> MorletFourierTransform(T omega)
    {
        T term1 = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Multiply(omega, omega)));
        T term2 = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Multiply(NumOps.Subtract(omega, _omega), NumOps.Subtract(omega, _omega))));
        T real = NumOps.Subtract(term2, term1);

        return new Complex<T>(real, NumOps.Zero);
    }
}
