namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Shannon wavelet function implementation for signal processing and frequency analysis.
/// </summary>
/// <remarks>
/// <para>
/// The Shannon wavelet is a band-limited wavelet that is perfectly localized in the frequency domain.
/// It is defined as the product of a sinc function and a cosine modulation. This wavelet is particularly
/// useful for signal analysis where precise frequency localization is required, though it has poor
/// time localization due to the slow decay of the sinc function.
/// </para>
/// <para><b>For Beginners:</b> The Shannon wavelet is like a specialized frequency analyzer.
/// 
/// Think of the Shannon wavelet as a musical tuning fork that:
/// - Can precisely identify specific frequencies in your data
/// - Has perfect frequency localization (it knows exactly which frequencies are present)
/// - Is less precise about when those frequencies occur in the signal
/// 
/// This wavelet is particularly useful when you need to know exactly which frequencies are
/// in your data, but don't need to know precisely when they occur. It's like having a
/// perfect pitch detector that can tell you exactly which notes are being played, but is
/// less precise about when each note starts and stops.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ShannonWavelet<T> : WaveletFunctionBase<T>
{

    /// <summary>
    /// Initializes a new instance of the <see cref="ShannonWavelet{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Shannon wavelet and sets up the numeric operations helper
    /// for the specified numeric type T. The Shannon wavelet doesn't require additional parameters
    /// as it has a fixed, well-defined mathematical form.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Shannon wavelet for use.
    /// 
    /// When creating a Shannon wavelet:
    /// - No parameters are needed because it has a fixed mathematical definition
    /// - It's ready to use immediately after creation
    /// 
    /// The Shannon wavelet has a specific mathematical form that doesn't need customization.
    /// This makes it simple to set up, but it also means you can't adjust its properties
    /// to better match specific types of data as you can with some other wavelets.
    /// </para>
    /// </remarks>
    public ShannonWavelet()
    {
    }

    /// <summary>
    /// Calculates the Shannon wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated Shannon wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Shannon wavelet function at the given input point.
    /// The Shannon wavelet is defined as sinc(x) * cos(x/2), where sinc(x) = sin(x)/x for x ? 0
    /// and sinc(0) = 1. This function has perfect frequency localization but poor time localization.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the height of the Shannon wavelet at a specific point.
    /// 
    /// When you use this method:
    /// - You provide a point (x) on the horizontal axis
    /// - The method returns the height of the Shannon wavelet at that point
    /// - It combines a sinc function (sin(x)/x) with a cosine modulation
    /// 
    /// The sinc function gives the wavelet its frequency properties, while the
    /// cosine modulation shifts the frequency band to be centered around a non-zero frequency.
    /// This creates a wavelet that's excellent at isolating specific frequency bands.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        if (NumOps.Equals(x, NumOps.Zero))
            return NumOps.One;
        T sinc = NumOps.Divide(MathHelper.Sin(x), x);
        T cos = MathHelper.Cos(NumOps.Divide(x, NumOps.FromDouble(2)));

        return NumOps.Multiply(sinc, cos);
    }

    /// <summary>
    /// Decomposes an input signal using the Shannon wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Shannon wavelet transform, which decomposes the input signal into
    /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
    /// It uses the Fast Fourier Transform to implement the transform efficiently in the frequency domain,
    /// which is natural for the Shannon wavelet since it has a simple rectangular spectrum.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into low-frequency and high-frequency parts.
    /// 
    /// When decomposing a signal with the Shannon wavelet:
    /// - First, the data is converted to the frequency domain using FFT
    /// - Then, the frequencies are simply split into two halves
    /// - Low frequencies go into the approximation, high frequencies into the detail
    /// - The results are converted back to the time domain
    /// 
    /// This is like having two perfect filters: one that keeps all the bass notes and
    /// nothing else, and another that keeps all the treble notes and nothing else.
    /// The Shannon wavelet is unique because it makes this perfect frequency separation possible.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int n = input.Length;
        var approximation = new Vector<T>((n + 1) / 2);
        var detail = new Vector<T>((n + 1) / 2);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        // Perform FFT
        var fft = new FastFourierTransform<T>();
        var spectrum = fft.Forward(input);
        int halfN = n / 2;
        var approxSpectrum = new Vector<Complex<T>>(halfN + 1);
        var detailSpectrum = new Vector<Complex<T>>(halfN + 1);
        for (int i = 0; i < halfN; i++)
        {
            approxSpectrum[i] = spectrum[i];
            detailSpectrum[i] = spectrum[halfN + i];
        }

        // If n is odd, handle the middle frequency
        if (n % 2 != 0)
        {
            T scaleFactor = NumOps.FromDouble(Math.Sqrt(0.5));
            Complex<T> complexScaleFactor = new Complex<T>(scaleFactor, NumOps.Zero);
            approxSpectrum[halfN] = complexOps.Multiply(spectrum[halfN], complexScaleFactor);
            detailSpectrum[halfN] = complexOps.Multiply(spectrum[halfN], complexScaleFactor);
        }
        // Perform inverse FFT on both approximation and detail
        var approxResult = fft.Inverse(approxSpectrum);
        var detailResult = fft.Inverse(detailSpectrum);

        return (approxResult, detailResult);
    }

    /// <summary>
    /// Reconstructs the original signal from approximation and detail coefficients.
    /// </summary>
    /// <param name="approximation">The approximation coefficients from decomposition.</param>
    /// <param name="detail">The detail coefficients from decomposition.</param>
    /// <returns>The reconstructed signal.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method reverses the decomposition process to get back the original signal.
    ///
    /// For Shannon wavelets, reconstruction combines the frequency bands:
    /// 1. Transform both approximation and detail to frequency domain using FFT
    /// 2. Place approximation spectrum in the low frequency half
    /// 3. Place detail spectrum in the high frequency half
    /// 4. Transform the combined spectrum back to time domain
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        var fft = new FastFourierTransform<T>();

        // Transform both to frequency domain
        Vector<Complex<T>> approxSpectrum = fft.Forward(approximation);
        Vector<Complex<T>> detailSpectrum = fft.Forward(detail);

        // Combine spectrums: approximation in low frequencies, detail in high frequencies
        int outputLength = approximation.Length + detail.Length;
        var combined = new Vector<Complex<T>>(outputLength);

        int halfN = outputLength / 2;
        for (int i = 0; i < approxSpectrum.Length && i < halfN; i++)
        {
            combined[i] = approxSpectrum[i];
        }
        for (int i = 0; i < detailSpectrum.Length && halfN + i < outputLength; i++)
        {
            combined[halfN + i] = detailSpectrum[i];
        }

        // Transform back to time domain
        return fft.Inverse(combined);
    }

    /// <summary>
    /// Gets the scaling coefficients used in the Shannon wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients used in the Shannon wavelet transform, which
    /// represent the low-pass filter in the frequency domain. For the Shannon wavelet, these
    /// coefficients form an ideal low-pass filter with a sharp cutoff at half the Nyquist frequency.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the values used to extract low frequencies.
    /// 
    /// The scaling coefficients in the Shannon wavelet:
    /// - Form an ideal low-pass filter in the frequency domain
    /// - Are based on the sinc function, which is the Fourier transform of a rectangular window
    /// - Keep all frequencies below a certain threshold and completely reject all frequencies above it
    /// 
    /// This is like a perfect audio equalizer that can completely separate bass from treble
    /// without any overlap or distortion. In most real-world filters, there's always some
    /// blurring at the boundary, but the Shannon wavelet achieves a theoretically perfect separation.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int n = 1024;
        var coeffs = new T[n];
        T twoPi = NumOps.Multiply(NumOps.FromDouble(2), MathHelper.Pi<T>());
        for (int k = -n / 2; k < n / 2; k++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(k), NumOps.FromDouble(n));
            if (k == 0)
            {
                coeffs[k + n / 2] = NumOps.FromDouble(Math.Sqrt(0.5));
            }
            else
            {
                T sinc = NumOps.Divide(MathHelper.Sin(NumOps.Multiply(twoPi, x)), NumOps.Multiply(twoPi, x));
                coeffs[k + n / 2] = NumOps.Multiply(NumOps.FromDouble(Math.Sqrt(0.5)), sinc);
            }
        }

        return new Vector<T>(coeffs);
    }

    /// <summary>
    /// Gets the wavelet coefficients used in the Shannon wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients used in the Shannon wavelet transform, which
    /// represent the high-pass filter in the frequency domain. For the Shannon wavelet, these
    /// coefficients form an ideal high-pass filter with a sharp cutoff at half the Nyquist frequency.
    /// The coefficients are modulated by a cosine to shift the passband to the appropriate frequency range.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the values used to extract high frequencies.
    /// 
    /// The wavelet coefficients in the Shannon wavelet:
    /// - Form an ideal high-pass filter in the frequency domain
    /// - Are based on the sinc function modulated by a cosine
    /// - Keep all frequencies above a certain threshold and completely reject all frequencies below it
    /// 
    /// This is the complement to the scaling coefficients - together they form a perfect
    /// pair of filters that split the frequency spectrum exactly in half with no overlap or gap.
    /// This perfect separation is a unique property of the Shannon wavelet.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int n = 1024;
        var coeffs = new T[n];
        T twoPi = NumOps.Multiply(NumOps.FromDouble(2), MathHelper.Pi<T>());
        for (int k = -n / 2; k < n / 2; k++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(k), NumOps.FromDouble(n));
            if (k == 0)
            {
                coeffs[k + n / 2] = NumOps.FromDouble(Math.Sqrt(0.5));
            }
            else
            {
                T sinc = NumOps.Divide(MathHelper.Sin(NumOps.Multiply(twoPi, x)), NumOps.Multiply(twoPi, x));
                T modulation = MathHelper.Cos(NumOps.Multiply(twoPi, x));
                coeffs[k + n / 2] = NumOps.Multiply(NumOps.FromDouble(Math.Sqrt(0.5)), NumOps.Multiply(sinc, modulation));
            }
        }

        return new Vector<T>(coeffs);
    }
}
