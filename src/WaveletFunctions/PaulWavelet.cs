namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Paul wavelet function implementation for complex signal analysis and processing.
/// </summary>
/// <remarks>
/// <para>
/// The Paul wavelet is a complex-valued wavelet that belongs to the family of analytic wavelets.
/// It is particularly useful for analyzing oscillatory behaviors and phase information in signals.
/// The Paul wavelet is defined in terms of complex functions and provides good time resolution
/// with moderate frequency resolution, making it suitable for extracting instantaneous phases
/// and detecting transient phenomena in signals.
/// </para>
/// <para><b>For Beginners:</b> The Paul wavelet is a special mathematical tool for analyzing phases and oscillations in data.
/// 
/// Think of the Paul wavelet like a specialized detector that:
/// - Can identify not just when oscillations occur, but also their phase (position in the cycle)
/// - Is particularly good at finding short-lived patterns in your data
/// - Works with complex numbers (numbers with both real and imaginary parts)
/// 
/// This wavelet is especially useful when you need to track how phases change over time,
/// such as in brain wave analysis, fluid dynamics, or geophysical signal processing.
/// Unlike some other wavelets, the Paul wavelet is asymmetric, which makes it particularly
/// sensitive to the direction of changes in your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class PaulWavelet<T> : WaveletFunctionBase<T>
{

    /// <summary>
    /// The order parameter that controls the properties of the Paul wavelet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the order of the Paul wavelet, which affects its time-frequency
    /// resolution characteristics. Higher orders provide better frequency resolution but poorer
    /// time resolution, while lower orders do the opposite.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how precisely the wavelet can identify frequencies versus timing.
    /// 
    /// Think of the order parameter like adjusting the focus on a camera:
    /// - Lower values (2-4): Better at pinpointing exactly when something happens, but less precise about frequencies
    /// - Higher values (6-8+): Better at distinguishing between similar frequencies, but less precise about timing
    /// 
    /// The default value of 4 provides a balanced approach that works well for many applications.
    /// You can adjust this value based on whether you care more about when something happens or
    /// exactly what frequencies are present.
    /// </para>
    /// </remarks>
    private readonly int _order;

    /// <summary>
    /// Provides Fast Fourier Transform capabilities for frequency domain analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of the Fast Fourier Transform (FFT) algorithm, which is used
    /// to convert signals between time and frequency domains. The Paul wavelet can be applied in
    /// both domains, and FFT allows for efficient computation of the transform in the frequency domain.
    /// </para>
    /// <para><b>For Beginners:</b> This is a tool that converts data between time and frequency representations.
    /// 
    /// The Fast Fourier Transform (FFT):
    /// - Converts data from "how it changes over time" to "what frequencies it contains"
    /// - Makes certain calculations much faster and more efficient
    /// - Allows the wavelet to work in the frequency domain when that's more convenient
    /// 
    /// This is like having a special translator that can convert a piece of music from
    /// sound waves to musical notes and back again. Each representation has advantages
    /// for different types of analysis.
    /// </para>
    /// </remarks>
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new instance of the <see cref="PaulWavelet{T}"/> class with the specified order.
    /// </summary>
    /// <param name="order">The order of the Paul wavelet. Defaults to 4.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Paul wavelet with the specified order parameter,
    /// which controls the time-frequency resolution characteristics of the wavelet.
    /// The default value of 4 provides a balanced wavelet suitable for general-purpose analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Paul wavelet with your chosen settings.
    /// 
    /// When creating a Paul wavelet:
    /// - The order parameter controls the balance between time and frequency precision
    /// - The default value (4) works well for many common applications
    /// - You can experiment with different values to match your specific needs
    /// 
    /// For example, if you're analyzing brain waves and need to know exactly when certain
    /// patterns occur, you might use a lower order. If you're more interested in identifying
    /// precise frequencies in a vibration analysis, you might use a higher order.
    /// </para>
    /// </remarks>
    public PaulWavelet(int order = 4)
    {
        _fft = new FastFourierTransform<T>();
        _order = order;
    }

    /// <summary>
    /// Calculates the Paul wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated Paul wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Paul wavelet function at the given input point.
    /// The Paul wavelet is defined as a complex-valued function with specific properties that
    /// make it suitable for analyzing phase information in signals. The result may be complex,
    /// but is converted to the appropriate real or complex representation depending on the context.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the value of the Paul wavelet at a specific point.
    /// 
    /// When you use this method:
    /// - You provide a point (x) on the horizontal axis
    /// - The method returns the value of the Paul wavelet at that point
    /// - The calculation involves complex numbers (numbers with both real and imaginary parts)
    /// 
    /// The Paul wavelet has a specific mathematical formula that makes it especially
    /// good at detecting oscillatory patterns and phase changes in your data. The result
    /// may be a complex number, which captures both amplitude and phase information.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        Complex<T> i = new Complex<T>(NumOps.Zero, NumOps.One);
        Complex<T> xComplex = new Complex<T>(x, NumOps.Zero);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Calculate (2^m * i^m * m!) / sqrt(p * (2m)!)
        double m = _order;
        double numerator = Math.Pow(2, m) * Convert.ToDouble(MathHelper.Factorial<T>(_order));
        double denominator = Math.Sqrt(Math.PI * Convert.ToDouble(MathHelper.Factorial<T>(2 * _order)));
        Complex<T> factor = complexOps.Power(i, complexOps.FromDouble(_order)) * complexOps.FromDouble(numerator / denominator);

        // Calculate (1 - ix)^(-m-1)
        Complex<T> base_term = new Complex<T>(NumOps.One, NumOps.Zero) - i * xComplex;
        T exponentT = NumOps.FromDouble(-m - 1);
        Complex<T> denominator_term = complexOps.Power(base_term, new Complex<T>(exponentT, exponentT));
        Complex<T> result = factor * denominator_term;

        return result.ToRealOrComplex();
    }

    /// <summary>
    /// Decomposes an input signal using the Paul wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Paul wavelet transform, which decomposes the input signal into
    /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
    /// It uses the Fast Fourier Transform to convert the signal to the frequency domain, applies appropriate
    /// filters based on the Paul wavelet, and then converts the results back to the time domain.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into low-frequency and high-frequency parts.
    /// 
    /// When decomposing a signal with the Paul wavelet:
    /// - First, the data is converted to the frequency domain using FFT
    /// - Then, the frequencies are separated into "low" (approximation) and "high" (detail) components
    /// - Finally, these components are converted back to the time domain
    /// 
    /// This is like analyzing a complex sound by separating it into its fundamental tone (approximation)
    /// and its overtones (detail). The approximation captures the general trend of your data,
    /// while the detail captures the rapid changes and fine structures.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Perform FFT
        var spectrum = _fft.Forward(input);

        // Apply Paul wavelet in frequency domain
        var scalingSpectrum = new Vector<Complex<T>>(size);
        var waveletSpectrum = new Vector<Complex<T>>(size);

        for (int i = 0; i < size; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size));
            Complex<T> psi = PaulFourierTransform(freq);

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
    /// Gets the scaling coefficients (low-pass filter) used in the Paul wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients used in the Paul wavelet transform, which
    /// represent the low-pass filter in the frequency domain. These coefficients are derived from
    /// the Fourier transform of the Paul wavelet and are designed to extract low-frequency components
    /// from the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern used to extract low frequencies.
    /// 
    /// The scaling coefficients in the Paul wavelet:
    /// - Act as a low-pass filter that keeps frequencies below a certain threshold
    /// - Are based on the Fourier transform of the Paul wavelet
    /// - Have stronger values for low frequencies and zero values for high frequencies
    /// 
    /// This is like a filter that lets you focus on the slow, gradual changes in your data
    /// while ignoring the rapid fluctuations. These coefficients help extract the overall
    /// trends and patterns that evolve slowly over time.
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
                ? PaulFourierTransform(freq).Magnitude
                : NumOps.Zero;
        }

        return coeffs;
    }

    /// <summary>
    /// Gets the wavelet coefficients (high-pass filter) used in the Paul wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients used in the Paul wavelet transform, which
    /// represent the high-pass filter in the frequency domain. These coefficients are designed
    /// to complement the scaling coefficients, extracting high-frequency components from the
    /// signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern used to extract high frequencies.
    /// 
    /// The wavelet coefficients in the Paul wavelet:
    /// - Act as a high-pass filter that keeps rapid changes and fine details
    /// - Are designed to complement the scaling coefficients
    /// - Have stronger values for high frequencies and weaker values for low frequencies
    /// 
    /// This is like a filter that lets you focus on the quick changes and subtle variations
    /// in your data. These coefficients help extract the details, transitions, and short-lived
    /// patterns that might be hidden within the broader trends.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coeffs = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size));
            coeffs[i] = NumOps.Subtract(NumOps.One, PaulFourierTransform(freq).Magnitude);
        }

        return coeffs;
    }

    /// <summary>
    /// Calculates the Fourier transform of the Paul wavelet at the specified frequency.
    /// </summary>
    /// <param name="omega">The frequency at which to calculate the Fourier transform.</param>
    /// <returns>The complex-valued Fourier transform of the Paul wavelet.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Fourier transform of the Paul wavelet at the given frequency.
    /// The transform is used in the frequency domain implementation of the wavelet transform,
    /// providing a more efficient way to apply the wavelet to signals in some cases.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method calculates how the Paul wavelet behaves in the frequency domain.
    /// 
    /// The Fourier transform calculation:
    /// - Converts the Paul wavelet from the time domain to the frequency domain
    /// - Has a specific mathematical form that makes it efficient for filtering
    /// - Returns zero for negative frequencies (the Paul wavelet is analytic)
    /// 
    /// This method is particularly important for the internal workings of the wavelet transform.
    /// The Paul wavelet has the special property that its Fourier transform is zero for negative
    /// frequencies, which makes it an "analytic wavelet" - a property that makes it especially
    /// good at analyzing phase information.
    /// </para>
    /// </remarks>
    private Complex<T> PaulFourierTransform(T omega)
    {
        if (NumOps.LessThanOrEquals(omega, NumOps.Zero))
        {
            return new Complex<T>(NumOps.Zero, NumOps.Zero);
        }

        T factor = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Power(omega, NumOps.FromDouble(_order)));
        T expTerm = NumOps.Exp(NumOps.Negate(omega));

        T real = NumOps.Multiply(factor, expTerm);
        T imag = NumOps.Zero;

        return new Complex<T>(real, imag);
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
    /// For Paul wavelets, reconstruction combines the frequency components:
    /// 1. Transform both approximation and detail to frequency domain using FFT
    /// 2. Add the spectrums together (since decomposition split them)
    /// 3. Transform the combined spectrum back to time domain
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        int size = approximation.Length;
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Transform both to frequency domain
        Vector<Complex<T>> approxSpectrum = _fft.Forward(approximation);
        Vector<Complex<T>> detailSpectrum = _fft.Forward(detail);

        // Combine spectrums
        var combined = new Vector<Complex<T>>(size);
        for (int i = 0; i < size; i++)
        {
            combined[i] = complexOps.Add(approxSpectrum[i], detailSpectrum[i]);
        }

        // Transform back to time domain
        return _fft.Inverse(combined);
    }
}
