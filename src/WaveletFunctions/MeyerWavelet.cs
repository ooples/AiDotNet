namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Meyer wavelet function implementation for frequency domain analysis and signal processing.
/// </summary>
/// <remarks>
/// <para>
/// The Meyer wavelet is a frequency domain wavelet that is infinitely differentiable with compact
/// support in the frequency domain. Unlike many other wavelets that are defined primarily in the
/// time domain, the Meyer wavelet is more naturally defined in the frequency domain, making it
/// particularly useful for spectral analysis. This implementation uses Fast Fourier Transform (FFT)
/// for efficient computation.
/// </para>
/// <para><b>For Beginners:</b> The Meyer wavelet is a special mathematical tool for analyzing signals in the frequency domain.
/// 
/// Think of the Meyer wavelet like a musical tuning fork that:
/// - Works especially well for analyzing what frequencies are present in your data
/// - Has very clean frequency separation (it doesn't mix up different frequencies)
/// - Is infinitely smooth, which makes it good for analyzing continuous signals
/// 
/// Unlike other wavelets that work directly with your data points (time domain), the Meyer wavelet
/// works with the frequencies in your data (frequency domain). This is like looking at a piece of music
/// as a collection of different notes played simultaneously, rather than as sounds that change over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MeyerWavelet<T> : WaveletFunctionBase<T>
{

    /// <summary>
    /// Provides Fast Fourier Transform capabilities for frequency domain analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of the Fast Fourier Transform (FFT) algorithm, which is used
    /// to convert signals between time and frequency domains. The Meyer wavelet operates primarily
    /// in the frequency domain, so FFT is used to transform input signals for processing and then
    /// transform the results back to the time domain.
    /// </para>
    /// <para><b>For Beginners:</b> This is a tool that converts data between time and frequency representations.
    /// 
    /// The Fast Fourier Transform (FFT):
    /// - Is like a translator between two different ways of looking at the same data
    /// - Converts data from "how it changes over time" to "what frequencies it contains"
    /// - Allows the Meyer wavelet to work in the frequency domain, where it's most natural
    /// 
    /// This is similar to how sheet music is another way of representing a song - the same information
    /// is there, but organized by notes (frequencies) rather than by sounds changing over time.
    /// </para>
    /// </remarks>
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new instance of the <see cref="MeyerWavelet{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Meyer wavelet and sets up the numeric operations helper
    /// and Fast Fourier Transform implementation for the specified numeric type T. Unlike other wavelets,
    /// the Meyer wavelet doesn't require additional parameters as it has a fixed, well-defined form
    /// in the frequency domain.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Meyer wavelet for use.
    /// 
    /// When creating a Meyer wavelet:
    /// - No parameters are needed because the Meyer wavelet has a fixed mathematical definition
    /// - It initializes the mathematical helpers needed for calculations
    /// - It prepares the FFT tool that will convert between time and frequency domains
    /// 
    /// This is like setting up your workbench with all the tools you'll need before starting a project.
    /// Once created, the wavelet is immediately ready to use for signal analysis.
    /// </para>
    /// </remarks>
    public MeyerWavelet()
    {
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Calculates the Meyer wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated Meyer wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Meyer wavelet function at the given input point in the
    /// time domain. The Meyer wavelet is defined piecewise based on the magnitude of the input value,
    /// with different formulas applied for different ranges. The result is a function that is
    /// infinitely differentiable yet has compact support in the frequency domain.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the value of the Meyer wavelet at a specific point.
    /// 
    /// The Meyer wavelet function:
    /// - Has a complex shape defined by mathematical formulas
    /// - Returns different values depending on where the input point falls
    /// - Is zero for many input values (outside its support region)
    /// 
    /// While the Meyer wavelet is more naturally defined in the frequency domain, this method
    /// gives you its value in the time domain at a specific point. This is useful for visualizing
    /// the wavelet or for direct application to signals in the time domain.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        return NumOps.FromDouble(MeyerFunction(t));
    }

    /// <summary>
    /// Decomposes an input signal using the Meyer wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Meyer wavelet transform, which decomposes the input signal into
    /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
    /// It uses the Fast Fourier Transform to convert the signal to the frequency domain, applies the Meyer
    /// wavelet filters, and then converts the results back to the time domain.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into low-frequency and high-frequency parts.
    /// 
    /// When decomposing a signal with the Meyer wavelet:
    /// - First, the data is converted to the frequency domain using FFT
    /// - Then, the frequencies are separated into "low" (approximation) and "high" (detail) components
    /// - Finally, these components are converted back to the time domain
    /// 
    /// This is like separating the bass notes from the treble notes in a piece of music.
    /// The approximation coefficients represent the bass notes (slow changes, overall trends),
    /// while the detail coefficients represent the treble notes (quick changes, fine details).
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;

        // Perform FFT on the input
        Vector<Complex<T>> spectrum = _fft.Forward(input);

        // Apply Meyer wavelet in frequency domain
        var lowPass = new Vector<Complex<T>>(size);
        var highPass = new Vector<Complex<T>>(size);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        for (int i = 0; i < size; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(size));
            if (NumOps.LessThanOrEquals(freq, NumOps.FromDouble(1.0 / 3)))
            {
                lowPass[i] = spectrum[i];
            }
            else if (NumOps.LessThanOrEquals(freq, NumOps.FromDouble(2.0 / 3)))
            {
                T v = NumOps.Multiply(NumOps.FromDouble(Math.PI), NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(3), freq), NumOps.One));
                T psi = NumOps.FromDouble(Math.Cos(Math.PI / 2 * Vf(Convert.ToDouble(v))));
                Complex<T> complexPsi = new Complex<T>(psi, NumOps.Zero);
                lowPass[i] = complexOps.Multiply(spectrum[i], complexPsi);
                T sqrtTerm = NumOps.Sqrt(NumOps.Subtract(NumOps.One, NumOps.Multiply(psi, psi)));
                Complex<T> complexSqrtTerm = new Complex<T>(sqrtTerm, NumOps.Zero);
                highPass[i] = complexOps.Multiply(spectrum[i], complexSqrtTerm);
            }
            else if (NumOps.LessThanOrEquals(freq, NumOps.FromDouble(4.0 / 3)))
            {
                T v = NumOps.Multiply(NumOps.FromDouble(Math.PI), NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(3.0 / 2), freq), NumOps.One));
                T psi = NumOps.FromDouble(Math.Sin(Math.PI / 2 * Vf(Convert.ToDouble(v))));
                Complex<T> complexPsi = new Complex<T>(psi, NumOps.Zero);
                highPass[i] = complexOps.Multiply(spectrum[i], complexPsi);
            }
        }

        // Perform inverse FFT
        Vector<T> approximation = _fft.Inverse(lowPass);
        Vector<T> detail = _fft.Inverse(highPass);

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients (low-pass filter) used in the Meyer wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients used in the Meyer wavelet transform, which
    /// represent the low-pass filter in the frequency domain. These coefficients follow a specific
    /// pattern with a flat region for low frequencies and a smooth transition band that ensures
    /// the filter has desirable mathematical properties.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern used to extract low frequencies.
    /// 
    /// The scaling coefficients in the Meyer wavelet:
    /// - Keep all the low frequencies (below 1/3 of the maximum frequency) unchanged
    /// - Gradually reduce frequencies between 1/3 and 2/3 using a smooth function
    /// - Remove all frequencies above 2/3
    /// 
    /// This is like a special audio filter that keeps the bass and mid-range sounds
    /// but gradually removes the higher pitches. The smooth transition ensures that
    /// the filtering doesn't create artificial artifacts in your results.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coefficients = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            double freq = (double)i / size;
            if (freq <= 1.0 / 3)
            {
                coefficients[i] = NumOps.FromDouble(1);
            }
            else if (freq <= 2.0 / 3)
            {
                double v = Math.PI * (3 * freq - 1);
                double psi = Math.Cos(Math.PI / 2 * Vf(v));
                coefficients[i] = NumOps.FromDouble(psi);
            }
            else
            {
                coefficients[i] = NumOps.FromDouble(0);
            }
        }

        return coefficients;
    }

    /// <summary>
    /// Gets the wavelet coefficients (high-pass filter) used in the Meyer wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients in the frequency domain.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients used in the Meyer wavelet transform, which
    /// represent the high-pass filter in the frequency domain. These coefficients are designed
    /// to complement the scaling coefficients, passing high frequencies while blocking low ones,
    /// with smooth transition bands that ensure the filter has desirable mathematical properties.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern used to extract high frequencies.
    /// 
    /// The wavelet coefficients in the Meyer wavelet:
    /// - Block all the very low frequencies (below 1/3 of the maximum frequency)
    /// - Gradually increase frequencies between 1/3 and 2/3 using a smooth function
    /// - Keep frequencies between 2/3 and 4/3 with another smooth transition
    /// - Block frequencies above 4/3
    /// 
    /// This is like an audio filter that focuses on the treble sounds while removing the bass.
    /// The smooth transitions ensure that the filtering process doesn't create artificial
    /// distortions in your results. Together with the scaling coefficients, they cover
    /// the entire frequency spectrum of your data.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coefficients = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            double freq = (double)i / size;
            if (freq <= 1.0 / 3)
            {
                coefficients[i] = NumOps.FromDouble(0);
            }
            else if (freq <= 2.0 / 3)
            {
                double v = Math.PI * (3 * freq - 1);
                double psi = Math.Sin(Math.PI / 2 * Vf(v));
                coefficients[i] = NumOps.FromDouble(psi);
            }
            else if (freq <= 4.0 / 3)
            {
                double v = Math.PI * (3 / 2 * freq - 1);
                double psi = Math.Sin(Math.PI / 2 * Vf(v));
                coefficients[i] = NumOps.FromDouble(psi);
            }
            else
            {
                coefficients[i] = NumOps.FromDouble(0);
            }
        }

        return coefficients;
    }

    /// <summary>
    /// Implements the Meyer wavelet function for time domain calculation.
    /// </summary>
    /// <param name="t">The input value in the time domain.</param>
    /// <returns>The calculated Meyer wavelet value.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the piecewise definition of the Meyer wavelet in the time domain.
    /// It evaluates the wavelet function based on which region the input value falls into, with
    /// different formulas applied for different ranges to ensure the wavelet has the desired properties.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method calculates the actual Meyer wavelet value at a point.
    /// 
    /// The Meyer function calculation:
    /// - Divides the input range into different regions
    /// - Applies different mathematical formulas depending on the region
    /// - Uses the auxiliary function to ensure smooth transitions between regions
    /// 
    /// This creates the specific shape of the Meyer wavelet in the time domain.
    /// While the Meyer wavelet is more naturally defined in the frequency domain,
    /// this time-domain representation is useful for direct application to signals.
    /// </para>
    /// </remarks>
    private double MeyerFunction(double t)
    {
        double abst = Math.Abs(t);

        if (abst < 2 * Math.PI / 3)
        {
            return 0;
        }
        else if (abst < 4 * Math.PI / 3)
        {
            double y = 9 / 4.0 * Math.Pow(abst / (2 * Math.PI) - 1 / 3.0, 2);
            return Math.Sin(2 * Math.PI * y) * Math.Sqrt(2 / 3.0 * AuxiliaryFunction(y));
        }
        else if (abst < 8 * Math.PI / 3)
        {
            double y = 9 / 4.0 * Math.Pow(2 / 3.0 - abst / (2 * Math.PI), 2);
            return Math.Sin(2 * Math.PI * y) * Math.Sqrt(2 / 3.0 * AuxiliaryFunction(1 - y));
        }
        else
        {
            return 0;
        }
    }

    /// <summary>
    /// Provides a smooth transition function for the Meyer wavelet.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>A value that smoothly transitions from 0 to 1 as x goes from 0 to 1.</returns>
    /// <remarks>
    /// <para>
    /// This method implements an auxiliary function that provides a smooth transition from 0 to 1
    /// as the input goes from 0 to 1. This function has continuous derivatives of all orders,
    /// which helps ensure the Meyer wavelet is infinitely differentiable.
    /// </para>
    /// <para><b>For Beginners:</b> This helper creates smooth transitions between different regions of the wavelet.
    /// 
    /// The auxiliary function:
    /// - Returns 0 for inputs less than or equal to 0
    /// - Returns 1 for inputs greater than or equal to 1
    /// - Creates a smooth S-shaped curve for inputs between 0 and 1
    /// 
    /// This is like creating a gentle ramp instead of a sudden step between different
    /// regions of the wavelet. The smoothness is important for ensuring the wavelet
    /// has desirable mathematical properties and doesn't introduce artifacts during analysis.
    /// </para>
    /// </remarks>
    private double AuxiliaryFunction(double x)
    {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;

        return x * x * (3 - 2 * x);
    }

    /// <summary>
    /// Provides a polynomial function used in the frequency domain definitions.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The result of applying the polynomial function to the input.</returns>
    /// <remarks>
    /// <para>
    /// This static method implements a specific polynomial function v(x) = x²(3-2x) that is used
    /// in the frequency domain definitions of the Meyer wavelet filters. This function creates
    /// a smooth transition from 0 to 1 as x goes from 0 to 1, with zero derivatives at both endpoints.
    /// </para>
    /// <para><b>For Beginners:</b> This helper creates smooth transitions in the frequency domain filters.
    /// 
    /// The Vf function:
    /// - Creates an S-shaped curve that goes smoothly from 0 to 1 as x goes from 0 to 1
    /// - Has zero slope at both ends, making for very smooth transitions
    /// - Is used when defining how different frequencies are filtered
    /// 
    /// This function helps ensure that the filters don't create artificial artifacts
    /// in your results by avoiding sharp transitions between frequency regions.
    /// </para>
    /// </remarks>
    private static double Vf(double x)
    {
        return x * x * (3 - 2 * x);
    }
}
