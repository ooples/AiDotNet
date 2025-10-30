namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Gabor wavelet function implementation for time-frequency analysis and signal processing.
/// </summary>
/// <remarks>
/// <para>
/// The Gabor wavelet is a complex wavelet defined as a sinusoidal function multiplied by a Gaussian window.
/// It provides excellent time-frequency localization and is widely used in image processing, computer vision, 
/// texture analysis, and various signal processing applications. This implementation supports
/// customization of frequency, bandwidth, and phase parameters.
/// </para>
/// <para><b>For Beginners:</b> A Gabor wavelet is a special mathematical tool that helps analyze patterns in data.
/// 
/// Think of it like a musical note with a specific pitch and duration:
/// - It can detect specific patterns (like frequencies) in your data
/// - It's especially good at finding where in your data a certain pattern occurs
/// - It's widely used in image processing to detect edges and textures
/// 
/// For example, in image recognition, Gabor wavelets can help detect specific features like edges
/// oriented in particular directions, making them useful for tasks like fingerprint recognition,
/// face detection, and texture classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GaborWavelet<T> : IWaveletFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of numeric operations that can work with the generic type T.
    /// It provides methods for basic arithmetic operations, comparisons, trigonometric functions, and 
    /// conversions that are used throughout the wavelet calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that lets us do math with different number types.
    /// 
    /// Because this class can work with different types of numbers (like float, double, or decimal),
    /// we need a special helper that knows how to:
    /// - Perform all the necessary math operations
    /// - Convert between different number formats
    /// - Handle complex calculations like exponentials and trigonometric functions
    /// 
    /// This allows the wavelet code to work with whatever number type you choose,
    /// without having to write separate code for each number type.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The central frequency of the Gabor wavelet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the frequency of the sinusoidal component of the Gabor wavelet.
    /// Higher values result in faster oscillations and detection of higher-frequency components
    /// in the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how rapidly the wavelet oscillates.
    /// 
    /// Think of it like the pitch of a musical note:
    /// - Higher values create a higher "pitch" wavelet that detects rapid changes
    /// - Lower values create a lower "pitch" wavelet that detects slower changes
    /// 
    /// For example, if you're analyzing an image, a high omega would help detect fine details
    /// and textures, while a low omega would help identify larger structures and shapes.
    /// </para>
    /// </remarks>
    private readonly T _omega;

    /// <summary>
    /// The standard deviation of the Gaussian envelope.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the width of the Gaussian envelope that modulates the sinusoidal component.
    /// It determines how localized the wavelet is in the time/space domain. Smaller values create a
    /// narrower wavelet that provides better time localization but poorer frequency resolution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how wide or narrow the wavelet is.
    /// 
    /// Think of it like the duration of a musical note:
    /// - Smaller values create a shorter, more precise wavelet (good for pinpointing exactly where something happens)
    /// - Larger values create a longer, more spread-out wavelet (good for determining exactly what frequency is present)
    /// 
    /// This represents the fundamental trade-off in signal analysis: you can't simultaneously have perfect
    /// precision in both time and frequency. It's like trying to say exactly when and exactly what note was
    /// played on a piano - the more precisely you know one, the less precisely you know the other.
    /// </para>
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// The wavelength parameter that determines oscillation frequency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the wavelength of the sinusoidal component of the Gabor wavelet.
    /// It is inversely proportional to the frequency of oscillations and affects how the wavelet
    /// responds to different frequencies in the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the distance between peaks in the wavelet.
    /// 
    /// Think of it like the wavelength of a water wave:
    /// - Smaller values mean more compressed waves (more oscillations in a given space)
    /// - Larger values mean more stretched-out waves (fewer oscillations in a given space)
    /// 
    /// This parameter gives you another way to tune how the wavelet responds to different
    /// patterns in your data. It works together with omega to define the oscillation properties.
    /// </para>
    /// </remarks>
    private readonly T _lambda;

    /// <summary>
    /// The phase offset of the sinusoidal component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the phase offset of the sinusoidal component of the Gabor wavelet.
    /// It shifts the starting point of the oscillation and can be used to create phase-sensitive
    /// analysis or to generate a family of wavelets with different phases for more comprehensive analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the starting position of the oscillation.
    /// 
    /// Think of it like the initial position of a swing:
    /// - A value of 0 means the swing starts at the center position
    /// - Other values shift the starting position (like pulling the swing back before release)
    /// 
    /// By using different phase values, you can detect patterns that have the same frequency
    /// but different alignments in your data. This is often used to create a complete family
    /// of wavelets that can analyze a signal from different "perspectives."
    /// </para>
    /// </remarks>
    private readonly T _psi;

    /// <summary>
    /// Initializes a new instance of the <see cref="GaborWavelet{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="omega">The central frequency of the wavelet. Defaults to 5.</param>
    /// <param name="sigma">The standard deviation of the Gaussian envelope. Defaults to 1.</param>
    /// <param name="lambda">The wavelength parameter. Defaults to 4.</param>
    /// <param name="psi">The phase offset of the sinusoidal component. Defaults to 0.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Gabor wavelet with the specified parameters, which control
    /// the shape, frequency, and localization properties of the wavelet. The default values provide
    /// a balanced wavelet suitable for general-purpose analysis, but they can be adjusted to target
    /// specific characteristics in the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Gabor wavelet with your chosen settings.
    /// 
    /// When creating a Gabor wavelet:
    /// - You can customize it for different types of pattern detection
    /// - The default values work well for many common applications
    /// - You can experiment with different values to optimize for your specific needs
    /// 
    /// For example, if you're looking for rapid changes in your data, you might increase omega,
    /// while if you need to precisely locate where changes occur, you might decrease sigma.
    /// </para>
    /// </remarks>
    public GaborWavelet(double omega = 5, double sigma = 1, double lambda = 4, double psi = 0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = _numOps.FromDouble(omega);
        _sigma = _numOps.FromDouble(sigma);
        _psi = _numOps.FromDouble(psi);
        _lambda = _numOps.FromDouble(lambda);
    }

    /// <summary>
    /// Calculates the wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Gabor wavelet function at the given input point.
    /// It combines a Gaussian envelope with a cosine wave to create the wavelet. The result
    /// represents how much the signal at that point contributes to the frequency band
    /// represented by this wavelet.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how strongly your data at a specific point
    /// matches the Gabor wavelet pattern.
    /// 
    /// When you use this method:
    /// - You provide a point (x) in your data
    /// - The method returns a value showing how well the wavelet matches your data at that point
    /// - The calculation combines a bell-shaped curve (Gaussian) with a wave pattern (cosine)
    /// 
    /// This is like asking: "How much does my data at this point look like this specific
    /// oscillation pattern with this specific width?"
    /// </para>
    /// </remarks>
    public T Calculate(T x)
    {
        T gaussianTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(x, x), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(_numOps.Multiply(_omega, x));
        return _numOps.Multiply(gaussianTerm, cosTerm);
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the Gabor transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Gabor transform, which decomposes the input signal into two components:
    /// real (approximation) and imaginary (detail) parts. The real part uses the cosine component of the 
    /// Gabor function, while the imaginary part uses the sine component. This complex representation
    /// captures both amplitude and phase information in the signal.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into two complementary parts using
    /// the Gabor wavelet.
    /// 
    /// When decomposing a signal:
    /// - The approximation coefficients (real part) capture the parts of your data that match the cosine wave
    /// - The detail coefficients (imaginary part) capture the parts that match the sine wave
    /// - Together, they provide a complete representation of the signal at this specific frequency and scale
    /// 
    /// This is similar to describing the motion of a swing using both its position (real part)
    /// and its velocity (imaginary part) - together they give you complete information about the swing.
    /// </para>
    /// </remarks>
    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var approximation = new Vector<T>(size);
        var detail = new Vector<T>(size);
        for (int x = 0; x < size; x++)
        {
            T gaborReal = GaborFunction(x, true);
            T gaborImag = GaborFunction(x, false);
            approximation[x] = _numOps.Multiply(gaborReal, input[x]);
            detail[x] = _numOps.Multiply(gaborImag, input[x]);
        }
        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients (real part of the Gabor function) used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the real part (cosine component) of the Gabor function sampled at 100 points.
    /// These coefficients represent the real part of the complex Gabor wavelet and are used to calculate
    /// the approximation coefficients during signal decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the "cosine-based" part of the Gabor wavelet.
    /// 
    /// The scaling coefficients:
    /// - Represent the real part of the complex Gabor function
    /// - Are based on cosine waves modified by a Gaussian envelope
    /// - Help capture how much your data matches this specific pattern
    /// 
    /// These coefficients are one half of the complete Gabor wavelet representation.
    /// They're sampled at 100 points to provide a discrete approximation of the continuous wavelet.
    /// </para>
    /// </remarks>
    public Vector<T> GetScalingCoefficients()
    {
        int size = 100;
        var coefficients = new Vector<T>(size);
        for (int x = 0; x < size; x++)
        {
            coefficients[x] = GaborFunction(x, true);
        }
        return coefficients;
    }

    /// <summary>
    /// Gets the wavelet coefficients (imaginary part of the Gabor function) used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the imaginary part (sine component) of the Gabor function sampled at 100 points.
    /// These coefficients represent the imaginary part of the complex Gabor wavelet and are used to calculate
    /// the detail coefficients during signal decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the "sine-based" part of the Gabor wavelet.
    /// 
    /// The wavelet coefficients:
    /// - Represent the imaginary part of the complex Gabor function
    /// - Are based on sine waves modified by a Gaussian envelope
    /// - Capture the phase information in your data
    /// 
    /// These coefficients are the other half of the complete Gabor wavelet representation.
    /// Together with the scaling coefficients, they provide a complete description of how
    /// your data matches the Gabor pattern, including both amplitude and phase information.
    /// </para>
    /// </remarks>
    public Vector<T> GetWaveletCoefficients()
    {
        int size = 100;
        var coefficients = new Vector<T>(size);
        for (int x = 0; x < size; x++)
        {
            coefficients[x] = GaborFunction(x, false);
        }
        return coefficients;
    }

    /// <summary>
    /// Calculates the Gabor function value at the specified point, either real (cosine) or imaginary (sine) part.
    /// </summary>
    /// <param name="x">The position at which to calculate the Gabor function.</param>
    /// <param name="real">If true, calculates the real (cosine) part; otherwise, calculates the imaginary (sine) part.</param>
    /// <returns>The calculated Gabor function value.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core Gabor function calculation, which combines a Gaussian envelope
    /// with either a cosine wave (real part) or a sine wave (imaginary part). It applies a rotation
    /// to the input coordinate by p/4 radians before calculating the function, which is a common
    /// approach in image processing applications to create oriented filters.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method calculates either the cosine or sine version of the Gabor function.
    /// 
    /// The Gabor function calculation:
    /// - First rotates the input coordinate (useful for detecting patterns at specific angles)
    /// - Then applies a Gaussian envelope (bell-shaped curve) to localize the function
    /// - Finally multiplies by either a cosine wave (if real=true) or a sine wave (if real=false)
    /// 
    /// This is the mathematical core of the Gabor wavelet. The combination of the Gaussian envelope
    /// and the trigonometric function creates a localized oscillation that's excellent for
    /// detecting specific patterns in specific locations.
    /// </para>
    /// </remarks>
    private T GaborFunction(int x, bool real)
    {
        T xT = _numOps.FromDouble(x);
        T xTheta = _numOps.Multiply(xT, MathHelper.Cos(_numOps.Divide(MathHelper.Pi<T>(), _numOps.FromDouble(4))));
        T expTerm = _numOps.Exp(_numOps.Divide(_numOps.Negate(_numOps.Multiply(xTheta, xTheta)), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma))));
        T angle = _numOps.Add(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2.0 * Math.PI), xTheta), _lambda), _psi);
        T trigTerm = real ? MathHelper.Cos(angle) : MathHelper.Sin(angle);
        return _numOps.Multiply(expTerm, trigTerm);
    }
}