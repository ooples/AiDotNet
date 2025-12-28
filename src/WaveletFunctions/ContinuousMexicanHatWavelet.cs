namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements the Mexican Hat wavelet (also known as the Ricker wavelet or the second derivative of a Gaussian),
/// which is commonly used for continuous wavelet transforms and feature detection.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Mexican Hat wavelet is the negative normalized second derivative of a Gaussian function.
/// It has a central peak with symmetric valleys on either side, resembling a Mexican hat.
/// This wavelet is particularly useful for detecting peaks and edges in signals.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Mexican Hat wavelet looks like a sombrero or a mountain with valleys on each side.
/// This shape makes it excellent at finding "bumps" or peaks in your data.
/// 
/// Key features of the Mexican Hat wavelet:
/// - It has a central positive peak with negative valleys on either side
/// - It's symmetric (the same on both sides of the center)
/// - It's good at detecting sudden changes or peaks in signals
/// - It has no scaling function in the traditional sense
/// 
/// Think of it as a template that you slide over your data, looking for places where
/// the data has a similar "bump" shape. When the wavelet aligns with a bump in your data,
/// it produces a strong response.
/// 
/// These wavelets are particularly useful for:
/// - Finding peaks in spectra
/// - Edge detection in images
/// - Scale-space analysis
/// - Feature detection in various signals
/// - Analyzing data where you need to identify local maxima or minima
/// 
/// Unlike some other wavelets, the Mexican Hat is primarily used for continuous wavelet
/// transforms rather than discrete transforms, though this implementation provides
/// both capabilities.
/// </para>
/// </remarks>
public class ContinuousMexicanHatWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the ContinuousMexicanHatWavelet class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Unlike some other wavelets, the Mexican Hat wavelet doesn't have parameters to adjust
    /// its shape. It has a fixed mathematical form as the second derivative of a Gaussian function.
    /// 
    /// This means:
    /// - You don't need to specify any parameters when creating it
    /// - It always has the same characteristic "Mexican hat" shape
    /// - The scale (width) of the wavelet is adjusted during the transform process, not in the constructor
    /// 
    /// The Mexican Hat wavelet is defined by the formula:
    /// ?(t) = (2/v3) · p^(-1/4) · (1-t²) · e^(-t²/2)
    /// 
    /// This formula creates the characteristic central peak with symmetric valleys on either side.
    /// </para>
    /// </remarks>
    public ContinuousMexicanHatWavelet()
    {
    }

    /// <summary>
    /// Calculates the value of the Mexican Hat wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the Mexican Hat wavelet at a specific point.
    /// 
    /// The Mexican Hat wavelet is defined by the formula:
    /// ?(t) = (2/v3) · p^(-1/4) · (1-t²) · e^(-t²/2)
    /// 
    /// Breaking this down:
    /// 1. (1-t²): This term creates the basic shape with a positive center and negative sides
    /// 2. e^(-t²/2): This is the Gaussian envelope that makes the function decay to zero as t moves away from the center
    /// 3. (2/v3) · p^(-1/4): This is a normalization factor that ensures the wavelet has unit energy
    /// 
    /// The result is a function that:
    /// - Equals 1 at x=0 (after normalization)
    /// - Has negative valleys at x = ±v2
    /// - Approaches zero as x moves further from the center
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        T x2 = NumOps.Square(x);
        T exp_term = NumOps.Exp(NumOps.Negate(NumOps.Divide(x2, NumOps.FromDouble(2))));

        T term1 = NumOps.Subtract(NumOps.One, x2);
        T result = NumOps.Multiply(term1, exp_term);

        // Normalization factor
        double norm_factor = 2.0 / (Math.Sqrt(3) * Math.Pow(Math.PI, 0.25));
        result = NumOps.Multiply(result, NumOps.FromDouble(norm_factor));

        return result;
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the Mexican Hat wavelet.
    /// </summary>
    /// <param name="input">The input signal vector to decompose.</param>
    /// <returns>A tuple containing the approximation coefficients and detail coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs one level of wavelet decomposition on your signal, splitting it into:
    /// 
    /// - Approximation coefficients: Represent the low-frequency components (the overall shape)
    /// - Detail coefficients: Represent the high-frequency components (the fine details)
    /// 
    /// The process works like this:
    /// 1. The input signal is convolved (filtered) with a low-pass filter to get the approximation
    /// 2. The input signal is convolved with the Mexican Hat wavelet to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes the Mexican Hat wavelet special for this task is:
    /// - It's excellent at detecting peaks and sudden changes in the signal
    /// - It has good localization properties in both time and frequency
    /// - It's symmetric, which helps preserve the phase of the original signal
    /// 
    /// Note that while the Mexican Hat wavelet is primarily used for continuous wavelet transforms,
    /// this method implements a discrete transform approach to fit within the common wavelet framework.
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which makes wavelet decomposition efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        var waveletCoeffs = GetWaveletCoefficients();
        var scalingCoeffs = GetScalingCoefficients();

        var approximation = Convolve(input, scalingCoeffs);
        var detail = Convolve(input, waveletCoeffs);

        // Downsample by 2
        approximation = Downsample(approximation, 2);
        detail = Downsample(detail, 2);

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Mexican Hat wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    /// 
    /// Interestingly, the Mexican Hat wavelet doesn't have a true scaling function in the
    /// traditional wavelet sense. However, for practical implementation of the discrete
    /// wavelet transform, we need a low-pass filter.
    /// 
    /// This method creates a sinc function-based low-pass filter:
    /// 
    /// sinc(x) = sin(px)/(px)
    /// 
    /// The sinc function is the ideal low-pass filter in signal processing theory.
    /// It lets through all frequencies below a cutoff point and blocks all frequencies above it.
    /// 
    /// The method:
    /// 1. Creates a discretized sinc function of specified length
    /// 2. Normalizes the coefficients so they sum to 1
    /// 
    /// This approach provides a reasonable low-pass filter that complements the
    /// Mexican Hat wavelet's ability to detect high-frequency components.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int length = 64;
        var coeffs = new T[length];
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(i - length / 2), NumOps.FromDouble(length / 4));
            T value = NumOps.Equals(x, NumOps.Zero)
                ? NumOps.One
                : NumOps.Divide(MathHelper.Sin(NumOps.Divide(MathHelper.Pi<T>(), x)), NumOps.Multiply(MathHelper.Pi<T>(), x));
            coeffs[i] = value;
            sum = NumOps.Add(sum, NumOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = NumOps.Divide(coeffs[i], sum);
        }

        return new Vector<T>(coeffs);
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Mexican Hat wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For the Mexican Hat wavelet, these coefficients are a discretized version of the
    /// Mexican Hat function:
    /// 
    /// ?(t) = (1-t²) · e^(-t²/2)
    /// 
    /// This method:
    /// 1. Creates a discretized Mexican Hat wavelet of specified length
    /// 2. Samples the function at regular intervals
    /// 3. Normalizes the coefficients to ensure proper energy preservation
    /// 
    /// The resulting filter is excellent at detecting peaks, edges, and sudden changes
    /// in the signal. The characteristic shape with a positive center and negative sides
    /// makes it respond strongly to local maxima or minima in the data.
    /// 
    /// These coefficients determine how the wavelet will identify and extract the
    /// detail information from your signal.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int length = 256;
        var coeffs = new T[length];
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T t = NumOps.Divide(NumOps.FromDouble(i - length / 2), NumOps.FromDouble(length / 4));
            T t2 = NumOps.Multiply(t, t);
            T value = NumOps.Multiply(
                NumOps.Subtract(NumOps.One, t2),
                NumOps.Exp(NumOps.Divide(NumOps.Negate(t2), NumOps.FromDouble(2.0)))
            );
            coeffs[i] = value;
            sum = NumOps.Add(sum, NumOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = NumOps.Divide(coeffs[i], sum);
        }

        return new Vector<T>(coeffs);
    }

    /// <summary>
    /// Performs convolution of an input signal with a filter.
    /// </summary>
    /// <param name="input">The input signal vector.</param>
    /// <param name="kernel">The filter vector.</param>
    /// <returns>The convolved signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Convolution is a mathematical operation that combines two functions to produce a third function.
    /// In signal processing, it's like sliding a filter over a signal and calculating a weighted sum
    /// at each position.
    /// 
    /// The process works like this:
    /// 1. For each position in the output:
    ///    a. Center the filter at that position
    ///    b. Multiply each filter coefficient by the corresponding signal value
    ///    c. Sum these products to get the output value at that position
    /// 
    /// This implementation produces a result with length equal to input.length + kernel.length - 1,
    /// which is the full convolution without truncation. This ensures that no information is lost
    /// at the boundaries.
    /// 
    /// Convolution is fundamental to wavelet transforms and many other signal processing operations.
    /// It's how filters are applied to signals to extract specific frequency components.
    /// </para>
    /// </remarks>
    private Vector<T> Convolve(Vector<T> input, Vector<T> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new T[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[i - j], kernel[j]));
                }
            }

            result[i] = sum;
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Downsamples a signal by keeping only every nth sample.
    /// </summary>
    /// <param name="input">The input signal vector.</param>
    /// <param name="factor">The downsampling factor.</param>
    /// <returns>The downsampled signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Downsampling reduces the sampling rate of a signal by keeping only a subset of the samples.
    /// 
    /// For example, with a factor of 2:
    /// - Original signal: [10, 20, 30, 40, 50, 60]
    /// - Downsampled signal: [10, 30, 50]
    /// 
    /// This method keeps every nth sample (where n is the factor) and discards the rest.
    /// In wavelet decomposition, downsampling by 2 is standard after filtering, because:
    /// 
    /// 1. The filters have already removed frequency components that would cause aliasing
    /// 2. It reduces the data size by half at each decomposition level
    /// 3. It ensures that the total size of approximation and detail coefficients equals the input size
    /// 
    /// Downsampling is crucial for the efficiency of wavelet transforms, especially for
    /// multi-level decomposition and compression applications.
    /// </para>
    /// </remarks>
    private Vector<T> Downsample(Vector<T> input, int factor)
    {
        int newLength = input.Length / factor;
        var result = new T[newLength];

        for (int i = 0; i < newLength; i++)
        {
            result[i] = input[i * factor];
        }

        return new Vector<T>(result);
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
    /// For Continuous Mexican Hat wavelets, reconstruction works by:
    /// 1. Upsampling (inserting zeros between each coefficient)
    /// 2. Convolving with time-reversed reconstruction filters
    /// 3. Adding the approximation and detail contributions together
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        // Upsample by 2
        var approxUpsampled = Upsample(approximation, 2);
        var detailUpsampled = Upsample(detail, 2);

        // Convolve with time-reversed filters
        var scalingCoeffs = GetScalingCoefficients();
        var waveletCoeffs = GetWaveletCoefficients();

        var approxRecon = ConvolveReversed(approxUpsampled, scalingCoeffs);
        var detailRecon = ConvolveReversed(detailUpsampled, waveletCoeffs);

        // Add contributions
        int outputLength = Math.Min(approxRecon.Length, detailRecon.Length);
        var reconstructed = new Vector<T>(outputLength);
        for (int i = 0; i < outputLength; i++)
        {
            reconstructed[i] = NumOps.Add(approxRecon[i], detailRecon[i]);
        }

        return reconstructed;
    }

    /// <summary>
    /// Upsamples a signal by inserting zeros between samples.
    /// </summary>
    private Vector<T> Upsample(Vector<T> input, int factor)
    {
        int newLength = input.Length * factor;
        var result = new T[newLength];

        // Initialize all elements to zero
        for (int i = 0; i < newLength; i++)
        {
            result[i] = NumOps.Zero;
        }

        // Place input values at every factor-th position
        for (int i = 0; i < input.Length; i++)
        {
            result[i * factor] = input[i];
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Convolves a signal with a time-reversed kernel.
    /// </summary>
    private Vector<T> ConvolveReversed(Vector<T> input, Vector<T> kernel)
    {
        int resultLength = input.Length;
        var result = new T[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < kernel.Length; j++)
            {
                int inputIndex = i - j + kernel.Length / 2;
                if (inputIndex >= 0 && inputIndex < input.Length)
                {
                    int revJ = kernel.Length - 1 - j;
                    sum = NumOps.Add(sum, NumOps.Multiply(input[inputIndex], kernel[revJ]));
                }
            }
            result[i] = sum;
        }

        return new Vector<T>(result);
    }
}
