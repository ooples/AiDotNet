namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Mexican Hat wavelet function implementation for signal processing and analysis.
/// </summary>
/// <remarks>
/// <para>
/// The Mexican Hat wavelet, also known as the Ricker wavelet or the second derivative of the Gaussian function,
/// is a wavelet consisting of a negative normalized second derivative of a Gaussian function.
/// It is commonly used in image processing, computer vision, and various signal analysis applications
/// due to its ability to detect transitions and edges at multiple scales.
/// </para>
/// <para><b>For Beginners:</b> The Mexican Hat wavelet looks like a sombrero or a bell curve with a dip in the middle.
/// 
/// Think of the Mexican Hat wavelet like a special pattern-matching tool that:
/// - Has a distinctive shape with a central peak surrounded by two valleys, then tapering to zero
/// - Is excellent at detecting edges, boundaries, and "blob-like" features in your data
/// - Gets its name because its shape resembles a Mexican sombrero hat viewed from above
/// 
/// This wavelet is particularly useful for finding places where your data changes its direction
/// twice in a row (like going up, then down, then up again). This makes it ideal for detecting
/// objects in images, finding peaks in signals, or identifying boundaries between different regions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MexicanHatWavelet<T> : WaveletFunctionBase<T>
{

    /// <summary>
    /// The standard deviation parameter that controls the width of the Mexican Hat wavelet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the width of the Mexican Hat wavelet. It affects how localized
    /// the wavelet is in both time and frequency domains. A smaller sigma creates a narrower
    /// wavelet that provides better time localization but poorer frequency resolution, while
    /// a larger sigma does the opposite.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how wide or narrow the "sombrero" shape is.
    /// 
    /// Think of sigma like adjusting the size of a magnifying glass:
    /// - Smaller values create a narrower wavelet (a smaller sombrero) that's good at finding small features
    /// - Larger values create a wider wavelet (a larger sombrero) that's good at finding broader features
    /// 
    /// If you're looking for small, detailed features in your data, use a smaller sigma.
    /// If you're looking for larger patterns or broader features, use a larger sigma.
    /// This parameter gives you the flexibility to analyze your data at different scales.
    /// </para>
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Initializes a new instance of the <see cref="MexicanHatWavelet{T}"/> class with the specified sigma.
    /// </summary>
    /// <param name="sigma">The standard deviation that controls the width of the Mexican Hat wavelet. Defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Mexican Hat wavelet with the specified sigma parameter,
    /// which controls the width of the wavelet. The default value of 1.0 provides a balanced wavelet
    /// suitable for general-purpose analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Mexican Hat wavelet with your chosen width.
    /// 
    /// When creating a Mexican Hat wavelet:
    /// - The sigma parameter controls how wide the sombrero shape will be
    /// - The default value (1.0) works well for many common applications
    /// - You can experiment with different values to match your specific needs
    /// 
    /// For example, if you're analyzing an image and want to detect small objects or fine details,
    /// you might use a smaller sigma. If you're more interested in detecting larger structures or
    /// broader features, you might use a larger sigma.
    /// </para>
    /// </remarks>
    public MexicanHatWavelet(double sigma = 1.0)
    {
        _sigma = NumOps.FromDouble(sigma);
    }

    /// <summary>
    /// Calculates the Mexican Hat wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated Mexican Hat wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Mexican Hat wavelet function at the given input point.
    /// The Mexican Hat wavelet is defined as (2 - x²/s²) * e^(-x²/2s²), which is proportional to
    /// the second derivative of a Gaussian function. This wavelet has a distinctive shape with a
    /// central peak flanked by two symmetric valleys.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the height of the sombrero shape at a specific point.
    /// 
    /// When you use this method:
    /// - You provide a point (x) on the horizontal axis
    /// - The method returns the height of the Mexican Hat wavelet at that point
    /// - The height is positive at the center, negative in the surrounding regions, and approaches zero further out
    /// 
    /// The Mexican Hat function creates the distinctive sombrero shape with a central peak
    /// surrounded by two symmetric dips. This shape makes it excellent for detecting features
    /// that have a transition from one state to another and back again.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        T x2 = NumOps.Square(x);
        T sigma2 = NumOps.Square(_sigma);
        T term1 = NumOps.Subtract(NumOps.FromDouble(2.0), NumOps.Divide(x2, sigma2));
        T term2 = NumOps.Exp(NumOps.Negate(NumOps.Divide(x2, NumOps.Multiply(NumOps.FromDouble(2.0), sigma2))));
        return NumOps.Multiply(term1, term2);
    }

    /// <summary>
    /// Decomposes an input signal using the Mexican Hat wavelet and its derivative.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method decomposes the input signal by applying the Mexican Hat wavelet function for
    /// approximation coefficients and its derivative for detail coefficients. The approximation
    /// coefficients capture features that match the Mexican Hat pattern, while the detail coefficients
    /// highlight areas where the signal changes in a way that matches the derivative pattern.
    /// </para>
    /// <para><b>For Beginners:</b> This method analyses your data using two different patterns.
    /// 
    /// When decomposing a signal:
    /// - The approximation coefficients (using the Mexican Hat function) detect peaks and valleys
    /// - The detail coefficients (using the derivative) detect points where the curvature changes
    /// - The process centers the wavelet around each point in your data
    /// 
    /// This is like analyzing a landscape with two different tools:
    /// one that finds hills and valleys (approximation),
    /// and another that finds places where flat land starts to curve (detail).
    /// Together, they give you complementary information about the features in your data.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var approximation = new Vector<T>(size);
        var detail = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T x = NumOps.FromDouble(i - size / 2);
            T waveletValue = Calculate(x);
            T derivativeValue = CalculateDerivative(x);
            approximation[i] = NumOps.Multiply(waveletValue, input[i]);
            detail[i] = NumOps.Multiply(derivativeValue, input[i]);
        }

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients (Mexican Hat function) used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the Mexican Hat function sampled at 101 points centered around zero
    /// and scaled to fit within a reasonable range. These coefficients represent the pattern
    /// used to calculate the approximation coefficients during signal decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the sombrero-shaped pattern used in the transform.
    /// 
    /// The scaling coefficients:
    /// - Form the distinctive Mexican Hat shape with a central peak and surrounding valleys
    /// - Are used to detect features that match this specific pattern in your data
    /// - Are sampled at 101 points to provide a good approximation of the continuous wavelet
    /// 
    /// These coefficients are scaled to ensure they cover an appropriate range that captures
    /// the important features of the Mexican Hat shape. The odd number of points ensures there's
    /// a center point exactly at zero.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size / 4.0));
            coefficients[i] = Calculate(x);
        }

        return coefficients;
    }

    /// <summary>
    /// Gets the wavelet coefficients (derivative of Mexican Hat function) used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the derivative of the Mexican Hat function sampled at 101 points
    /// centered around zero and scaled to fit within a reasonable range. These coefficients represent
    /// the pattern used to calculate the detail coefficients during signal decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the pattern that detects changes in curvature.
    /// 
    /// The wavelet coefficients:
    /// - Represent how the Mexican Hat function changes as you move along the x-axis
    /// - Have a more complex shape with multiple peaks and valleys
    /// - Are particularly sensitive to points where the curvature of your data changes
    /// 
    /// These coefficients complement the scaling coefficients by detecting different types
    /// of features in your data. While the scaling coefficients detect sombrero-shaped patterns,
    /// these coefficients detect places where those patterns are starting to form or dissolve.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(i - size / 2), NumOps.FromDouble(size / 4.0));
            coefficients[i] = CalculateDerivative(x);
        }

        return coefficients;
    }

    /// <summary>
    /// Calculates the derivative of the Mexican Hat function at the specified point.
    /// </summary>
    /// <param name="x">The position at which to calculate the derivative.</param>
    /// <returns>The calculated derivative value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the first derivative of the Mexican Hat function at the given input point.
    /// The derivative provides information about how the Mexican Hat function changes at that point
    /// and is useful for detecting changes in curvature in the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method calculates how quickly the Mexican Hat function is changing at a specific point.
    /// 
    /// The derivative calculation:
    /// - Measures the rate of change of the Mexican Hat function at point x
    /// - Has a complex pattern with multiple zero-crossings
    /// - Is particularly sensitive to changes in the curvature of your data
    /// 
    /// This derivative function helps detect more subtle features in your data that might not
    /// be captured by the Mexican Hat function alone. It's particularly good at finding points
    /// where the behavior of your data changes in complex ways.
    /// </para>
    /// </remarks>
    private T CalculateDerivative(T x)
    {
        // Derivative of f(x) = (2 - x²/σ²) * exp(-x²/(2σ²))
        // f'(x) = (x³/σ⁴ - 4x/σ²) * exp(-x²/(2σ²))
        T x2 = NumOps.Square(x);
        T sigma2 = NumOps.Square(_sigma);
        T sigma4 = NumOps.Square(sigma2);
        T cubicTerm = NumOps.Divide(NumOps.Multiply(x, x2), sigma4); // x³/σ⁴
        T linearTerm = NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(4.0), x), sigma2); // 4x/σ²
        T polynomial = NumOps.Subtract(cubicTerm, linearTerm); // x³/σ⁴ - 4x/σ²
        T expTerm = NumOps.Exp(NumOps.Negate(NumOps.Divide(x2, NumOps.Multiply(NumOps.FromDouble(2.0), sigma2))));

        return NumOps.Multiply(polynomial, expTerm);
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
    /// For Mexican Hat wavelets, decomposition multiplies the input by wavelet values,
    /// so reconstruction uses a weighted combination of the coefficients based on
    /// the original wavelet and derivative values at each position.
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should approximate the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        int size = approximation.Length;
        var reconstructed = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            T x = NumOps.FromDouble(i - size / 2);
            T waveletValue = Calculate(x);
            T derivativeValue = CalculateDerivative(x);

            // Use weighted least-squares approach to recover original
            T waveletSq = NumOps.Square(waveletValue);
            T derivativeSq = NumOps.Square(derivativeValue);
            T denominator = NumOps.Add(waveletSq, derivativeSq);

            if (NumOps.GreaterThan(NumOps.Abs(denominator), NumOps.FromDouble(1e-10)))
            {
                T numerator = NumOps.Add(
                    NumOps.Multiply(approximation[i], waveletValue),
                    NumOps.Multiply(detail[i], derivativeValue)
                );
                reconstructed[i] = NumOps.Divide(numerator, denominator);
            }
            else
            {
                // When both wavelet and derivative are near zero, use sum of coefficients
                reconstructed[i] = NumOps.Add(approximation[i], detail[i]);
            }
        }

        return reconstructed;
    }
}
