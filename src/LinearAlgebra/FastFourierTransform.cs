namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Implements the Fast Fourier Transform (FFT) algorithm for converting between time domain and frequency domain representations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> The Fast Fourier Transform is a mathematical technique that breaks down a signal (like sound or image data)
/// into its component frequencies. Think of it like analyzing a musical chord to identify which individual notes are being played.
/// 
/// For example, if you have audio data that represents a recording of multiple instruments playing together,
/// the FFT can help separate the different frequencies that make up that sound. This is useful in many applications
/// like audio processing, image compression, and pattern recognition.
/// </remarks>
public readonly struct FastFourierTransform<T>
{
    /// <summary>
    /// Provides operations for the numeric type T (addition, multiplication, etc.).
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the FastFourierTransform struct.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor prepares the FFT calculator by setting up the necessary
    /// mathematical operations for the specific number type you're using (like double or float).
    /// </remarks>
    public FastFourierTransform()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Performs a forward Fast Fourier Transform, converting from time domain to frequency domain.
    /// </summary>
    /// <param name="input">The input vector in time domain.</param>
    /// <returns>A vector of complex numbers representing the frequency domain.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes your original data (like a sound wave over time) and
    /// converts it to show which frequencies are present and how strong each frequency is.
    /// 
    /// For example, if your input represents a sound recording, the output will tell you which
    /// musical notes (frequencies) are present in that recording and how loud each note is.
    /// </remarks>
    public Vector<Complex<T>> Forward(Vector<T> input)
    {
        int n = input.Length;
        var output = new Vector<Complex<T>>(n);

        for (int i = 0; i < n; i++)
        {
            output[i] = new Complex<T>(input[i], _numOps.Zero);
        }

        return FFTInternal(output, false);
    }

    /// <summary>
    /// Performs an inverse Fast Fourier Transform, converting from frequency domain back to time domain.
    /// </summary>
    /// <param name="input">The input vector in frequency domain (complex numbers).</param>
    /// <returns>A vector representing the time domain.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method does the opposite of the Forward method. It takes frequency information
    /// (which frequencies are present and how strong they are) and converts it back to the original form.
    /// 
    /// For example, if you have information about which musical notes are in a chord and how loud each note is,
    /// this method can reconstruct the actual sound wave of that chord.
    /// </remarks>
    public Vector<T> Inverse(Vector<Complex<T>> input)
    {
        int n = input.Length;
        var complexOutput = FFTInternal(input, true);
        var result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            result[i] = _numOps.Divide(complexOutput[i].Real, _numOps.FromDouble(n));
        }

        return result;
    }

    /// <summary>
    /// Internal recursive implementation of the FFT algorithm using the Cooley-Tukey method.
    /// </summary>
    /// <param name="input">The input vector of complex numbers.</param>
    /// <param name="inverse">Whether to perform the inverse transform.</param>
    /// <returns>The transformed vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the core algorithm that makes the Fast Fourier Transform "fast".
    /// It uses a clever approach called "divide and conquer" - breaking the problem into smaller pieces,
    /// solving each piece, and then combining the results.
    /// 
    /// The method separates the input into even and odd-indexed elements, processes each group separately,
    /// and then combines them in a special way. This approach dramatically reduces the computation time
    /// compared to more straightforward methods.
    /// </remarks>
    private Vector<Complex<T>> FFTInternal(Vector<Complex<T>> input, bool inverse)
    {
        int n = input.Length;
        if (n <= 1) return input;

        var even = new Vector<Complex<T>>(n / 2);
        var odd = new Vector<Complex<T>>(n / 2);

        for (int i = 0; i < n / 2; i++)
        {
            even[i] = input[2 * i];
            odd[i] = input[2 * i + 1];
        }

        even = FFTInternal(even, inverse);
        odd = FFTInternal(odd, inverse);

        var output = new Vector<Complex<T>>(n);
        T angleSign = inverse ? _numOps.One : _numOps.Negate(_numOps.One);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        for (int k = 0; k < n / 2; k++)
        {
            T angle = _numOps.Multiply(angleSign, _numOps.Multiply(_numOps.FromDouble(2 * Math.PI * k), _numOps.FromDouble(1.0 / n)));
            var t = complexOps.Multiply(Complex<T>.FromPolarCoordinates(_numOps.One, angle), odd[k]);
            output[k] = complexOps.Add(even[k], t);
            output[k + n / 2] = complexOps.Subtract(even[k], t);
        }

        return output;
    }
}
