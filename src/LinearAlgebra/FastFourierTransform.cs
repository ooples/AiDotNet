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
    /// Internal FFT implementation. Power-of-two inputs use radix-2 Cooley-Tukey;
    /// all other lengths use Bluestein's chirp-z reduction to a power-of-two convolution.
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

        return IsPowerOfTwo(n)
            ? Radix2Transform(input, inverse)
            : BluesteinTransform(input, inverse);
    }

    /// <summary>
    /// Computes an FFT whose length is known to be a power of two.
    /// The inverse transform is intentionally unnormalized; <see cref="Inverse"/>
    /// applies the public 1/N normalization once at the end.
    /// </summary>
    private Vector<Complex<T>> Radix2Transform(Vector<Complex<T>> input, bool inverse)
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

        even = Radix2Transform(even, inverse);
        odd = Radix2Transform(odd, inverse);

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

    /// <summary>
    /// Computes an arbitrary-length DFT with Bluestein's algorithm.
    /// This preserves exact FFT sizes used by audio models such as Whisper's
    /// 400-sample analysis window instead of silently padding them to 512.
    /// </summary>
    private Vector<Complex<T>> BluesteinTransform(Vector<Complex<T>> input, bool inverse)
    {
        int n = input.Length;
        long requiredLength = (2L * n) - 1L;
        const int maxPowerOfTwo = 1 << 30;
        if (requiredLength > maxPowerOfTwo)
        {
            throw new ArgumentOutOfRangeException(
                nameof(input),
                $"FFT length {n} is too large for Bluestein's convolution buffer.");
        }

        int convolutionLength = 1;
        while (convolutionLength < requiredLength)
            convolutionLength <<= 1;

        var a = new Vector<Complex<T>>(convolutionLength);
        var b = new Vector<Complex<T>>(convolutionLength);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        double direction = inverse ? 1.0 : -1.0;

        for (int i = 0; i < n; i++)
        {
            double phase = direction * Math.PI * ChirpExponent(i, n);
            var inputChirp = Complex<T>.FromPolarCoordinates(_numOps.One, _numOps.FromDouble(phase));
            var convolutionChirp = Complex<T>.FromPolarCoordinates(_numOps.One, _numOps.FromDouble(-phase));

            a[i] = complexOps.Multiply(input[i], inputChirp);
            b[i] = convolutionChirp;
            if (i != 0)
                b[convolutionLength - i] = convolutionChirp;
        }

        var spectrumA = Radix2Transform(a, inverse: false);
        var spectrumB = Radix2Transform(b, inverse: false);
        var product = new Vector<Complex<T>>(convolutionLength);
        for (int i = 0; i < convolutionLength; i++)
            product[i] = complexOps.Multiply(spectrumA[i], spectrumB[i]);

        var convolution = Radix2Transform(product, inverse: true);
        var scale = new Complex<T>(_numOps.FromDouble(convolutionLength), _numOps.Zero);
        var output = new Vector<Complex<T>>(n);
        for (int i = 0; i < n; i++)
        {
            double phase = direction * Math.PI * ChirpExponent(i, n);
            var outputChirp = Complex<T>.FromPolarCoordinates(_numOps.One, _numOps.FromDouble(phase));
            var normalized = complexOps.Divide(convolution[i], scale);
            output[i] = complexOps.Multiply(normalized, outputChirp);
        }

        return output;
    }
    /// <summary>
    /// The Bluestein chirp exponent <c>i^2 / n</c>, with <c>i^2</c> reduced modulo <c>2n</c> first.
    /// </summary>
    /// <remarks>
    /// The reduction is what keeps a large transform accurate. Computing <c>(double)i * i / n</c>
    /// directly overflows double's 53-bit exact integer range once <c>i^2</c> passes about 9.0e15
    /// (i above roughly 9.5e7), and relative error grows well before that. Only the FRACTIONAL part
    /// of <c>i^2 / n</c> matters here -- the chirp is periodic, and the integer part is discarded by
    /// the multiply against Math.PI -- so the leading bits are thrown away anyway and every bit of
    /// error they carried lands directly in the phase. Reducing <c>i^2</c> into <c>[0, 2n)</c> before
    /// the division keeps the whole product exact for every representable n.
    ///
    /// Modulo 2n rather than n because the chirp has period 2n in this exponent: exp(i*pi*k/n)
    /// repeats every 2n, not every n.
    ///
    /// Both chirp loops call this, which is also why it is a method: the two must stay identical,
    /// and they previously repeated the expression.
    /// </remarks>
    private static double ChirpExponent(int i, int n)
    {
        // long is exact for i^2 at every int i (max ~4.6e18, well inside long's range).
        long squared = (long)i * i;
        long reduced = squared % (2L * n);
        return (double)reduced / n;
    }

    /// <summary>
    /// True when <paramref name="value"/> is a positive power of two.
    /// </summary>
    /// <remarks>
    /// The positivity test is not redundant: <c>(0 &amp; -1) == 0</c>, so a bare bit-trick reports
    /// zero as a power of two. The current call site is safe only because FFTInternal returns early
    /// for <c>n &lt;= 1</c>, which makes this a trap for the next caller rather than a live bug.
    /// </remarks>
    private static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;
}
