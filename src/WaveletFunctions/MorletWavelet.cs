namespace AiDotNet.WaveletFunctions;

public class MorletWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _omega;
    private readonly FastFourierTransform<T> _fft;

    public MorletWavelet(double omega = 5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = _numOps.FromDouble(omega);
        _fft = new FastFourierTransform<T>();
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        T cosine = MathHelper.Cos(_numOps.Multiply(_omega, x));
        T gaussian = _numOps.Exp(_numOps.Divide(_numOps.Negate(x2), _numOps.FromDouble(2)));

        return _numOps.Multiply(cosine, gaussian);
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
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
            T freq = _numOps.Divide(_numOps.FromDouble(i - size / 2), _numOps.FromDouble(size));
            Complex<T> psi = MorletFourierTransform(freq);
            
            // Low-pass (scaling) filter
            if (_numOps.LessThanOrEquals(_numOps.Abs(freq), _numOps.FromDouble(0.5)))
            {
                scalingSpectrum[i] = complexOps.Multiply(spectrum[i], psi);
            }

            // High-pass (wavelet) filter
            waveletSpectrum[i] = complexOps.Multiply(spectrum[i], 
                complexOps.Subtract(new Complex<T>(_numOps.One, _numOps.Zero), psi));
        }

        // Perform inverse FFT
        var approximation = _fft.Inverse(scalingSpectrum);
        var detail = _fft.Inverse(waveletSpectrum);

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coeffs = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            T freq = _numOps.Divide(_numOps.FromDouble(i - size / 2), _numOps.FromDouble(size));
            coeffs[i] = _numOps.LessThanOrEquals(_numOps.Abs(freq), _numOps.FromDouble(0.5)) 
                ? MorletFourierTransform(freq).Magnitude 
                : _numOps.Zero;
        }

        return coeffs;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coeffs = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            T freq = _numOps.Divide(_numOps.FromDouble(i - size / 2), _numOps.FromDouble(size));
            coeffs[i] = _numOps.Subtract(_numOps.One, MorletFourierTransform(freq).Magnitude);
        }

        return coeffs;
    }

    private Complex<T> MorletFourierTransform(T omega)
    {
        T term1 = _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.Multiply(omega, omega)));
        T term2 = _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.Multiply(_numOps.Subtract(omega, _omega), _numOps.Subtract(omega, _omega))));
        T real = _numOps.Subtract(term2, term1);

        return new Complex<T>(real, _numOps.Zero);
    }
}