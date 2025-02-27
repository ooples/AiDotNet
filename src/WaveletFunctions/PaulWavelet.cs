namespace AiDotNet.WaveletFunctions;

public class PaulWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;
    private readonly FastFourierTransform<T> _fft;

    public PaulWavelet(int order = 4)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fft = new FastFourierTransform<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        Complex<T> i = new Complex<T>(_numOps.Zero, _numOps.One);
        Complex<T> xComplex = new Complex<T>(x, _numOps.Zero);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Calculate (2^m * i^m * m!) / sqrt(π * (2m)!)
        double m = _order;
        double numerator = Math.Pow(2, m) * Convert.ToDouble(MathHelper.Factorial<T>(_order));
        double denominator = Math.Sqrt(Math.PI * Convert.ToDouble(MathHelper.Factorial<T>(2 * _order)));
        // Calculate (2^m * i^m * m!) / sqrt(π * (2m)!)
        Complex<T> factor = complexOps.Power(i, complexOps.FromDouble(_order)) * complexOps.FromDouble(numerator / denominator);

        // Calculate (1 - ix)^(-m-1)
        Complex<T> base_term = new Complex<T>(_numOps.One, _numOps.Zero) - i * xComplex;
        T exponentT = _numOps.FromDouble(-m - 1);
        Complex<T> denominator_term = complexOps.Power(base_term, new Complex<T>(exponentT, exponentT));
        Complex<T> result = factor * denominator_term;

        return result.ToRealOrComplex();
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
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
            T freq = _numOps.Divide(_numOps.FromDouble(i - size / 2), _numOps.FromDouble(size));
            Complex<T> psi = PaulFourierTransform(freq);
            
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
                ? PaulFourierTransform(freq).Magnitude 
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
            coeffs[i] = _numOps.Subtract(_numOps.One, PaulFourierTransform(freq).Magnitude);
        }

        return coeffs;
    }

    private Complex<T> PaulFourierTransform(T omega)
    {
        if (_numOps.LessThanOrEquals(omega, _numOps.Zero))
        {
            return new Complex<T>(_numOps.Zero, _numOps.Zero);
        }

        T factor = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Power(omega, _numOps.FromDouble(_order)));
        T expTerm = _numOps.Exp(_numOps.Negate(omega));

        T real = _numOps.Multiply(factor, expTerm);
        T imag = _numOps.Zero;

        return new Complex<T>(real, imag);
    }
}