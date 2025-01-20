namespace AiDotNet.WaveletFunctions;

public class ShannonWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ShannonWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        if (_numOps.Equals(x, _numOps.Zero))
            return _numOps.One;

        T sinc = _numOps.Divide(MathHelper.Sin(x), x);
        T cos = MathHelper.Cos(_numOps.Divide(x, _numOps.FromDouble(2)));

        return _numOps.Multiply(sinc, cos);
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
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
            T scaleFactor = _numOps.FromDouble(Math.Sqrt(0.5));
            Complex<T> complexScaleFactor = new Complex<T>(scaleFactor, _numOps.Zero);
            approxSpectrum[halfN] = complexOps.Multiply(spectrum[halfN], complexScaleFactor);
            detailSpectrum[halfN] = complexOps.Multiply(spectrum[halfN], complexScaleFactor);
        }

        // Perform inverse FFT on both approximation and detail
        var approxResult = fft.Inverse(approxSpectrum);
        var detailResult = fft.Inverse(detailSpectrum);

        return (approxResult, detailResult);
    }

    public Vector<T> GetScalingCoefficients()
    {
        int n = 1024;
        var coeffs = new T[n];
        T twoPi = _numOps.Multiply(_numOps.FromDouble(2), MathHelper.Pi<T>());

        for (int k = -n/2; k < n/2; k++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(k), _numOps.FromDouble(n));
            if (k == 0)
            {
                coeffs[k + n/2] = _numOps.FromDouble(Math.Sqrt(0.5));
            }
            else
            {
                T sinc = _numOps.Divide(MathHelper.Sin(_numOps.Multiply(twoPi, x)), _numOps.Multiply(twoPi, x));
                coeffs[k + n/2] = _numOps.Multiply(_numOps.FromDouble(Math.Sqrt(0.5)), sinc);
            }
        }

        return new Vector<T>(coeffs);
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int n = 1024;
        var coeffs = new T[n];
        T twoPi = _numOps.Multiply(_numOps.FromDouble(2), MathHelper.Pi<T>());

        for (int k = -n/2; k < n/2; k++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(k), _numOps.FromDouble(n));
            if (k == 0)
            {
                coeffs[k + n/2] = _numOps.FromDouble(Math.Sqrt(0.5));
            }
            else
            {
                T sinc = _numOps.Divide(MathHelper.Sin(_numOps.Multiply(twoPi, x)), _numOps.Multiply(twoPi, x));
                T modulation = MathHelper.Cos(_numOps.Multiply(twoPi, x));
                coeffs[k + n/2] = _numOps.Multiply(_numOps.FromDouble(Math.Sqrt(0.5)), _numOps.Multiply(sinc, modulation));
            }
        }

        return new Vector<T>(coeffs);
    }
}