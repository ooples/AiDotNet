namespace AiDotNet.WaveletFunctions;

public class MeyerWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly FastFourierTransform<T> _fft;

    public MeyerWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fft = new FastFourierTransform<T>();
    }

    public T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        return _numOps.FromDouble(MeyerFunction(t));
    }

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

    private double AuxiliaryFunction(double x)
    {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;

        return x * x * (3 - 2 * x);
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
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
            T freq = _numOps.Divide(_numOps.FromDouble(i), _numOps.FromDouble(size));
            if (_numOps.LessThanOrEquals(freq, _numOps.FromDouble(1.0 / 3)))
            {
                lowPass[i] = spectrum[i];
            }
            else if (_numOps.LessThanOrEquals(freq, _numOps.FromDouble(2.0 / 3)))
            {
                T v = _numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(3), freq), _numOps.One));
                T psi = _numOps.FromDouble(Math.Cos(Math.PI / 2 * Vf(Convert.ToDouble(v))));
                Complex<T> complexPsi = new Complex<T>(psi, _numOps.Zero);
                lowPass[i] = complexOps.Multiply(spectrum[i], complexPsi);
                T sqrtTerm = _numOps.Sqrt(_numOps.Subtract(_numOps.One, _numOps.Multiply(psi, psi)));
                Complex<T> complexSqrtTerm = new Complex<T>(sqrtTerm, _numOps.Zero);
                highPass[i] = complexOps.Multiply(spectrum[i], complexSqrtTerm);
            }
            else if (_numOps.LessThanOrEquals(freq, _numOps.FromDouble(4.0 / 3)))
            {
                T v = _numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(3.0 / 2), freq), _numOps.One));
                T psi = _numOps.FromDouble(Math.Sin(Math.PI / 2 * Vf(Convert.ToDouble(v))));
                Complex<T> complexPsi = new Complex<T>(psi, _numOps.Zero);
                highPass[i] = complexOps.Multiply(spectrum[i], complexPsi);
            }
        }

        // Perform inverse FFT
        Vector<T> approximation = _fft.Inverse(lowPass);
        Vector<T> detail = _fft.Inverse(highPass);

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coefficients = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            double freq = (double)i / size;
            if (freq <= 1.0 / 3)
            {
                coefficients[i] = _numOps.FromDouble(1);
            }
            else if (freq <= 2.0 / 3)
            {
                double v = Math.PI * (3 * freq - 1);
                double psi = Math.Cos(Math.PI / 2 * Vf(v));
                coefficients[i] = _numOps.FromDouble(psi);
            }
            else
            {
                coefficients[i] = _numOps.FromDouble(0);
            }
        }

        return coefficients;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int size = 1024; // Use a power of 2 for efficient FFT
        var coefficients = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            double freq = (double)i / size;
            if (freq <= 1.0 / 3)
            {
                coefficients[i] = _numOps.FromDouble(0);
            }
            else if (freq <= 2.0 / 3)
            {
                double v = Math.PI * (3 * freq - 1);
                double psi = Math.Sin(Math.PI / 2 * Vf(v));
                coefficients[i] = _numOps.FromDouble(psi);
            }
            else if (freq <= 4.0 / 3)
            {
                double v = Math.PI * (3 / 2 * freq - 1);
                double psi = Math.Sin(Math.PI / 2 * Vf(v));
                coefficients[i] = _numOps.FromDouble(psi);
            }
            else
            {
                coefficients[i] = _numOps.FromDouble(0);
            }
        }

        return coefficients;
    }

    private static double Vf(double x)
    {
        return x * x * (3 - 2 * x);
    }
}