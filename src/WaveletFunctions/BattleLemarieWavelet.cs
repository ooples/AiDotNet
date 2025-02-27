namespace AiDotNet.WaveletFunctions;

public class BattleLemarieWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public BattleLemarieWavelet(int order = 1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        T result = _numOps.Zero;
        for (int k = -_order; k <= _order; k++)
        {
            T term = BSpline(_numOps.Add(x, _numOps.FromDouble(k)));
            result = _numOps.Add(result, _numOps.Multiply(_numOps.FromDouble(Math.Pow(-1, k)), term));
        }

        return result;
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Battle-Lemarie wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        var scalingCoeffs = GetScalingCoefficients();
        var waveletCoeffs = GetWaveletCoefficients();

        for (int i = 0; i < halfLength; i++)
        {
            T approx = _numOps.Zero;
            T det = _numOps.Zero;

            for (int j = 0; j < scalingCoeffs.Length; j++)
            {
                int index = (2 * i + j) % input.Length;
                approx = _numOps.Add(approx, _numOps.Multiply(scalingCoeffs[j], input[index]));
                det = _numOps.Add(det, _numOps.Multiply(waveletCoeffs[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        int order = _order;
        int numCoeffs = 4 * order + 1; // Increased support for better accuracy
        Vector<T> coeffs = new Vector<T>(numCoeffs);

        // Calculate Battle-Lemarie coefficients using Fourier transform
        int numSamples = 1024; // Number of samples for FFT, adjust as needed
        Vector<Complex<T>> fftInput = new Vector<Complex<T>>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            T omega = _numOps.Multiply(_numOps.FromDouble(2 * Math.PI * i), _numOps.FromDouble(1.0 / numSamples));
            Complex<T> bSplineFourier = BSplineFourier(omega, order);
            T denominator = _numOps.Sqrt(SumSquaredBSplineFourier(omega, order));
            fftInput[i] = new Complex<T>(_numOps.Divide(bSplineFourier.Real, denominator), _numOps.Divide(bSplineFourier.Imaginary, denominator));
        }

        // Perform inverse FFT
        Vector<Complex<T>> fftOutput = InverseFFT(fftInput);

        // Extract and normalize coefficients
        int center = numSamples / 2;
        for (int i = 0; i < numCoeffs; i++)
        {
            coeffs[i] = fftOutput[(center - numCoeffs / 2 + i + numSamples) % numSamples].Real;
        }

        // Normalize the coefficients
        T sum = coeffs.Sum();
        for (int i = 0; i < coeffs.Length; i++)
        {
            coeffs[i] = _numOps.Divide(coeffs[i], sum);
        }

        return coeffs;
    }

    private Complex<T> BSplineFourier(T omega, int order)
    {
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        Complex<T> result = new Complex<T>(_numOps.One, _numOps.Zero);
        for (int i = 0; i < order; i++)
        {
            T sinc = _numOps.Divide(MathHelper.Sin(_numOps.Divide(omega, _numOps.FromDouble(2))), _numOps.Divide(omega, _numOps.FromDouble(2)));
            result = complexOps.Multiply(result, new Complex<T>(sinc, _numOps.Zero));
        }

        return result;
    }

    private T SumSquaredBSplineFourier(T omega, int order)
    {
        T sum = _numOps.Zero;
        for (int k = -order; k <= order; k++)
        {
            T shiftedOmega = _numOps.Add(omega, _numOps.Multiply(_numOps.FromDouble(2 * Math.PI), _numOps.FromDouble(k)));
            Complex<T> bSpline = BSplineFourier(shiftedOmega, order);
            sum = _numOps.Add(sum, _numOps.Add(_numOps.Multiply(bSpline.Real, bSpline.Real), _numOps.Multiply(bSpline.Imaginary, bSpline.Imaginary)));
        }

        return sum;
    }

    private Vector<Complex<T>> InverseFFT(Vector<Complex<T>> input)
    {
        int n = input.Length;
        Vector<Complex<T>> output = new Vector<Complex<T>>(n);

        for (int k = 0; k < n; k++)
        {
            var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
            Complex<T> sum = new Complex<T>(_numOps.Zero, _numOps.Zero);
            for (int t = 0; t < n; t++)
            {
                T angle = _numOps.Multiply(_numOps.FromDouble(2 * Math.PI * t * k), _numOps.FromDouble(1.0 / n));
                Complex<T> exp = new Complex<T>(MathHelper.Cos(angle), MathHelper.Sin(angle));
                sum = complexOps.Add(sum, complexOps.Multiply(input[t], exp));
            }

            output[k] = new Complex<T>(_numOps.Divide(sum.Real, _numOps.FromDouble(n)), _numOps.Divide(sum.Imaginary, _numOps.FromDouble(n)));
        }

        return output;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        var scalingCoeffs = GetScalingCoefficients();
        int L = scalingCoeffs.Length;
        var waveletCoeffs = new T[L];

        for (int i = 0; i < L; i++)
        {
            waveletCoeffs[i] = _numOps.Multiply(_numOps.FromDouble(Math.Pow(-1, i)), scalingCoeffs[L - 1 - i]);
        }

        return new Vector<T>(waveletCoeffs);
    }

    private T BSpline(T x)
    {
        T absX = _numOps.Abs(x);
        if (_numOps.LessThan(absX, _numOps.FromDouble(0.5)))
        {
            return _numOps.One;
        }
        else if (_numOps.LessThanOrEquals(absX, _numOps.FromDouble(1.5)))
        {
            T temp = _numOps.Subtract(_numOps.FromDouble(1.5), absX);
            return _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(temp, temp));
        }

        return _numOps.Zero;
    }
}