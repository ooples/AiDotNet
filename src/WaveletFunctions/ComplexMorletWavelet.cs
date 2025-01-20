namespace AiDotNet.WaveletFunctions;

public class ComplexMorletWavelet<T> : IWaveletFunction<Complex<T>>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _omega;
    private readonly T _sigma;

    public ComplexMorletWavelet(double omega = 5.0, double sigma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = _numOps.FromDouble(omega);
        _sigma = _numOps.FromDouble(sigma);
    }

    public Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T expTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Add(_numOps.Multiply(x, x), _numOps.Multiply(y, y)), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(_numOps.Multiply(_omega, x));
        T sinTerm = MathHelper.Sin(_numOps.Multiply(_omega, x));

        T real = _numOps.Multiply(expTerm, cosTerm);
        T imag = _numOps.Multiply(expTerm, sinTerm);

        return new Complex<T>(real, imag);
    }

    public (Vector<Complex<T>> approximation, Vector<Complex<T>> detail) Decompose(Vector<Complex<T>> input)
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

    public Vector<Complex<T>> GetScalingCoefficients()
    {
        int length = 64; // Adjust as needed
        var coeffs = new Complex<T>[length];
        T sum = _numOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(i - length / 2), _numOps.FromDouble(length / 4));
            T value = _numOps.Equals(x, _numOps.Zero)
                ? _numOps.One
                : _numOps.Divide(MathHelper.Sin(_numOps.Divide(MathHelper.Pi<T>(), x)), _numOps.Multiply(MathHelper.Pi<T>(), x));
            coeffs[i] = new Complex<T>(value, _numOps.Zero);
            sum = _numOps.Add(sum, _numOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(_numOps.Divide(coeffs[i].Real, sum), _numOps.Zero);
        }

        return new Vector<Complex<T>>(coeffs);
    }

    public Vector<Complex<T>> GetWaveletCoefficients()
    {
        int length = 256; // Adjust based on desired precision
        var coeffs = new Complex<T>[length];
        T sum = _numOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T t = _numOps.Divide(_numOps.FromDouble(i - length / 2), _numOps.FromDouble(length / 4));
            T realPart = MathHelper.Cos(_numOps.Multiply(_omega, t));
            T imagPart = MathHelper.Sin(_numOps.Multiply(_omega, t));
            T envelope = _numOps.Exp(_numOps.Divide(_numOps.Negate(_numOps.Multiply(t, t)), _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(_sigma, _sigma))));

            coeffs[i] = new Complex<T>(
                _numOps.Multiply(envelope, realPart),
                _numOps.Multiply(envelope, imagPart)
            );
            sum = _numOps.Add(sum, _numOps.Abs(coeffs[i].Real));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(
                _numOps.Divide(coeffs[i].Real, sum),
                _numOps.Divide(coeffs[i].Imaginary, sum)
            );
        }

        return new Vector<Complex<T>>(coeffs);
    }

    private Vector<Complex<T>> Convolve(Vector<Complex<T>> input, Vector<Complex<T>> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new Complex<T>[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            Complex<T> sum = new(_numOps.Zero, _numOps.Zero);
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum += input[i - j] * kernel[j];
                }
            }

            result[i] = sum;
        }

        return new Vector<Complex<T>>(result);
    }

    private Vector<Complex<T>> Downsample(Vector<Complex<T>> input, int factor)
    {
        int newLength = input.Length / factor;
        var result = new Complex<T>[newLength];

        for (int i = 0; i < newLength; i++)
        {
            result[i] = input[i * factor];
        }

        return new Vector<Complex<T>>(result);
    }
}