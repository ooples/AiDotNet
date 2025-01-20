namespace AiDotNet.WaveletFunctions;

public class ComplexGaussianWavelet<T> : IWaveletFunction<Complex<T>>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public ComplexGaussianWavelet(int order = 1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T gaussianTerm = _numOps.Exp(_numOps.Negate(_numOps.Multiply(x, x)));
        T polynomialTerm = HermitePolynomial(x, _order);

        T real = _numOps.Multiply(gaussianTerm, polynomialTerm);
        T imag = _numOps.Zero; // The imaginary part is zero for real input

        return new Complex<T>(real, imag);
    }

    public (Vector<Complex<T>> approximation, Vector<Complex<T>> detail) Decompose(Vector<Complex<T>> input)
    {
        var lowPass = GetScalingCoefficients();
        var highPass = GetWaveletCoefficients();

        var approximation = Convolve(input, lowPass);
        var detail = Convolve(input, highPass);

        // Downsample by 2
        approximation = ComplexGaussianWavelet<T>.Downsample(approximation, 2);
        detail = ComplexGaussianWavelet<T>.Downsample(detail, 2);

        return (approximation, detail);
    }

    public Vector<Complex<T>> GetScalingCoefficients()
    {
        var errorTolerance = _numOps.FromDouble(1e-6);
        var sigma = _numOps.FromDouble(Math.Sqrt(_order));

        if (_numOps.LessThanOrEquals(sigma, _numOps.Zero))
            throw new ArgumentException("Sigma must be positive", nameof(sigma));

        if (_numOps.LessThanOrEquals(errorTolerance, _numOps.Zero))
            throw new ArgumentException("Error tolerance must be positive", nameof(errorTolerance));

        int length = DetermineAdaptiveLength(sigma, errorTolerance);
        var coeffs = new Complex<T>[length];

        T sum = _numOps.Zero;
        T centerIndex = _numOps.FromDouble((length - 1) / 2.0);

        for (int i = 0; i < length; i++)
        {
            T x = _numOps.Divide(_numOps.Subtract(_numOps.FromDouble(i), centerIndex), sigma);
            T gaussianValue = CalculateGaussianValue(x);
            coeffs[i] = new Complex<T>(gaussianValue, _numOps.Zero);
            sum = _numOps.Add(sum, gaussianValue);
        }

        // Normalize coefficients
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(_numOps.Divide(coeffs[i].Real, sum), _numOps.Zero);
        }

        return new Vector<Complex<T>>(coeffs);
    }

    private int DetermineAdaptiveLength(T sigma, T errorTolerance)
    {
        int length = 1;
        T x = _numOps.Zero;
        T gaussianValue;

        do
        {
            length += 2; // Ensure odd length for symmetry
            x = _numOps.Divide(_numOps.FromDouble(length / 2), sigma);
            gaussianValue = CalculateGaussianValue(x);
        } while (_numOps.GreaterThan(gaussianValue, errorTolerance));

        return length;
    }

    private T CalculateGaussianValue(T x)
    {
        return _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(x, x), _numOps.FromDouble(2))));
    }

    public Vector<Complex<T>> GetWaveletCoefficients()
    {
        int length = 10 * _order;
        var coeffs = new Complex<T>[length];
        T sigma = _numOps.FromDouble(Math.Sqrt(_order));

        for (int i = 0; i < length; i++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(i - length / 2), sigma);
            T gaussianValue = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(x, x), _numOps.FromDouble(2))));
            T sinValue = MathHelper.Sin(x);
            T cosValue = MathHelper.Cos(x);

            coeffs[i] = new Complex<T>(
                _numOps.Multiply(gaussianValue, cosValue),
                _numOps.Multiply(gaussianValue, sinValue)
            );
        }

        return new Vector<Complex<T>>(coeffs);
    }

    private Vector<Complex<T>> Convolve(Vector<Complex<T>> input, Vector<Complex<T>> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new Complex<T>[resultLength];
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        for (int i = 0; i < resultLength; i++)
        {
            Complex<T> sum = new Complex<T>(_numOps.Zero, _numOps.Zero);
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum = complexOps.Add(sum, complexOps.Multiply(input[i - j], kernel[j]));
                }
            }

            result[i] = sum;
        }

        return new Vector<Complex<T>>(result);
    }

    private static Vector<Complex<T>> Downsample(Vector<Complex<T>> input, int factor)
    {
        int resultLength = input.Length / factor;
        var result = new Complex<T>[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            result[i] = input[i * factor];
        }

        return new Vector<Complex<T>>(result);
    }

    private T HermitePolynomial(T x, int n)
    {
        if (n == 0) return _numOps.One;
        if (n == 1) return _numOps.Multiply(_numOps.FromDouble(2), x);

        T h0 = _numOps.One;
        T h1 = _numOps.Multiply(_numOps.FromDouble(2), x);

        for (int i = 2; i <= n; i++)
        {
            T hi = _numOps.Subtract(_numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(2), x), h1), _numOps.Multiply(_numOps.FromDouble(2 * (i - 1)), h0));
            h0 = h1;
            h1 = hi;
        }

        return h1;
    }
}