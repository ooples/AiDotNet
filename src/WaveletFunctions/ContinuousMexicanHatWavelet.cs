namespace AiDotNet.WaveletFunctions;

public class ContinuousMexicanHatWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ContinuousMexicanHatWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        T exp_term = _numOps.Exp(_numOps.Negate(_numOps.Divide(x2, _numOps.FromDouble(2))));

        T term1 = _numOps.Subtract(_numOps.One, x2);
        T result = _numOps.Multiply(term1, exp_term);

        // Normalization factor
        double norm_factor = 2.0 / (Math.Sqrt(3) * Math.Pow(Math.PI, 0.25));
        result = _numOps.Multiply(result, _numOps.FromDouble(norm_factor));

        return result;
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
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

    public Vector<T> GetScalingCoefficients()
    {
        int length = 64;
        var coeffs = new T[length];
        T sum = _numOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(i - length / 2), _numOps.FromDouble(length / 4));
            T value = _numOps.Equals(x, _numOps.Zero)
                ? _numOps.One
                : _numOps.Divide(MathHelper.Sin(_numOps.Divide(MathHelper.Pi<T>(), x)), _numOps.Multiply(MathHelper.Pi<T>(), x));
            coeffs[i] = value;
            sum = _numOps.Add(sum, _numOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = _numOps.Divide(coeffs[i], sum);
        }

        return new Vector<T>(coeffs);
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int length = 256;
        var coeffs = new T[length];
        T sum = _numOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T t = _numOps.Divide(_numOps.FromDouble(i - length / 2), _numOps.FromDouble(length / 4));
            T t2 = _numOps.Multiply(t, t);
            T value = _numOps.Multiply(
                _numOps.Subtract(_numOps.One, t2),
                _numOps.Exp(_numOps.Divide(_numOps.Negate(t2), _numOps.FromDouble(2.0)))
            );
            coeffs[i] = value;
            sum = _numOps.Add(sum, _numOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = _numOps.Divide(coeffs[i], sum);
        }

        return new Vector<T>(coeffs);
    }

    private Vector<T> Convolve(Vector<T> input, Vector<T> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new T[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(input[i - j], kernel[j]));
                }
            }

            result[i] = sum;
        }

        return new Vector<T>(result);
    }

    private Vector<T> Downsample(Vector<T> input, int factor)
    {
        int newLength = input.Length / factor;
        var result = new T[newLength];

        for (int i = 0; i < newLength; i++)
        {
            result[i] = input[i * factor];
        }

        return new Vector<T>(result);
    }
}