namespace AiDotNet.WaveletFunctions;

public class GaussianWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _sigma;

    public GaussianWavelet(double sigma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = _numOps.FromDouble(sigma);
    }

    public T Calculate(T x)
    {
        T exponent = _numOps.Divide(_numOps.Multiply(x, x), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)));
        return _numOps.Exp(_numOps.Negate(exponent));
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var approximation = new Vector<T>(size, _numOps);
        var detail = new Vector<T>(size, _numOps);

        for (int i = 0; i < size; i++)
        {
            T x = _numOps.FromDouble(i - size / 2);
            T waveletValue = Calculate(x);
            T derivativeValue = CalculateDerivative(x);

            approximation[i] = _numOps.Multiply(waveletValue, input[i]);
            detail[i] = _numOps.Multiply(derivativeValue, input[i]);
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size, _numOps);

        for (int i = 0; i < size; i++)
        {
            T x = _numOps.FromDouble(i - size / 2);
            coefficients[i] = Calculate(x);
        }

        return coefficients;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size, _numOps);

        for (int i = 0; i < size; i++)
        {
            T x = _numOps.FromDouble(i - size / 2);
            coefficients[i] = CalculateDerivative(x);
        }

        return coefficients;
    }

    private T CalculateDerivative(T x)
    {
        T gaussianValue = Calculate(x);
        T factor = _numOps.Divide(x, _numOps.Multiply(_sigma, _sigma));
        return _numOps.Multiply(_numOps.Negate(factor), gaussianValue);
    }
}