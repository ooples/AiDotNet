namespace AiDotNet.WaveletFunctions;

public class MexicanHatWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _sigma;

    public MexicanHatWavelet(double sigma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = _numOps.FromDouble(sigma);
    }

    public T Calculate(T x)
    {
        T x2 = _numOps.Square(x);
        T sigma2 = _numOps.Square(_sigma);
        T term1 = _numOps.Subtract(_numOps.FromDouble(2.0), _numOps.Divide(x2, sigma2));
        T term2 = _numOps.Exp(_numOps.Negate(_numOps.Divide(x2, _numOps.Multiply(_numOps.FromDouble(2.0), sigma2))));

        return _numOps.Multiply(term1, term2);
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var approximation = new Vector<T>(size);
        var detail = new Vector<T>(size);

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
        var coefficients = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(i - size / 2), _numOps.FromDouble(size / 4));
            coefficients[i] = Calculate(x);
        }

        return coefficients;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size);

        for (int i = 0; i < size; i++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(i - size / 2), _numOps.FromDouble(size / 4));
            coefficients[i] = CalculateDerivative(x);
        }

        return coefficients;
    }

    private T CalculateDerivative(T x)
    {
        T x2 = _numOps.Square(x);
        T sigma2 = _numOps.Square(_sigma);
        T term1 = _numOps.Multiply(x, _numOps.Divide(_numOps.FromDouble(3.0), sigma2));
        T term2 = _numOps.Subtract(_numOps.FromDouble(1.0), _numOps.Divide(x2, sigma2));
        T term3 = _numOps.Exp(_numOps.Negate(_numOps.Divide(x2, _numOps.Multiply(_numOps.FromDouble(2.0), sigma2))));

        return _numOps.Multiply(_numOps.Multiply(term1, term2), term3);
    }
}