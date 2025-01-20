namespace AiDotNet.WaveletFunctions;

public class GaborWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _omega;
    private readonly T _sigma;
    private readonly T _lambda;
    private readonly T _psi;

    public GaborWavelet(double omega = 5, double sigma = 1, double lambda = 4, double psi = 0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = _numOps.FromDouble(omega);
        _sigma = _numOps.FromDouble(sigma);
        _psi =  _numOps.FromDouble(psi);
        _lambda = _numOps.FromDouble(lambda);
    }

    public T Calculate(T x)
    {
        T gaussianTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(x, x), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(_numOps.Multiply(_omega, x));

        return _numOps.Multiply(gaussianTerm, cosTerm);
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var approximation = new Vector<T>(size, _numOps);
        var detail = new Vector<T>(size, _numOps);

        for (int x = 0; x < size; x++)
        {
            T gaborReal = GaborFunction(x, true);
            T gaborImag = GaborFunction(x, false);

            approximation[x] = _numOps.Multiply(gaborReal, input[x]);
            detail[x] = _numOps.Multiply(gaborImag, input[x]);
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        int size = 100;
        var coefficients = new Vector<T>(size, _numOps);

        for (int x = 0; x < size; x++)
        {
            coefficients[x] = GaborFunction(x, true);
        }

        return coefficients;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        int size = 100;
        var coefficients = new Vector<T>(size, _numOps);

        for (int x = 0; x < size; x++)
        {
            coefficients[x] = GaborFunction(x, false);
        }

        return coefficients;
    }

    private T GaborFunction(int x, bool real)
    {
        T xT = _numOps.FromDouble(x);
        T xTheta = _numOps.Multiply(xT, MathHelper.Cos(_numOps.Divide(MathHelper.Pi<T>(), _numOps.FromDouble(4))));
        T expTerm = _numOps.Exp(_numOps.Divide(_numOps.Negate(_numOps.Multiply(xTheta, xTheta)), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma))));
        T angle = _numOps.Add(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2.0 * Math.PI), xTheta), _lambda), _psi);
        T trigTerm = real ? MathHelper.Cos(angle) : MathHelper.Sin(angle);

        return _numOps.Multiply(expTerm, trigTerm);
    }
}