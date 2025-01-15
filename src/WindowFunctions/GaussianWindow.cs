namespace AiDotNet.WindowFunctions;

public class GaussianWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _sigma;

    public GaussianWindow(double sigma = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = _numOps.FromDouble(sigma);
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        T center = _numOps.Divide(_numOps.FromDouble(windowSize - 1), _numOps.FromDouble(2.0));

        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.Subtract(_numOps.FromDouble(n), center), _numOps.Multiply(_sigma, _numOps.FromDouble(2.0)));
            window[n] = _numOps.Exp(_numOps.Negate(_numOps.Multiply(x, x)));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Gaussian;
}