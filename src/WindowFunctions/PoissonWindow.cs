namespace AiDotNet.WindowFunctions;

public class PoissonWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _alpha;

    public PoissonWindow(double alpha = 2.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _alpha = _numOps.FromDouble(alpha);
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        T halfN = _numOps.Divide(_numOps.FromDouble(windowSize - 1), _numOps.FromDouble(2));

        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.Abs(_numOps.Subtract(_numOps.FromDouble(n), halfN)), halfN);
            window[n] = _numOps.Exp(_numOps.Multiply(_numOps.Negate(_alpha), x));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Poisson;
}