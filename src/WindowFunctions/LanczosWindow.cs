namespace AiDotNet.WindowFunctions;

public class LanczosWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public LanczosWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T x = _numOps.Multiply(_numOps.FromDouble(2 * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)));
            x = _numOps.Subtract(x, _numOps.One);
            window[n] = MathHelper.Sinc(x);
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Lanczos;
}