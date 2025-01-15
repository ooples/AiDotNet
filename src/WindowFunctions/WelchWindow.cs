namespace AiDotNet.WindowFunctions;

public class WelchWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public WelchWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        T N = _numOps.FromDouble(windowSize - 1);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term = _numOps.Divide(_numOps.Subtract(nT, _numOps.Divide(N, _numOps.FromDouble(2))), _numOps.Divide(N, _numOps.FromDouble(2)));
            window[n] = _numOps.Subtract(_numOps.One, _numOps.Multiply(term, term));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Welch;
}