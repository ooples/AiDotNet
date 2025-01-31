namespace AiDotNet.WindowFunctions;

public class BartlettWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public BartlettWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T N = _numOps.FromDouble(windowSize - 1);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = _numOps.Subtract(_numOps.One, _numOps.Abs(_numOps.Multiply(_numOps.FromDouble(2), _numOps.Divide(_numOps.Subtract(nT, _numOps.Divide(N, _numOps.FromDouble(2))), N))));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Bartlett;
}