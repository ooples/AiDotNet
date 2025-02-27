namespace AiDotNet.WindowFunctions;

public class HammingWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public HammingWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = _numOps.Add(_numOps.FromDouble(0.54), _numOps.Multiply(_numOps.FromDouble(0.46), 
                MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1))))));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Hamming;
}