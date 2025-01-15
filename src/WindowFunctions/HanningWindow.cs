namespace AiDotNet.WindowFunctions;

public class HanningWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public HanningWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Subtract(_numOps.One, 
                MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1))))));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Hanning;
}