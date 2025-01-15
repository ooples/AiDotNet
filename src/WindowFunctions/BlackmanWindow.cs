namespace AiDotNet.WindowFunctions;

public class BlackmanWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public BlackmanWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.Multiply(_numOps.FromDouble(0.42), _numOps.One);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.5), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.08), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            window[n] = _numOps.Subtract(_numOps.Subtract(term1, term2), term3);
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Blackman;
}