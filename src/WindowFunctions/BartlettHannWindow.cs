namespace AiDotNet.WindowFunctions;

public class BartlettHannWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public BartlettHannWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.FromDouble(0.62);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.48), _numOps.Abs(_numOps.Subtract(_numOps.Divide(nT, _numOps.FromDouble(windowSize - 1)), _numOps.FromDouble(0.5))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.38), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI), _numOps.Subtract(_numOps.Divide(nT, _numOps.FromDouble(windowSize - 1)), _numOps.FromDouble(0.5)))));
            window[n] = _numOps.Subtract(_numOps.Subtract(term1, term2), term3);
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.BartlettHann;
}