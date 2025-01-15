namespace AiDotNet.WindowFunctions;

public class CosineWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public CosineWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = MathHelper.Sin(_numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.Divide(nT, _numOps.FromDouble(windowSize - 1))));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Cosine;
}