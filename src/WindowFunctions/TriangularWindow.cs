namespace AiDotNet.WindowFunctions;

public class TriangularWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public TriangularWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T L = _numOps.FromDouble(windowSize - 1);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = _numOps.Subtract(_numOps.One, _numOps.Abs(_numOps.Divide(_numOps.Subtract(_numOps.Multiply(nT, _numOps.FromDouble(2)), L), L)));
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Triangular;
}