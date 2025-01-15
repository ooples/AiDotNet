namespace AiDotNet.WindowFunctions;

public class BohmanWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public BohmanWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        T N = _numOps.FromDouble(windowSize - 1);

        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(2 * n - windowSize + 1), N);
            T absX = _numOps.Abs(x);

            T term1 = _numOps.Subtract(_numOps.One, absX);
            T term2 = _numOps.Multiply(_numOps.FromDouble(Math.PI), absX);
            T cosTerm = MathHelper.Cos(term2);

            window[n] = _numOps.Multiply(
                _numOps.Subtract(_numOps.One, cosTerm),
                _numOps.Add(_numOps.FromDouble(0.84), _numOps.Multiply(_numOps.FromDouble(0.16), cosTerm))
            );
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Bohman;
}