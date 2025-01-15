namespace AiDotNet.WindowFunctions;

public class KaiserWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _beta;

    public KaiserWindow(double beta = 5.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = _numOps.FromDouble(beta);
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        T alphaSq = _numOps.Multiply(_numOps.FromDouble(((windowSize - 1) / 2.0)), _numOps.FromDouble(((windowSize - 1) / 2.0)));

        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(2 * n - windowSize + 1), _numOps.FromDouble(windowSize - 1));
            T arg = _numOps.Sqrt(_numOps.Subtract(_numOps.One, _numOps.Multiply(x, x)));
            window[n] = MathHelper.BesselI0(_numOps.Multiply(_beta, arg));
        }

        // Normalize
        T maxVal = window.Max();
        for (int n = 0; n < windowSize; n++)
        {
            window[n] = _numOps.Divide(window[n], maxVal);
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Kaiser;
}