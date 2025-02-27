namespace AiDotNet.WindowFunctions;

public class BlackmanNuttallWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public BlackmanNuttallWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.Multiply(_numOps.FromDouble(0.3635819), _numOps.One);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.4891775), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.1365995), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term4 = _numOps.Multiply(_numOps.FromDouble(0.0106411), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(6 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            window[n] = _numOps.Subtract(_numOps.Subtract(_numOps.Add(term1, term2), term3), term4);
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.BlackmanNuttall;
}