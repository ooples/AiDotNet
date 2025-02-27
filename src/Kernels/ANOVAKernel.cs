namespace AiDotNet.Kernels;

public class ANOVAKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly int _degree;
    private readonly INumericOperations<T> _numOps;

    public ANOVAKernel(T? sigma = default, int degree = 2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
        _degree = degree;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T result = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T term = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Square(_numOps.Subtract(x1[i], x2[i])), _numOps.Multiply(_numOps.FromDouble(2), _numOps.Square(_sigma)))));
            result = _numOps.Add(result, _numOps.Power(term, _numOps.FromDouble(_degree)));
        }

        return result;
    }
}