global using AiDotNet.WaveletFunctions;

namespace AiDotNet.Kernels;

public class WaveletKernel<T> : IKernelFunction<T>
{
    private readonly T _a;
    private readonly T _c;
    private readonly INumericOperations<T> _numOps;
    private readonly IWaveletFunction<T> _waveletFunction;

    public WaveletKernel(IWaveletFunction<T>? waveletFunction = null, T? a = default, T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _a = a ?? _numOps.One;
        _c = c ?? _numOps.One;
        _waveletFunction = waveletFunction ?? new MexicanHatWavelet<T>();
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T product = _numOps.One;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Subtract(x1[i], x2[i]);
            T scaledDiff = _numOps.Divide(diff, _a);
            product = _numOps.Multiply(product, _numOps.Multiply(_waveletFunction.Calculate(scaledDiff), _numOps.Sqrt(_c)));
        }

        return product;
    }
}