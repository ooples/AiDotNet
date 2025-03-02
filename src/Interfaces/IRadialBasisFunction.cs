namespace AiDotNet.Interfaces;

public interface IRadialBasisFunction<T>
{
    T Compute(T r);
    T ComputeDerivative(T r);
    T ComputeWidthDerivative(T r);
}