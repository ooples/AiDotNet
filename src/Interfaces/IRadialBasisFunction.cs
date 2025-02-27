namespace AiDotNet.Interfaces;

public interface IRadialBasisFunction<T>
{
    T Compute(T r);
}