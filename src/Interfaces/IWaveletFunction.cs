namespace AiDotNet.Interfaces;

public interface IWaveletFunction<T>
{
    T Calculate(T x);
    (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input);
    Vector<T> GetScalingCoefficients();
    Vector<T> GetWaveletCoefficients();
}