namespace AiDotNet.Interfaces;

public interface IWaveletFunction<T>
{
    T Calculate(T x);
}