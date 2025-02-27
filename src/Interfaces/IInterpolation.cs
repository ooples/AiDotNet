namespace AiDotNet.Interfaces;

public interface IInterpolation<T>
{
    T Interpolate(T x);
}