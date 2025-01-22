namespace AiDotNet.Interfaces;

public interface I2DInterpolation<T>
{
    T Interpolate(T x, T y);
}