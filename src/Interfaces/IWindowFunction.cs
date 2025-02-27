namespace AiDotNet.Interfaces;

public interface IWindowFunction<T>
{
    Vector<T> Create(int windowSize);
    WindowFunctionType GetWindowFunctionType();
}