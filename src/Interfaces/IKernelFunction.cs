namespace AiDotNet.Interfaces;

public interface IKernelFunction<T>
{
    T Calculate(Vector<T> x1, Vector<T> x2);
}