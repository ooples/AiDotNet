namespace AiDotNet.Interfaces;

public interface IActivationFunction<T>
{
    T Activate(T input);
    T Derivative(T input);
}