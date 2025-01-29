namespace AiDotNet.Interfaces;

public interface IVectorActivationFunction<T>
{
    Vector<T> Activate(Vector<T> input);
    Matrix<T> Derivative(Vector<T> input);
}