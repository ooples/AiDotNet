namespace AiDotNet.Interfaces;

public interface IVectorActivationFunction<T>
{
    Vector<T> Activate(Vector<T> input);
    Matrix<T> Derivative(Vector<T> input);
    Tensor<T> Activate(Tensor<T> input);
    Tensor<T> Derivative(Tensor<T> input);
}