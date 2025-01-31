namespace AiDotNet.Interfaces;

public interface ILayer<T>
{
    Tensor<T> Forward(Tensor<T> input);
    Tensor<T> Backward(Tensor<T> outputGradient);
    void UpdateParameters(T learningRate);
}