namespace AiDotNet.Interfaces;

public interface ILayer<T>
{
    int[] GetInputShape();
    int[] GetOutputShape();
    Tensor<T> Forward(Tensor<T> input);
    Tensor<T> Backward(Tensor<T> outputGradient);
    void UpdateParameters(T learningRate);
    void UpdateParameters(Vector<T> parameters);
    int ParameterCount { get; }
    void Serialize(BinaryWriter writer);
    void Deserialize(BinaryReader reader);
}