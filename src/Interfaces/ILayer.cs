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
    IEnumerable<ActivationFunction> GetActivationTypes();
    Vector<T> GetParameters();
    bool SupportsTraining { get; }
    void SetTrainingMode(bool isTraining);
    Vector<T> GetParameterGradients();
    void ClearGradients();
    void SetParameters(Vector<T> parameters);
    void ResetState();
}