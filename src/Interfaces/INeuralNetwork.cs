namespace AiDotNet.Interfaces;

public interface INeuralNetwork<T>
{
    Vector<T> Predict(Vector<T> input);
    void UpdateParameters(Vector<T> parameters);
    void Serialize(BinaryWriter writer);
    void Deserialize(BinaryReader reader);
}