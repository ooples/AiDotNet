namespace AiDotNet.NeuralNetworks;

public abstract class NeuralNetworkBase<T> : INeuralNetwork<T>
{
    protected readonly List<Layer<T>> Layers;
    protected readonly NeuralNetworkArchitecture<T> Architecture;

    protected NeuralNetworkBase(NeuralNetworkArchitecture<T> architecture)
    {
        Architecture = architecture;
        Layers = [];
        InitializeLayers();
    }

    protected abstract void InitializeLayers();

    public abstract Vector<T> Predict(Vector<T> input);

    public abstract void UpdateParameters(Vector<T> parameters);

    public abstract void Serialize(BinaryWriter writer);

    public abstract void Deserialize(BinaryReader reader);
}