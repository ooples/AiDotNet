namespace AiDotNet.NeuralNetworks;

public class NeuralNetworkArchitecture<T>
{
    public List<int> LayerSizes { get; set; } = [];
    public List<IActivationFunction<T>>? ActivationFunctions { get; set; } 
    public List<IVectorActivationFunction<T>>? VectorActivationFunctions { get; set; }
}