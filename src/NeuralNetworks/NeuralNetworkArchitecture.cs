namespace AiDotNet.NeuralNetworks;
public class NeuralNetworkArchitecture<T>
{
    public List<int> LayerSizes { get; set; }
    public List<IActivationFunction<T>>? ActivationFunctions { get; set; }
    public List<IVectorActivationFunction<T>>? VectorActivationFunctions { get; set; }
    public List<ILayer<T>>? CustomLayers { get; set; }
    public int InputHeight { get; set; }
    public int InputWidth { get; set; }

    public NeuralNetworkArchitecture(
        List<int> layerSizes, 
        List<IActivationFunction<T>>? activationFunctions = null,
        List<IVectorActivationFunction<T>>? vectorActivationFunctions = null,
        List<ILayer<T>>? customLayers = null,
        int inputHeight = 0,
        int inputWidth = 0)
    {
        LayerSizes = layerSizes;
        ActivationFunctions = activationFunctions;
        VectorActivationFunctions = vectorActivationFunctions;
        CustomLayers = customLayers;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
    }
}