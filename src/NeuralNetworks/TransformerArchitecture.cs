namespace AiDotNet.NeuralNetworks;

public class TransformerArchitecture<T> : NeuralNetworkArchitecture<T>
{
    public int NumEncoderLayers { get; }
    public int NumDecoderLayers { get; }
    public int NumHeads { get; }
    public int ModelDimension { get; }
    public int FeedForwardDimension { get; }
    public double DropoutRate { get; }
    public int MaxSequenceLength { get; }
    public int VocabularySize { get; }
    public bool UsePositionalEncoding { get; }
    public double Temperature { get; } // For text generation only

    public TransformerArchitecture(
        InputType inputType,
        NeuralNetworkTaskType taskType,
        int numEncoderLayers,
        int numDecoderLayers,
        int numHeads,
        int modelDimension,
        int feedForwardDimension,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int inputHeight = 0,
        int inputWidth = 0,
        int inputDepth = 1,
        int outputSize = 0,
        double dropoutRate = 0.1,
        int maxSequenceLength = 512,
        int vocabularySize = 0,
        bool usePositionalEncoding = true,
        double temperature = 1.0,
        List<ILayer<T>>? layers = null,
        List<RestrictedBoltzmannMachine<T>>? rbmLayers = null)
        : base(
            inputType: inputType, 
            taskType: taskType, 
            complexity: complexity,
            inputSize: inputSize, 
            outputSize: outputSize,
            layers: layers, 
            rbmLayers: rbmLayers)
    {
        NumEncoderLayers = numEncoderLayers;
        NumDecoderLayers = numDecoderLayers;
        NumHeads = numHeads;
        ModelDimension = modelDimension;
        FeedForwardDimension = feedForwardDimension;
        DropoutRate = dropoutRate;
        MaxSequenceLength = maxSequenceLength;
        VocabularySize = vocabularySize;
        UsePositionalEncoding = usePositionalEncoding;
        Temperature = temperature;
    }
}