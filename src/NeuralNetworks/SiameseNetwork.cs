namespace AiDotNet.NeuralNetworks;

public class SiameseNetwork<T> : NeuralNetworkBase<T>
{
    private ConvolutionalNeuralNetwork<T> _subnetwork;
    private DenseLayer<T> _outputLayer;

    public SiameseNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        _subnetwork = new ConvolutionalNeuralNetwork<T>(architecture);
        int embeddingSize = architecture.GetOutputShape()[0];
        _outputLayer = new DenseLayer<T>(embeddingSize * 2, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    protected override void InitializeLayers()
    {
        // The layers are initialized in the subnetwork constructor
    }

    public Vector<T> PredictPair(Vector<T> input1, Vector<T> input2)
    {
        var embedding1 = GetEmbedding(input1);
        var embedding2 = GetEmbedding(input2);
        var combinedEmbedding = CombineEmbeddings(embedding1, embedding2);

        return _outputLayer.Forward(Tensor<T>.FromVector(combinedEmbedding)).ToVector();
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return GetEmbedding(input);
    }

    private Vector<T> GetEmbedding(Vector<T> input)
    {
        return _subnetwork.Predict(input);
    }

    private Vector<T> CombineEmbeddings(Vector<T> embedding1, Vector<T> embedding2)
    {
        var combined = new Vector<T>(embedding1.Length * 2);
        for (int i = 0; i < embedding1.Length; i++)
        {
            combined[i] = embedding1[i];
            combined[i + embedding1.Length] = embedding2[i];
        }

        return combined;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int subnetworkParameterCount = _subnetwork.GetParameterCount();
        Vector<T> subnetworkParameters = parameters.SubVector(0, subnetworkParameterCount);
        _subnetwork.UpdateParameters(subnetworkParameters);

        Vector<T> outputLayerParameters = parameters.SubVector(subnetworkParameterCount, _outputLayer.ParameterCount);
        _outputLayer.UpdateParameters(outputLayerParameters);
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        _subnetwork.Serialize(writer);
        _outputLayer.Serialize(writer);
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        _subnetwork.Deserialize(reader);
        _outputLayer.Deserialize(reader);
    }

    public override int GetParameterCount()
    {
        return _subnetwork.GetParameterCount() + _outputLayer.ParameterCount;
    }
}