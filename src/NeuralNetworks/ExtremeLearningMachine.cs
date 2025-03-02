namespace AiDotNet.NeuralNetworks;

public class ExtremeLearningMachine<T> : NeuralNetworkBase<T>
{
    private readonly int HiddenLayerSize;

    public ExtremeLearningMachine(NeuralNetworkArchitecture<T> architecture, int hiddenLayerSize) 
        : base(architecture)
    {
        HiddenLayerSize = hiddenLayerSize;

        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultELMLayers(Architecture, HiddenLayerSize));
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    public void Train(Matrix<T> X, Matrix<T> Y)
    {
        // Forward pass through random hidden layer
        var H = X;
        for (int i = 0; i < 2; i++)  // First two layers: Dense and Activation
        {
            H = Layers[i].Forward(Tensor<T>.FromMatrix(H)).ToMatrix();
        }

        // Calculate output weights using pseudo-inverse
        var HTranspose = H.Transpose();
        var HHTranspose = HTranspose.Multiply(H);
        var HHTransposeInverse = HHTranspose.Inverse();
        var outputWeights = HHTransposeInverse.Multiply(HTranspose).Multiply(Y);

        // Set the calculated weights to the output layer
        ((DenseLayer<T>)Layers[2]).SetWeights(outputWeights);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // ELM doesn't update parameters in the traditional sense
        throw new NotImplementedException("ELM does not support traditional parameter updates.");
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }
}