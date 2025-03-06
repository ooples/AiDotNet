namespace AiDotNet.NeuralNetworks;

public class GraphNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private IVectorActivationFunction<T>? GraphConvolutionalVectorActivation { get; set; }
    private IVectorActivationFunction<T>? ActivationLayerVectorActivation { get; set; }
    private IVectorActivationFunction<T>? FinalDenseLayerVectorActivation { get; set; }
    private IVectorActivationFunction<T>? FinalActivationLayerVectorActivation { get; set; }
    private IActivationFunction<T>? GraphConvolutionalScalarActivation { get; set; }
    private IActivationFunction<T>? ActivationLayerScalarActivation { get; set; }
    private IActivationFunction<T>? FinalDenseLayerScalarActivation { get; set; }
    private IActivationFunction<T>? FinalActivationLayerScalarActivation { get; set; }

    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture, IVectorActivationFunction<T>? graphConvolutionalVectorActivation = null, 
        IVectorActivationFunction<T>? activationLayerVectorActivation = null, IVectorActivationFunction<T>? finalDenseLayerVectorActivation = null, 
        IVectorActivationFunction<T>? finalActivationLayerVectorActivation = null) : base(architecture)
    {
        GraphConvolutionalVectorActivation = graphConvolutionalVectorActivation;
        ActivationLayerVectorActivation = activationLayerVectorActivation;
        FinalDenseLayerVectorActivation = finalDenseLayerVectorActivation;
        FinalActivationLayerVectorActivation = finalActivationLayerVectorActivation;
    }

    public GraphNeuralNetwork(NeuralNetworkArchitecture<T> architecture, IActivationFunction<T>? graphConvolutionalActivation = null, 
        IActivationFunction<T>? activationLayerActivation = null, IActivationFunction<T>? finalDenseLayerActivation = null, 
        IActivationFunction<T>? finalActivationLayerActivation = null) : base(architecture)
    {
        GraphConvolutionalScalarActivation = graphConvolutionalActivation;
        ActivationLayerScalarActivation = activationLayerActivation;
        FinalDenseLayerScalarActivation = finalDenseLayerActivation;
        FinalActivationLayerScalarActivation = finalActivationLayerActivation;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultGNNLayers(Architecture));
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

    public Tensor<T> PredictGraph(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        if (nodeFeatures == null || adjacencyMatrix == null)
            throw new ArgumentNullException(nodeFeatures == null ? nameof(nodeFeatures) : nameof(adjacencyMatrix));

        if (nodeFeatures.Shape[0] != adjacencyMatrix.Shape[0] || nodeFeatures.Shape[1] != adjacencyMatrix.Shape[1])
            throw new ArgumentException("Node features and adjacency matrix dimensions are incompatible.");

        var current = nodeFeatures;
        foreach (var layer in Layers)
        {
            if (layer is GraphConvolutionalLayer<T> graphLayer)
            {
                current = graphLayer.Forward(current, adjacencyMatrix);
            }
            else if (layer is ILayer<T> standardLayer)
            {
                // Handle non-graph layers (e.g., Dense, Activation)
                current = standardLayer.Forward(current);
            }
            else
            {
                throw new InvalidOperationException($"Unsupported layer type: {layer.GetType().Name}");
            }

            // Ensure the output maintains the expected shape
            if (current.Rank < 2)
                throw new InvalidOperationException($"Layer {layer.GetType().Name} produced an invalid output shape.");
        }

        // Implement hybrid pooling
        return HybridPooling(current);
    }

    private Tensor<T> HybridPooling(Tensor<T> nodeFeatures)
    {
        // Perform different types of pooling
        var meanPooled = nodeFeatures.MeanOverAxis(1);
        var maxPooled = nodeFeatures.MaxOverAxis(1);
        var sumPooled = nodeFeatures.SumOverAxis(1);

        // Concatenate the pooling results
        var concatenated = Tensor<T>.Concatenate([meanPooled, maxPooled, sumPooled], axis: 1);

        // Apply a small neural network to learn the best combination
        var denseLayer = new DenseLayer<T>(concatenated.Shape[1], concatenated.Shape[1] / 2, 
            FinalDenseLayerVectorActivation ?? new ReLUActivation<T>());
        // No separate activation layer needed since activation is included in the dense layer
        var outputLayer = new DenseLayer<T>(concatenated.Shape[1] / 2, meanPooled.Shape[1], 
            FinalActivationLayerVectorActivation ?? new IdentityActivation<T>());
        var activation = new ReLUActivation<T>();

        var hidden = denseLayer.Forward(concatenated);
        hidden = activation.Activate(hidden);
        var output = outputLayer.Forward(hidden);

        return output;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
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