
namespace AiDotNet.NeuralNetworks;

public class ConvolutionalNeuralNetwork<T> : NeuralNetworkBase<T>
{
    public ConvolutionalNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
    }

    protected override void InitializeLayers()
    {
        if (Architecture.LayerSizes.Count < 2)
        {
            throw new InvalidOperationException("The network must have at least an input and an output layer.");
        }

        for (int i = 0; i < Architecture.LayerSizes.Count - 1; i++)
        {
            if (Architecture.CustomLayers != null && i < Architecture.CustomLayers.Count)
            {
                Layers.Add(Architecture.CustomLayers[i]);
            }
            else
            {
                int inputSize = Architecture.LayerSizes[i];
                int nextSize = Architecture.LayerSizes[i + 1];

                // Add Convolutional Layer
                Layers.Add(new ConvolutionalLayer<T>(
                    inputDepth: inputSize,
                    outputDepth: nextSize,
                    kernelSize: 3,
                    inputHeight: Architecture.InputHeight,
                    inputWidth: Architecture.InputWidth,
                    stride: 1,
                    padding: 1,
                    activation: new ReLUActivation<T>()
                ));

                // Add Activation Layer
                Layers.Add(new ActivationLayer<T>([nextSize], (IActivationFunction<T>)new ReLUActivation<T>()));

                // If not the last layer, add a Pooling Layer
                if (i < Architecture.LayerSizes.Count - 2)
                {
                    Layers.Add(new PoolingLayer<T>(nextSize, nextSize, 2, 2, 2, PoolingType.Max));
                }
            }
        }

        // Add a Flatten Layer before the final Dense Layer
        int lastLayerSize = Architecture.LayerSizes[Architecture.LayerSizes.Count - 2];
        Layers.Add(new FlattenLayer<T>(new int[] { lastLayerSize }));

        // Add the final Dense Layer
        int finalOutputSize = Architecture.LayerSizes[Architecture.LayerSizes.Count - 1];
        Layers.Add(new DenseLayer<T>(lastLayerSize, finalOutputSize));

        // Add the final Activation Layer (typically Softmax for classification tasks)
        // Using IActivationFunction<T> to resolve ambiguity
        Layers.Add(new ActivationLayer<T>(new int[] { finalOutputSize }, (IActivationFunction<T>)new SoftmaxActivation<T>()));
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