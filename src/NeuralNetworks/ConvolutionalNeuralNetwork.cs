namespace AiDotNet.NeuralNetworks;

public class ConvolutionalNeuralNetwork<T> : NeuralNetworkBase<T>
{
    public ConvolutionalNeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        if (architecture.InputType != InputType.ThreeDimensional)
        {
            throw new ArgumentException("Convolutional Neural Network requires three-dimensional input.");
        }
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCNNLayers(Architecture));
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // Convert the input Vector to a Tensor with the correct shape
        var inputShape = Architecture.GetInputShape();
        var totalSize = inputShape.Aggregate(1, (a, b) => a * b);
    
        if (input.Length != totalSize)
        {
            throw new ArgumentException("Input vector length must match the product of input dimensions.");
        }

        var inputTensor = new Tensor<T>(inputShape, input);

        // Perform forward pass
        var output = Forward(inputTensor);

        // Flatten the output Tensor to a Vector
        return new Vector<T>([.. output]);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        if (!input.Shape.SequenceEqual(Architecture.GetInputShape()))
        {
            throw new ArgumentException("Input shape does not match the expected input shape.");
        }

        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            outputGradient = Layers[i].Backward(outputGradient);
        }
        return outputGradient;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            writer.Write(layer.GetType().FullName ?? throw new InvalidOperationException("Layer type name is null"));
            layer.Serialize(writer);
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
            {
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");
            }

            ILayer<T> layer = (ILayer<T>)Activator.CreateInstance(layerType)!;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }

    public override int GetParameterCount()
    {
        return Layers.Sum(layer => layer.ParameterCount);
    }
}