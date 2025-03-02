namespace AiDotNet.NeuralNetworks;

public class Autoencoder<T> : NeuralNetworkBase<T>
{
    public int EncodedSize { get; private set; }

    public Autoencoder(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        EncodedSize = 0;

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAutoEncoderLayers(Architecture));
        }

        // Set EncodedSize based on the middle layer
        EncodedSize = Layers[Layers.Count / 2].GetOutputShape()[0];
    }

    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
        {
            throw new ArgumentException("Autoencoder must have at least 3 layers (input, encoded, and output).");
        }

        // Check if input and output layers have the same size
        if (!Enumerable.SequenceEqual(layers[0].GetInputShape(), layers[layers.Count - 1].GetOutputShape()))
        {
            throw new ArgumentException("Input and output layer sizes must be the same for an autoencoder.");
        }

        // Ensure the architecture is symmetric
        for (int i = 0; i < layers.Count / 2; i++)
        {
            if (!Enumerable.SequenceEqual(layers[i].GetOutputShape(), layers[layers.Count - 1 - i].GetInputShape()))
            {
                throw new ArgumentException($"Layer sizes must be symmetric. Mismatch at position {i} and {layers.Count - i - 1}");
            }
        }

        // Validate activation functions
        for (int i = 0; i < layers.Count / 2; i++)
        {
            var leftActivation = layers[i].GetActivationTypes();
            var rightActivation = layers[layers.Count - 1 - i].GetActivationTypes();

            if (!Enumerable.SequenceEqual(leftActivation, rightActivation))
            {
                throw new ArgumentException($"Activation functions must be symmetric. Mismatch at position {i} and {layers.Count - i - 1}");
            }
        }
    }

    public Vector<T> Encode(Vector<T> input)
    {
        var current = input;
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    public Vector<T> Decode(Vector<T> encodedInput)
    {
        var current = encodedInput;
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return Decode(Encode(input));
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