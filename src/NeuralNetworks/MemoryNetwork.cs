namespace AiDotNet.NeuralNetworks;

public class MemoryNetwork<T> : NeuralNetworkBase<T>
{
    private readonly int _memorySize;
    private readonly int _embeddingSize;
    private Matrix<T> _memory;

    public MemoryNetwork(NeuralNetworkArchitecture<T> architecture, int memorySize, int embeddingSize) : base(architecture)
    {
        _memorySize = memorySize;
        _embeddingSize = embeddingSize;
        _memory = new Matrix<T>(_memorySize, _embeddingSize);

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMemoryNetworkLayers(Architecture, _memorySize, _embeddingSize));
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            if (layer is MemoryReadLayer<T> memoryReadLayer)
            {
                // Convert Matrix<T> to Tensor<T> before passing to Forward
                Tensor<T> memoryTensor = Tensor<T>.FromMatrix(_memory);
                current = memoryReadLayer.Forward(Tensor<T>.FromVector(current), memoryTensor).ToVector();
            }
            else if (layer is MemoryWriteLayer<T> memoryWriteLayer)
            {
                // Convert Matrix<T> to Tensor<T> before passing to Forward
                Tensor<T> memoryTensor = Tensor<T>.FromMatrix(_memory);
                Tensor<T> updatedMemoryTensor = memoryWriteLayer.Forward(Tensor<T>.FromVector(current), memoryTensor);
            
                // Convert the result back to Matrix<T>
                _memory = updatedMemoryTensor.ToMatrix();
                // The output of MemoryWriteLayer is typically not used in the forward pass
            }
            else
            {
                current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            }
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

        // Serialize memory
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                // Fix: Handle potential null values by using the null-coalescing operator
                string valueStr = _memory[i, j]?.ToString() ?? string.Empty;
                writer.Write(valueStr);
            }
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

        // Deserialize memory
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                _memory[i, j] = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
            }
        }
    }
}