namespace AiDotNet.NeuralNetworks;

public class NeuralTuringMachine<T> : NeuralNetworkBase<T>
{
    private int MemorySize;
    private int MemoryVectorSize;
    private int ControllerSize;
    private Matrix<T> Memory;

    public NeuralTuringMachine(NeuralNetworkArchitecture<T> architecture, int memorySize, int memoryVectorSize, int controllerSize) 
        : base(architecture)
    {
        MemorySize = memorySize;
        MemoryVectorSize = memoryVectorSize;
        ControllerSize = controllerSize;
        Memory = new Matrix<T>(MemorySize, MemoryVectorSize);

        InitializeMemory();
    }

    private void InitializeMemory()
    {
        // Initialize memory with small random values
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemoryVectorSize; j++)
            {
                Memory[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1);
            }
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNTMLayers(Architecture, MemorySize, MemoryVectorSize, ControllerSize));
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        var memoryReadOutput = new Vector<T>(MemoryVectorSize);

        foreach (var layer in Layers)
        {
            if (layer is MemoryReadLayer<T> memoryReadLayer)
            {
                // Convert Matrix to Tensor before passing to Forward method
                memoryReadOutput = memoryReadLayer.Forward(
                    Tensor<T>.FromVector(current), 
                    Tensor<T>.FromMatrix(Memory)).ToVector();
            }
            else if (layer is MemoryWriteLayer<T> memoryWriteLayer)
            {
                // Convert Matrix to Tensor before passing to Forward method
                // and convert the result back to Matrix
                Memory = memoryWriteLayer.Forward(
                    Tensor<T>.FromVector(current), 
                    Tensor<T>.FromMatrix(Memory)).ToMatrix();
            }
            else
            {
                current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            }
        }

        // Concatenate controller output with memory read output
        current = Vector<T>.Concatenate(current, memoryReadOutput);

        // Process through the final layers
        for (int i = Layers.Count - 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
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
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemoryVectorSize; j++)
            {
                // Handle potential null values in Memory
                T value = Memory[i, j];
                string valueString = value?.ToString() ?? string.Empty;
                writer.Write(valueString);
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
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemoryVectorSize; j++)
            {
                Memory[i, j] = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
            }
        }
    }
}