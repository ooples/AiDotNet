namespace AiDotNet.NeuralNetworks;

public class DifferentiableNeuralComputer<T> : NeuralNetworkBase<T>
{
    private int MemorySize;
    private int MemoryWordSize;
    private int ControllerSize;
    private int ReadHeads;
    private Matrix<T> Memory;
    private Vector<T> UsageFree;
    private Vector<T> WriteWeighting;
    private List<Vector<T>> ReadWeightings;
    private Vector<T> PrecedenceWeighting;
    private Matrix<T> TemporalLinkMatrix;
    private List<Vector<T>> ReadVectors;

    public DifferentiableNeuralComputer(NeuralNetworkArchitecture<T> architecture, int memorySize, int memoryWordSize, int controllerSize, int readHeads) 
        : base(architecture)
    {
        MemorySize = memorySize;
        MemoryWordSize = memoryWordSize;
        ControllerSize = controllerSize;
        ReadHeads = readHeads;
        Memory = new Matrix<T>(MemorySize, MemoryWordSize);
        UsageFree = new Vector<T>(MemorySize);
        WriteWeighting = new Vector<T>(MemorySize);
        ReadWeightings = [];
        ReadVectors = [];

        for (int i = 0; i < ReadHeads; i++)
        {
            ReadWeightings.Add(new Vector<T>(MemorySize));
        }

        PrecedenceWeighting = new Vector<T>(MemorySize);
        TemporalLinkMatrix = new Matrix<T>(MemorySize, MemorySize);

        InitializeMemory();
        InitializeLayers();
    }

    private void InitializeMemory()
    {
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemoryWordSize; j++)
            {
                Memory[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1);
            }

            UsageFree[i] = NumOps.FromDouble(1.0);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDNCLayers(Architecture, ControllerSize, MemoryWordSize, ReadHeads, CalculateDNCInterfaceSize(MemoryWordSize, ReadHeads)));
        }
    }

    private static int CalculateDNCInterfaceSize(int memoryWordSize, int readHeads)
    {
        return memoryWordSize + // Write vector
               memoryWordSize + // Erase vector
               readHeads * memoryWordSize + // Read vectors
               3 + // Write gate, allocation gate, write mode
               3 * readHeads; // Read modes
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var controllerState = input;
        var readVectors = new List<Vector<T>>();

        for (int i = 0; i < ReadHeads; i++)
        {
            readVectors.Add(new Vector<T>(MemoryWordSize));
        }

        // Controller
        for (int i = 0; i < 2; i++)
        {
            controllerState = Layers[i].Forward(Tensor<T>.FromVector(controllerState)).ToVector();
        }

        // Memory interface
        var interfaceVector = Layers[2].Forward(Tensor<T>.FromVector(controllerState)).ToVector();
        ProcessMemoryInterface(interfaceVector, readVectors);

        // Concatenate controller output with read vectors
        var concatenated = Vector<T>.Concatenate(controllerState, Vector<T>.Concatenate(readVectors));

        // Output layers
        var output = Layers[3].Forward(Tensor<T>.FromVector(concatenated)).ToVector();
        output = Layers[4].Forward(Tensor<T>.FromVector(output)).ToVector();

        return output;
    }

    private void ProcessMemoryInterface(Vector<T> interfaceVector, List<Vector<T>> readVectors)
    {
        int offset = 0;

        // Read vectors
        for (int i = 0; i < ReadHeads; i++)
        {
            readVectors[i] = interfaceVector.SubVector(offset, MemoryWordSize);
            offset += MemoryWordSize;
        }

        // Write vector
        var writeVector = interfaceVector.SubVector(offset, MemoryWordSize);
        offset += MemoryWordSize;

        // Erase vector
        var eraseVector = interfaceVector.SubVector(offset, MemoryWordSize);
        offset += MemoryWordSize;

        // Write gate, allocation gate, write mode
        var writeGate = interfaceVector[offset++];
        var allocateGate = interfaceVector[offset++];
        var writeMode = interfaceVector[offset++];

        // Read modes
        var readModes = new List<Vector<T>>();
        for (int i = 0; i < ReadHeads; i++)
        {
            readModes.Add(interfaceVector.SubVector(offset, 3));
            offset += 3;
        }

        // Update usage vector
        UpdateUsageVector();

        // Allocation weighting
        var allocationWeighting = CalculateAllocationWeighting();

        // Write weighting
        WriteWeighting = CalculateWriteWeighting(writeMode, allocationWeighting);

        // Write to memory
        WriteToMemory(writeVector, eraseVector, writeGate);

        // Update temporal linkage
        UpdateTemporalLinkage();

        // Read from memory
        ReadFromMemory(readModes);
    }

    private void UpdateUsageVector()
    {
        for (int i = 0; i < MemorySize; i++)
        {
            UsageFree[i] = NumOps.Multiply(
                UsageFree[i],
                NumOps.Subtract(NumOps.One, WriteWeighting[i])
            );
        }
    }

    private Vector<T> CalculateAllocationWeighting()
    {
        var sortedUsageFree = UsageFree.OrderBy(x => x).ToList();
        var allocationWeighting = Vector<T>.CreateDefault(MemorySize, NumOps.Zero);

        for (int i = 0; i < MemorySize; i++)
        {
            var product = NumOps.One;
            for (int j = 0; j < i; j++)
            {
                product = NumOps.Multiply(product, NumOps.Subtract(NumOps.One, UsageFree[j]));
            }
            allocationWeighting[i] = NumOps.Multiply(UsageFree[i], product);
        }

        return allocationWeighting;
    }

    private Vector<T> CalculateWriteWeighting(T writeMode, Vector<T> allocationWeighting)
    {
        var contentWeighting = ContentBasedAddressing(WriteWeighting, Memory);
        return allocationWeighting.Multiply(NumOps.Subtract(NumOps.One, writeMode))
            .Add(contentWeighting.Multiply(writeMode));
    }

    private void WriteToMemory(Vector<T> writeVector, Vector<T> eraseVector, T writeGate)
    {
        for (int i = 0; i < MemorySize; i++)
        {
            var eraseAmount = eraseVector.Multiply(WriteWeighting[i]);
            var writeAmount = writeVector.Multiply(WriteWeighting[i]);
            for (int j = 0; j < MemoryWordSize; j++)
            {
                Memory[i, j] = NumOps.Add(
                    NumOps.Multiply(
                        Memory[i, j],
                        NumOps.Subtract(NumOps.One, NumOps.Multiply(eraseAmount[j], writeGate))
                    ),
                    NumOps.Multiply(writeAmount[j], writeGate)
                );
            }
        }
    }

    private void UpdateTemporalLinkage()
    {
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemorySize; j++)
            {
                if (i != j)
                {
                    TemporalLinkMatrix[i, j] = NumOps.Add(
                        NumOps.Multiply(
                            NumOps.Subtract(NumOps.One, WriteWeighting[i]),
                            NumOps.Subtract(NumOps.One, WriteWeighting[j])
                        ),
                        NumOps.Multiply(PrecedenceWeighting[i], WriteWeighting[j])
                    );
                }
            }
        }

        PrecedenceWeighting = PrecedenceWeighting.Multiply(NumOps.Subtract(NumOps.One, WriteWeighting.Sum()))
            .Add(WriteWeighting);
    }

    private void ReadFromMemory(List<Vector<T>> readModes)
    {
        var newReadVectors = new List<Vector<T>>();
        for (int i = 0; i < ReadHeads; i++)
        {
            var backwardWeighting = TemporalLinkMatrix.Multiply(ReadWeightings[i]);
            var forwardWeighting = TemporalLinkMatrix.Transpose().Multiply(ReadWeightings[i]);
            var contentWeighting = ContentBasedAddressing(ReadWeightings[i], Memory);

            ReadWeightings[i] = backwardWeighting.Multiply(readModes[i][0])
                .Add(contentWeighting.Multiply(readModes[i][1]))
                .Add(forwardWeighting.Multiply(readModes[i][2]));

            var readVector = Vector<T>.CreateDefault(MemoryWordSize, NumOps.Zero);
            for (int j = 0; j < MemorySize; j++)
            {
                readVector = readVector.Add(Memory.GetRow(j).Multiply(ReadWeightings[i][j]));
            }
            newReadVectors.Add(readVector);
        }

        ReadVectors = newReadVectors;
    }

    private Vector<T> ContentBasedAddressing(Vector<T> key, Matrix<T> memory)
    {
        var similarities = Vector<T>.CreateDefault(MemorySize, NumOps.Zero);
        for (int i = 0; i < MemorySize; i++)
        {
            similarities[i] = CosineSimilarity(key, memory.GetRow(i));
        }
        return Softmax(similarities);
    }

    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        var dotProduct = a.DotProduct(b);
        var normA = a.Norm();
        var normB = b.Norm();

        return NumOps.Divide(dotProduct, NumOps.Multiply(normA, normB));
    }

    private Vector<T> Softmax(Vector<T> vector)
    {
        var expVector = new Vector<T>(vector.Select(x => NumOps.Exp(x)).ToArray());
        var sum = expVector.Sum();

        return expVector.Divide(sum);
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

        // Serialize memory and other DNC-specific components
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemoryWordSize; j++)
            {
                // Fix null reference warning
                string valueStr = Memory[i, j]?.ToString() ?? string.Empty;
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

        // Deserialize memory and other DNC-specific components
        for (int i = 0; i < MemorySize; i++)
        {
            for (int j = 0; j < MemoryWordSize; j++)
            {
                Memory[i, j] = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
            }
        }
    }
}