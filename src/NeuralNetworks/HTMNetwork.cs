namespace AiDotNet.NeuralNetworks;

public class HTMNetwork<T> : NeuralNetworkBase<T>
{
    private int InputSize { get; }
    private int ColumnCount { get; }
    private int CellsPerColumn { get; }
    private double SparsityThreshold { get; }

    /// <summary>
    /// Creates a new Hierarchical Temporal Memory (HTM) network.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="columnCount">The number of columns in the spatial pooler (default: 2048).</param>
    /// <param name="cellsPerColumn">The number of cells per column in the temporal memory (default: 32).</param>
    /// <param name="sparsityThreshold">The target sparsity for the spatial pooler output (default: 0.02).</param>
    public HTMNetwork(
        NeuralNetworkArchitecture<T> architecture, 
        int columnCount = 2048, 
        int cellsPerColumn = 32, 
        double sparsityThreshold = 0.02) 
        : base(architecture)
    {
        var inputShape = Architecture.GetInputShape();
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for HTM network.");
        }
    
        InputSize = inputShape[0];
        ColumnCount = columnCount;
        CellsPerColumn = cellsPerColumn;
        SparsityThreshold = sparsityThreshold;

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultHTMLayers(Architecture, ColumnCount, CellsPerColumn, SparsityThreshold));
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

    public void Learn(Vector<T> input)
    {
        if (input.Length != InputSize)
            throw new ArgumentException($"Input size mismatch. Expected {InputSize}, got {input.Length}.");

        // Forward pass through Spatial Pooler
        if (!(Layers[0] is SpatialPoolerLayer<T> spatialPoolerLayer))
            throw new InvalidOperationException("The first layer is not a SpatialPoolerLayer.");
        var spatialPoolerOutput = spatialPoolerLayer.Forward(Tensor<T>.FromVector(input)).ToVector();

        // Forward pass through Temporal Memory
        if (!(Layers[1] is TemporalMemoryLayer<T> temporalMemoryLayer))
            throw new InvalidOperationException("The second layer is not a TemporalMemoryLayer.");
        var temporalMemoryOutput = temporalMemoryLayer.Forward(Tensor<T>.FromVector(spatialPoolerOutput)).ToVector();

        // Learning in Temporal Memory
        temporalMemoryLayer.Learn(spatialPoolerOutput, temporalMemoryLayer.PreviousState);

        // Update the previous state for the next iteration
        temporalMemoryLayer.PreviousState = temporalMemoryOutput;

        // Learning in Spatial Pooler
        spatialPoolerLayer.Learn(input);

        // Forward pass through remaining layers
        var current = temporalMemoryOutput;
        for (int i = 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }
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