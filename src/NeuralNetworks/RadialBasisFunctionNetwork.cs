namespace AiDotNet.NeuralNetworks;

public class RadialBasisFunctionNetwork<T> : NeuralNetworkBase<T>
{
    private int InputSize { get; set; }
    private int HiddenSize { get; set; }
    private int OutputSize { get; set; }
    private IRadialBasisFunction<T> RadialBasisFunction { get; set; }

    public RadialBasisFunctionNetwork(NeuralNetworkArchitecture<T> architecture, IRadialBasisFunction<T>? radialBasisFunction = null) : base(architecture)
    {
        // Get the input shape and output size from the architecture
        var inputShape = architecture.GetInputShape();
        int outputSize = architecture.OutputSize;
    
        // For RBF networks, we need to determine the hidden layer size
        // If the architecture has custom layers defined, we can extract it from there
        // Otherwise, we'll use a default or specified value
        int hiddenSize;
        if (architecture.Layers != null && architecture.Layers.Count >= 2)
        {
            // Extract hidden size from the architecture's layers if available
            hiddenSize = architecture.Layers[1].GetOutputShape()[0];
        }
        else
        {
            hiddenSize = 64; // Default to 64 if not specified
        }
    
        // Validate the network structure
        if (inputShape == null || inputShape.Length == 0 || outputSize <= 0)
        {
            throw new ArgumentException("RBFN requires valid input shape and output size specifications.");
        }
    
        // Set the properties
        InputSize = inputShape[0]; // Assuming 1D input for simplicity
        HiddenSize = hiddenSize;
        OutputSize = outputSize;
    
        // Default to Gaussian RBF if not specified
        RadialBasisFunction = radialBasisFunction ?? new GaussianRBF<T>();
    
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultRBFNetworkLayers(Architecture));
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

        // Serialize the RBF function type
        writer.Write(RadialBasisFunction.GetType().FullName ?? throw new InvalidOperationException("Unable to get full name for RBF function type"));
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

        // Deserialize the RBF function type
        string rbfTypeName = reader.ReadString();
        Type? rbfType = Type.GetType(rbfTypeName);
        if (rbfType == null)
            throw new InvalidOperationException($"Cannot find type {rbfTypeName}");

        if (!typeof(IRadialBasisFunction<T>).IsAssignableFrom(rbfType))
            throw new InvalidOperationException($"Type {rbfTypeName} does not implement IRadialBasisFunction<T>");

        object? rbfInstance = Activator.CreateInstance(rbfType);
        if (rbfInstance == null)
            throw new InvalidOperationException($"Failed to create an instance of {rbfTypeName}");

        RadialBasisFunction = (IRadialBasisFunction<T>)rbfInstance;
    }
}