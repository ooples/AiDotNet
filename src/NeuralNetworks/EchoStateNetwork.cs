namespace AiDotNet.NeuralNetworks;

public class EchoStateNetwork<T> : NeuralNetworkBase<T>
{
    private readonly int ReservoirSize;
    private readonly double SpectralRadius;
    private readonly double Sparsity;
    private Vector<T> ReservoirState;
    private IVectorActivationFunction<T>? ReservoirInputVectorActivation { get; set; }
    private IVectorActivationFunction<T>? ReservoirOutputVectorActivation { get; set; }
    private IVectorActivationFunction<T>? ReservoirVectorActivation { get; set; }
    private IVectorActivationFunction<T>? OutputVectorActivation { get; set; }
    private IActivationFunction<T>? ReservoirInputScalarActivation { get; set; }
    private IActivationFunction<T>? ReservoirOutputScalarActivation { get; set; }
    private IActivationFunction<T>? ReservoirScalarActivation { get; set; }
    private IActivationFunction<T>? OutputScalarActivation { get; set; }

    public EchoStateNetwork(NeuralNetworkArchitecture<T> architecture, int reservoirSize, double spectralRadius = 0.9, double sparsity = 0.1, 
        IVectorActivationFunction<T>? reservoirInputVectorActivation = null, IVectorActivationFunction<T>? reservoirOutputVectorActivation = null, 
        IVectorActivationFunction<T>? reservoirVectorActivation = null, IVectorActivationFunction<T>? outputVectorActivation = null) 
        : base(architecture)
    {
        ReservoirSize = reservoirSize;
        SpectralRadius = spectralRadius;
        Sparsity = sparsity;
        ReservoirState = new Vector<T>(ReservoirSize);
        ReservoirInputVectorActivation = reservoirInputVectorActivation;
        ReservoirOutputVectorActivation = reservoirOutputVectorActivation;
        ReservoirVectorActivation = reservoirVectorActivation;
        OutputVectorActivation = outputVectorActivation;

        InitializeLayers();
    }

    public EchoStateNetwork(NeuralNetworkArchitecture<T> architecture, int reservoirSize, double spectralRadius = 0.9, double sparsity = 0.1, 
        IActivationFunction<T>? reservoirInputScalarActivation = null, IActivationFunction<T>? reservoirOutputScalarActivation = null, 
        IActivationFunction<T>? reservoirScalarActivation = null, IActivationFunction<T>? outputScalarActivation = null) 
        : base(architecture)
    {
        ReservoirSize = reservoirSize;
        SpectralRadius = spectralRadius;
        Sparsity = sparsity;
        ReservoirState = new Vector<T>(ReservoirSize);
        ReservoirInputScalarActivation = reservoirInputScalarActivation;
        ReservoirOutputScalarActivation = reservoirOutputScalarActivation;
        ReservoirScalarActivation = reservoirScalarActivation;
        OutputScalarActivation = outputScalarActivation;

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
            int inputSize = Architecture.GetInputShape()[0];
            int outputSize = Architecture.OutputSize;
        
            Layers.AddRange(LayerHelper<T>.CreateDefaultESNLayers(
                inputSize: inputSize,
                outputSize: outputSize,
                reservoirSize: ReservoirSize,
                spectralRadius: SpectralRadius,
                sparsity: Sparsity
            ));
        }
    }

    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
        {
            throw new InvalidOperationException("Echo State Network must have at least 3 layers: input, reservoir, and output.");
        }

        // ESN-specific validation
        bool hasInputLayer = false;
        bool hasReservoirLayer = false;
        bool hasOutputLayer = false;

        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];

            if (layer is ReservoirLayer<T>)
            {
                if (hasReservoirLayer)
                {
                    throw new InvalidOperationException("Echo State Network should have only one Reservoir Layer.");
                }
                hasReservoirLayer = true;
            }
            else if (layer is DenseLayer<T>)
            {
                if (i == 0)
                {
                    hasInputLayer = true;
                }
                else if (!hasOutputLayer)
                {
                    hasOutputLayer = true;
                }
            }
        }

        if (!hasInputLayer)
        {
            throw new InvalidOperationException("Echo State Network must start with an input layer (DenseLayer).");
        }

        if (!hasReservoirLayer)
        {
            throw new InvalidOperationException("Echo State Network must contain a Reservoir Layer.");
        }

        if (!hasOutputLayer)
        {
            throw new InvalidOperationException("Echo State Network must contain an output layer (DenseLayer).");
        }

        // Ensure the reservoir layer is not the first or last layer
        int reservoirIndex = layers.FindIndex(l => l is ReservoirLayer<T>);
        if (reservoirIndex == 0 || reservoirIndex == layers.Count - 1)
        {
            throw new InvalidOperationException("The Reservoir Layer cannot be the first or last layer in the network.");
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            if (i == 1) // Reservoir layer
            {
                var reservoirLayer = (ReservoirLayer<T>)Layers[i];
                ReservoirState = reservoirLayer.Forward(Tensor<T>.FromVector(current), Tensor<T>.FromVector(ReservoirState)).ToVector();
                current = ReservoirState;
            }
            else
            {
                current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
            }
        }

        return current;
    }

    public void Train(Matrix<T> X, Matrix<T> Y)
    {
        // Collect reservoir states
        var states = new List<Vector<T>>();
        for (int i = 0; i < X.Rows; i++)
        {
            var input = X.GetRow(i);
            Predict(input); // This updates the ReservoirState
            states.Add(ReservoirState);
        }

        // Concatenate states into a matrix
        var stateMatrix = new Matrix<T>(states.Count, ReservoirSize);
        for (int i = 0; i < states.Count; i++)
        {
            stateMatrix.SetRow(i, states[i]);
        }

        // Calculate output weights using ridge regression
        var regularization = NumOps.FromDouble(1e-8); // Small regularization term
        var stateTranspose = stateMatrix.Transpose();
        var outputWeights = stateTranspose.Multiply(stateMatrix)
            .Add(Matrix<T>.CreateIdentity(ReservoirSize).Multiply(regularization))
            .Inverse()
            .Multiply(stateTranspose)
            .Multiply(Y);

        // Set the calculated weights to the output layer
        ((DenseLayer<T>)Layers[3]).UpdateParameters(outputWeights.Flatten());
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // ESN doesn't update parameters in the traditional sense
        throw new NotImplementedException("ESN does not support traditional parameter updates.");
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