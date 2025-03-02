namespace AiDotNet.NeuralNetworks;

public class DeepBeliefNetwork<T> : NeuralNetworkBase<T>
{
    private List<RestrictedBoltzmannMachine<T>> RBMLayers { get; set; }

    public DeepBeliefNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        RBMLayers = architecture.RbmLayers;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepBeliefNetworkLayers(Architecture));
        }

        ValidateAndInitializeRbmLayers();
    }

    private void ValidateAndInitializeRbmLayers()
    {
        if (Architecture.RbmLayers == null || Architecture.RbmLayers.Count == 0)
        {
            throw new InvalidOperationException("RBM layers are required for a Deep Belief Network but none were provided.");
        }

        // Validate RBM layers
        for (int i = 0; i < Architecture.RbmLayers.Count; i++)
        {
            var rbm = Architecture.RbmLayers[i];
            if (rbm == null)
            {
                throw new InvalidOperationException($"RBM layer at index {i} is null.");
            }

            if (i > 0)
            {
                var prevRbm = Architecture.RbmLayers[i - 1];
                if (rbm.VisibleSize != prevRbm.HiddenSize)
                {
                    throw new InvalidOperationException($"Mismatch in RBM layer dimensions. Layer {i-1} hidden size ({prevRbm.HiddenSize}) " +
                        $"do not match layer {i} visible size ({rbm.VisibleSize}).");
                }
            }
            else
            {
                // Check if the first RBM layer matches the input dimension
                if (rbm.VisibleSize != Architecture.CalculatedInputSize)
                {
                    throw new InvalidOperationException($"The first RBM layer's visible units ({rbm.VisibleSize}) " +
                        $"do not match the network's calculated input size ({Architecture.CalculatedInputSize}).");
                }
            }
        }

        // If validation passes, initialize RBMLayers
        RBMLayers = [.. Architecture.RbmLayers];
    }

    public void PretrainRBMs(Tensor<T> trainingData, int epochs, T learningRate)
    {
        for (int i = 0; i < RBMLayers.Count; i++)
        {
            Console.WriteLine($"Pretraining RBM layer {i + 1}");
            var rbm = RBMLayers[i];

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                rbm.Train(trainingData, 1, learningRate);
            }

            // Transform the data through this RBM for the next layer
            trainingData = rbm.GetHiddenLayerActivation(trainingData);
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
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();
        RBMLayers.Clear();

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

            // Reconstruct RBM layers
            if (layer is RestrictedBoltzmannMachine<T> rbm)
            {
                RBMLayers.Add(rbm);
            }
        }
    }
}