namespace AiDotNet.NeuralNetworks;

public class SpikingNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private double TimeStep { get; set; }
    private int SimulationSteps { get; set; }
    private IVectorActivationFunction<T>? VectorActivation { get; set; }
    private IActivationFunction<T>? ScalarActivation { get; set; }

    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100, IVectorActivationFunction<T>? vectorActivation = null) 
        : base(architecture)
    {
        TimeStep = timeStep;
        SimulationSteps = simulationSteps;
        VectorActivation = vectorActivation;
    }

    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100, IActivationFunction<T>? scalarActivation = null) 
        : base(architecture)
    {
        TimeStep = timeStep;
        SimulationSteps = simulationSteps;
        ScalarActivation = scalarActivation;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpikingLayers(Architecture));
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        var spikeTrain = new List<Vector<T>>();

        for (int t = 0; t < SimulationSteps; t++)
        {
            foreach (var layer in Layers)
            {
                current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            }
            spikeTrain.Add(current);
        }

        // Aggregate spike train to produce final output
        return AggregateSpikeTrainToOutput(spikeTrain);
    }

    private Vector<T> AggregateSpikeTrainToOutput(List<Vector<T>> spikeTrain)
    {
        int outputSize = spikeTrain[0].Length;
        var output = new Vector<T>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            T sum = NumOps.Zero;
            for (int t = 0; t < spikeTrain.Count; t++)
            {
                sum = NumOps.Add(sum, spikeTrain[t][i]);
            }
            output[i] = NumOps.Divide(sum, NumOps.FromDouble(spikeTrain.Count));
        }

        return output;
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
        writer.Write(TimeStep);
        writer.Write(SimulationSteps);

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
        TimeStep = reader.ReadDouble();
        SimulationSteps = reader.ReadInt32();

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