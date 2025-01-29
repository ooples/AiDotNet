namespace AiDotNet.NeuralNetworks;

public class NeuralNetwork<T> : NeuralNetworkBase<T>
{
    private readonly List<Layer<T>> _layers;
    private readonly NeuralNetworkArchitecture<T> _architecture;

    public NeuralNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        _architecture = architecture;
        _layers = [];
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        for (int i = 0; i < _architecture.LayerSizes.Count - 1; i++)
        {
            int inputSize = _architecture.LayerSizes[i];
            int outputSize = _architecture.LayerSizes[i + 1];
            
            if (_architecture.VectorActivationFunctions != null && 
                _architecture.VectorActivationFunctions[i] != null)
            {
                _layers.Add(new Layer<T>(inputSize, outputSize, _architecture.VectorActivationFunctions[i]));
            }
            else if (_architecture.ActivationFunctions != null && 
                     _architecture.ActivationFunctions[i] != null)
            {
                _layers.Add(new Layer<T>(inputSize, outputSize, _architecture.ActivationFunctions[i]));
            }
            else
            {
                throw new ArgumentException($"No activation function specified for layer {i}");
            }
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        Vector<T> output = input;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int paramIndex = 0;
        foreach (var layer in _layers)
        {
            int layerParamCount = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(paramIndex, layerParamCount));
            paramIndex += layerParamCount;
        }
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_layers.Count);
        foreach (var layer in _layers)
        {
            // Write layer type (0 for IActivationFunction, 1 for IVectorActivationFunction)
            writer.Write(layer.IsVectorActivation ? 1 : 0);
        
            // Write input and output sizes
            writer.Write(layer.InputSize);
            writer.Write(layer.OutputSize);

            layer.Serialize(writer);
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        int layerCount = reader.ReadInt32();
        _layers.Clear();
        for (int i = 0; i < layerCount; i++)
        {
            // Read layer type (0 for IActivationFunction, 1 for IVectorActivationFunction)
            int layerType = reader.ReadInt32();
        
            // Read input and output sizes
            int inputSize = reader.ReadInt32();
            int outputSize = reader.ReadInt32();

            Layer<T> layer;
            if (layerType == 0)
            {
                layer = new Layer<T>(inputSize, outputSize, (IActivationFunction<T>?)null);
            }
            else
            {
                layer = new Layer<T>(inputSize, outputSize, (IVectorActivationFunction<T>?)null);
            }

            layer.Deserialize(reader);
            _layers.Add(layer);
        }
    }
}