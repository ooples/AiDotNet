namespace AiDotNet.NeuralNetworks.Layers;

public class ReconstructionLayer<T> : LayerBase<T>
{
    private readonly FullyConnectedLayer<T> _fc1;
    private readonly FullyConnectedLayer<T> _fc2;
    private readonly FullyConnectedLayer<T> _fc3;
    private bool _useVectorActivation;

    public override int ParameterCount =>
        _fc1.ParameterCount + _fc2.ParameterCount + _fc3.ParameterCount;

    public override bool SupportsTraining => true;

    public ReconstructionLayer(
        int inputDimension,
        int hidden1Dimension,
        int hidden2Dimension,
        int outputDimension,
        IActivationFunction<T>? hiddenActivation = null,
        IActivationFunction<T>? outputActivation = null)
        : base([inputDimension], [outputDimension])
    {
        _useVectorActivation = false;
        hiddenActivation ??= new ReLUActivation<T>();
        outputActivation ??= new SigmoidActivation<T>();

        _fc1 = new FullyConnectedLayer<T>(inputDimension, hidden1Dimension, hiddenActivation);
        _fc2 = new FullyConnectedLayer<T>(hidden1Dimension, hidden2Dimension, hiddenActivation);
        _fc3 = new FullyConnectedLayer<T>(hidden2Dimension, outputDimension, outputActivation);
    }

    public ReconstructionLayer(
        int inputDimension,
        int hidden1Dimension,
        int hidden2Dimension,
        int outputDimension,
        IVectorActivationFunction<T>? hiddenVectorActivation = null,
        IVectorActivationFunction<T>? outputVectorActivation = null)
        : base([inputDimension], [outputDimension])
    {
        _useVectorActivation = true;
        hiddenVectorActivation ??= new ReLUActivation<T>();
        outputVectorActivation ??= new SigmoidActivation<T>();

        _fc1 = new FullyConnectedLayer<T>(inputDimension, hidden1Dimension, hiddenVectorActivation);
        _fc2 = new FullyConnectedLayer<T>(hidden1Dimension, hidden2Dimension, hiddenVectorActivation);
        _fc3 = new FullyConnectedLayer<T>(hidden2Dimension, outputDimension, outputVectorActivation);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var x = _fc1.Forward(input);
        x = _fc2.Forward(x);
        return _fc3.Forward(x);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradient = _fc3.Backward(outputGradient);
        gradient = _fc2.Backward(gradient);

        return _fc1.Backward(gradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        _fc1.UpdateParameters(learningRate);
        _fc2.UpdateParameters(learningRate);
        _fc3.UpdateParameters(learningRate);
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_useVectorActivation);
        _fc1.Serialize(writer);
        _fc2.Serialize(writer);
        _fc3.Serialize(writer);
    }

    public override void Deserialize(BinaryReader reader)
    {
        _useVectorActivation = reader.ReadBoolean();
        _fc1.Deserialize(reader);
        _fc2.Deserialize(reader);
        _fc3.Deserialize(reader);
    }

    public override Vector<T> GetParameters()
    {
        // Get parameters from all sublayers
        var fc1Params = _fc1.GetParameters();
        var fc2Params = _fc2.GetParameters();
        var fc3Params = _fc3.GetParameters();
    
        // Create a combined parameter vector
        int totalParams = fc1Params.Length + fc2Params.Length + fc3Params.Length;
        var parameters = new Vector<T>(totalParams);
    
        // Copy parameters from each layer
        int index = 0;
    
        for (int i = 0; i < fc1Params.Length; i++)
        {
            parameters[index++] = fc1Params[i];
        }
    
        for (int i = 0; i < fc2Params.Length; i++)
        {
            parameters[index++] = fc2Params[i];
        }
    
        for (int i = 0; i < fc3Params.Length; i++)
        {
            parameters[index++] = fc3Params[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Get parameter counts for each sublayer
        int fc1ParamCount = _fc1.ParameterCount;
        int fc2ParamCount = _fc2.ParameterCount;
        int fc3ParamCount = _fc3.ParameterCount;
        int totalParams = fc1ParamCount + fc2ParamCount + fc3ParamCount;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        // Extract and set parameters for each sublayer
        int index = 0;
    
        var fc1Params = new Vector<T>(fc1ParamCount);
        for (int i = 0; i < fc1ParamCount; i++)
        {
            fc1Params[i] = parameters[index++];
        }
        _fc1.SetParameters(fc1Params);
    
        var fc2Params = new Vector<T>(fc2ParamCount);
        for (int i = 0; i < fc2ParamCount; i++)
        {
            fc2Params[i] = parameters[index++];
        }
        _fc2.SetParameters(fc2Params);
    
        var fc3Params = new Vector<T>(fc3ParamCount);
        for (int i = 0; i < fc3ParamCount; i++)
        {
            fc3Params[i] = parameters[index++];
        }
        _fc3.SetParameters(fc3Params);
    }

    public override void ResetState()
    {
        // Reset state in all sublayers
        _fc1.ResetState();
        _fc2.ResetState();
        _fc3.ResetState();
    }
}