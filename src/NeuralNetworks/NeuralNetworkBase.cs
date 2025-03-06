namespace AiDotNet.NeuralNetworks;

public abstract class NeuralNetworkBase<T> : INeuralNetwork<T>
{
    protected readonly List<ILayer<T>> Layers;
    protected readonly NeuralNetworkArchitecture<T> Architecture;
    protected readonly INumericOperations<T> NumOps;
    protected Dictionary<int, Tensor<T>> LayerInputs = [];
    protected Dictionary<int, Tensor<T>> LayerOutputs = [];

    protected Random Random => new();
    protected bool IsTrainingMode = true;
    public virtual bool SupportsTraining => false;

    protected NeuralNetworkBase(NeuralNetworkArchitecture<T> architecture)
    {
        Architecture = architecture;
        Layers = [];
        InitializeLayers();
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    public virtual Vector<T> GetParameters()
    {
        int totalParameterCount = GetParameterCount();
        var parameters = new Vector<T>(totalParameterCount);
    
        int currentIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = layer.GetParameters();
                for (int i = 0; i < layerParameterCount; i++)
                {
                    parameters[currentIndex + i] = layerParameters[i];
                }

                currentIndex += layerParameterCount;
            }
        }
    
        return parameters;
    }

    public virtual Vector<T> Backpropagate(Vector<T> outputGradients)
    {
        if (!IsTrainingMode)
        {
            throw new InvalidOperationException("Cannot backpropagate when network is not in training mode");
        }
        
        if (!SupportsTraining)
        {
            throw new InvalidOperationException("This network does not support backpropagation");
        }
        
        // Convert output gradients to tensor format
        var gradientTensor = Tensor<T>.FromVector(outputGradients);
        
        // Backpropagate through layers in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }
        
        // Convert input gradients back to vector format
        return gradientTensor.ToVector();
    }

    public virtual Vector<T> ForwardWithMemory(Vector<T> input)
    {
        if (!SupportsTraining)
        {
            throw new InvalidOperationException("This network does not support training mode");
        }
        
        var current = input;
        
        for (int i = 0; i < Layers.Count; i++)
        {
            // Store input to each layer for backpropagation
            LayerInputs[i] = Tensor<T>.FromVector(current);
            
            // Forward pass through layer
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
            
            // Store output from each layer for backpropagation
            LayerOutputs[i] = Tensor<T>.FromVector(current);
        }
        
        return current;
    }

    public virtual int GetParameterCount()
    {
        return Layers.Sum(layer => layer.ParameterCount);
    }

    protected virtual void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        if (layers == null || layers.Count < 2)
        {
            throw new ArgumentException("Neural network must have at least 2 layers (1 input layer and 1 output layer).");
        }

        var errors = new List<string>();

        // Check input layer
        if (!IsValidInputLayer(layers[0]))
        {
            errors.Add("The first layer must be a valid input layer.");
        }

        // Check layer connections
        for (int i = 1; i < layers.Count; i++)
        {
            var prevLayer = layers[i - 1];
            var currentLayer = layers[i];

            if (!AreLayersCompatible(prevLayer, currentLayer))
            {
                errors.Add($"Layer {i - 1} is not compatible with Layer {i}.");
            }
        }

        // Check output layer
        if (!IsValidOutputLayer(layers[layers.Count - 1]))
        {
            errors.Add("The last layer must be a valid output layer.");
        }

        // Throw exception if any errors were found
        if (errors.Count > 0)
        {
            throw new ArgumentException($"Invalid layer configuration:\n{string.Join("\n", errors)}");
        }
    }

    protected virtual bool IsValidInputLayer(ILayer<T> layer)
    {
        // Check if the layer is specifically designed as an input layer
        if (layer is InputLayer<T>)
            return true;

        // For convolutional networks, the first layer is often a ConvolutionalLayer
        if (layer is ConvolutionalLayer<T>)
            return true;

        // For simple feedforward networks, the first layer might be Dense
        if (layer is DenseLayer<T> denseLayer)
        {
            // Ensure the dense layer doesn't have any inputs (it's the first layer)
            return denseLayer.GetInputShape().Length == 1 && denseLayer.GetInputShape()[0] > 0;
        }

        // For recurrent networks, the first layer might be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // If none of the above, it's not a valid input layer
        return false;
    }

    protected virtual bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Most commonly, the output layer is a Dense layer
        if (layer is DenseLayer<T> denseLayer)
        {
            // Ensure the dense layer has an output (it's not empty)
            return denseLayer.GetOutputShape().Length == 1 && denseLayer.GetOutputShape()[0] > 0;
        }

        // For some specific tasks, the output might be from other layer types
        // For example, in sequence-to-sequence models, it could be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // For image segmentation tasks, it might be a Convolutional layer
        if (layer is ConvolutionalLayer<T>)
            return true;

        // Check if the layer has an activation function typically used in output layers
        if (layer is ActivationLayer<T> activationLayer)
        {
            // Check if the layer has an activation function typically used in output layers
            var activationTypes = layer.GetActivationTypes();
            return activationTypes.Any(type => type == ActivationFunction.Softmax || type == ActivationFunction.Sigmoid);
        }

        // If none of the above, it's not a valid output layer
        return false;
    }

    protected virtual bool AreLayersCompatible(ILayer<T> prevLayer, ILayer<T> currentLayer)
    {
        // Check if the output shape of the previous layer matches the input shape of the current layer
        if (!Enumerable.SequenceEqual(prevLayer.GetOutputShape(), currentLayer.GetInputShape()))
            return false;

        // Special checks for specific layer combinations
        if (prevLayer is ConvolutionalLayer<T> && currentLayer is DenseLayer<T>)
        {
            // Ensure there's a Flatten layer between Conv and Dense
            return false;
        }

        if (prevLayer is PoolingLayer<T> && currentLayer is LSTMLayer<T>)
        {
            // Pooling directly to LSTM is usually not valid
            return false;
        }

        // Check for dimension compatibility in case of Reshape or Flatten layers
        if (prevLayer is ReshapeLayer<T> reshapeLayer)
        {
            return reshapeLayer.GetOutputShape().Aggregate((a, b) => a * b) == 
                   currentLayer.GetInputShape().Aggregate((a, b) => a * b);
        }

        // If no incompatibilities found, layers are considered compatible
        return true;
    }

    public virtual Vector<T> Backpropagate(Vector<T> outputGradients, Vector<T> inputs)
    {
        // Store the original input for later use
        var originalInput = inputs;
    
        // Forward pass to compute all intermediate activations
        var activations = new List<Vector<T>> { inputs };
        var current = inputs;
    
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            activations.Add(current);
        }
    
        // Backward pass
        var gradient = outputGradients;
    
        // Go through layers in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(Tensor<T>.FromVector(gradient)).ToVector();
        }
    
        // Return gradient with respect to inputs
        return gradient;
    }

    public virtual Vector<T> GetParameterGradients()
    {
        // Collect gradients from all layers
        List<Vector<T>> allGradients = [];
    
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                allGradients.Add(layer.GetParameterGradients());
            }
        }
    
        // Concatenate all gradients into a single vector
        if (allGradients.Count == 0)
        {
            return new Vector<T>(0);
        }
    
        return Vector<T>.Concatenate(allGradients.ToArray());
    }

    protected abstract void InitializeLayers();

    public abstract Vector<T> Predict(Vector<T> input);

    public abstract void UpdateParameters(Vector<T> parameters);

    public abstract void Serialize(BinaryWriter writer);

    public abstract void Deserialize(BinaryReader reader);

    public virtual void SetTrainingMode(bool isTraining)
    {
        if (SupportsTraining)
        {
            IsTrainingMode = isTraining;
        }
    }
}