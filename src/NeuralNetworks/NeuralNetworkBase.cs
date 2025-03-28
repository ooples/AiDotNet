namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Base class for all neural network implementations in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A neural network is a computing system inspired by the human brain. It consists of 
/// interconnected "layers" of artificial neurons that process information and learn patterns from data.
/// This class provides the foundation for building different types of neural networks.
/// </para>
/// </remarks>
public abstract class NeuralNetworkBase<T> : INeuralNetwork<T>
{
    /// <summary>
    /// The collection of layers that make up this neural network.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Layers are the building blocks of neural networks. Each layer contains 
    /// neurons that process information and pass it to the next layer. A typical network has 
    /// an input layer (receives data), hidden layers (process data), and an output layer (produces results).
    /// </remarks>
    protected readonly List<ILayer<T>> Layers;
    
    /// <summary>
    /// The architecture definition for this neural network.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The architecture defines the structure of your neural network - how many layers it has,
    /// how many neurons are in each layer, and how they're connected. Think of it as the blueprint for your network.
    /// </remarks>
    protected readonly NeuralNetworkArchitecture<T> Architecture;
    
    /// <summary>
    /// Mathematical operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;
    
    /// <summary>
    /// Stores the input values for each layer during forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When data flows through the network, we need to remember what values went into each layer.
    /// This is necessary for the learning process (backpropagation).
    /// </remarks>
    protected Dictionary<int, Tensor<T>> _layerInputs = [];
    
    /// <summary>
    /// Stores the output values from each layer during forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Similar to layer inputs, we also need to remember what values came out of each layer
    /// during the learning process.
    /// </remarks>
    protected Dictionary<int, Tensor<T>> _layerOutputs = [];

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    protected Random Random => new();
    
    /// <summary>
    /// Indicates whether the network is currently in training mode.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Neural networks behave differently during training versus when they're making predictions.
    /// In training mode, the network keeps track of additional information needed for learning.
    /// </remarks>
    protected bool IsTrainingMode = true;
    
    /// <summary>
    /// Indicates whether this network supports training (learning from data).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Not all neural networks can learn. Some are designed only for making predictions
    /// with pre-set parameters. This property tells you if the network can learn from data.
    /// </remarks>
    public virtual bool SupportsTraining => false;

    /// <summary>
    /// Creates a new neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the network.</param>
    protected NeuralNetworkBase(NeuralNetworkArchitecture<T> architecture)
    {
        Architecture = architecture;
        Layers = [];
        InitializeLayers();
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets all trainable parameters of the network as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks learn by adjusting their "parameters" (also called weights and biases).
    /// This method collects all those adjustable values into a single list so they can be updated during training.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Performs backpropagation to compute gradients for network parameters.
    /// </summary>
    /// <param name="outputGradients">The gradients of the loss with respect to the network outputs.</param>
    /// <returns>The gradients of the loss with respect to the network inputs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation is how neural networks learn. After making a prediction, the network
    /// calculates how wrong it was (the error). Then it works backward through the layers to figure out
    /// how each parameter contributed to that error. This method handles that backward flow of information.
    /// </para>
    /// <para>
    /// The "gradients" are numbers that tell us how to adjust each parameter to reduce the error.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the network is not in training mode or doesn't support training.</exception>
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

    /// <summary>
    /// Performs a forward pass through the network while storing intermediate values for backpropagation.
    /// </summary>
    /// <param name="input">The input data to the network.</param>
    /// <returns>The output of the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method passes data through the network from input to output, but also
    /// remembers all the intermediate values. This is necessary for the learning process, as the network
    /// needs to know these values when figuring out how to improve.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support training.</exception>
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
            _layerInputs[i] = Tensor<T>.FromVector(current);
            
            // Forward pass through layer
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
            
            // Store output from each layer for backpropagation
            _layerOutputs[i] = Tensor<T>.FromVector(current);
        }
        
        return current;
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    /// <returns>The total parameter count.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many adjustable values (weights and biases) your neural network has.
    /// More complex networks typically have more parameters and can learn more complex patterns, but also
    /// require more data to train effectively.
    /// </remarks>
    public virtual int GetParameterCount()
    {
        return Layers.Sum(layer => layer.ParameterCount);
    }

    /// <summary>
    /// Validates that the provided layers form a valid neural network architecture.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Not all combinations of layers make a valid neural network. This method checks that
    /// the layers can properly connect to each other (like making sure puzzle pieces fit together).
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the layer configuration is invalid.</exception>
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

    /// <summary>
    /// Determines if a layer can serve as a valid input layer for the neural network.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer can be used as an input layer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The input layer is the first layer of your neural network. It receives the raw data 
    /// you want to process (like image pixels or text features). This method checks if a layer is suitable 
    /// to be the first layer in your network.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for the neural network.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer can be used as an output layer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The output layer is the last layer of your neural network. It produces the final result 
    /// (like a prediction or classification). This method checks if a layer is suitable to be the final layer 
    /// in your network. Different tasks need different types of output layers - for example, image classification 
    /// might use a Softmax activation, while regression might use a linear activation.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Checks if two consecutive layers can be connected in a neural network.
    /// </summary>
    /// <param name="prevLayer">The preceding layer.</param>
    /// <param name="currentLayer">The current layer to check compatibility with.</param>
    /// <returns>True if the layers can be connected; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks work by connecting layers in sequence. For two layers to connect properly, 
    /// the output of one layer must match what the next layer expects as input. This is like making sure puzzle 
    /// pieces fit together. This method checks if two layers can be properly connected.
    /// </para>
    /// <para>
    /// For example, if a layer outputs 100 values, the next layer should expect 100 values as input. Some layer 
    /// combinations also have special rules - like needing a "Flatten" layer between image processing layers and 
    /// regular dense layers.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Performs backpropagation through the network using explicit input values.
    /// </summary>
    /// <param name="outputGradients">The gradients of the loss with respect to the network outputs.</param>
    /// <param name="inputs">The input data used for the forward pass.</param>
    /// <returns>The gradients of the loss with respect to the network inputs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation is how neural networks learn. This method takes both the network's inputs 
    /// and the error gradients from the output, then calculates how to adjust the network's internal values to 
    /// reduce errors. Think of it as the network figuring out which knobs to turn (and by how much) to get better results.
    /// </para>
    /// <para>
    /// This version of backpropagation requires you to provide the original inputs because it needs to recalculate 
    /// all the intermediate values in the network.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Retrieves the gradients for all trainable parameters in the network.
    /// </summary>
    /// <returns>A vector containing all parameter gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When a neural network learns, it needs to know how to adjust each of its internal values 
    /// (parameters). These adjustments are called "gradients" - they tell the network which direction and how much 
    /// to change each parameter. This method collects all those adjustment values into a single list.
    /// </para>
    /// <para>
    /// Think of gradients as a recipe for improvement: "increase this weight by 0.01, decrease that one by 0.03," etc.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the layers of the neural network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method sets up all the layers in your neural network according to the architecture 
    /// you've defined. It's like assembling the parts of your network before you can use it.
    /// </remarks>
    protected abstract void InitializeLayers();

    /// <summary>
    /// Makes a prediction using the neural network.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>The network's prediction.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main method you'll use to get results from your trained neural network. 
    /// You provide some input data (like an image or text), and the network processes it through all its 
    /// layers to produce an output (like a classification or prediction).
    /// </remarks>
    public abstract Vector<T> Predict(Vector<T> input);

    /// <summary>
    /// Updates the network's parameters with new values.
    /// </summary>
    /// <param name="parameters">The new parameter values to set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, a neural network's internal values (parameters) get adjusted to improve 
    /// its performance. This method allows you to update all those values at once by providing a complete set 
    /// of new parameters.
    /// </para>
    /// <para>
    /// This is typically used by optimization algorithms that calculate better parameter values based on 
    /// training data.
    /// </para>
    /// </remarks>
    public abstract void UpdateParameters(Vector<T> parameters);

    /// <summary>
    /// Saves the neural network's state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to save to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method saves your trained neural network to a file so you can use it later 
    /// without having to train it again. It's like saving your progress in a video game.
    /// </remarks>
    public abstract void Serialize(BinaryWriter writer);

    /// <summary>
    /// Loads the neural network's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to load from.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads a previously saved neural network from a file. It's like loading 
    /// a saved game so you can continue where you left off, without having to start over.
    /// </remarks>
    public abstract void Deserialize(BinaryReader reader);

    /// <summary>
    /// Sets the neural network to either training or inference mode.
    /// </summary>
    /// <param name="isTraining">True to enable training mode; false to enable inference mode.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks behave differently during training versus when making predictions.
    /// </para>
    /// <para>
    /// When in training mode (isTraining = true):
    /// - The network keeps track of intermediate calculations needed for learning
    /// - Certain layers like Dropout and BatchNormalization behave differently
    /// - The network uses more memory but can learn from its mistakes
    /// </para>
    /// <para>
    /// When in inference/prediction mode (isTraining = false):
    /// - The network only performs forward calculations
    /// - It uses less memory and runs faster
    /// - It cannot learn or update its parameters
    /// </para>
    /// <para>
    /// Think of it like the difference between taking a practice test (training mode) where you 
    /// can check your answers and learn from mistakes, versus taking the actual exam (inference mode)
    /// where you just give your best answers based on what you've already learned.
    /// </para>
    /// </remarks>
    public virtual void SetTrainingMode(bool isTraining)
    {
        if (SupportsTraining)
        {
            IsTrainingMode = isTraining;
        }
    }
}