namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network implementation that processes data through multiple layers to make predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double)</typeparam>
/// <remarks>
/// <para>
/// Neural networks are computing systems inspired by the human brain. They consist of multiple layers
/// of interconnected nodes (neurons) that process input data to produce predictions. This class provides
/// a straightforward implementation that can be used for various machine learning tasks.
/// </para>
/// <para><b>For Beginners:</b> A neural network is like a brain-inspired system that learns from examples.
/// 
/// Think of a neural network as an assembly line for information:
/// - Input data enters the "factory" (like features of an image or text)
/// - It passes through several processing stations (layers of neurons)
/// - Each station transforms the information in specific ways
/// - Finally, it produces an output (like a prediction or classification)
/// 
/// For example, if you want to classify images of cats and dogs:
/// - The input would be the pixel values of the image
/// - The neural network processes these values through its layers
/// - Each layer learns to recognize different patterns (edges, shapes, textures, etc.)
/// - The output tells you the probability of the image containing a cat or dog
/// 
/// The network "learns" by adjusting its internal parameters based on examples,
/// gradually improving its predictions through a process called training.
/// </para>
/// </remarks>
public class NeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Indicates whether this network supports training (learning from data).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A neural network is considered trainable when at least one layer supports training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => Layers.Any(layer => layer.SupportsTraining);
    /// <summary>
    /// Creates a new neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure and configuration of the neural network</param>
    /// <remarks>
    /// <para>
    /// The architecture determines important aspects of the neural network such as:
    /// - The number and types of layers
    /// - The number of neurons in each layer
    /// - The activation functions used
    /// - Other configuration parameters
    /// 
    /// After creating the neural network, it automatically initializes the layers based on the provided architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new neural network with your desired structure.
    /// 
    /// When creating a neural network, you need to define its "architecture" - the blueprint that specifies:
    /// - How many inputs it will accept (like the number of features in your data)
    /// - How many layers it has (more layers can learn more complex patterns)
    /// - How many neurons are in each layer (more neurons can capture more information)
    /// - What activation functions to use (these add non-linearity, allowing the network to learn complex patterns)
    /// 
    /// Think of it like designing a building - you're establishing the foundation and framework
    /// before you start "training" it (like furnishing the rooms).
    /// 
    /// For example, a simple network for classifying handwritten digits might have:
    /// - 784 inputs (for a 28x28 pixel image)
    /// - 2 hidden layers with 128 neurons each
    /// - 10 outputs (one for each digit 0-9)
    /// </para>
    /// </remarks>
    public NeuralNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network's structure by either:
    /// 1. Using custom layers provided in the architecture, or
    /// 2. Creating default layers if none were specified
    /// 
    /// The layers determine how data flows through the network and how computations are performed.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your neural network.
    /// 
    /// Think of this as assembling the components of your network:
    /// - If you've specified exactly what layers you want, those are used
    /// - If not, standard layers are created based on your architecture settings
    /// 
    /// Layers are the key processing units in a neural network. Common types include:
    /// - Input Layer: Receives your data
    /// - Hidden Layers: Process the information, extracting patterns
    /// - Output Layer: Produces the final prediction
    /// 
    /// Each layer contains neurons that apply mathematical operations and activation functions
    /// to transform the data as it flows through the network.
    /// </para>
    /// </remarks>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters (weights and biases) of the neural network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the entire network</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameters to each layer of the neural network.
    /// It's typically used during training when an optimization algorithm has calculated
    /// new parameter values to improve the network's performance.
    /// 
    /// The parameters vector must contain values for all trainable parameters in the network,
    /// arranged in the same order as the layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the internal values that the network has learned.
    /// 
    /// Neural networks learn by adjusting their "parameters" (weights and biases):
    /// - Weights determine how strongly neurons are connected to each other
    /// - Biases allow neurons to activate more or less easily
    /// 
    /// During training, the network figures out what parameters work best by:
    /// 1. Making predictions on training examples
    /// 2. Comparing predictions to correct answers
    /// 3. Calculating how to change parameters to improve accuracy
    /// 4. Using this method to update those parameters
    /// 
    /// Think of it like adjusting the settings on a complex machine to improve its performance.
    /// This method takes a long list of new parameter values and distributes them to the right
    /// places throughout the network.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Makes a prediction using the neural network.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through all layers of the neural network without updating any internal states.
    /// It's used for making predictions on new data after the network has been trained.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and gives you the network's prediction.
    /// 
    /// When making a prediction:
    /// - Your input data (like an image or set of features) enters the network
    /// - It passes through each layer, being transformed along the way
    /// - Each neuron applies its weights, bias, and activation function
    /// - The final layer produces the output (like a classification or regression value)
    /// 
    /// This is the main method you'll use when applying your trained network to new data.
    /// For example, if you've trained a network to recognize handwritten digits,
    /// you would use this method to classify new digit images.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Set network to inference mode (not training)
        bool originalTrainingMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            // CPU path: forward pass through all layers
            Tensor<T> current = input;

            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }
        finally
        {
            // Restore original training mode
            SetTrainingMode(originalTrainingMode);
        }
    }

    /// <summary>
    /// Trains the neural network on input-output pairs.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method performs one step of training on a single input-output pair or batch.
    /// It computes the forward pass, calculates the error, and backpropagates to update the network's parameters.
    /// For full training, this method should be called repeatedly with different inputs from the training dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the network to make better predictions.
    /// 
    /// The training process works like this:
    /// 1. Input data is fed into the network
    /// 2. The network makes a prediction (forward pass)
    /// 3. The prediction is compared to the expected output to calculate error
    /// 4. The error is propagated backward through the network (backpropagation)
    /// 5. The network's parameters are adjusted to reduce the error
    /// 
    /// Think of it like learning from mistakes:
    /// - The network makes a guess
    /// - It sees how far off it was
    /// - It adjusts its approach to do better next time
    /// 
    /// This method performs one iteration of this process. To fully train a network,
    /// you'd typically call this method many times with different examples from your training data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure we're in training mode
        SetTrainingMode(true);

        // Step 1: Forward pass with memory for backpropagation
        Vector<T> outputVector = ForwardWithMemory(input).ToVector();

        // Step 2: Calculate loss/error (e.g., mean squared error)
        Vector<T> expectedVector = expectedOutput.ToVector();
        Vector<T> errorVector = new(expectedVector.Length);

        for (int i = 0; i < expectedVector.Length; i++)
        {
            // Error = expected - actual
            errorVector[i] = NumOps.Subtract(expectedVector[i], outputVector[i]);
        }

        // Calculate and store the loss value
        LastLoss = LossFunction.CalculateLoss(outputVector, expectedVector);

        // Step 3: Backpropagation to compute gradients
        Backpropagate(Tensor<T>.FromVector(errorVector));

        // Step 4: Update parameters using gradients and learning rate
        T learningRate = NumOps.FromDouble(0.01);

        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }


    /// <summary>
    /// Gets metadata about the neural network.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the neural network, including its architecture,
    /// layer configuration, and other relevant parameters. This information is useful for model
    /// management, tracking experiments, and reporting.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your neural network.
    /// 
    /// The metadata includes:
    /// - The type of neural network
    /// - Details about its structure (layers, neurons, etc.)
    /// - The total number of parameters (weights and biases)
    /// - Additional configuration information
    /// 
    /// This information is useful for:
    /// - Documentation
    /// - Comparing different network architectures
    /// - Debugging and analyzing network behavior
    /// - Creating reports or visualizations
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Count parameters by layer type
        Dictionary<string, int> layerCounts = new Dictionary<string, int>();

        foreach (var layer in Layers)
        {
            string layerType = layer.GetType().Name;

            if (layerCounts.ContainsKey(layerType))
            {
                layerCounts[layerType]++;
            }
            else
            {
                layerCounts[layerType] = 1;
            }
        }

        // Get layer sizes
        int[] layerSizes = Architecture.GetLayerSizes();

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize },
                { "TotalParameters", ParameterCount },
                { "TotalLayers", Layers.Count },
                { "LayerTypes", layerCounts },
                { "LayerSizes", layerSizes },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() },
                { "TaskType", Architecture.TaskType.ToString() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes neural network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves any neural network-specific data to the binary stream.
    /// In this implementation, there is no additional data beyond what the base class serializes,
    /// but this method could be extended for specialized neural network types.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves neural network-specific information.
    /// 
    /// When saving a neural network to a file:
    /// - The base class already saves the basic structure and weights
    /// - This method saves any additional information specific to this type of network
    /// 
    /// For a standard neural network, there's typically no additional information needed
    /// beyond what the base class already saves.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
    }

    /// <summary>
    /// Deserializes neural network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads any neural network-specific data from the binary stream.
    /// In this implementation, there is no additional data beyond what the base class deserializes,
    /// but this method could be extended for specialized neural network types.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads neural network-specific information.
    /// 
    /// When loading a neural network from a file:
    /// - The base class already loads the basic structure and weights
    /// - This method loads any additional information specific to this type of network
    /// 
    /// For a standard neural network, there's typically no additional information needed
    /// beyond what the base class already loads.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
    }

    /// <summary>
    /// Creates a new instance of the neural network with the same architecture.
    /// </summary>
    /// <returns>A new instance of the neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new neural network with the same architecture as the current instance.
    /// The new instance is initialized with fresh layers and parameters, making it useful for
    /// creating multiple networks with the same structure or for resetting a network while preserving its architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a brand new neural network with the same structure.
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh network but keep the same structure
    /// - Create multiple networks with identical layouts
    /// - Reset a network to its initial state
    /// 
    /// The new network will have:
    /// - The same number of layers and neurons
    /// - The same activation functions
    /// - Newly initialized weights and biases
    /// 
    /// Think of it like creating a twin of your neural network, but with a "blank slate" -
    /// it has the same structure but hasn't learned anything yet.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuralNetwork<T>(Architecture);
    }
}
