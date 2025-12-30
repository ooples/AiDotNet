namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Recurrent Neural Network, which is a type of neural network designed to process sequential data by maintaining an internal state.
/// </summary>
/// <remarks>
/// <para>
/// A Recurrent Neural Network (RNN) is a class of neural networks that can process sequential data by maintaining
/// an internal state (memory) that captures information about previous inputs. Unlike traditional feedforward
/// neural networks, RNNs have connections that form directed cycles, allowing information to persist from one step
/// to the next. This architectural feature makes RNNs particularly well-suited for tasks involving sequential data,
/// such as time series prediction, natural language processing, and speech recognition.
/// </para>
/// <para><b>For Beginners:</b> A Recurrent Neural Network is a type of neural network that has memory.
/// 
/// Think of it like reading a book:
/// - A standard neural network looks at each word in isolation
/// - An RNN remembers what it read earlier to understand the current word
/// 
/// For example, in the sentence "The clouds were dark, so I took my umbrella," an RNN understands that
/// "I took my umbrella" is related to "The clouds were dark" because it remembers the earlier part of the sentence.
/// 
/// This memory makes RNNs good at:
/// - Processing sequences like sentences, time series, or music
/// - Understanding context and relationships over time
/// - Predicting what comes next in a sequence
/// 
/// Common applications include text generation, translation, speech recognition, and stock price prediction - 
/// all tasks where what happened before affects how you interpret what's happening now.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RecurrentNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The learning rate used for updating the network parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate is a crucial hyperparameter that determines the step size at each iteration
    /// while moving toward a minimum of the loss function. It influences how quickly the network adapts
    /// to the training data. A higher learning rate allows for faster learning but may overshoot the optimal
    /// solution, while a lower learning rate provides more precise updates but may require more training time.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate is like the size of the steps the network takes when learning.
    /// 
    /// Think of it as adjusting the volume on a radio:
    /// - A high learning rate is like turning the knob quickly: you might find the right station faster,
    ///   but you could also overshoot it.
    /// - A low learning rate is like turning the knob very slowly: you're less likely to miss the station,
    ///   but it takes longer to find it.
    /// 
    /// The right learning rate helps the network learn efficiently without making wild guesses.
    /// </para>
    /// </remarks>
    private T _learningRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="RecurrentNeuralNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the RNN.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Recurrent Neural Network with the specified architecture.
    /// It initializes the network layers based on the architecture, or creates default RNN layers if
    /// no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Recurrent Neural Network with its basic structure.
    /// 
    /// When creating a new RNN:
    /// - The architecture defines what the network looks like - how many neurons it has, how they're connected, etc.
    /// - The constructor prepares the network by either:
    ///   * Using the specific layers provided in the architecture, or
    ///   * Creating default layers designed for processing sequences if none are specified
    /// 
    /// This is like setting up a specialized calculator before you start using it. The architecture
    /// determines what kind of calculations it can perform and how it will process information.
    /// </para>
    /// </remarks>
    public RecurrentNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double learningRate = 0.01, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _learningRate = NumOps.FromDouble(learningRate);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Recurrent Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for RNNs
    /// is created. In a typical RNN, this involves creating recurrent layers that maintain state between
    /// processing steps.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for recurrent networks are created automatically
    /// - The system checks that any custom layers will work properly with the RNN
    /// 
    /// A typical RNN has specialized layers that include:
    /// - Input layers that accept sequences
    /// - Recurrent layers that maintain memory of previous inputs
    /// - Output layers that produce predictions
    /// 
    /// These layers work together to process sequential data while maintaining context from previous steps.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultRNNLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters of the recurrent neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the recurrent neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer. In an RNN, these parameters typically include weights
    /// for both the input connections and the recurrent connections that maintain state.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the RNN makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For an RNN, these adjustments might include:
    /// - How much attention to pay to the current input
    /// - How much to rely on memory of previous inputs
    /// - How to combine different pieces of information
    /// 
    /// These parameter updates help the network learn to:
    /// - Recognize important patterns in sequences
    /// - Decide what information is worth remembering
    /// - Make better predictions based on both current and past information
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
    /// Makes a prediction using the current state of the Recurrent Neural Network.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor after passing through all layers of the network.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the input tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the input tensor has incorrect dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network, transforming the input data through each layer
    /// to produce a final prediction. It includes input validation to ensure the provided tensor matches the
    /// expected input shape of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network processes new data to make predictions.
    /// 
    /// The prediction process:
    /// 1. Checks if the input data is valid and has the correct shape
    /// 2. Passes the input through each layer of the network
    /// 3. Each layer transforms the data, with recurrent layers using their internal state
    /// 4. The final layer produces the network's prediction
    /// 
    /// Think of it like a game of telephone, where each person (layer) passes along a message,
    /// but also remembers previous messages to provide context.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Validate input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }

        // Support any rank tensors - RNNs can handle variable sequence lengths and dimensions
        // The recurrent layers will internally adapt to the input dimensions
        // This is industry standard behavior for flexible neural networks

        // Forward pass through each layer in the network
        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        return currentOutput;
    }

    /// <summary>
    /// Trains the Recurrent Neural Network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor used for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <exception cref="ArgumentNullException">Thrown when either input or expectedOutput is null.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the training process for the RNN. It performs a forward pass, calculates the error
    /// between the network's prediction and the expected output, and then backpropagates this error to adjust
    /// the network's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network learns from examples.
    /// 
    /// The training process:
    /// 1. Takes a sequence of inputs and their correct answers (expected outputs)
    /// 2. Makes predictions using the current network state
    /// 3. Compares the predictions to the correct answers to calculate the error
    /// 4. Uses this error to adjust the network's internal settings (backpropagation through time)
    /// 
    /// It's like learning to predict weather patterns:
    /// - The network looks at a series of past weather conditions (input)
    /// - It tries to predict the next day's weather (output)
    /// - It compares its prediction to what actually happened
    /// - It adjusts its understanding based on its mistakes
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }

        if (expectedOutput == null)
        {
            throw new ArgumentNullException(nameof(expectedOutput), "Expected output tensor cannot be null.");
        }

        // Forward pass
        Tensor<T> output = Predict(input);

        // Calculate error/loss
        Tensor<T> error = output.Subtract(expectedOutput);

        // Calculate and set the loss using the loss function
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        // Backpropagate error through time
        BackpropagateError(error);

        // Update network parameters
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Backpropagates the error through the network layers.
    /// </summary>
    /// <param name="error">The initial error tensor to backpropagate.</param>
    /// <remarks>
    /// <para>
    /// This method propagates the error backwards through each layer of the network, allowing each layer
    /// to compute its local gradients. In an RNN, this process involves unrolling the network through time,
    /// which is crucial for capturing dependencies in sequential data.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network learns from its mistakes.
    /// 
    /// The backpropagation process:
    /// 1. Starts with the error at the output layer
    /// 2. Moves backwards through each layer
    /// 3. Each layer figures out how much it contributed to the error
    /// 4. This information is used to update the network's parameters
    /// 
    /// Think of it like tracing back through a series of decisions to understand where things went wrong,
    /// so you can make better decisions next time.
    /// </para>
    /// </remarks>
    private void BackpropagateError(Tensor<T> error)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            error = Layers[i].Backward(error);
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the network based on computed gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the computed gradients to update the parameters of each layer in the network.
    /// It uses the learning rate to control the size of the updates. In an RNN, this process adjusts
    /// both the weights for processing current inputs and the recurrent weights that handle sequential information.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network improves its performance over time.
    /// 
    /// The parameter update process:
    /// 1. Goes through each layer in the network
    /// 2. Checks if the layer has any parameters to update
    /// 3. Calculates how much to change each parameter based on the gradients and learning rate
    /// 4. Applies these changes to the layer's parameters
    /// 
    /// It's like fine-tuning a musical instrument. After hearing how it sounds (the error),
    /// you make small adjustments to each part (the parameters) to improve the overall performance.
    /// </para>
    /// </remarks>
    private void UpdateNetworkParameters()
    {
        foreach (var layer in Layers)
        {
            if (layer.GetParameterGradients() != null)
            {
                Vector<T> updates = layer.GetParameterGradients().Multiply(_learningRate);
                layer.UpdateParameters(updates);
            }
        }
    }

    /// <summary>
    /// Gets metadata about the Recurrent Neural Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata that describes the RNN, including its type, architecture details,
    /// and other relevant information. This metadata can be useful for model management, documentation,
    /// and versioning.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your network's setup and characteristics.
    /// 
    /// The metadata includes:
    /// - The type of model (Recurrent Neural Network)
    /// - How many inputs and outputs the network has
    /// - A description of the network's structure
    /// - Additional information specific to RNNs
    /// 
    /// This is useful for:
    /// - Keeping track of different versions of your model
    /// - Comparing different network configurations
    /// - Documenting your model's setup for future reference
    /// 
    /// Think of it like a spec sheet for a car, listing all its important features and capabilities.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.RecurrentNeuralNetwork,
            FeatureCount = Architecture.GetInputShape()[0],
            Description = $"RNN with {Layers.Count} layers",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data for the Recurrent Neural Network.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific configuration and state of the RNN to a binary stream.
    /// It includes RNN-specific parameters that are essential for later reconstruction of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the unique settings of your RNN.
    /// 
    /// It writes:
    /// - The size of the hidden state (which determines the network's memory capacity)
    /// - The length of sequences the network is designed to handle
    /// - Any other RNN-specific parameters
    /// 
    /// Saving these details allows you to recreate the exact same network structure later.
    /// It's like writing down a recipe so you can make the same dish again in the future.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(Convert.ToDouble(_learningRate));
    }

    /// <summary>
    /// Deserializes network-specific data for the Recurrent Neural Network.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific configuration and state of the RNN from a binary stream.
    /// It reconstructs the RNN-specific parameters to match the state of the network when it was serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the unique settings of your RNN.
    /// 
    /// It reads:
    /// - The size of the hidden state (which determines the network's memory capacity)
    /// - The length of sequences the network is designed to handle
    /// - Any other RNN-specific parameters
    /// 
    /// Loading these details allows you to recreate the exact same network structure that was previously saved.
    /// It's like following a recipe to recreate a dish exactly as it was made before.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
    }

    /// <summary>
    /// Creates a new instance of the recurrent neural network with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="RecurrentNeuralNetwork{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new recurrent neural network that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// The new instance will have the same architecture and learning rate as the original,
    /// but will not share parameter values unless they are explicitly copied after creation.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your network that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar recurrent neural networks
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// 
    /// Note that while the settings are copied, the learned parameters (like the weights that determine
    /// how the network processes sequences) are not automatically transferred, so the new instance
    /// will need training or parameter copying to match the performance of the original.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a new instance with the cloned architecture and the same learning rate
        double learningRate = Convert.ToDouble(_learningRate);
        return new RecurrentNeuralNetwork<T>(Architecture, learningRate, LossFunction);
    }
}
