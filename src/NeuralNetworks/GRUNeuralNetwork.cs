namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Gated Recurrent Unit (GRU) Neural Network for processing sequential data.
/// </summary>
/// <remarks>
/// <para>
/// A GRU Neural Network is a type of recurrent neural network designed to effectively model sequential data.
/// GRU networks use gating mechanisms to control information flow through the network, allowing them to
/// capture long-term dependencies in sequence data while avoiding the vanishing gradient problem
/// that affects simple recurrent networks.
/// </para>
/// <para><b>For Beginners:</b> A GRU Neural Network is a special type of neural network that's good at processing data that comes in sequences.
/// 
/// Think of it like reading a book:
/// - A regular neural network would look at each word in isolation
/// - A GRU network remembers what it read earlier and uses that context to understand each new word
/// 
/// GRU networks have special "gates" that control what information to remember and what to forget:
/// - This helps them understand patterns that stretch over long sequences
/// - For example, in a sentence like "John went to the store because he needed milk," a GRU can connect "he" with "John"
/// 
/// GRU networks are useful for:
/// - Text processing and generation
/// - Time series prediction (like stock prices or weather)
/// - Speech recognition
/// - Any task where the order and context of data matters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GRUNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GRUNeuralNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a GRU neural network with the specified architecture. The architecture defines
    /// important properties like input size, hidden layer sizes, and output size of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU neural network with your chosen design.
    /// 
    /// When you create a GRU network, you specify its architecture, which is like a blueprint that defines:
    /// - How many inputs the network accepts
    /// - How many hidden units to use (the network's "memory capacity")
    /// - How many outputs the network produces
    /// - Other structural aspects of the network
    /// 
    /// Think of it like defining the floor plan before building a house - you're setting up the basic structure
    /// that will determine how information flows through your network.
    /// </para>
    /// </remarks>
    public GRUNeuralNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        InitializeGRULayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the GRU neural network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for a GRU neural network are created.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your neural network.
    /// 
    /// When initializing layers:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates a standard set of GRU layers automatically
    /// 
    /// This is like assembling all the components of your network before training begins.
    /// The standard GRU layers typically include:
    /// - Input layers to receive your data
    /// - GRU layers that process sequential information
    /// - Output layers that produce the final prediction
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        InitializeGRULayers();
    }

    private void InitializeGRULayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayersInternal(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultGRULayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the network using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing updated parameters for all layers.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter values to each layer in the network. It extracts
    /// the appropriate segment of the parameter vector for each layer based on the layer's parameter count.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learned values in the network.
    /// 
    /// During training, a neural network adjusts its internal values (parameters) to make
    /// better predictions. This method:
    /// 
    /// 1. Takes a long list of new parameter values
    /// 2. Figures out which values belong to which layers
    /// 3. Updates each layer with its corresponding values
    /// 
    /// Think of it like fine-tuning different parts of a machine based on how well it performed.
    /// GRU networks have several important parameters:
    /// - Update gate parameters: control what information to add from the current input
    /// - Reset gate parameters: control what past information to forget
    /// - Memory parameters: store information across the sequence
    /// 
    /// This method ensures all these parameters get updated correctly during training.
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
    /// Performs a forward pass through the network and generates predictions.
    /// </summary>
    /// <param name="input">The input tensor to the network, typically a sequence.</param>
    /// <returns>The output tensor produced by the network.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input tensor through all layers of the GRU network in sequence,
    /// applying the appropriate transformations at each step. For sequential data, the input is typically
    /// a 3D tensor with dimensions [batch_size, sequence_length, feature_size].
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and runs it through the neural network to get a prediction.
    /// 
    /// For example, if your input is a sequence of words:
    /// 1. Each word is passed through the network one at a time
    /// 2. The GRU remembers information from previous words
    /// 3. After processing the entire sequence, the network produces its prediction
    /// 
    /// This is similar to how you might read a sentence and understand its meaning
    /// by considering each word in context with the ones before it.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        var current = input;

        // Forward pass through all layers
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Trains the GRU network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for calculating error.</param>
    /// <remarks>
    /// <para>
    /// This method implements the training process for GRU networks using backpropagation through time (BPTT).
    /// It forward propagates the input, calculates the error by comparing with the expected output,
    /// and then backpropagates the error to update the network parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the network learns from examples.
    /// 
    /// The training process works like this:
    /// 1. The network makes a prediction based on the input sequence
    /// 2. The prediction is compared to the expected output to calculate the error
    /// 3. The error is used to adjust the network's internal values (parameters)
    /// 4. Over time, these adjustments help the network make better predictions
    /// 
    /// In GRU networks, training is more complex because the error needs to flow backwards
    /// through time (across the sequence), but this complexity is handled internally.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Set network to training mode
        SetTrainingMode(true);

        // Forward pass with memory
        var predictions = ForwardWithMemory(input);

        // Calculate error/loss
        var outputGradients = predictions.Subtract(expectedOutput);

        // Backpropagate error
        Backpropagate(outputGradients);

        // Calculate learning rate - could be more sophisticated in production
        T learningRate = NumOps.FromDouble(0.001);

        // Update parameters based on gradients
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }

        // Calculate and store the loss using the loss function
        LastLoss = LossFunction.CalculateLoss(predictions.ToVector(), expectedOutput.ToVector());

        // Set network back to inference mode
        SetTrainingMode(false);
    }

    /// <summary>
    /// Performs a forward pass while storing intermediate values for backpropagation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor from the forward pass.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method processes the input through the network while
    /// remembering intermediate values needed for learning.
    ///
    /// Think of it like solving a math problem and showing your work - the network
    /// needs to keep track of intermediate steps to understand how to improve.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        var current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            // Store input to each layer for backpropagation
            _layerInputs[i] = current;

            // Forward pass through layer
            current = Layers[i].Forward(current);

            // Store output from each layer for backpropagation
            _layerOutputs[i] = current;
        }

        return current;
    }

    /// <summary>
    /// Gets metadata about this GRU Neural Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its name, description, architecture,
    /// and other relevant information that might be useful for model management and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about this neural network model.
    /// 
    /// The metadata includes:
    /// - The type of model (GRU Neural Network)
    /// - The network architecture (how many layers, neurons, etc.)
    /// - Configuration details specific to GRU networks
    /// 
    /// This information is useful for documentation, debugging, and when saving/loading models.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GRUNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize },
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Saves GRU-specific data to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to save to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes any GRU-specific data that isn't part of the base neural network.
    /// In the case of a GRU network, this might include sequence-specific settings or state.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves special GRU settings to a file.
    /// 
    /// When saving the model:
    /// - The base neural network parts are saved by other methods
    /// - This method saves any GRU-specific settings or state
    /// 
    /// This ensures that when you reload the model, it will have all the same settings
    /// and capabilities as the original.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
    }

    /// <summary>
    /// Loads GRU-specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to load from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes GRU-specific data that was previously saved using SerializeNetworkSpecificData.
    /// It restores any special configuration or state that is unique to GRU networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads special GRU settings from a file.
    /// 
    /// When loading a saved model:
    /// - The base neural network parts are loaded by other methods
    /// - This method loads any GRU-specific settings or state
    /// 
    /// This ensures that the loaded model functions exactly like the original one that was saved.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
    }

    /// <summary>
    /// Creates a new instance of the GRU Neural Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new GRU Neural Network instance with the same architecture and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the GRU Neural Network with the same architecture as the current instance.
    /// It's used in scenarios where a fresh copy of the model is needed while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the neural network with the same setup.
    /// 
    /// Think of it like creating a clone of the network:
    /// - The new network has the same architecture (structure)
    /// - But it's a completely separate instance with its own parameters and learning state
    /// 
    /// This is useful when you need multiple instances of the same GRU model,
    /// such as for ensemble learning or comparing different training approaches.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GRUNeuralNetwork<T>(this.Architecture);
    }
}
