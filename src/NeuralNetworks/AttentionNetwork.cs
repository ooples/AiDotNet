namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a neural network that utilizes attention mechanisms for sequence processing.
/// </summary>
/// <remarks>
/// <para>
/// An attention network is a specialized neural network architecture designed for sequence processing tasks.
/// It uses attention mechanisms to dynamically focus on different parts of the input sequence when generating
/// outputs. This allows the network to capture long-range dependencies and relationships between elements in 
/// the sequence, making it particularly effective for tasks like natural language processing, time series analysis,
/// and other sequence-to-sequence problems.
/// </para>
/// <para><b>For Beginners:</b> This network mimics how humans pay attention to different parts of information.
/// 
/// Think of it like reading a complex paragraph:
/// - When you try to understand a sentence, you don't focus equally on all words
/// - You focus more on the important words that carry meaning
/// - You also connect related words even if they're far apart
/// 
/// For example, in the sentence "The cat, which had a white spot on its tail, chased the mouse":
/// - An attention network would connect "cat" with "chased" even though they're separated
/// - It would assign different importance to different words based on context
/// - This helps it understand the overall meaning better than networks that process words in isolation
/// 
/// This ability to selectively focus and connect distant information makes attention networks
/// powerful for language tasks, time series prediction, and many other sequence-based problems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class AttentionNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The maximum length of sequences this network can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the maximum length of sequences that the attention network is configured to process.
    /// It determines the size of attention matrices and positional encodings.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many elements (like words) the network can handle at once.
    /// 
    /// For example:
    /// - For text processing, this might be 512 tokens (roughly equivalent to paragraphs of text)
    /// - For time series data, this might be 100 time steps
    /// - The network cannot process sequences longer than this limit without truncation or splitting
    /// 
    /// This limit exists because attention mechanisms need to compare each element with every other element,
    /// which becomes computationally expensive for very long sequences.
    /// </para>
    /// </remarks>
    private int _sequenceLength;

    /// <summary>
    /// The size of the embeddings used to represent each element in the sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimensionality of the embedding vectors used to represent each element in the input sequence.
    /// Higher dimensional embeddings can capture more nuanced representations but require more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many features or dimensions each element has.
    /// 
    /// Think of it as the "richness" of information for each element in the sequence:
    /// - Larger values (like 768 or 1024) give more capacity to represent complex patterns
    /// - Common values range from 128 for simple tasks to 1024+ for complex language models
    /// - Higher values require more memory and processing power
    /// 
    /// For example, when processing text, each word might be represented by a 512-dimensional
    /// vector that captures various aspects of its meaning and context.
    /// </para>
    /// </remarks>
    private int _embeddingSize;

    /// <summary>
    /// The loss function used to measure the network's performance during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the loss function that the network uses to quantify its error during training.
    /// The loss function plays a crucial role in guiding the network's learning process.
    /// </para>
    /// <para><b>For Beginners:</b> The loss function is like a scorekeeper for the network's performance.
    /// 
    /// It does the following:
    /// - Measures how far off the network's predictions are from the correct answers
    /// - Provides a single number that represents how well (or poorly) the network is doing
    /// - Guides the network in adjusting its internal values to improve its performance
    /// 
    /// For attention networks, especially in tasks like language translation or text summarization,
    /// Cross-Entropy Loss is often used as the default. This loss function is particularly good at
    /// handling tasks where the network needs to choose from a set of possible outputs, which is
    /// common in language-related tasks.
    /// </para>
    /// </remarks>
    private readonly ILossFunction<T> _lossFunction = default!;

    /// <summary>
    /// Initializes a new instance of the <see cref="AttentionNetwork{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture specification for the network.</param>
    /// <param name="sequenceLength">The maximum length of sequences this network can process.</param>
    /// <param name="embeddingSize">The size of the embeddings used to represent each element in the sequence.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, a default Cross-Entropy loss function will be used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an attention network with the specified architecture, sequence length, and embedding size.
    /// It initializes the network's layers according to the architecture specification or uses default layers if none are provided.
    /// If no loss function is specified, it uses Cross-Entropy Loss, which is commonly used for attention networks in tasks like
    /// machine translation or text summarization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new attention network with the specified settings.
    /// 
    /// The parameters you provide determine:
    /// - architecture: The overall design of the network (layers, connections, etc.)
    /// - sequenceLength: How many elements (like words) the network can process at once
    /// - embeddingSize: How rich the representation of each element is
    /// - lossFunction: How the network measures its mistakes during training (optional)
    /// 
    /// These settings control the capacity, expressiveness, and computational requirements of the network.
    /// Larger values for sequenceLength and embeddingSize give the network more capacity to handle
    /// complex tasks but require more memory and processing power.
    /// 
    /// The loss function helps the network learn by measuring how far off its predictions are.
    /// Cross-Entropy Loss is used by default because it works well for many language-related tasks.
    /// </para>
    /// </remarks>
    public AttentionNetwork(NeuralNetworkArchitecture<T> architecture, int sequenceLength, int embeddingSize, ILossFunction<T>? lossFunction = null) : 
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _sequenceLength = sequenceLength;
        _embeddingSize = embeddingSize;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the attention network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the layers of the attention network either by using the layers provided by the user
    /// in the architecture specification or by creating default attention layers if none are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the attention network.
    /// 
    /// It does one of two things:
    /// 1. If you provided specific layers in the architecture, it uses those
    /// 2. If you didn't provide layers, it creates a default set of attention layers
    /// 
    /// The default layers typically include:
    /// - Embedding layers to convert inputs to vector representations
    /// - Attention layers to focus on relevant parts of the sequence
    /// - Feed-forward layers to process the attended information
    /// - Output layers to produce the final results
    /// 
    /// This flexibility allows both beginners and experts to use the network effectively.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAttentionLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters of the attention network.
    /// </summary>
    /// <param name="parameters">The parameters to update the network with.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the network with the provided parameter values.
    /// It distributes the parameters to each layer based on the number of parameters in each layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's internal values to improve its performance.
    /// 
    /// During training:
    /// - The learning algorithm calculates how the parameters should change
    /// - This method applies those changes to the actual parameters
    /// - Each layer gets its own portion of the parameter updates
    /// 
    /// Think of it like fine-tuning all the components of the network based on feedback:
    /// - Attention mechanisms learn to focus on more relevant parts
    /// - Embedding layers learn better representations of the input
    /// - Feed-forward layers learn to process the information more effectively
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
    /// Makes a prediction using the current state of the Attention Network.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor after passing through all layers of the network.</returns>
    /// <exception cref="ArgumentException">Thrown when the input sequence length exceeds the maximum allowed length.</exception>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network, transforming the input data through each layer
    /// to produce a final prediction. It includes input validation to ensure the provided tensor matches the
    /// expected input shape of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network processes new data to make predictions.
    /// 
    /// The prediction process:
    /// 1. Checks if the input data is valid and not too long
    /// 2. Passes the input through each layer of the network
    /// 3. Each layer transforms the data, with attention layers focusing on relevant parts
    /// 4. The final layer produces the network's prediction
    /// 
    /// Think of it like a series of experts each looking at the data and passing their insights
    /// to the next expert, with the last one making the final decision.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (input.Shape[1] > _sequenceLength)
            throw new ArgumentException($"Input sequence length {input.Shape[1]} exceeds maximum sequence length {_sequenceLength}");

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Trains the Attention Network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor used for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method implements the training process for the Attention Network. It performs a forward pass,
    /// calculates the loss between the network's prediction and the expected output, and then backpropagates
    /// this error to adjust the network's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network learns from examples.
    /// 
    /// The training process:
    /// 1. Makes a prediction using the current network state
    /// 2. Compares the prediction to the correct answer to calculate the error
    /// 3. Figures out how to adjust the network to reduce this error
    /// 4. Updates the network's internal settings to improve future predictions
    /// 
    /// It's like a student doing practice problems, checking their answers, and learning
    /// from their mistakes to do better next time.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var output = Predict(input);
        var flattenedOutput = output.ToVector();
        var flattenedExpectedOutput = expectedOutput.ToVector();

        // Calculate loss
        var loss = _lossFunction.CalculateLoss(flattenedOutput, flattenedExpectedOutput);
        LastLoss = loss;

        // Backward pass
        var flattenedGradient = _lossFunction.CalculateDerivative(flattenedOutput, flattenedExpectedOutput);
    
        // Unflatten the gradient to match the output shape
        var gradient = new Tensor<T>(output.Shape).Unflatten(flattenedGradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        // Update parameters
        UpdateParameters(GetParameters());
    }

    /// <summary>
    /// Gets metadata about the Attention Network model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata that describes the Attention Network, including its type, architecture details,
    /// and other relevant information. This metadata can be useful for model management, documentation,
    /// and versioning.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your network's setup and characteristics.
    /// 
    /// The metadata includes:
    /// - The type of model (Attention Network)
    /// - Details about the network's structure and capacity
    /// - Information about the input and output shapes
    /// 
    /// It's like a spec sheet for your network, useful for keeping track of different versions
    /// or comparing different network configurations.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.AttentionNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "SequenceLength", _sequenceLength },
                { "EmbeddingSize", _embeddingSize },
                { "NumberOfLayers", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToList() },
                { "ParameterCount", Layers.Sum(l => l.ParameterCount) },
                { "InputShape", new[] { _sequenceLength, _embeddingSize } },
                { "OutputShape", Layers[Layers.Count - 1].GetOutputShape() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data for the Attention Network.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific configuration and state of the Attention Network to a binary stream.
    /// It includes network-specific parameters that are essential for later reconstruction of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the unique settings of your Attention Network.
    /// 
    /// It writes:
    /// - The sequence length and embedding size
    /// - The configuration of each layer
    /// - Any other Attention Network-specific parameters
    /// 
    /// Saving these details allows you to recreate the exact same network structure later.
    /// It's like writing down a detailed recipe so you can make the same dish again in the future.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_embeddingSize);
    }

    /// <summary>
    /// Deserializes network-specific data for the Attention Network.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific configuration and state of the Attention Network from a binary stream.
    /// It reconstructs the network-specific parameters to match the state of the network when it was serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the unique settings of your Attention Network.
    /// 
    /// It reads:
    /// - The sequence length and embedding size
    /// - The configuration of each layer
    /// - Any other Attention Network-specific parameters
    /// 
    /// Loading these details allows you to recreate the exact same network structure that was previously saved.
    /// It's like following a detailed recipe to recreate a dish exactly as it was made before.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _embeddingSize = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of the attention network model.
    /// </summary>
    /// <returns>A new instance of the attention network model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the attention network model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the model structure without copying the learned data.
    /// 
    /// Think of it like creating a blueprint of the network's architecture:
    /// - It includes the same structure (layers, connections, sizes)
    /// - It preserves the configuration settings (sequence length, embedding size)
    /// - It doesn't copy over any of the learned knowledge (weights, biases)
    /// 
    /// This is particularly useful when you want to save or load models, as it provides the framework
    /// that learned parameters can be loaded into.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AttentionNetwork<T>(Architecture, _sequenceLength, _embeddingSize, _lossFunction);
    }
}