namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Transformer neural network architecture, which is particularly effective for 
/// sequence-based tasks like natural language processing.
/// </summary>
/// <remarks>
/// <para>
/// The Transformer architecture is a type of neural network design that uses self-attention mechanisms
/// instead of recurrence or convolution. This approach allows the model to weigh the importance of 
/// different parts of the input sequence when producing each part of the output sequence.
/// </para>
/// <para>
/// The key components of a Transformer include:
/// - Multi-head attention layers: Allow the model to focus on different parts of the input
/// - Feed-forward networks: Process the attended information
/// - Layer normalization: Stabilize the network during training
/// - Residual connections: Help information flow through the network
/// </para>
/// <para><b>For Beginners:</b> A Transformer is a modern type of neural network that excels at 
/// understanding sequences of data, like sentences or time series.
/// 
/// Think of it like reading a book:
/// - When you read a sentence, some words are more important than others for understanding the meaning
/// - A Transformer can "pay attention" to different words based on their importance
/// - It can look at the entire context at once, rather than reading one word at a time
/// 
/// For example, in the sentence "The animal didn't cross the street because it was too wide",
/// the Transformer can figure out that "it" refers to "the street" by paying attention to the
/// relationship between these words.
/// 
/// Transformers are behind many recent AI advances, including large language models like GPT and BERT.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class Transformer<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (attention regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention regularization aggregates auxiliary losses from all MultiHeadAttentionLayers in the network.
    /// This includes both entropy regularization and head diversity penalties.
    /// </para>
    /// <para><b>For Beginners:</b> This controls attention quality across the entire Transformer.
    ///
    /// When enabled, the Transformer:
    /// - Collects regularization from all attention layers
    /// - Prevents attention collapse across the network
    /// - Encourages diverse attention patterns at all levels
    ///
    /// This is especially important for:
    /// - Deep transformers (many layers)
    /// - Models with many attention heads
    /// - Tasks requiring robust attention patterns
    ///
    /// The auxiliary loss helps maintain attention quality throughout training.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the attention regularization auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much network-level attention regularization contributes to the total loss.
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to encourage good attention throughout the network.
    ///
    /// Common values:
    /// - 0.005 (default): Balanced network-level regularization
    /// - 0.001-0.003: Light regularization
    /// - 0.008-0.01: Strong regularization
    ///
    /// Higher values enforce stronger attention quality constraints.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastAttentionRegularizationLoss;

    /// <summary>
    /// The configuration settings for this Transformer network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the architecture configuration which defines the structure and properties
    /// of this Transformer network, including settings like embedding size, number of attention 
    /// heads, and feed-forward dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the blueprint for our Transformer.
    /// 
    /// It contains all the important settings that determine how the Transformer works:
    /// - How many attention mechanisms to use
    /// - How large each part of the network should be
    /// - How information flows through the network
    /// 
    /// Just like a house blueprint defines the structure of a house, this architecture
    /// defines the structure of our Transformer neural network.
    /// </para>
    /// </remarks>
    private readonly TransformerArchitecture<T> _transformerArchitecture;

    /// <summary>
    /// Gets or sets the attention mask used in the Transformer.
    /// </summary>
    /// <remarks>
    /// This mask is used to control which positions are attended to in the self-attention layers.
    /// It's particularly useful for tasks like sequence generation where future tokens should be masked.
    /// </remarks>
    public Tensor<T>? AttentionMask { get; set; }

    /// <summary>
    /// The optimizer used to update the Transformer's parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The optimizer determines how the Transformer's parameters should be adjusted based on the calculated gradients.
    /// It's responsible for the learning process, controlling how quickly and in what manner the model improves.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is like a coach for the Transformer.
    /// 
    /// Think of training the Transformer as teaching it to play a sport:
    /// - The optimizer decides how to adjust the Transformer's technique (its parameters)
    /// - It looks at how the Transformer performed (the loss) and suggests improvements
    /// - Different optimizers have different strategies, like focusing on quick improvements or steady, consistent progress
    /// 
    /// The choice of optimizer can significantly affect how well and how quickly the Transformer learns.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a new Transformer neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how this Transformer will be structured.
    /// This includes settings like embedding size, number of attention heads, and feed-forward dimensions.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Transformer neural network with the provided architecture
    /// configuration. It passes the architecture to the base class constructor and also stores it
    /// for use in initializing the Transformer-specific layers.
    /// </para>
    /// <para><b>For Beginners:</b> This is where we create our Transformer network.
    /// 
    /// When you create a new Transformer, you provide a blueprint (the architecture) that specifies:
    /// - How many layers it should have
    /// - How attention works in the network
    /// - How large the various components should be
    /// 
    /// This is similar to how you might specify the size, number of rooms, and layout when building a house.
    /// </para>
    /// </remarks>
    public Transformer(TransformerArchitecture<T> architecture, ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _transformerArchitecture = architecture;
        _optimizer = optimizer ?? new GradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAttentionRegularizationLoss = NumOps.Zero;

        InitializeLayers();
    }

    /// <summary>
    /// Sets up the layers of the Transformer network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided by the user or creates default Transformer layers.
    /// A typical Transformer consists of attention mechanisms, normalization layers, and feed-forward networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the Transformer.
    /// 
    /// It works in one of two ways:
    /// - If you've provided your own custom layers, it uses those
    /// - Otherwise, it creates a standard set of Transformer layers
    /// 
    /// These layers typically include:
    /// - Attention layers (which let the model focus on relevant parts of the input)
    /// - Normalization layers (which keep the numbers from getting too large or small)
    /// - Feed-forward layers (which process the information)
    /// 
    /// It's like assembling the rooms and sections of a house according to the blueprint.
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
            // Use default transformer layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultTransformerLayers(_transformerArchitecture));
        }
    }

    /// <summary>
    /// Ensures that custom layers provided for the Transformer meet the minimum requirements.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <remarks>
    /// <para>
    /// A valid Transformer must include at least one attention layer and one normalization layer.
    /// Attention layers allow the model to focus on different parts of the input sequence.
    /// Normalization layers help stabilize training by normalizing the activations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if your custom layers will actually work as a Transformer.
    /// 
    /// For a Transformer to function properly, it needs at minimum:
    /// - An attention layer (which helps the model focus on important parts of the input)
    /// - A normalization layer (which keeps the numbers stable during training)
    /// 
    /// If either of these is missing, it's like trying to build a house without walls or a foundation - it won't work!
    /// 
    /// This method checks for these essential components and raises an error if they're missing.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the custom layers don't include required layer types.
    /// </exception>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        bool hasAttentionLayer = false;
        bool hasLayerNorm = false;

        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is MultiHeadAttentionLayer<T>)
            {
                hasAttentionLayer = true;
            }
            else if (layers[i] is LayerNormalizationLayer<T>)
            {
                hasLayerNorm = true;
            }
        }

        if (!hasAttentionLayer)
        {
            throw new InvalidOperationException("Custom Transformer must include at least one MultiHeadAttentionLayer.");
        }

        if (!hasLayerNorm)
        {
            throw new InvalidOperationException("Custom Transformer must include at least one LayerNormalizationLayer.");
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for attention regularization across all attention layers.
    /// </summary>
    /// <returns>The computed attention regularization auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method aggregates auxiliary losses from all MultiHeadAttentionLayers in the Transformer.
    /// It collects both entropy regularization and head diversity penalties from each attention layer.
    /// Formula: L = (1/N) * Σ_layers auxloss_i where N = number of attention layers
    /// </para>
    /// <para><b>For Beginners:</b> This calculates network-wide attention quality.
    ///
    /// Transformer attention regularization works by:
    /// 1. Finding all attention layers in the network
    /// 2. Computing auxiliary loss for each layer (if enabled)
    /// 3. Averaging these losses across all layers
    /// 4. Returning the network-level regularization penalty
    ///
    /// This helps because:
    /// - Maintains attention quality throughout the entire network
    /// - Prevents attention collapse at any level
    /// - Encourages diverse attention patterns across all layers
    /// - Improves interpretability and robustness
    ///
    /// The auxiliary loss is added to the main task loss during training.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastAttentionRegularizationLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        T totalAuxLoss = NumOps.Zero;
        int attentionLayerCount = 0;

        // Aggregate auxiliary losses from all attention layers
        foreach (var layer in Layers)
        {
            if (layer is IAuxiliaryLossLayer<T> auxLayer && auxLayer.UseAuxiliaryLoss)
            {
                T layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
                totalAuxLoss = NumOps.Add(totalAuxLoss, layerAuxLoss);
                attentionLayerCount++;
            }
        }

        // Average over all attention layers
        if (attentionLayerCount > 0)
        {
            totalAuxLoss = NumOps.Divide(totalAuxLoss, NumOps.FromDouble(attentionLayerCount));
        }

        _lastAttentionRegularizationLoss = totalAuxLoss;
        return totalAuxLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the attention regularization auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about network-level attention regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about attention regularization across the Transformer,
    /// including aggregated losses, layer counts, and configuration parameters.
    /// This information is useful for monitoring training progress and debugging attention issues.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how attention works across the network.
    ///
    /// The diagnostics include:
    /// - Total attention regularization loss (averaged across layers)
    /// - Weight applied to the regularization
    /// - Number of attention layers with regularization enabled
    /// - Whether network-level regularization is enabled
    ///
    /// This helps you:
    /// - Monitor attention quality throughout the network
    /// - Debug issues with attention collapse
    /// - Understand the impact of regularization at the network level
    /// - Track which layers are contributing to regularization
    ///
    /// You can use this information to adjust regularization settings for better training results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "TotalAttentionRegularizationLoss", _lastAttentionRegularizationLoss?.ToString() ?? "0" },
            { "AttentionRegularizationWeight", AuxiliaryLossWeight?.ToString() ?? "0.005" },
            { "UseAttentionRegularization", UseAuxiliaryLoss.ToString() }
        };

        // Count attention layers with regularization enabled
        int attentionLayerCount = Layers.OfType<IAuxiliaryLossLayer<T>>()
            .Count(l => l.UseAuxiliaryLoss);
        diagnostics["AttentionLayersWithRegularization"] = attentionLayerCount.ToString();

        // Total attention layers
        int totalAttentionLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        diagnostics["TotalAttentionLayers"] = totalAttentionLayers.ToString();

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Updates the parameters of all layers in the Transformer network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer based on their parameter counts.
    /// It's typically used during training when applying gradient updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the Transformer's internal values during training.
    /// 
    /// Think of parameters as the "settings" of the Transformer:
    /// - Each layer needs a certain number of parameters to function
    /// - During training, these parameters are constantly adjusted to improve performance
    /// - This method takes a big list of new parameter values and gives each layer its share
    /// 
    /// It's like distributing updated parts to each section of a machine so it works better.
    /// Each layer gets exactly the number of parameters it needs.
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
    /// Performs a forward pass through the Transformer network to generate predictions.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the predictions.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input through each layer of the Transformer sequentially.
    /// It handles both the encoder and decoder parts of the Transformer if present.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and runs it through the entire Transformer.
    /// 
    /// It's like sending a message through a complex machine:
    /// - The input goes through each part of the Transformer in order
    /// - Each layer processes the data in its own way (attention, normalization, etc.)
    /// - The final output is the Transformer's prediction or transformation of your input
    /// 
    /// This is used when you want to use a trained Transformer to process new data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> output = input;
        Tensor<T>? encoderOutput = null;
        Tensor<T> mask = AttentionMask ?? Tensor<T>.CreateDefault(input.Shape, NumOps.One); // Default to all ones if no mask is provided

        // Process all layers sequentially
        // The layer list structure: input projection, positional encoding, dropout, then encoder/decoder blocks
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is DecoderLayer<T> decoderLayer)
            {
                // Decoder layer with cross-attention needs encoder output
                output = decoderLayer.Forward(output, encoderOutput ?? output, mask);
            }
            else if (Layers[i] is AttentionLayer<T> attentionLayer)
            {
                output = attentionLayer.Forward(output, mask);
            }
            else
            {
                output = Layers[i].Forward(output);
            }

            // Track encoder output for cross-attention in decoders
            // The last attention layer before any decoder layer is the encoder output
            if (Layers[i] is MultiHeadAttentionLayer<T> && encoderOutput is null)
            {
                // Check if there are decoder layers ahead
                for (int j = i + 1; j < Layers.Count; j++)
                {
                    if (Layers[j] is DecoderLayer<T>)
                    {
                        encoderOutput = output;
                        break;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Trains the Transformer network on a single batch of data.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass, calculates the loss, and then backpropagates
    /// the error to update the network's parameters. It uses the specified loss function and optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the Transformer using example data.
    /// 
    /// The process works like this:
    /// 1. The Transformer makes a prediction based on the input
    /// 2. We compare this prediction to the expected output
    /// 3. We calculate how wrong the prediction was (the "loss")
    /// 4. We adjust the Transformer's internal values to make it a little more accurate next time
    /// 
    /// This process is repeated many times with different examples to train the Transformer.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        Tensor<T> prediction = Predict(input);

        var flattenedPredictions = prediction.ToVector();
        var flattenedOutput = expectedOutput.ToVector();

        // Calculate main loss
        LastLoss = LossFunction.CalculateLoss(flattenedPredictions, flattenedOutput);

        // Add auxiliary loss if enabled
        if (UseAuxiliaryLoss)
        {
            T auxLoss = ComputeAuxiliaryLoss();
            T weightedAuxLoss = NumOps.Multiply(AuxiliaryLossWeight, auxLoss);
            LastLoss = NumOps.Add(LastLoss, weightedAuxLoss);
        }

        // Backward pass
        var outputGradients = LossFunction.CalculateDerivative(flattenedPredictions, flattenedOutput);

        // Backpropagate to get gradients for all layers
        Backpropagate(Tensor<T>.FromVector(outputGradients));

        // Get parameter gradients
        Vector<T> parameterGradients = GetParameterGradients();

        // Apply gradient clipping if necessary
        parameterGradients = ClipGradient(parameterGradients);

        // Update parameters
        Vector<T> currentParameters = GetParameters();
        Vector<T> updatedParameters = _optimizer.UpdateParameters(currentParameters, parameterGradients);

        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Sets the attention mask for the Transformer.
    /// </summary>
    /// <param name="mask">The attention mask to be used in self-attention layers.</param>
    /// <remarks>
    /// Call this method before training or prediction to set a mask for controlling attention.
    /// </remarks>
    public void SetAttentionMask(Tensor<T> mask)
    {
        AttentionMask = mask;
    }

    /// <summary>
    /// Retrieves metadata about the Transformer model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the Transformer.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the Transformer,
    /// including its type, architecture details, and current state.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a summary of the Transformer's current state and structure.
    /// 
    /// It's like creating a report card for the Transformer, including:
    /// - What type of model it is (Transformer)
    /// - How it's structured (number of layers, size of each layer, etc.)
    /// - Its current training progress
    /// - Other important details about its configuration
    /// 
    /// This information is useful for keeping track of different models, especially when you're
    /// experimenting with multiple configurations.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Transformer,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumHeads", _transformerArchitecture.NumHeads },
                { "NumEncoderLayers", _transformerArchitecture.NumEncoderLayers },
                { "NumDecoderLayers", _transformerArchitecture.NumDecoderLayers },
                { "MaxSequenceLength", _transformerArchitecture.MaxSequenceLength },
                { "VocabularySize", _transformerArchitecture.VocabularySize },
                { "DropoutRate", _transformerArchitecture.DropoutRate },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() },
                { "LossFunction", LossFunction.GetType().Name },
                { "Optimizer", _optimizer.GetType().Name }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Transformer-specific data to a binary stream.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes Transformer-specific configuration and state data to a binary stream.
    /// It allows the Transformer's current state to be saved and later reconstructed.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important details about the Transformer to a file.
    /// 
    /// It's like taking a snapshot of the Transformer's current state, including:
    /// - Its configuration (how it's set up)
    /// - Its current parameter values (what it has learned so far)
    /// 
    /// This allows you to save your trained Transformer and use it again later without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write Transformer-specific architecture details
        writer.Write(_transformerArchitecture.NumHeads);
        writer.Write(_transformerArchitecture.NumEncoderLayers);
        writer.Write(_transformerArchitecture.NumDecoderLayers);
        writer.Write(_transformerArchitecture.MaxSequenceLength);
        writer.Write(_transformerArchitecture.VocabularySize);
        writer.Write(Convert.ToDouble(_transformerArchitecture.DropoutRate));

        // Write loss function and optimizer types
        SerializationHelper<T>.SerializeInterface(writer, LossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes Transformer-specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads Transformer-specific configuration and state data from a binary stream.
    /// It reconstructs the Transformer's state from previously serialized data.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads all the important details about a Transformer from a file.
    /// 
    /// It's like reconstructing the Transformer from a saved snapshot, including:
    /// - Rebuilding its configuration (how it was set up)
    /// - Restoring its parameter values (what it had learned)
    /// 
    /// This allows you to load a previously trained Transformer and use it immediately without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read Transformer-specific architecture details
        int numHeads = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int maxSequenceLength = reader.ReadInt32();
        int vocabularySize = reader.ReadInt32();
        T dropoutRate = NumOps.FromDouble(reader.ReadDouble());

        // Read and reconstruct loss function and optimizer (must match serialization order).
        LossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader)
            ?? LossFunction;

        _optimizer = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader)
            ?? new GradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    /// <summary>
    /// Creates a new instance of the Transformer with the same architecture and configuration.
    /// </summary>
    /// <returns>A new instance of the Transformer with the same configuration as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new Transformer neural network with the same architecture, loss function,
    /// and optimizer as the current instance. The new instance has freshly initialized parameters,
    /// making it useful for creating separate instances with identical configurations or for
    /// resetting a network while preserving its structure.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a brand new Transformer with the same setup.
    /// 
    /// Think of it like creating a blueprint copy:
    /// - It has the same architecture (number of layers, attention heads, etc.)
    /// - It uses the same loss function to measure performance
    /// - It uses the same optimizer to learn from data
    /// - But it starts with fresh parameters (weights and biases)
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh network but keep the same design
    /// - Create multiple networks with identical settings for comparison
    /// - Reset a network to its initial state
    /// 
    /// The new Transformer will need to be trained from scratch, as it doesn't
    /// inherit any of the learned knowledge from the original.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Transformer<T>(
            _transformerArchitecture,
            LossFunction,
            _optimizer);
    }
}
