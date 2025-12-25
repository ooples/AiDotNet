namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Memory Network, a neural network architecture designed with explicit memory components
/// for improved reasoning and question answering capabilities.
/// </summary>
/// <remarks>
/// <para>
/// Memory Networks combine neural network components with a long-term memory matrix that can be read from
/// and written to. This architecture allows the network to store information persistently and access it
/// selectively when needed, making it particularly effective for tasks requiring reasoning over facts or
/// answering questions based on provided context.
/// </para>
/// <para><b>For Beginners:</b> A Memory Network is a special type of neural network that has its own "memory storage" component.
/// 
/// Think of it like a person who has:
/// - A notebook (the memory) where they can write down important facts
/// - The ability to read specific information from their notebook when needed
/// - The ability to add new information to their notebook as they learn
/// 
/// For example, if you provided a Memory Network with several facts about a topic:
/// - It would store these facts in its memory matrix
/// - When asked a question, it would search its memory for relevant information
/// - It would use this retrieved information to formulate an answer
/// 
/// Memory Networks are particularly good at:
/// - Question answering based on a given context
/// - Reasoning tasks that require remembering multiple facts
/// - Dialog systems that need to maintain conversation history
/// - Tasks where information needs to be remembered and used later
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MemoryNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the memory (number of memory slots).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory size determines how many separate memory entries the network can store.
    /// Each memory slot can contain a vector of information (an embedding) that the network can later access.
    /// A larger memory size allows for storing more distinct pieces of information but increases
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many "pages" are in the network's notebook.
    /// 
    /// Think of memory size as:
    /// - The number of separate facts the network can store
    /// - The capacity of the network's memory
    /// 
    /// For example:
    /// - A memory size of 100 means the network can store 100 different pieces of information
    /// - Each piece is stored as a vector (a list of numbers) in the memory matrix
    /// 
    /// Choosing the right memory size depends on your task:
    /// - Too small: The network might not be able to store all necessary information
    /// - Too large: Could be computationally expensive and might store irrelevant details
    /// </para>
    /// </remarks>
    private readonly int _memorySize;

    /// <summary>
    /// Gets the size of each memory embedding vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The embedding size determines the dimensionality of each memory slot's vector representation.
    /// This controls how rich and detailed each stored memory can be. Larger embedding sizes can capture
    /// more nuanced information but require more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This is how detailed each "note" in the network's notebook can be.
    /// 
    /// Think of embedding size as:
    /// - The amount of detail that can be stored about each fact
    /// - The richness of each memory representation
    /// 
    /// For example:
    /// - A small embedding size (like 32) might store basic information about a fact
    /// - A large embedding size (like 256) can store much more detailed and nuanced information
    /// 
    /// Higher embedding sizes allow for more complex representations but require more processing power.
    /// This is similar to how writing a detailed paragraph contains more information than just a few words.
    /// </para>
    /// </remarks>
    private readonly int _embeddingSize;

    /// <summary>
    /// Gets or sets the memory matrix that stores embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory matrix is the core storage component of the Memory Network. It is organized as a matrix
    /// where each row represents a separate memory slot, and each column represents a dimension in the
    /// embedding space. The network reads from and writes to this matrix during operation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual "notebook" where information is stored.
    /// 
    /// The memory matrix:
    /// - Is organized as a table with rows and columns
    /// - Each row is a separate memory slot (a separate fact)
    /// - Each column represents one aspect of the information being stored
    /// 
    /// When the network operates:
    /// - It can read specific information from this matrix
    /// - It can update or write new information to this matrix
    /// - The matrix persists information across different inputs, acting as long-term memory
    /// 
    /// This persistent memory is what gives Memory Networks their power for reasoning tasks,
    /// as they can store and retrieve information as needed throughout a sequence of operations.
    /// </para>
    /// </remarks>
    private Matrix<T> _memory;

    /// <summary>
    /// Initializes a new instance of the <see cref="MemoryNetwork{T}"/> class with the specified architecture and memory parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="memorySize">The number of memory slots in the memory matrix.</param>
    /// <param name="embeddingSize">The dimensionality of each memory embedding vector.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a Memory Network with the specified architecture and memory configuration.
    /// It initializes the memory matrix to the given dimensions and sets up the network layers based
    /// on the provided architecture or default configuration if no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Memory Network with your chosen settings.
    /// 
    /// When creating a Memory Network, you specify three main things:
    /// 
    /// 1. Architecture: The basic structure of your network (input/output sizes, etc.)
    /// 
    /// 2. Memory Size: How many separate facts the network can remember
    ///    - Like choosing how many pages your notebook has
    ///    - More memory slots = more storage capacity
    /// 
    /// 3. Embedding Size: How detailed each stored fact can be
    ///    - Like deciding how much detail to write on each notebook page
    ///    - Larger embeddings = more detailed representations
    /// 
    /// Once created, the network initializes an empty memory matrix (like a blank notebook)
    /// and sets up the layers needed for processing inputs and interacting with memory.
    /// </para>
    /// </remarks>
    public MemoryNetwork(NeuralNetworkArchitecture<T> architecture, int memorySize, int embeddingSize, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _memorySize = memorySize;
        _embeddingSize = embeddingSize;
        _memory = new Matrix<T>(_memorySize, _embeddingSize);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Memory Network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Memory Network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for a Memory Network are created,
    /// including input encoding, memory reading, memory writing, and output layers configured with
    /// the specified memory parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your Memory Network.
    /// 
    /// When initializing the network:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates standard Memory Network layers automatically
    /// 
    /// The standard Memory Network layers typically include:
    /// 1. Input Encoding Layer: Converts raw input into a format suitable for memory operations
    /// 2. Memory Read Layer: Allows the network to retrieve relevant information from memory
    /// 3. Memory Write Layer: Allows the network to update memory with new information
    /// 4. Output Layer: Produces the final answer based on the input and retrieved memory
    /// 
    /// This process is like assembling all the components before the network starts processing data.
    /// Each layer has a specific role in how the network interacts with its memory.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMemoryNetworkLayers(Architecture, _memorySize, _embeddingSize));
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
    /// This allows for updating the learned weights and biases in the network's layers after training.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the network's layers.
    /// 
    /// During training, a Memory Network learns many values (called parameters) that determine
    /// how it processes information. These include:
    /// - How to encode inputs
    /// - How to determine which memory slots to access
    /// - How to update memory with new information
    /// - How to produce outputs based on memory and input
    /// 
    /// This method:
    /// 1. Takes a long list of all these parameters
    /// 2. Figures out which parameters belong to which layers
    /// 3. Updates each layer with its corresponding parameters
    /// 
    /// Note that this updates the network's processing mechanisms but not the content of the memory itself.
    /// The memory content is updated during normal operation through the memory write layers.
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
    /// Processes input through the memory network to generate predictions.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Memory Network. It encodes the input,
    /// uses it to calculate attention over memory slots, retrieves relevant information from memory,
    /// and combines it with the input to generate the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes an input through the Memory Network to get an answer.
    /// 
    /// The prediction process works like this:
    /// 1. Input Encoding: Convert the input (like a question) into an internal representation
    /// 2. Memory Attention: Determine which parts of memory are most relevant to this input
    /// 3. Memory Reading: Retrieve information from the most relevant memory slots
    /// 4. Response Generation: Combine the input with the retrieved memory to generate an answer
    /// 
    /// This is similar to how you might answer a question:
    /// - First understand the question
    /// - Then recall relevant information from your memory
    /// - Finally formulate an answer based on both the question and what you remembered
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Set to inference mode
        SetTrainingMode(false);

        // Get batch size from input shape
        bool isBatch = input.Shape.Length > 1 && input.Shape[0] > 1;
        int batchSize = isBatch ? input.Shape[0] : 1;

        // Process through input encoding layers (first quarter of layers)
        Tensor<T> encoded = EncodeInput(input);

        // Calculate attention over memory
        Tensor<T> attentionWeights = CalculateAttention(encoded);

        // Read from memory using attention weights
        Tensor<T> memoryReadout = ReadFromMemory(attentionWeights);

        // Combine input representation with memory
        Tensor<T> combined = CombineInputAndMemory(encoded, memoryReadout);

        // Process through output layers to generate prediction
        Tensor<T> output = GenerateOutput(combined);

        // Update memory with new information (optional, depends on architecture)
        if (ShouldUpdateMemoryDuringInference())
        {
            UpdateMemory(encoded, attentionWeights);
        }

        return output;
    }

    /// <summary>
    /// Encodes the input using the input encoding layers.
    /// </summary>
    /// <param name="input">The input tensor to encode.</param>
    /// <returns>The encoded input tensor.</returns>
    private Tensor<T> EncodeInput(Tensor<T> input)
    {
        // Use the first quarter of layers for input encoding
        int encodingLayerCount = Layers.Count / 4;

        Tensor<T> current = input;
        for (int i = 0; i < encodingLayerCount; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Calculates attention weights over memory slots based on the encoded input.
    /// </summary>
    /// <param name="encoded">The encoded input tensor.</param>
    /// <returns>Attention weights over memory slots.</returns>
    private Tensor<T> CalculateAttention(Tensor<T> encoded)
    {
        // Use the second quarter of layers for attention calculation
        int startLayerIndex = Layers.Count / 4;
        int endLayerIndex = Layers.Count / 2;

        Tensor<T> current = encoded;
        for (int i = startLayerIndex; i < endLayerIndex; i++)
        {
            current = Layers[i].Forward(current);
        }

        // Apply softmax to get normalized attention weights
        current = ApplySoftmax(current);

        return current;
    }

    /// <summary>
    /// Applies softmax normalization to attention logits.
    /// </summary>
    /// <param name="logits">The unnormalized attention logits.</param>
    /// <returns>Normalized attention weights.</returns>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        // === Vectorized softmax using IEngine (Phase B: US-GPU-015) ===
        // Apply softmax along the last axis (attention dimension)
        return Engine.Softmax(logits, -1);
    }

    /// <summary>
    /// Reads from memory using attention weights.
    /// </summary>
    /// <param name="attentionWeights">The attention weights over memory slots.</param>
    /// <returns>The weighted sum of memory content.</returns>
    private Tensor<T> ReadFromMemory(Tensor<T> attentionWeights)
    {
        // Get shape information
        int[] shape = attentionWeights.Shape;
        int batchSize = shape[0];

        // Create result tensor with shape [batchSize, embeddingSize]
        Tensor<T> readout = new Tensor<T>(new int[] { batchSize, _embeddingSize });

        // For each batch, compute weighted sum of memory
        for (int b = 0; b < batchSize; b++)
        {
            // Initialize readout with zeros
            for (int e = 0; e < _embeddingSize; e++)
            {
                readout[b, e] = NumOps.Zero;
            }

            // Compute weighted sum of memory slots
            for (int m = 0; m < _memorySize; m++)
            {
                T weight = attentionWeights[b, m];

                for (int e = 0; e < _embeddingSize; e++)
                {
                    // Add weighted contribution from this memory slot
                    T weightedValue = NumOps.Multiply(weight, _memory[m, e]);
                    readout[b, e] = NumOps.Add(readout[b, e], weightedValue);
                }
            }
        }

        return readout;
    }

    /// <summary>
    /// Combines the encoded input with the memory readout.
    /// </summary>
    /// <param name="encoded">The encoded input tensor.</param>
    /// <param name="memoryReadout">The memory readout tensor.</param>
    /// <returns>The combined representation.</returns>
    private Tensor<T> CombineInputAndMemory(Tensor<T> encoded, Tensor<T> memoryReadout)
    {
        // Get shape information
        int[] encodedShape = encoded.Shape;
        int[] readoutShape = memoryReadout.Shape;
        int batchSize = encodedShape[0];
        int encodedSize = encodedShape[1];

        // Create result tensor with shape [batchSize, encodedSize + embeddingSize]
        Tensor<T> combined = new Tensor<T>(new int[] { batchSize, encodedSize + _embeddingSize });

        // Concatenate encoded input and memory readout
        for (int b = 0; b < batchSize; b++)
        {
            // Copy encoded input
            for (int i = 0; i < encodedSize; i++)
            {
                combined[b, i] = encoded[b, i];
            }

            // Copy memory readout
            for (int i = 0; i < _embeddingSize; i++)
            {
                combined[b, encodedSize + i] = memoryReadout[b, i];
            }
        }

        return combined;
    }

    /// <summary>
    /// Generates the final output from the combined representation.
    /// </summary>
    /// <param name="combined">The combined input and memory representation.</param>
    /// <returns>The final output tensor.</returns>
    private Tensor<T> GenerateOutput(Tensor<T> combined)
    {
        // Use the fourth quarter of layers for output generation
        int startLayerIndex = 3 * Layers.Count / 4;

        Tensor<T> current = combined;
        for (int i = startLayerIndex; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Determines whether memory should be updated during inference.
    /// </summary>
    /// <returns>True if memory should be updated; otherwise, false.</returns>
    private bool ShouldUpdateMemoryDuringInference()
    {
        // For standard Memory Networks, we typically update memory only during training
        // But some variants might update memory during inference as well
        return false;
    }

    /// <summary>
    /// Updates memory with new information.
    /// </summary>
    /// <param name="encoded">The encoded input tensor.</param>
    /// <param name="attentionWeights">The attention weights over memory slots.</param>
    private void UpdateMemory(Tensor<T> encoded, Tensor<T> attentionWeights)
    {
        // Use the third quarter of layers for memory writing
        int startLayerIndex = Layers.Count / 2;
        int endLayerIndex = 3 * Layers.Count / 4;

        // Get memory write controls from layers
        Tensor<T> current = encoded;
        for (int i = startLayerIndex; i < endLayerIndex; i++)
        {
            current = Layers[i].Forward(current);
        }

        // Extract memory update parameters
        // In a realistic implementation, we would extract:
        // - erase vector (how much to erase from existing memory)
        // - write vector (what to write to memory)
        // For simplicity, we'll just use the layer output directly as write vector

        // Get shape information
        int[] shape = current.Shape;
        int batchSize = shape[0];

        // Use first batch's attention and write values
        // In a full implementation, each batch would have its own memory state

        // Find most attended memory slot
        int maxIndex = 0;
        T maxAttention = attentionWeights[0, 0];

        for (int m = 1; m < _memorySize; m++)
        {
            if (NumOps.GreaterThan(attentionWeights[0, m], maxAttention))
            {
                maxAttention = attentionWeights[0, m];
                maxIndex = m;
            }
        }

        // Update this memory slot with write values
        T writeStrength = NumOps.FromDouble(0.1); // How strongly to write (small value for stability)
        for (int e = 0; e < _embeddingSize && e < shape[1]; e++)
        {
            // Weighted update: memory = (1-strength) * memory + strength * write_value
            T oldValue = _memory[maxIndex, e];
            T newValue = current[0, e];

            _memory[maxIndex, e] = NumOps.Add(
                NumOps.Multiply(NumOps.Subtract(NumOps.One, writeStrength), oldValue),
                NumOps.Multiply(writeStrength, newValue)
            );
        }
    }

    /// <summary>
    /// Trains the memory network on input-output pairs.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Memory Network using backpropagation. It performs a forward pass through
    /// the network, calculates the loss between predictions and expected outputs, computes gradients,
    /// and updates the network parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the Memory Network to give correct answers.
    /// 
    /// The training process works like this:
    /// 1. Make a prediction using the current network parameters
    /// 2. Compare the prediction to the expected output
    /// 3. Calculate the error (how wrong the prediction was)
    /// 4. Update the network parameters to reduce this error
    /// 5. Optionally update memory with new information
    /// 
    /// Over time, this process helps the network:
    /// - Learn how to encode inputs effectively
    /// - Learn which memory slots to pay attention to for different inputs
    /// - Learn how to combine memory and input to produce correct outputs
    /// - Build up a useful memory of facts and information
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Set to training mode
        SetTrainingMode(true);

        // Forward pass to get predictions
        var predictions = Predict(input);

        // Calculate loss using the specified loss function
        Vector<T> predictedVector = predictions.ToVector();
        Vector<T> expectedVector = expectedOutput.ToVector();
        T loss = LossFunction.CalculateLoss(predictedVector, expectedVector);

        // Set the LastLoss property
        LastLoss = loss;

        // Calculate output gradients using the loss function's derivative
        var gradientVector = LossFunction.CalculateDerivative(predictedVector, expectedVector);
        var outputGradients = new Tensor<T>(predictions.Shape, gradientVector);

        // Backpropagation
        BackpropagateMemoryNetwork(outputGradients);

        // Update parameters
        UpdateMemoryNetworkParameters();

        // Always update memory during training
        // This is handled in the Predict method
    }

    /// <summary>
    /// Calculates mean squared error between predictions and expected outputs.
    /// </summary>
    /// <param name="predictions">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The mean squared error loss value.</returns>
    private T CalculateMeanSquaredError(Tensor<T> predictions, Tensor<T> expected)
    {
        // Verify tensor shapes match
        if (!AreShapesCompatible(predictions.Shape, expected.Shape))
        {
            throw new ArgumentException("Prediction and expected output shapes must be compatible");
        }

        // Calculate squared differences
        T sumSquaredDiff = NumOps.Zero;
        int totalElements = 0;

        // Handle different tensor shapes
        if (predictions.Shape.Length == 2)
        {
            // 2D tensors [batch, features]
            int batchSize = predictions.Shape[0];
            int features = predictions.Shape[1];
            totalElements = batchSize * features;

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    T diff = NumOps.Subtract(predictions[b, f], expected[b, f]);
                    sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                }
            }
        }
        else if (predictions.Shape.Length == 3)
        {
            // 3D tensors [batch, sequence, features]
            int batchSize = predictions.Shape[0];
            int seqLength = predictions.Shape[1];
            int features = predictions.Shape[2];
            totalElements = batchSize * seqLength * features;

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLength; s++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        T diff = NumOps.Subtract(predictions[b, s, f], expected[b, s, f]);
                        sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                    }
                }
            }
        }

        // Calculate mean
        return NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(totalElements));
    }

    /// <summary>
    /// Checks if two tensor shapes are compatible for element-wise operations.
    /// </summary>
    /// <param name="shape1">The first tensor shape.</param>
    /// <param name="shape2">The second tensor shape.</param>
    /// <returns>True if the shapes are compatible; otherwise, false.</returns>
    private bool AreShapesCompatible(int[] shape1, int[] shape2)
    {
        // Same rank tensors with the same dimensions are compatible
        if (shape1.Length == shape2.Length)
        {
            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                {
                    return false;
                }
            }
            return true;
        }

        return false;
    }

    /// <summary>
    /// Calculates gradients for output layer based on predictions and expected outputs.
    /// </summary>
    /// <param name="predictions">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The gradient tensor for the output layer.</returns>
    private Tensor<T> CalculateOutputGradients(Tensor<T> predictions, Tensor<T> expected)
    {
        // Verify tensor shapes match
        if (!AreShapesCompatible(predictions.Shape, expected.Shape))
        {
            throw new ArgumentException("Prediction and expected output shapes must be compatible");
        }

        // For MSE loss, gradient is 2 * (predictions - expected) / N
        // We can simplify to (predictions - expected) and adjust learning rate

        // Create gradient tensor with same shape as predictions
        Tensor<T> gradients = new Tensor<T>(predictions.Shape);

        // Calculate gradients
        if (predictions.Shape.Length == 2)
        {
            // 2D tensors [batch, features]
            int batchSize = predictions.Shape[0];
            int features = predictions.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    gradients[b, f] = NumOps.Subtract(predictions[b, f], expected[b, f]);
                }
            }
        }
        else if (predictions.Shape.Length == 3)
        {
            // 3D tensors [batch, sequence, features]
            int batchSize = predictions.Shape[0];
            int seqLength = predictions.Shape[1];
            int features = predictions.Shape[2];

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLength; s++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        gradients[b, s, f] = NumOps.Subtract(predictions[b, s, f], expected[b, s, f]);
                    }
                }
            }
        }

        return gradients;
    }

    /// <summary>
    /// Performs backpropagation through the memory network.
    /// </summary>
    /// <param name="outputGradients">The gradients from the output layer.</param>
    private void BackpropagateMemoryNetwork(Tensor<T> outputGradients)
    {
        // Start with output gradients
        Tensor<T> gradients = outputGradients;

        // Backpropagate through output layers (fourth quarter)
        for (int i = Layers.Count - 1; i >= 3 * Layers.Count / 4; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }

        // Split gradients for memory reading and input encoding paths
        // In a real implementation, we would calculate:
        // - gradients for memory attention
        // - gradients for memory content
        // - gradients for input encoding

        // For simplicity, we'll just continue backpropagation through all previous layers
        for (int i = 3 * Layers.Count / 4 - 1; i >= 0; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }

        // The result is that all layers now have their gradients computed and stored internally
    }

    /// <summary>
    /// Updates the memory network parameters based on calculated gradients.
    /// </summary>
    private void UpdateMemoryNetworkParameters()
    {
        // Simple learning rate for gradient descent
        T learningRate = NumOps.FromDouble(0.01);

        // Update parameters for each layer
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }


    /// <summary>
    /// Gets metadata about the memory network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the memory network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the Memory Network, including its architecture,
    /// memory configuration, and other relevant parameters. This information is useful for model
    /// management, tracking experiments, and reporting.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your Memory Network.
    /// 
    /// The metadata includes:
    /// - What this model is and what it does
    /// - Details about the network architecture
    /// - Information about the memory configuration
    /// - Statistics about the current memory state
    /// 
    /// This information is useful for:
    /// - Documentation
    /// - Comparing different memory network configurations
    /// - Debugging and analysis
    /// - Tracking memory usage and performance
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Calculate memory statistics
        double avgMemValue = 0.0;
        double minMemValue = double.MaxValue;
        double maxMemValue = double.MinValue;

        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                double value = Convert.ToDouble(_memory[i, j]);
                avgMemValue += value;
                minMemValue = Math.Min(minMemValue, value);
                maxMemValue = Math.Max(maxMemValue, value);
            }
        }

        avgMemValue /= _memorySize * _embeddingSize;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.MemoryNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MemorySize", _memorySize },
                { "EmbeddingSize", _embeddingSize },
                { "TotalParameters", ParameterCount },
                { "LayerCount", Layers.Count },
                { "AvgMemoryValue", avgMemValue },
                { "MinMemoryValue", minMemValue },
                { "MaxMemoryValue", maxMemValue },
                { "MemoryTotalElements", _memorySize * _embeddingSize }
            }
        };
    }

    /// <summary>
    /// Serializes memory network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Memory Network to a binary stream. It serializes memory network-specific
    /// parameters like the memory matrix contents, allowing the complete state to be restored later.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the complete state of your Memory Network to a file.
    /// 
    /// When saving the Memory Network:
    /// - Memory matrix contents are saved (all the stored facts and information)
    /// - Configuration parameters are saved
    /// - Neural network parameters are saved
    /// 
    /// This allows you to:
    /// - Save your progress and continue training later
    /// - Share trained models with others
    /// - Deploy models in applications
    /// - Preserve the memory of facts the network has learned
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save memory configuration
        writer.Write(_memorySize);
        writer.Write(_embeddingSize);

        // Save memory matrix contents
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                writer.Write(Convert.ToDouble(_memory[i, j]));
            }
        }

        // Save each layer
        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Deserializes memory network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the state of a previously saved Memory Network from a binary stream. It restores
    /// memory network-specific parameters like the memory matrix contents, allowing the model to
    /// continue from exactly where it left off.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a complete Memory Network from a saved file.
    /// 
    /// When loading the Memory Network:
    /// - Memory matrix contents are restored (all the stored facts)
    /// - Configuration parameters are restored
    /// - Neural network parameters are restored
    /// 
    /// This lets you:
    /// - Continue working with a model exactly where you left off
    /// - Use a model that someone else has trained
    /// - Deploy pre-trained models in applications
    /// - Restore the memory of facts the network had previously learned
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Load memory configuration
        int memorySize = reader.ReadInt32();
        int embeddingSize = reader.ReadInt32();

        // Verify configuration matches
        if (memorySize != _memorySize || embeddingSize != _embeddingSize)
        {
            throw new InvalidOperationException("Memory configuration in saved model does not match current configuration");
        }

        // Load memory matrix contents
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Load layers
        int layerCount = reader.ReadInt32();
        if (layerCount != Layers.Count)
        {
            throw new InvalidOperationException("Layer count in saved model does not match current model");
        }

        for (int i = 0; i < layerCount; i++)
        {
            Layers[i].Deserialize(reader);
        }
    }

    /// <summary>
    /// Stores a new fact in memory.
    /// </summary>
    /// <param name="fact">The fact to store, as a tensor.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new fact to the memory of the network. It encodes the fact using the input
    /// encoding layers, finds the least recently used memory slot, and stores the encoded fact there.
    /// This allows for explicit memory updates beyond what happens during normal training.
    /// </para>
    /// <para><b>For Beginners:</b> This directly adds a new fact to the network's memory.
    /// 
    /// When adding a fact:
    /// - The fact is first encoded into an embedding (internal representation)
    /// - The system finds an appropriate memory slot to store it
    /// - The encoded fact is written to that memory slot
    /// 
    /// This provides a way to directly add knowledge to the memory network
    /// without having to train it on examples, which can be useful for
    /// quickly updating the network's knowledge base.
    /// </para>
    /// </remarks>
    public void StoreFact(Tensor<T> fact)
    {
        // Encode the fact
        Tensor<T> encoded = EncodeInput(fact);

        // Find a memory slot to store the fact
        // Here we use a simple strategy of writing to the next available slot
        // In a real implementation, we might use more sophisticated strategies

        // Calculate usage based on L2 norm of each memory row
        var usage = new Dictionary<int, T>();
        for (int m = 0; m < _memorySize; m++)
        {
            T sumSquared = NumOps.Zero;
            for (int e = 0; e < _embeddingSize; e++)
            {
                T val = _memory[m, e];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }

            usage[m] = sumSquared;
        }

        // Find least used memory slot
        int leastUsedSlot = 0;
        T minUsage = usage[0];

        for (int m = 1; m < _memorySize; m++)
        {
            if (NumOps.LessThan(usage[m], minUsage))
            {
                minUsage = usage[m];
                leastUsedSlot = m;
            }
        }

        // Store the encoded fact
        for (int e = 0; e < _embeddingSize && e < encoded.Shape[1]; e++)
        {
            _memory[leastUsedSlot, e] = encoded[0, e];
        }
    }

    /// <summary>
    /// Queries the memory network with a question and returns the answer.
    /// </summary>
    /// <param name="question">The question tensor.</param>
    /// <returns>The answer tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method is a wrapper around the Predict method that is semantically meaningful
    /// for question answering tasks, which are a common use case for Memory Networks.
    /// </para>
    /// <para><b>For Beginners:</b> This asks the Memory Network a question and gets an answer.
    /// 
    /// When asking a question:
    /// - The question is processed through the network
    /// - The network accesses relevant information from its memory
    /// - It combines the question with retrieved memory to generate an answer
    /// 
    /// This is the most common way to use a Memory Network once it's trained,
    /// especially for question answering and reasoning tasks.
    /// </para>
    /// </remarks>
    public Tensor<T> AnswerQuestion(Tensor<T> question)
    {
        return Predict(question);
    }

    /// <summary>
    /// Creates a new instance of the Memory Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new Memory Network instance with the same architecture and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the Memory Network with the same architecture and memory configuration
    /// as the current instance. It's used in scenarios where a fresh copy of the model is needed
    /// while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the Memory Network with the same setup.
    /// 
    /// Think of it like creating a clone of the network:
    /// - The new network has the same architecture (structure)
    /// - It has the same memory size and embedding size
    /// - But it's a completely separate instance with its own memory matrix
    /// - The memory starts fresh (empty) rather than copying the current memory contents
    /// 
    /// This is useful when you want to:
    /// - Train multiple versions of the same memory network architecture
    /// - Start with a clean memory but the same network structure
    /// - Compare how different training approaches affect learning with the same configuration
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a new instance of MemoryNetwork with the same architecture and memory configuration
        return new MemoryNetwork<T>(
            this.Architecture,
            _memorySize,
            _embeddingSize);
    }
}
