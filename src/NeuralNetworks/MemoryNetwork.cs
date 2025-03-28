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
    public MemoryNetwork(NeuralNetworkArchitecture<T> architecture, int memorySize, int embeddingSize) : base(architecture)
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
    /// Performs a forward pass through the network to generate a prediction from an input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input vector through all layers of the Memory Network. It handles special
    /// memory-specific layers differently from standard layers. For memory read layers, it provides access
    /// to the current memory matrix. For memory write layers, it updates the memory matrix based on the layer's
    /// output. For standard layers, it processes the input normally. This allows the network to read from and
    /// write to its persistent memory during operation.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your input through the network, using the memory when needed.
    /// 
    /// When you pass input to the network:
    /// 
    /// 1. The input goes through each layer in sequence
    /// 
    /// 2. When it reaches a Memory Read Layer:
    ///    - The layer looks at the input and determines what information to retrieve from memory
    ///    - It's like looking up relevant facts in a notebook based on a question
    ///    - The retrieved information becomes part of the processing flow
    /// 
    /// 3. When it reaches a Memory Write Layer:
    ///    - The layer decides what new information to store in memory
    ///    - It's like writing new notes in the notebook
    ///    - The memory gets updated for future reference
    /// 
    /// 4. Regular layers process the information normally
    /// 
    /// This combination of reading from and writing to memory, along with standard neural network
    /// processing, allows the network to reason with facts and maintain information across multiple
    /// inputs or questions.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            if (layer is MemoryReadLayer<T> memoryReadLayer)
            {
                // Convert Matrix<T> to Tensor<T> before passing to Forward
                Tensor<T> memoryTensor = Tensor<T>.FromMatrix(_memory);
                current = memoryReadLayer.Forward(Tensor<T>.FromVector(current), memoryTensor).ToVector();
            }
            else if (layer is MemoryWriteLayer<T> memoryWriteLayer)
            {
                // Convert Matrix<T> to Tensor<T> before passing to Forward
                Tensor<T> memoryTensor = Tensor<T>.FromMatrix(_memory);
                Tensor<T> updatedMemoryTensor = memoryWriteLayer.Forward(Tensor<T>.FromVector(current), memoryTensor);
            
                // Convert the result back to Matrix<T>
                _memory = updatedMemoryTensor.ToMatrix();
                // The output of MemoryWriteLayer is typically not used in the forward pass
            }
            else
            {
                current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            }
        }

        return current;
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
    /// Serializes the Memory Network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized network to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or when a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the Memory Network's structure, parameters, and memory content to a binary format
    /// that can be stored and later loaded. It first serializes the network layers and then serializes the
    /// entire memory matrix, preserving both the network's learned weights and its accumulated knowledge.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your Memory Network to a file, including both its structure and memory contents.
    /// 
    /// When you save a Memory Network, two important things are preserved:
    /// 
    /// 1. The network structure and parameters:
    ///    - How many layers there are and what type each layer is
    ///    - All the learned values that determine how the network processes information
    ///    - This is similar to saving the "brain" of the system
    /// 
    /// 2. The memory matrix contents:
    ///    - All the information stored in the network's memory
    ///    - All the "notes" in the network's notebook
    ///    - This is what makes Memory Networks special - they can retain accumulated knowledge
    /// 
    /// This comprehensive saving ensures that when you reload the network later,
    /// it will have both the same processing capabilities and the same stored knowledge.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }

        // Serialize memory
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                // Fix: Handle potential null values by using the null-coalescing operator
                string valueStr = _memory[i, j]?.ToString() ?? string.Empty;
                writer.Write(valueStr);
            }
        }
    }

    /// <summary>
    /// Deserializes the Memory Network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized network from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when an empty layer type name is encountered, when a layer type cannot be found, when a type does not implement the required interface, or when a layer instance cannot be created.</exception>
    /// <remarks>
    /// <para>
    /// This method loads a previously serialized Memory Network from a binary format. It first reads and
    /// recreates the network layers, then loads the memory matrix contents. This restores both the network's
    /// structure and its accumulated knowledge, allowing it to continue operation exactly where it left off.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved Memory Network from a file, restoring both its structure and memory contents.
    /// 
    /// When you load a Memory Network, the process rebuilds two key components:
    /// 
    /// 1. The network structure and parameters:
    ///    - Recreates all the layers with their correct types
    ///    - Restores all the learned values that determine how the network processes information
    ///    - This is like restoring the "brain" of the system
    /// 
    /// 2. The memory matrix contents:
    ///    - Restores all the information that was stored in the network's memory
    ///    - Repopulates all the "notes" in the network's notebook
    ///    - This allows the network to remember everything it knew before being saved
    /// 
    /// After loading, the network can continue operation exactly as if it had never been saved,
    /// with all its processing capabilities and accumulated knowledge intact.
    /// 
    /// For example, if you trained a Memory Network to answer questions about a specific topic
    /// and saved it, when you load it later it will still remember all the facts it learned
    /// about that topic.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }

        // Deserialize memory
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                _memory[i, j] = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
            }
        }
    }
}