namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Neural Turing Machine, which is a neural network architecture that combines a neural network with external memory.
/// </summary>
/// <remarks>
/// <para>
/// A Neural Turing Machine (NTM) extends traditional neural networks by adding an external memory component that
/// the network can read from and write to. This allows the network to store and retrieve information over long
/// sequences, making it particularly effective for tasks requiring complex memory operations.
/// </para>
/// <para><b>For Beginners:</b> A Neural Turing Machine is like a neural network with a "notebook" that it can write to and read from.
/// 
/// Think of it like a student solving a math problem:
/// - The student (neural network) can process information directly
/// - But for complex problems, the student needs to write down intermediate steps in a notebook (external memory)
/// - The student can later refer back to these notes when needed
/// 
/// This memory capability helps the network:
/// - Remember information over long periods
/// - Store and retrieve specific pieces of data
/// - Learn more complex patterns that require step-by-step reasoning
/// 
/// For example, a standard neural network might struggle to add two long numbers, but an NTM can learn to write down 
/// partial results and carry digits, similar to how humans solve addition problems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NeuralTuringMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the size of the external memory matrix (number of memory locations).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory size determines the number of separate memory locations available to the neural network.
    /// A larger memory size allows the network to store more distinct pieces of information but increases
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the number of pages in the network's notebook.
    /// 
    /// More memory locations (larger MemorySize) means:
    /// - The network can store more separate pieces of information
    /// - It can keep track of more things at once
    /// - It might perform better on complex tasks that require remembering many details
    /// 
    /// However, a larger memory also requires more computing power to process.
    /// </para>
    /// </remarks>
    private int _memorySize;

    /// <summary>
    /// Gets or sets the size of each memory vector (the amount of information stored at each memory location).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory vector size determines how much information can be stored at each memory location.
    /// Larger vector sizes allow more detailed information to be stored at each location but increase
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like how much information you can write on each page of the notebook.
    /// 
    /// A larger vector size means:
    /// - Each memory location can store more detailed information
    /// - The network can capture more complex patterns at each location
    /// - It might provide better performance for tasks requiring nuanced memory
    /// 
    /// Think of it as the difference between taking brief notes versus detailed notes on each page.
    /// </para>
    /// </remarks>
    private int _memoryVectorSize;

    /// <summary>
    /// Gets or sets the size of the controller network that manages memory operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The controller size determines the complexity of the neural network that decides how to interact with
    /// the external memory. A larger controller can implement more sophisticated memory access strategies
    /// but requires more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the brain power of the system that decides when and how to use the notebook.
    /// 
    /// A larger controller size means:
    /// - The network can make more sophisticated decisions about using memory
    /// - It can develop more complex strategies for storing and retrieving information
    /// - It might learn more effectively on difficult tasks
    /// 
    /// Think of it as having a smarter student who knows better strategies for taking and using notes.
    /// </para>
    /// </remarks>
    private int _controllerSize;

    /// <summary>
    /// Gets or sets the external memory matrix used by the Neural Turing Machine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory matrix stores all the information that the Neural Turing Machine can access. It is organized
    /// as a matrix where each row represents a memory location and each column represents a dimension of the
    /// stored information vector.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual notebook where information is stored.
    /// 
    /// The memory matrix:
    /// - Has _memorySize rows (like pages in a notebook)
    /// - Has _memoryVectorSize columns (like the amount of information on each page)
    /// - Can be read from and written to by the neural network
    /// - Persists information across processing steps, allowing the network to "remember"
    /// 
    /// When the network processes information, it can store results here and retrieve them later,
    /// which is what gives the Neural Turing Machine its powerful memory capabilities.
    /// </para>
    /// </remarks>
    private Matrix<T> _memory;

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralTuringMachine{T}"/> class with the specified architecture and memory parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the NTM.</param>
    /// <param name="memorySize">The number of memory locations (rows in the memory matrix).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (columns in the memory matrix).</param>
    /// <param name="controllerSize">The size of the controller network that manages memory operations.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Neural Turing Machine with the specified architecture and memory parameters.
    /// It initializes the memory matrix with small random values to facilitate learning during training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Neural Turing Machine with its basic components.
    /// 
    /// When creating a new NTM:
    /// - architecture: Defines the overall structure of the neural network
    /// - memorySize: Sets how many separate memory locations are available
    /// - memoryVectorSize: Sets how much information can be stored at each location
    /// - controllerSize: Sets how complex the control system is
    /// 
    /// The constructor also initializes the memory with small random values as a starting point,
    /// similar to how you might prepare a notebook with light markings before actually using it.
    /// </para>
    /// </remarks>
    public NeuralTuringMachine(NeuralNetworkArchitecture<T> architecture, int memorySize, int memoryVectorSize, int controllerSize) 
        : base(architecture)
    {
        _memorySize = memorySize;
        _memoryVectorSize = memoryVectorSize;
        _controllerSize = controllerSize;
        _memory = new Matrix<T>(_memorySize, _memoryVectorSize);

        InitializeMemory();
    }

    /// <summary>
    /// Initializes the memory matrix with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method populates the memory matrix with small random values between 0 and 0.1. Using small
    /// initial values helps with the training process by providing a stable starting point that allows
    /// gradual learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the notebook with light random markings.
    /// 
    /// When initializing memory:
    /// - Each cell in the memory matrix gets a small random value
    /// - These values are between 0 and 0.1 (very small)
    /// - Starting with small random values helps the network learn more effectively
    /// - It's like having faint pencil marks that can be easily modified during learning
    /// 
    /// Without this initialization, the network might have trouble starting the learning process.
    /// </para>
    /// </remarks>
    private void InitializeMemory()
    {
        // Initialize memory with small random values
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryVectorSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1);
            }
        }
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Neural Turing Machine. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration is created based on
    /// the memory parameters. The method also validates that custom layers are compatible with NTM requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for an NTM are created automatically
    /// - The system checks that any custom layers will work properly with the NTM
    /// 
    /// Layers are like the different processing stages in the neural network.
    /// Each layer performs a specific operation on the data as it flows through the network.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNTMLayers(Architecture, _memorySize, _memoryVectorSize, _controllerSize));
        }
    }

    /// <summary>
    /// Processes the input through the neural network and memory operations to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through the network and memory operations.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Neural Turing Machine. It processes the input through
    /// the neural network layers, interacting with the memory as specified by memory read and write layers.
    /// The final output combines the controller output with the memory read output to produce the prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the NTM processes information and makes predictions.
    /// 
    /// During the prediction process:
    /// - The input data flows through the network layers one by one
    /// - Special memory read layers can look up information from the memory
    /// - Special memory write layers can save information to the memory
    /// - The final output combines what the network processed directly with what it read from memory
    /// 
    /// This is similar to a student solving a problem by both thinking directly and referring to their notes.
    /// The ability to read from and write to memory allows the network to handle more complex tasks
    /// than a standard neural network could.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        var memoryReadOutput = new Vector<T>(_memoryVectorSize);

        foreach (var layer in Layers)
        {
            if (layer is MemoryReadLayer<T> memoryReadLayer)
            {
                // Convert Matrix to Tensor before passing to Forward method
                memoryReadOutput = memoryReadLayer.Forward(
                    Tensor<T>.FromVector(current), 
                    Tensor<T>.FromMatrix(_memory)).ToVector();
            }
            else if (layer is MemoryWriteLayer<T> memoryWriteLayer)
            {
                // Convert Matrix to Tensor before passing to Forward method
                // and convert the result back to Matrix
                _memory = memoryWriteLayer.Forward(
                    Tensor<T>.FromVector(current), 
                    Tensor<T>.FromMatrix(_memory)).ToMatrix();
            }
            else
            {
                current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            }
        }

        // Concatenate controller output with memory read output
        current = Vector<T>.Concatenate(current, memoryReadOutput);

        // Process through the final layers
        for (int i = Layers.Count - 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Updates the parameters of the neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the neural network makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - It takes a vector of parameter updates and distributes them to the correct layers
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// Think of it like updating the decision-making rules for each part of the network
    /// based on what was learned during training.
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
    /// Saves the state of the Neural Turing Machine to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer serialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the Neural Turing Machine, including all layers and the
    /// memory matrix. It writes the number of layers, the type and state of each layer, and the contents of
    /// the memory matrix to the provided binary writer.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the NTM to a file.
    /// 
    /// When serializing:
    /// - All the network's layers are saved (their types and internal values)
    /// - The entire memory matrix is saved (what the network has "written down")
    /// - The saved file can later be used to restore the exact same network state
    /// 
    /// This is useful for:
    /// - Saving a trained model to use later
    /// - Sharing a model with others
    /// - Creating backups during long training processes
    /// 
    /// Think of it like taking a complete snapshot of the network that can be restored later.
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
            for (int j = 0; j < _memoryVectorSize; j++)
            {
                // Handle potential null values in Memory
                T value = _memory[i, j];
                string valueString = value?.ToString() ?? string.Empty;
                writer.Write(valueString);
            }
        }
    }

    /// <summary>
    /// Loads the state of the Neural Turing Machine from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the Neural Turing Machine from a binary reader. It reads
    /// the number of layers, recreates each layer based on its type, deserializes the layer state, and
    /// finally reconstructs the memory matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved NTM state from a file.
    /// 
    /// When deserializing:
    /// - The number and types of layers are read from the file
    /// - Each layer is recreated and its state is restored
    /// - The memory matrix is reconstructed with all its saved values
    /// 
    /// This allows you to:
    /// - Load a previously trained model
    /// - Continue using or training a model from where you left off
    /// - Use models created by others
    /// 
    /// Think of it like restoring a complete snapshot of the network that was saved earlier.
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
            for (int j = 0; j < _memoryVectorSize; j++)
            {
                _memory[i, j] = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
            }
        }
    }
}