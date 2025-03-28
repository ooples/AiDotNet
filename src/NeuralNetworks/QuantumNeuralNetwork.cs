namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Quantum Neural Network, which combines quantum computing principles with neural network architecture.
/// </summary>
/// <remarks>
/// <para>
/// A Quantum Neural Network (QNN) is a neural network architecture that leverages quantum computing principles
/// to potentially solve certain problems more efficiently than classical neural networks. It uses quantum bits (qubits)
/// instead of classical bits, allowing it to process information in ways not possible with traditional neural networks.
/// </para>
/// <para><b>For Beginners:</b> A Quantum Neural Network combines ideas from quantum computing with neural networks.
/// 
/// Think of it like upgrading from a regular calculator to a special calculator with new abilities:
/// - Regular neural networks use normal bits (0 or 1)
/// - Quantum neural networks use quantum bits or "qubits" that can be 0, 1, or both at the same time
/// - This "both at the same time" property (called superposition) gives quantum networks special abilities
/// - These networks might solve certain problems much faster than regular neural networks
/// 
/// For example, a quantum neural network might find patterns in complex data or optimize solutions
/// in ways that would be extremely difficult for traditional neural networks.
/// 
/// While the math behind quantum computing is complex, you can think of a quantum neural network
/// as having the potential to explore many possible solutions simultaneously rather than one at a time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class QuantumNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the number of qubits used in the quantum neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of qubits determines the dimensionality of the quantum state space that the network
    /// can operate in. Each additional qubit doubles the size of this state space, allowing the network
    /// to represent more complex quantum states but also increasing computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many quantum bits (qubits) the network uses.
    /// 
    /// The number of qubits is important because:
    /// - Each qubit exponentially increases the computing power
    /// - With n qubits, the network can represent 2^n states simultaneously
    /// - More qubits allow the network to solve more complex problems
    /// - But more qubits also make the network harder to simulate on classical computers
    /// 
    /// For example, with just 10 qubits, a quantum neural network can represent 1,024 states at once.
    /// With 20 qubits, it can represent over 1 million states simultaneously!
    /// </para>
    /// </remarks>
    private int _numQubits;

    /// <summary>
    /// Initializes a new instance of the <see cref="QuantumNeuralNetwork{T}"/> class with the specified architecture and number of qubits.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the QNN.</param>
    /// <param name="numQubits">The number of qubits to use in the quantum neural network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Quantum Neural Network with the specified architecture and number of qubits.
    /// It initializes the network layers based on the architecture, or creates default quantum network layers if
    /// no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Quantum Neural Network with its basic components.
    /// 
    /// When creating a new QNN:
    /// - architecture: Defines the overall structure of the neural network
    /// - numQubits: Sets how many quantum bits the network will use
    /// 
    /// The constructor prepares the network by either:
    /// - Using the specific layers provided in the architecture, or
    /// - Creating default layers designed for quantum processing if none are specified
    /// 
    /// This is like setting up a specialized calculator before you start using it for calculations.
    /// </para>
    /// </remarks>
    public QuantumNeuralNetwork(NeuralNetworkArchitecture<T> architecture, int numQubits) : base(architecture)
    {
        _numQubits = numQubits;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Quantum Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for quantum
    /// processing is created based on the number of qubits specified during initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for quantum neural networks are created automatically
    /// - The system checks that any custom layers will work properly with quantum computations
    /// 
    /// Layers are like the different processing stages in the neural network.
    /// For a quantum neural network, these layers are designed to work with quantum principles,
    /// allowing the network to take advantage of quantum effects like superposition and entanglement.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultQuantumNetworkLayers(Architecture, _numQubits));
        }
    }

    /// <summary>
    /// Processes the input through the quantum neural network to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through the quantum neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Quantum Neural Network. It processes the input
    /// through each layer of the network in sequence, transforming it according to quantum-inspired
    /// operations defined in each layer. The final transformed vector represents the network's prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the quantum neural network processes information and makes predictions.
    /// 
    /// During the prediction process:
    /// - The input data enters the network
    /// - The data flows through each layer in sequence
    /// - Each layer performs its quantum-inspired operations on the data
    /// - The final output represents the network's prediction or answer
    /// 
    /// Unlike a regular neural network, the layers in a quantum neural network may perform operations
    /// that simulate quantum behaviors, potentially allowing the network to find patterns or solutions
    /// that would be difficult to discover with classical methods.
    /// 
    /// This is conceptually similar to how a regular neural network makes predictions,
    /// but the internal processing takes advantage of quantum principles.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }
        return current;
    }

    /// <summary>
    /// Updates the parameters of the quantum neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the quantum neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the quantum neural network makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - It takes a vector of parameter updates and distributes them to the correct layers
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For a quantum neural network, these parameters might control operations like quantum rotations,
    /// entanglement settings, or other quantum-inspired transformations.
    /// 
    /// This process allows the quantum neural network to improve its performance over time
    /// by adjusting how it processes information.
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
    /// Saves the state of the Quantum Neural Network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer serialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the Quantum Neural Network, including all layers and the
    /// number of qubits. It writes the number of layers, the number of qubits, and the type and state of each layer
    /// to the provided binary writer.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the quantum neural network to a file.
    /// 
    /// When serializing:
    /// - The number of qubits is saved
    /// - All the network's layers are saved (their types and internal values)
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
        writer.Write(_numQubits);
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
    }

    /// <summary>
    /// Loads the state of the Quantum Neural Network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the Quantum Neural Network from a binary reader. It reads
    /// the number of layers, the number of qubits, recreates each layer based on its type, and deserializes
    /// the layer state.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved quantum neural network state from a file.
    /// 
    /// When deserializing:
    /// - The number of layers and qubits are read from the file
    /// - Each layer is recreated and its state is restored
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
        _numQubits = reader.ReadInt32();
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
    }
}