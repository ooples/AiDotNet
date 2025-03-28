namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Liquid State Machine (LSM), a type of reservoir computing neural network.
/// </summary>
/// <remarks>
/// <para>
/// A Liquid State Machine is a form of reservoir computing that uses a recurrent neural network with 
/// randomly connected neurons (the "reservoir") to process temporal information. The reservoir transforms 
/// input signals into higher-dimensional representations, which are then processed by trained readout 
/// functions. LSMs are particularly effective for processing time-varying inputs and are inspired by 
/// the dynamics of biological neural networks.
/// </para>
/// <para><b>For Beginners:</b> A Liquid State Machine is a neural network inspired by how real brains process information over time.
/// 
/// Think of it like dropping different objects into a pool of water:
/// - Each object creates unique ripple patterns when it hits the water
/// - The ripples interact with each other in complex ways
/// - By observing these ripple patterns, you can determine what objects were dropped in
/// 
/// In a Liquid State Machine:
/// - The "reservoir" is like the pool of water with randomly connected neurons
/// - Input signals create "ripples" of activity through the connected neurons
/// - The network learns to recognize patterns in how these ripples develop over time
/// 
/// LSMs are particularly good at:
/// - Processing sequential data (like speech or sensor readings)
/// - Handling inputs that change over time
/// - Working with noisy or incomplete information
/// - Learning temporal patterns without needing extensive training
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LiquidStateMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the reservoir (number of neurons).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The reservoir size determines the number of neurons in the reservoir component of the Liquid State Machine.
    /// A larger reservoir size increases the computational capacity of the network, allowing it to represent more
    /// complex temporal patterns, but also increases memory requirements and computational load.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many neurons are in the main processing part of the network.
    /// 
    /// Think of it like the size of the pool in our water analogy:
    /// - A larger reservoir has more neurons to create complex patterns
    /// - More neurons can represent more complex information
    /// - But a larger reservoir also requires more computing power
    /// 
    /// Choosing the right size depends on your task:
    /// - Simple tasks might need only a few hundred neurons
    /// - Complex tasks might need thousands of neurons
    /// 
    /// This is one of the most important parameters to adjust when setting up a Liquid State Machine.
    /// </para>
    /// </remarks>
    private readonly int _reservoirSize;

    /// <summary>
    /// Gets the probability of connection between neurons in the reservoir.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter determines how densely connected the neurons in the reservoir are. A value of 0 means
    /// no connections, while a value of 1 means fully connected. Typical values range from 0.1 to 0.3.
    /// The connectivity affects the dynamics of the reservoir and its ability to process temporal information.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how densely connected the neurons are within the reservoir.
    /// 
    /// In our pool of water analogy, this is like controlling how easily ripples travel through the water:
    /// - A low value (like 0.1) means each neuron connects to only about 10% of other neurons
    /// - A high value means more connections between neurons
    /// 
    /// The default value is 0.1 (10% connectivity), which typically works well because:
    /// - Too few connections (low value) means information doesn't flow well through the network
    /// - Too many connections (high value) can cause the network to become chaotic
    /// - Sparse connectivity (like 10%) is actually similar to biological brains
    /// 
    /// Finding the right balance helps the network process complex patterns efficiently.
    /// </para>
    /// </remarks>
    private readonly double _connectionProbability;

    /// <summary>
    /// Gets the spectral radius of the reservoir weight matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The spectral radius controls the stability of the reservoir dynamics. It is the largest absolute eigenvalue
    /// of the reservoir weight matrix. Values less than 1.0 generally lead to stable dynamics, while values
    /// greater than 1.0 can lead to chaotic behavior. The default value of 0.9 provides a good balance between
    /// stability and computational capacity.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how long information echoes through the reservoir.
    /// 
    /// In our water analogy, this is like controlling how long ripples last before fading away:
    /// - A value close to 0 means ripples die out quickly
    /// - A value close to 1 means ripples persist longer
    /// - A value greater than 1 can make the system chaotic (like a continuous splash)
    /// 
    /// The default value is 0.9, which means:
    /// - Information persists long enough to be useful
    /// - But not so long that older information interferes too much with new information
    /// - The network remains stable rather than becoming chaotic
    /// 
    /// This parameter helps balance the network's memory capacity with its stability.
    /// </para>
    /// </remarks>
    private readonly double _spectralRadius;

    /// <summary>
    /// Gets the scaling factor applied to input signals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input scaling factor determines how strongly input signals influence the reservoir dynamics.
    /// Higher values mean stronger influence, potentially causing more dramatic changes in the reservoir state.
    /// Lower values result in more subtle influences, which can be helpful for processing subtle patterns
    /// in the input data.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly input signals affect the reservoir.
    /// 
    /// In our water analogy, this is like controlling how hard objects hit the water surface:
    /// - A high value means inputs create big, dramatic ripples
    /// - A low value means inputs create small, subtle ripples
    /// 
    /// The default value is 1.0, which provides a balanced impact, but you might adjust it:
    /// - Increase it if inputs are very subtle and need amplification
    /// - Decrease it if inputs are already strong and would overwhelm the system
    /// 
    /// Finding the right input scaling helps the network properly respond to the strength of your signals.
    /// </para>
    /// </remarks>
    private readonly double _inputScaling;

    /// <summary>
    /// Gets the leaking rate of the reservoir neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The leaking rate controls how quickly neuron activations decay over time. A value of 1.0 means no decay
    /// (neurons maintain their activation), while lower values cause faster decay. This parameter affects
    /// the memory capacity of the reservoir and its sensitivity to the timing of input signals.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how quickly neuron activations fade over time.
    /// 
    /// In our water analogy, this is like controlling how quickly the water returns to stillness:
    /// - A value of 1.0 means ripples maintain their strength (no fading)
    /// - A lower value means ripples fade away more quickly
    /// 
    /// The default value is 1.0, which means:
    /// - Neuron activations persist fully between time steps
    /// - The network maintains a strong memory of recent inputs
    /// 
    /// You might adjust this value:
    /// - Decrease it to make the network more responsive to immediate inputs
    /// - Keep it high if longer-term memory is important for your task
    /// 
    /// This parameter helps control the network's memory duration.
    /// </para>
    /// </remarks>
    private readonly double _leakingRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="LiquidStateMachine{T}"/> class with the specified architecture and parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="reservoirSize">The number of neurons in the reservoir.</param>
    /// <param name="connectionProbability">The probability of connection between neurons in the reservoir. Default is 0.1.</param>
    /// <param name="spectralRadius">The spectral radius of the reservoir weight matrix. Default is 0.9.</param>
    /// <param name="inputScaling">The scaling factor applied to input signals. Default is 1.0.</param>
    /// <param name="leakingRate">The leaking rate of the reservoir neurons. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a Liquid State Machine with the specified architecture and parameters.
    /// It initializes the reservoir and readout layers based on the provided configuration values.
    /// The default parameters are based on common values used in Liquid State Machine research and applications.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Liquid State Machine with your chosen settings.
    /// 
    /// When creating a Liquid State Machine, you can customize several key aspects:
    /// 
    /// 1. Architecture: The basic structure of your network
    /// 
    /// 2. Reservoir Size: How many neurons to use in the "liquid" part
    ///    - More neurons = more capacity but slower processing
    /// 
    /// 3. Connection Probability: How densely connected the neurons are
    ///    - Default 0.1 means each neuron connects to about 10% of others
    /// 
    /// 4. Spectral Radius: How long information echoes in the system
    ///    - Default 0.9 provides good memory without becoming unstable
    /// 
    /// 5. Input Scaling: How strongly inputs affect the network
    ///    - Default 1.0 provides balanced influence
    /// 
    /// 6. Leaking Rate: How quickly neuron activations fade
    ///    - Default 1.0 means activations persist fully
    /// 
    /// These parameters work together to determine how your network processes temporal information.
    /// The default values work well for many applications, but you may need to adjust them based
    /// on your specific task.
    /// </para>
    /// </remarks>
    public LiquidStateMachine(
        NeuralNetworkArchitecture<T> architecture,
        int reservoirSize,
        double connectionProbability = 0.1,
        double spectralRadius = 0.9,
        double inputScaling = 1.0,
        double leakingRate = 1.0) : base(architecture)
    {
        _leakingRate = leakingRate;
        _inputScaling = inputScaling;
        _spectralRadius = spectralRadius;
        _reservoirSize = reservoirSize;
        _connectionProbability = connectionProbability;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Liquid State Machine based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Liquid State Machine. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for a Liquid State Machine are created,
    /// typically including an input projection layer, a reservoir layer, and a readout layer configured with
    /// the specified parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your Liquid State Machine.
    /// 
    /// When initializing the network:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates standard LSM layers automatically
    /// 
    /// The standard LSM layers typically include:
    /// 1. Input Layer: Projects the inputs into the reservoir
    /// 2. Reservoir Layer: The randomly connected "liquid" pool of neurons
    /// 3. Readout Layer: Interprets the reservoir states to produce outputs
    /// 
    /// This process is like setting up all the components before the network starts processing data.
    /// The method uses your specified parameters (reservoir size, connection probability, etc.)
    /// to configure these layers appropriately.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultLSMLayers(Architecture, _reservoirSize, _connectionProbability, _spectralRadius, _inputScaling, _leakingRate));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network to generate a prediction from an input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input vector through all layers of the Liquid State Machine sequentially, transforming
    /// it at each step according to the layer's function, and returns the final output vector. For Liquid State Machines,
    /// this typically means passing through the input projection, the reservoir, and the readout layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your input data through the network to get a prediction.
    /// 
    /// The process works like this:
    /// 1. Your input enters the network and is projected into the reservoir
    /// 2. The reservoir neurons respond to the input, creating complex patterns
    /// 3. These patterns evolve based on the connections between neurons
    /// 4. The readout layer interprets these patterns to produce the final output
    /// 
    /// Each time you call this method with new input:
    /// - The reservoir state changes based on both the new input and its previous state
    /// - This gives the network its ability to process sequences and remember context
    /// - The output is computed based on the current reservoir state
    /// 
    /// This temporal processing is what makes Liquid State Machines powerful for time-series data.
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
    /// Updates the parameters of all layers in the network using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing updated parameters for all layers.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter values to each layer in the network. It extracts
    /// the appropriate segment of the parameter vector for each layer based on the layer's parameter count.
    /// In Liquid State Machines, typically only the readout layer's parameters are updated during training,
    /// while the reservoir remains fixed after initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the changeable parts of your network.
    /// 
    /// In a Liquid State Machine:
    /// - The reservoir connections are typically fixed after random initialization
    /// - Only the readout layer (which interprets the reservoir state) is usually trained
    /// 
    /// This method:
    /// 1. Takes a vector of parameter values
    /// 2. Figures out which values belong to which layers
    /// 3. Updates each layer with its corresponding parameters
    /// 
    /// The unique aspect of LSMs is that they maintain random, fixed connections in the reservoir,
    /// which makes them simpler to train than many other recurrent neural networks. Only the
    /// readout mechanism typically needs optimization, which happens through this method.
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
    /// Serializes the Liquid State Machine to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write the serialized network to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or when a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the Liquid State Machine's structure and parameters to a binary format that can be stored
    /// and later loaded. It writes the number of layers and then serializes each layer individually,
    /// preserving the network's configuration and any trained readout weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your Liquid State Machine to a file.
    /// 
    /// After setting up and potentially training your network, you'll want to save it
    /// so you can use it later without recreating it. This method:
    /// 
    /// 1. Counts how many layers your network has
    /// 2. Writes this count to the file
    /// 3. For each layer:
    ///    - Writes the type of layer
    ///    - Saves the configuration and parameters for that layer
    /// 
    /// This is particularly important for Liquid State Machines because:
    /// - The randomly-initialized reservoir is a key part of the network
    /// - If you created a new random reservoir, it would behave differently
    /// - Saving ensures you can use exactly the same network later
    /// 
    /// The saved file captures both the fixed reservoir and any trained readout mechanisms.
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
    }

    /// <summary>
    /// Deserializes the Liquid State Machine from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read the serialized network from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when an empty layer type name is encountered, when a layer type cannot be found, when a type does not implement the required interface, or when a layer instance cannot be created.</exception>
    /// <remarks>
    /// <para>
    /// This method loads a previously serialized Liquid State Machine from a binary format. It reads the number of layers
    /// and then deserializes each layer individually, recreating the network's structure and parameters.
    /// This allows the network to continue with the exact same reservoir and readout configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved Liquid State Machine from a file.
    /// 
    /// When you want to use a network that was saved earlier, this method:
    /// 
    /// 1. Reads how many layers the network should have
    /// 2. Creates a new, empty network
    /// 3. For each layer:
    ///    - Reads what type of layer it should be
    ///    - Creates that type of layer
    ///    - Loads the configuration and parameters for that layer
    /// 
    /// This is particularly important for Liquid State Machines because:
    /// - It restores the exact same random reservoir that was saved
    /// - It maintains the exact same dynamics that were present before
    /// - Any trained readout mechanisms are preserved
    /// 
    /// After loading, the network will behave identically to when it was saved,
    /// which is essential for consistent predictions.
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
    }
}