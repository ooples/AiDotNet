namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Spiking Neural Network, which is a type of neural network that more closely models biological neurons with temporal dynamics.
/// </summary>
/// <remarks>
/// <para>
/// A Spiking Neural Network (SNN) is a more biologically realistic neural network model that incorporates the
/// concept of time and discrete spikes into the neural processing. Unlike traditional artificial neural networks
/// that transmit continuous values, SNNs communicate through discrete spikes, similar to how biological neurons
/// communicate with action potentials. This approach allows for temporal information processing and potentially
/// more efficient computation on specialized hardware.
/// </para>
/// <para><b>For Beginners:</b> A Spiking Neural Network is designed to work more like real brain cells (neurons).
/// 
/// Think of it like this:
/// - Regular neural networks use continuous signals (like dimming or brightening a light)
/// - Spiking neural networks use discrete pulses (like turning a light on and off)
/// 
/// In your brain, neurons don't constantly send signals. Instead, they:
/// - Build up electrical charge
/// - When the charge passes a threshold, they "fire" a signal (spike)
/// - Then reset and start charging again
/// 
/// This spiking behavior:
/// - Is more energy-efficient (only sending signals when needed)
/// - Can process time-based information (when spikes occur matters)
/// - May be better for certain problems like real-time processing
/// 
/// For example, when recognizing a moving object, the timing of signals between neurons
/// helps your brain understand the motion, not just the shape of the object.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpikingNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the simulation time step for the spiking neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The time step determines the temporal resolution of the simulation. A smaller time step provides
    /// more precise timing of spikes but increases computational cost. The time step is measured in
    /// arbitrary time units, typically milliseconds when modeling biological neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how finely the network slices time during simulation.
    /// 
    /// The time step:
    /// - Is like the "tick" of a clock in the simulation
    /// - Smaller values (e.g., 0.1) give more precise timing but require more computation
    /// - Larger values (e.g., 1.0) are faster but less precise
    /// 
    /// For example, with a time step of 0.1:
    /// - The simulation updates neuron states every 0.1 time units
    /// - This allows for detecting timing differences as small as 0.1 units
    /// - But requires 10 updates to simulate 1 full time unit
    /// 
    /// The right time step depends on the temporal precision needed for your specific problem.
    /// </para>
    /// </remarks>
    private double _timeStep { get; set; }

    /// <summary>
    /// Gets or sets the number of time steps to simulate when processing input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The simulation steps parameter determines how many time steps the network simulates when processing
    /// an input. A larger number of steps allows for capturing longer temporal patterns but increases
    /// computational cost. This value, combined with the time step, determines the total simulated time.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how long the simulation runs for each input.
    /// 
    /// The simulation steps:
    /// - Define how many "ticks" of the simulation clock to run
    /// - More steps allow neurons to interact for longer periods
    /// - The total simulated time is: timeStep × simulationSteps
    /// 
    /// For example, with 100 steps and a time step of 0.1:
    /// - The network simulates a total of 10 time units for each input
    /// - This gives neurons time to process the input, fire, and influence each other
    /// - Longer simulations can capture more complex temporal patterns
    /// 
    /// This is especially important for time-dependent tasks like processing audio or detecting
    /// sequences, where the timing of spikes contains valuable information.
    /// </para>
    /// </remarks>
    private int _simulationSteps { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The vector activation function applies non-linearity to entire vectors of neuron activations at once.
    /// In spiking neural networks, this may implement specialized spike-generating mechanisms that operate
    /// on multiple neurons simultaneously, potentially improving computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This function determines how groups of neurons fire together.
    /// 
    /// The vector activation:
    /// - Processes multiple neurons at once
    /// - Determines when they produce spikes based on their internal state
    /// - Can be more computationally efficient than processing neurons one by one
    /// 
    /// In spiking networks, activation functions are special because they:
    /// - May include thresholds for firing
    /// - Often produce binary (on/off) outputs
    /// - Can incorporate time-dependent behavior
    /// 
    /// This is one way to implement the activation function - as an operation on vectors
    /// of neurons. The alternative is to use the scalar activation function below.
    /// </para>
    /// </remarks>
    private IVectorActivationFunction<T>? _vectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The scalar activation function applies non-linearity to individual neuron activations. In spiking
    /// neural networks, this typically implements a neuron model that determines when a neuron fires based
    /// on its membrane potential. Common models include the Leaky Integrate-and-Fire or Hodgkin-Huxley models.
    /// </para>
    /// <para><b>For Beginners:</b> This function determines how each individual neuron fires.
    /// 
    /// The scalar activation:
    /// - Processes one neuron at a time
    /// - Determines when it produces a spike based on its internal state
    /// - Can model complex neuron behaviors like charging and discharging
    /// 
    /// In biological neurons, this is similar to:
    /// - Building up electrical charge (integration)
    /// - Firing when a threshold is reached
    /// - Resetting after firing
    /// 
    /// This is an alternative to the vector activation above - processing each
    /// neuron individually rather than in groups.
    /// </para>
    /// </remarks>
    private IActivationFunction<T>? _scalarActivation { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpikingNeuralNetwork{T}"/> class with the specified architecture and a vector activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the SNN.</param>
    /// <param name="timeStep">The simulation time step, defaults to 0.1.</param>
    /// <param name="simulationSteps">The number of time steps to simulate, defaults to 100.</param>
    /// <param name="vectorActivation">The vector activation function to use. If null, a default activation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Spiking Neural Network with the specified architecture, time parameters,
    /// and vector activation function. The time step and simulation steps determine the temporal resolution
    /// and duration of the simulation, respectively. The vector activation function processes multiple neurons
    /// at once, potentially improving computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Spiking Neural Network with a function that processes neurons in groups.
    /// 
    /// When creating a new SNN this way:
    /// - You specify the architecture (layer sizes, connections, etc.)
    /// - You set the time step (clock tick size, default 0.1)
    /// - You set the simulation steps (how long to run, default 100 ticks)
    /// - You provide a vector activation function (how groups of neurons generate spikes)
    /// 
    /// For example, with the defaults:
    /// - The network will simulate 10 time units (0.1 × 100) for each input
    /// - It will process neurons in groups using the provided vector activation
    /// 
    /// This constructor is useful when you have an activation function that can efficiently
    /// process multiple neurons together.
    /// </para>
    /// </remarks>
    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100, IVectorActivationFunction<T>? vectorActivation = null) 
        : base(architecture)
    {
        _timeStep = timeStep;
        _simulationSteps = simulationSteps;
        _vectorActivation = vectorActivation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpikingNeuralNetwork{T}"/> class with the specified architecture and a scalar activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the SNN.</param>
    /// <param name="timeStep">The simulation time step, defaults to 0.1.</param>
    /// <param name="simulationSteps">The number of time steps to simulate, defaults to 100.</param>
    /// <param name="scalarActivation">The scalar activation function to use. If null, a default activation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Spiking Neural Network with the specified architecture, time parameters,
    /// and scalar activation function. The time step and simulation steps determine the temporal resolution
    /// and duration of the simulation, respectively. The scalar activation function processes individual
    /// neurons one at a time, allowing for more specialized neuron models.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Spiking Neural Network with a function that processes neurons individually.
    /// 
    /// When creating a new SNN this way:
    /// - You specify the architecture (layer sizes, connections, etc.)
    /// - You set the time step (clock tick size, default 0.1)
    /// - You set the simulation steps (how long to run, default 100 ticks)
    /// - You provide a scalar activation function (how each neuron generates spikes)
    /// 
    /// For example, with the defaults:
    /// - The network will simulate 10 time units (0.1 × 100) for each input
    /// - It will process neurons one by one using the provided scalar activation
    /// 
    /// This constructor is useful when you want to apply specialized neuron models like
    /// Leaky Integrate-and-Fire or Hodgkin-Huxley to each neuron individually.
    /// </para>
    /// </remarks>
    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100, IActivationFunction<T>? scalarActivation = null) 
        : base(architecture)
    {
        _timeStep = timeStep;
        _simulationSteps = simulationSteps;
        _scalarActivation = scalarActivation;
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Spiking Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for spiking
    /// networks is created. Spiking neural networks typically use specialized layers that model temporal
    /// dynamics and discrete spike generation.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for spiking networks are created automatically
    /// - The system checks that any custom layers will work properly with the SNN
    /// 
    /// Spiking neural networks use specialized layers that:
    /// - Track neuron states over time
    /// - Generate discrete spikes rather than continuous values
    /// - May include features like refractory periods (recovery time after firing)
    /// 
    /// These specialized layers are what allow the network to process temporal information
    /// and operate in a more brain-like manner.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpikingLayers(Architecture));
        }
    }

    /// <summary>
    /// Processes the input through the spiking neural network to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through the network.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Spiking Neural Network. It simulates the network
    /// over multiple time steps, generating a spike train (sequence of spikes over time). The input
    /// is passed through each layer of the network for each time step, and the resulting spike train
    /// is aggregated to produce the final output. This temporal processing is a key feature of spiking
    /// neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input through the network over time.
    /// 
    /// During the prediction process:
    /// 1. The input enters the network
    /// 2. The network runs a simulation for multiple time steps (like watching neurons interact for a period)
    /// 3. Each layer processes the data at each time step, generating spikes
    /// 4. The pattern of spikes over time (spike train) is recorded
    /// 5. The spike train is converted into a final output
    /// 
    /// This is different from standard neural networks because:
    /// - The processing happens over multiple time steps, not just once
    /// - Information is encoded in when spikes occur, not just their magnitude
    /// - The final output summarizes the temporal activity of the network
    /// 
    /// This temporal processing allows spiking networks to handle time-based patterns
    /// that standard networks might struggle with.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        var spikeTrain = new List<Vector<T>>();

        for (int t = 0; t < _simulationSteps; t++)
        {
            foreach (var layer in Layers)
            {
                current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
            }
            spikeTrain.Add(current);
        }

        // Aggregate spike train to produce final output
        return AggregateSpikeTrainToOutput(spikeTrain);
    }

    /// <summary>
    /// Aggregates a spike train (sequence of spikes over time) into a final output vector.
    /// </summary>
    /// <param name="spikeTrain">The list of spike vectors over time.</param>
    /// <returns>The aggregated output vector.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a temporal sequence of spikes (spike train) into a single output vector.
    /// It does this by averaging the spike values for each neuron over all time steps. This simple
    /// aggregation method captures the average activity of each output neuron over the simulation
    /// period, which can be interpreted as the network's overall response to the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the sequence of spikes over time into a single result.
    /// 
    /// The spike train:
    /// - Contains the activity of each output neuron at each time step
    /// - Records when neurons fired and when they were silent
    /// 
    /// To get a final result:
    /// - The method averages each neuron's activity over all time steps
    /// - Higher values mean the neuron fired more frequently
    /// - Lower values mean the neuron fired less frequently or not at all
    /// 
    /// For example, if an output neuron fired in 80 out of 100 time steps:
    /// - Its final output value would be 0.8 (80%)
    /// - This indicates it responded strongly to the input pattern
    /// 
    /// This averaging approach is just one way to interpret the spike train;
    /// other methods might look at precise spike timing or patterns.
    /// </para>
    /// </remarks>
    private Vector<T> AggregateSpikeTrainToOutput(List<Vector<T>> spikeTrain)
    {
        int outputSize = spikeTrain[0].Length;
        var output = new Vector<T>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            T sum = NumOps.Zero;
            for (int t = 0; t < spikeTrain.Count; t++)
            {
                sum = NumOps.Add(sum, spikeTrain[t][i]);
            }
            output[i] = NumOps.Divide(sum, NumOps.FromDouble(spikeTrain.Count));
        }

        return output;
    }

    /// <summary>
    /// Updates the parameters of the spiking neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the spiking neural network based on the provided
    /// parameter updates. The parameters vector is divided into segments corresponding to each layer's
    /// parameter count, and each segment is applied to its respective layer. In spiking neural networks,
    /// these parameters might include weights, thresholds, and time constants that govern spike generation
    /// and propagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the SNN makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For a spiking neural network, these adjustments might include:
    /// - Connection strengths between neurons
    /// - Threshold values that determine when neurons fire
    /// - Time constants that control how quickly neurons respond
    /// 
    /// These parameter updates help the network learn to recognize patterns
    /// and generate appropriate responses based on the training data.
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
    /// Saves the state of the Spiking Neural Network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer serialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the Spiking Neural Network, including all layers,
    /// time step, simulation steps, and layer-specific parameters. It writes these values to the
    /// provided binary writer, allowing the SNN to be saved to a file or other storage medium and
    /// later restored.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the SNN to a file.
    /// 
    /// When serializing:
    /// - The number of layers is saved
    /// - The time step and simulation steps are saved
    /// - Each layer's type and internal state is saved
    /// 
    /// This is useful for:
    /// - Saving a trained SNN to use later
    /// - Sharing a model with others
    /// - Creating backups during long training processes
    /// 
    /// Think of it like taking a complete snapshot of the SNN that can be restored later.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        writer.Write(_timeStep);
        writer.Write(_simulationSteps);

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
    /// Loads the state of the Spiking Neural Network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the Spiking Neural Network from a binary reader. It reads
    /// the number of layers, time step, simulation steps, and recreates each layer based on its type and
    /// stored state. This allows a previously saved SNN to be restored and used for prediction or
    /// further training.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved SNN state from a file.
    /// 
    /// When deserializing:
    /// - The number of layers is read first
    /// - The time step and simulation steps are read
    /// - Each layer's type and internal state is loaded
    /// 
    /// This allows you to:
    /// - Load a previously trained SNN
    /// - Continue using or training a model from where you left off
    /// - Use models created by others
    /// 
    /// Think of it like restoring a complete snapshot of an SNN that was saved earlier.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        _timeStep = reader.ReadDouble();
        _simulationSteps = reader.ReadInt32();

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