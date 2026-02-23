using AiDotNet.NeuralNetworks.Options;

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
    private readonly LiquidStateMachineOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    private int _reservoirSize;

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
    private double _connectionProbability;

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
    private double _spectralRadius;

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
    private double _inputScaling;

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
    private double _leakingRate;

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
        double leakingRate = 1.0,
        ILossFunction<T>? lossFunction = null,
        LiquidStateMachineOptions? options = null) : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new LiquidStateMachineOptions();
        Options = _options;
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
    /// Performs a forward pass through the Liquid State Machine to make a prediction.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input data through the reservoir and readout layers to produce an output.
    /// The liquid state machine relies on the complex dynamics of the reservoir to transform the input
    /// signal into a higher-dimensional representation, which is then processed by the readout layers.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the LSM processes new data to make predictions.
    /// 
    /// It works like this:
    /// 1. Input data enters the "reservoir" (like dropping an object into a pool of water)
    /// 2. The reservoir creates complex ripple patterns based on its internal connections
    /// 3. The readout layers interpret these patterns to produce the final output
    /// 
    /// The key advantage is that even simple readout mechanisms can solve complex problems
    /// by leveraging the rich dynamics of the reservoir.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Clear any stored state from previous predictions (must be done before GPU path too)
        ResetState();

        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> current = input;

        // Process the input through each layer sequentially
        for (int i = 0; i < Layers.Count; i++)
        {
            if (!IsTrainingMode)
            {
                // For inference mode, just forward pass
                current = Layers[i].Forward(current);
            }
            else
            {
                // For training mode, store intermediate values
                _layerInputs[i] = current;
                current = Layers[i].Forward(current);
                _layerOutputs[i] = current;
            }
        }

        return current;
    }

    /// <summary>
    /// Resets the state of the Liquid State Machine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the Liquid State Machine and all its layers.
    /// It clears the stored inputs and outputs from previous forward passes and resets the state
    /// of any stateful layers like the reservoir. This is important when starting to process a new,
    /// unrelated sequence or when you want to clear the temporal memory of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This is like clearing the water surface to start fresh.
    /// 
    /// In our water analogy:
    /// - ResetState() is like calming the water so no ripples remain
    /// - This ensures that previous inputs don't affect how the network processes new inputs
    /// - It's important to call this when starting to process a completely new sequence
    /// 
    /// For example, if you've processed one audio clip and now want to process another unrelated clip,
    /// you should reset the state in between to clear the "memory" of the first clip.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear stored layer activations
        _layerInputs.Clear();
        _layerOutputs.Clear();

        // Reset all layers, especially important for reservoir layers
        foreach (var layer in Layers)
        {
            layer.ResetState();
        }

        // Initialize empty dictionaries for storing layer inputs/outputs
        // during future forward passes
        _layerInputs = new Dictionary<int, Tensor<T>>();
        _layerOutputs = new Dictionary<int, Tensor<T>>();
    }

    /// <summary>
    /// Trains the Liquid State Machine on a single input-output pair.
    /// </summary>
    /// <param name="input">The input tensor to learn from.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method trains the readout mechanism of the Liquid State Machine using the given input-output pair.
    /// In LSMs, typically only the readout layers are trained, while the reservoir remains fixed
    /// after initialization. This makes training efficient compared to fully recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the LSM learns from examples.
    /// 
    /// The unique aspect of Liquid State Machines is that:
    /// - The reservoir (internal pool of neurons) is randomly connected and stays fixed
    /// - Only the readout mechanism (which interprets the reservoir states) is trained
    /// 
    /// This makes training much simpler than in other recurrent networks, because:
    /// 1. Input data is passed through the fixed reservoir
    /// 2. The reservoir creates complex patterns
    /// 3. Only the readout layer is trained to map these patterns to correct outputs
    /// 
    /// It's like learning to read ripple patterns in water without changing how water behaves.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!IsTrainingMode)
        {
            SetTrainingMode(true);
        }

        // Forward pass through the network, storing intermediate values
        Tensor<T> prediction = Predict(input);

        // Calculate loss
        var flattenedPredictions = prediction.ToVector();
        var flattenedExpected = expectedOutput.ToVector();
        LastLoss = LossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        // Calculate output gradients
        var outputGradients = LossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);

        // Backpropagate to get parameter gradients
        Vector<T> gradients = Backpropagate(Tensor<T>.FromVector(outputGradients)).ToVector();

        // Get parameter gradients for all trainable layers
        Vector<T> parameterGradients = GetParameterGradients();

        // Clip gradients to prevent exploding gradients
        parameterGradients = ClipGradient(parameterGradients);

        // Create optimizer
        var optimizer = new GradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Get current parameters
        Vector<T> currentParameters = GetParameters();

        // Update parameters using the optimizer
        Vector<T> updatedParameters = optimizer.UpdateParameters(currentParameters, parameterGradients);

        // Apply updated parameters
        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Gets metadata about the Liquid State Machine model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the LSM, including its model type, architecture details,
    /// reservoir properties, and serialized model data. This information is useful for model management
    /// and for saving/loading models.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your LSM's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (LSM)
    /// - Details about the reservoir (size, connectivity, etc.)
    /// - Parameters that affect the model's behavior
    /// - Serialized data that can be used to save and reload the model
    /// 
    /// This is useful for keeping track of different models and their configurations,
    /// especially when experimenting with multiple settings.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.LiquidStateMachine,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ReservoirSize", _reservoirSize },
                { "ConnectionProbability", _connectionProbability },
                { "SpectralRadius", _spectralRadius },
                { "InputScaling", _inputScaling },
                { "LeakingRate", _leakingRate },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = Serialize()
        };
    }

    /// <summary>
    /// Serializes Liquid State Machine-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes LSM-specific configuration data to a binary stream. This includes
    /// properties such as reservoir size, connection probability, spectral radius, input scaling,
    /// and leaking rate. This data is needed to reconstruct the LSM when deserializing.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the special configuration of your LSM.
    /// 
    /// It's like writing down the recipe for how your specific LSM was built:
    /// - How big the reservoir is
    /// - How densely it's connected
    /// - How strongly inputs affect it
    /// - How quickly information fades
    /// 
    /// These details are crucial because they define how your LSM processes information,
    /// and they need to be saved along with the weights for the model to work correctly
    /// when loaded later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write LSM-specific properties
        writer.Write(_reservoirSize);
        writer.Write(_connectionProbability);
        writer.Write(_spectralRadius);
        writer.Write(_inputScaling);
        writer.Write(_leakingRate);

        // Write whether we're in training mode
        writer.Write(IsTrainingMode);
    }

    /// <summary>
    /// Deserializes Liquid State Machine-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads LSM-specific configuration data from a binary stream. It restores
    /// properties such as reservoir size, connection probability, spectral radius, input scaling,
    /// and leaking rate. After reading this data, the state of the LSM is fully restored.
    /// </para>
    /// <para><b>For Beginners:</b> This restores the special configuration of your LSM from saved data.
    /// 
    /// It's like following the recipe to rebuild your LSM exactly as it was:
    /// - Setting the reservoir to the right size
    /// - Configuring the connection density
    /// - Setting up how strongly inputs affect the network
    /// - Restoring how quickly information fades
    /// 
    /// By reading these details, the LSM can be reconstructed exactly as it was
    /// when it was saved, preserving all its behavior and learned patterns.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _reservoirSize = reader.ReadInt32();
        _connectionProbability = reader.ReadDouble();
        _spectralRadius = reader.ReadDouble();
        _inputScaling = reader.ReadDouble();
        _leakingRate = reader.ReadDouble();

        // Read training mode
        IsTrainingMode = reader.ReadBoolean();
    }

    /// <summary>
    /// Sets the training mode for the Liquid State Machine.
    /// </summary>
    /// <param name="isTraining">True to enable training mode; false to enable inference mode.</param>
    /// <remarks>
    /// <para>
    /// This method overrides the base class implementation to set the training mode for both
    /// the LSM itself and all its layers. In training mode, the network keeps track of intermediate
    /// values needed for backpropagation, while in inference mode it operates more efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This switches the LSM between learning mode and prediction mode.
    /// 
    /// In training mode (isTraining = true):
    /// - The network keeps track of more information to enable learning
    /// - It stores intermediate values needed for backpropagation
    /// - It uses more memory but can adjust its parameters
    /// 
    /// In inference mode (isTraining = false):
    /// - The network is more efficient
    /// - It doesn't need to store extra information
    /// - It's faster but cannot learn new patterns
    /// 
    /// Switching to the right mode is important for both efficient training and fast prediction.
    /// </para>
    /// </remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);

        // Also set training mode for all layers
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(isTraining);
        }

        // Clear stored states when switching modes
        if (IsTrainingMode != isTraining)
        {
            _layerInputs.Clear();
            _layerOutputs.Clear();
        }
    }

    /// <summary>
    /// Simulates the LSM with time-series data, allowing the reservoir state to evolve over time.
    /// </summary>
    /// <param name="timeSeriesInput">A sequence of input tensors representing a time series.</param>
    /// <returns>A sequence of output tensors corresponding to each input time step.</returns>
    /// <remarks>
    /// <para>
    /// This method processes a sequence of inputs through the LSM, maintaining the reservoir state
    /// between time steps. This allows the network to exhibit temporal memory and process time-series
    /// data effectively. The method returns the corresponding sequence of outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the LSM processes data that changes over time.
    /// 
    /// Think of it like continuously dropping objects into water:
    /// - Each input creates new ripples
    /// - But these ripples interact with existing ripples from previous inputs
    /// - The state of the water at any moment depends on both current and past inputs
    /// 
    /// This temporal memory makes LSMs excellent for:
    /// - Speech recognition (where sounds have meaning in sequence)
    /// - Time-series prediction (like stock prices or weather)
    /// - Any data where history matters
    /// </para>
    /// </remarks>
    public List<Tensor<T>> SimulateTimeSeries(List<Tensor<T>> timeSeriesInput)
    {
        var outputs = new List<Tensor<T>>();

        // Reset state before starting the simulation
        ResetState();

        // Process each time step
        foreach (var input in timeSeriesInput)
        {
            var output = Predict(input);
            outputs.Add(output);

            // Note: We don't reset state between time steps to maintain temporal dynamics
        }

        return outputs;
    }

    /// <summary>
    /// Performs online learning for time-series data, updating the network after each time step.
    /// </summary>
    /// <param name="timeSeriesInput">A sequence of input tensors representing a time series.</param>
    /// <param name="timeSeriesExpectedOutput">A sequence of expected output tensors.</param>
    /// <remarks>
    /// <para>
    /// This method implements online learning for time-series data, where the network is updated
    /// after each time step rather than after processing the entire sequence. This approach
    /// allows the LSM to adapt to changing dynamics in the time series.
    /// </para>
    /// <para><b>For Beginners:</b> This trains the LSM on data that changes over time, updating after each step.
    /// 
    /// In online learning:
    /// - The network processes each time step and makes a prediction
    /// - It immediately receives feedback on how accurate that prediction was
    /// - It updates its parameters before moving to the next time step
    /// 
    /// This is different from batch learning where the network would see the entire sequence
    /// before making any updates. Online learning can be better for:
    /// - Data streams that continually arrive
    /// - Systems that need to adapt to changing conditions
    /// - Learning from very long sequences
    /// </para>
    /// </remarks>
    public void TrainOnTimeSeries(List<Tensor<T>> timeSeriesInput, List<Tensor<T>> timeSeriesExpectedOutput)
    {
        if (timeSeriesInput.Count != timeSeriesExpectedOutput.Count)
        {
            throw new ArgumentException("Input and expected output sequences must have the same length");
        }

        // Reset state before starting the training
        ResetState();
        SetTrainingMode(true);

        // Process each time step
        for (int i = 0; i < timeSeriesInput.Count; i++)
        {
            // Forward pass and train on this time step
            Train(timeSeriesInput[i], timeSeriesExpectedOutput[i]);

            // Note: We don't reset state between time steps to maintain temporal dynamics
        }
    }

    /// <summary>
    /// Creates a new instance of the Liquid State Machine with the same architecture and configuration.
    /// </summary>
    /// <returns>A new Liquid State Machine instance with the same architecture and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the Liquid State Machine with the same architecture and LSM-specific
    /// parameters as the current instance. It's used in scenarios where a fresh copy of the model is needed
    /// while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the LSM with the same setup.
    /// 
    /// Think of it like creating a clone of the network:
    /// - The new network has the same architecture (structure)
    /// - It has the same reservoir size, connection probability, and other settings
    /// - But it's a completely separate instance with its own internal state
    /// - The reservoir will be randomly initialized again, creating a different random network
    /// 
    /// This is useful when you want to:
    /// - Train multiple networks with the same configuration
    /// - Compare how different random initializations affect learning
    /// - Create an ensemble of models with the same parameters
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LiquidStateMachine<T>(
            this.Architecture,
            _reservoirSize,
            _connectionProbability,
            _spectralRadius,
            _inputScaling,
            _leakingRate);
    }
}
