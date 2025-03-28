namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Neural Network specialized for occupancy detection and prediction in spaces.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// An Occupancy Neural Network processes sensor data to detect and predict the presence and number
/// of people in a given space. It can handle various types of sensor inputs including temperature,
/// humidity, CO2 levels, motion detection, and other environmental factors.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this network as a smart system that can "understand" when people
/// are present in a room or building by analyzing data from various sensors. Just like you might
/// determine if someone is in a room by noticing changes in temperature, sounds, or movement,
/// this neural network learns patterns in sensor data that indicate human presence. It's particularly
/// useful for smart buildings, energy management, security systems, and space utilization analysis.
/// </para>
/// </remarks>
public class OccupancyNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private bool _includeTemporalData { get; set; }
    private int _historyWindowSize { get; set; }
    
    /// <summary>
    /// Gets a value indicating whether this network processes temporal data.
    /// </summary>
    public bool IncludesTemporalData => _includeTemporalData;
    
    /// <summary>
    /// Gets the size of the time window used for temporal data.
    /// </summary>
    public int HistoryWindowSize => _historyWindowSize;

    /// <summary>
    /// Initializes a new instance of the OccupancyNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="includeTemporalData">Whether to include temporal data processing capabilities.</param>
    /// <param name="historyWindowSize">The number of previous time steps to include when using temporal data.</param>
    /// <exception cref="InvalidInputTypeException">Thrown when the input type does not match the required type based on configuration.</exception>
    /// <remarks>
    /// <para>
    /// Creates an occupancy neural network with the specified architecture and temporal settings.
    /// The input type is validated based on whether temporal data is included.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up your occupancy detection network based on
    /// your specifications. You provide the network's blueprint (architecture), tell it whether
    /// to analyze patterns over time, and if so, how many previous readings to consider.
    /// The constructor makes sure your input data format matches what the network expects.
    /// For example, if you want to include time patterns but provide only single-moment data,
    /// it will alert you that there's a mismatch.
    /// </para>
    /// </remarks>
    public OccupancyNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture, 
        bool includeTemporalData = false, 
        int historyWindowSize = 5) : base(architecture)
    {
        _includeTemporalData = includeTemporalData;
        _historyWindowSize = includeTemporalData ? Math.Max(1, historyWindowSize) : 0;
        
        if (includeTemporalData)
        {
            ArchitectureValidator.ValidateInputType(architecture, InputType.ThreeDimensional, nameof(OccupancyNeuralNetwork<T>));
        }
        else
        {
            ArchitectureValidator.ValidateInputType(architecture, InputType.OneDimensional, nameof(OccupancyNeuralNetwork<T>));
        }
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default layers appropriate for occupancy detection. If temporal data is included,
    /// the default configuration will include recurrent layers to capture time-series patterns.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the different processing stages (layers) of your
    /// occupancy detection network. If you've specified custom layers in your architecture, it uses those.
    /// If not, it creates a standard set of layers that work well for occupancy prediction.
    /// 
    /// When temporal data is included, it adds special layers (called recurrent layers) that can
    /// "remember" past data and detect patterns over time - like noticing gradual changes in CO2 levels
    /// when people enter a room. Think of this as either using your specific recipe for the network
    /// or falling back to a proven recipe that works well for most occupancy detection tasks.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            if (_includeTemporalData)
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultOccupancyTemporalLayers(Architecture, _historyWindowSize));
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultOccupancyLayers(Architecture));
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the neural network based on the provided input vector.
    /// </summary>
    /// <param name="input">The input vector containing the sensor data to process.</param>
    /// <returns>A vector containing the network's occupancy prediction.</returns>
    /// <exception cref="VectorLengthMismatchException">Thrown when the input vector length doesn't match the expected input dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method processes the input sensor data through the network and returns an occupancy prediction.
    /// The interpretation of the output depends on the specific task (binary occupancy detection,
    /// occupant counting, etc.) and how the network was trained.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use to get occupancy predictions from your network.
    /// You provide your sensor data (temperature, humidity, CO2, etc.) as a list of numbers, and this
    /// method processes it through the neural network and returns the prediction results.
    /// 
    /// The output could be a simple yes/no for occupancy, a count of people in the space, or probabilities
    /// for different occupancy levels, depending on how you trained your network. Before processing,
    /// the method checks that your input has the correct number of values for the network to use.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        if (input.Length != Architecture.InputSize)
        {
            throw new VectorLengthMismatchException(
                Architecture.InputSize, 
                input.Length,
                nameof(OccupancyNeuralNetwork<T>),
                nameof(Predict)
            );
        }

        var currentInput = new Tensor<T>([1, input.Length]);
        currentInput.SetRow(0, input);

        foreach (var layer in Layers)
        {
            currentInput = layer.Forward(currentInput);
        }

        return currentInput.GetRow(0);
    }

    /// <summary>
    /// Performs a forward pass through the network for temporal data.
    /// </summary>
    /// <param name="input">The input tensor to process with shape [batchSize, timeSteps, features].</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <exception cref="TensorShapeMismatchException">Thrown when the input shape doesn't match the expected shape.</exception>
    /// <remarks>
    /// <para>
    /// This method handles the processing of temporal (time-series) data through the network.
    /// It ensures the input has the correct time dimension and sequences the data appropriately
    /// through the temporal-aware layers.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method processes time-sequence sensor data through your network.
    /// It takes your sensor readings over time (already in the right format) and passes them through
    /// each layer in sequence. The network can detect patterns in how the readings change over time,
    /// which is often crucial for accurate occupancy detection.
    /// 
    /// For example, CO2 levels might rise gradually when people enter a room, and this method enables
    /// the network to recognize such temporal patterns. Think of it like analyzing a short video clip
    /// instead of just a single snapshot to determine occupancy.
    /// </para>
    /// </remarks>
    public Tensor<T> ForwardTemporal(Tensor<T> input)
    {
        if (!_includeTemporalData)
        {
            throw new InvalidOperationException("Temporal processing is not enabled for this network.");
        }

        var expectedShape = new[] { input.Shape[0], _historyWindowSize, Architecture.InputSize };
        if (input.Shape.Length != 3 || !input.Shape.SequenceEqual(expectedShape))
        {
            throw new TensorShapeMismatchException(
                expectedShape,
                input.Shape,
                nameof(OccupancyNeuralNetwork<T>),
                nameof(ForwardTemporal)
            );
        }

        var currentInput = input;
        foreach (var layer in Layers)
        {
            currentInput = layer.Forward(currentInput);
        }

        return currentInput;
    }

    /// <summary>
    /// Processes a new sensor reading and updates the prediction for real-time occupancy detection.
    /// </summary>
    /// <param name="currentReading">The latest sensor reading as a vector.</param>
    /// <param name="sensorHistory">Optional buffer of previous readings for temporal processing.</param>
    /// <returns>The updated occupancy prediction.</returns>
    /// <exception cref="InvalidOperationException">Thrown when temporal data is required but no history is provided.</exception>
    /// <remarks>
    /// <para>
    /// This method is optimized for real-time occupancy detection where new sensor readings arrive
    /// continuously. It can use a provided history buffer or maintain an internal one for temporal processing.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method is like a continuous room monitor. Each time you get a new
    /// sensor reading (like temperature or CO2 level), you feed it into this method. It combines this
    /// new information with recent past readings to give you an up-to-date prediction of room occupancy.
    /// It's particularly useful for systems that need to adjust in real-time, like smart thermostats
    /// or adaptive lighting systems.
    /// </para>
    /// </remarks>
    public Vector<T> UpdatePrediction(Vector<T> currentReading, Queue<Vector<T>>? sensorHistory = null)
    {
        if (_includeTemporalData && sensorHistory == null)
        {
            throw new InvalidOperationException("Sensor history is required for temporal processing.");
        }

        if (_includeTemporalData)
        {
            sensorHistory!.Enqueue(currentReading);
            if (sensorHistory.Count > _historyWindowSize)
            {
                sensorHistory.Dequeue();
            }

            var input = new Tensor<T>([1, _historyWindowSize, Architecture.InputSize]);
            int timeStep = 0;
            foreach (var reading in sensorHistory)
            {
                input.SetSlice(timeStep, reading);
                timeStep++;
            }

            return ForwardTemporal(input).GetRow(0);
        }
        else
        {
            return Predict(currentReading);
        }
    }

    /// <summary>
    /// Updates the parameters of the neural network based on computed gradients.
    /// </summary>
    /// <param name="gradients">The gradients to apply to the network parameters.</param>
    /// <remarks>
    /// <para>
    /// This method applies the provided gradients to update the network's parameters,
    /// typically as part of the training process.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network "learns" from its mistakes. After making
    /// predictions and calculating how far off they were, this method adjusts the network's
    /// internal settings to try and make better predictions next time. It's like fine-tuning
    /// an instrument based on feedback from a tuner.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <summary>
    /// Serializes the neural network to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the network state to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the entire state of the neural network, including its architecture
    /// and learned parameters, to a binary stream for later retrieval.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as saving your network's "brain" to a file. After
    /// training your network to recognize occupancy patterns, you can save all that learned
    /// knowledge so you can use it later without having to retrain. It's like taking a snapshot
    /// of the network's current state of understanding about occupancy detection.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        SerializationValidator.ValidateWriter(writer, nameof(OccupancyNeuralNetwork<T>));

        writer.Write(_includeTemporalData);
        writer.Write(_historyWindowSize);

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            string? layerTypeName = layer.GetType().FullName;
            SerializationValidator.ValidateLayerTypeName(layerTypeName);
            writer.Write(layerTypeName!);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Deserializes the neural network from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the network state from.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the entire state of the neural network, including its architecture
    /// and learned parameters, from a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like loading a saved "brain" back into your network. If you've
    /// previously trained and saved an occupancy detection network, you can use this method to
    /// recreate that network exactly as it was, complete with all its learned knowledge about
    /// occupancy patterns. It's like giving the network back its memories and experience.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        SerializationValidator.ValidateReader(reader, nameof(OccupancyNeuralNetwork<T>));

        _includeTemporalData = reader.ReadBoolean();
        _historyWindowSize = reader.ReadInt32();

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            SerializationValidator.ValidateLayerTypeName(layerTypeName);

            Type? layerType = Type.GetType(layerTypeName);
            SerializationValidator.ValidateLayerTypeExists(layerTypeName, layerType, nameof(OccupancyNeuralNetwork<T>));

            try
            {
                ILayer<T> layer = (ILayer<T>)Activator.CreateInstance(layerType!)!;
                layer.Deserialize(reader);
                Layers.Add(layer);
            }
            catch (Exception ex) when (ex is not SerializationException)
            {
                throw new SerializationException(
                    $"Failed to instantiate or deserialize layer of type {layerTypeName}",
                    nameof(OccupancyNeuralNetwork<T>),
                    "Deserialize",
                    ex);
            }
        }
    }
}