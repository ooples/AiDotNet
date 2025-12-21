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
    /// <summary>
    /// Gets or sets a value indicating whether this network processes temporal data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set to true, the network will consider historical data in its predictions,
    /// allowing it to detect patterns over time.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like telling the network whether to consider how sensor
    /// readings change over time. If set to true, the network can spot trends, like gradual
    /// increases in CO2 levels as people enter a room. If false, it only looks at the current
    /// moment's readings.
    /// </para>
    /// </remarks>
    private bool _includeTemporalData { get; set; }

    /// <summary>
    /// Gets or sets the size of the time window used for temporal data processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines how many previous time steps of data the network considers
    /// when temporal processing is enabled. A larger window allows the network to detect
    /// longer-term patterns but requires more memory and processing power.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like deciding how far back in time the network should
    /// "remember" when analyzing sensor data. If set to 5, for example, the network will
    /// consider the current reading plus the 4 previous readings when making its prediction.
    /// This helps in detecting gradual changes or recurring patterns in occupancy.
    /// </para>
    /// </remarks>
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
    /// Buffer to store historical sensor readings for temporal processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This internal buffer maintains the recent history of sensor readings when temporal
    /// processing is enabled, allowing the network to detect patterns over time.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as the network's short-term memory.
    /// It stores the most recent sensor readings so the network can analyze
    /// how values have changed over time rather than just looking at the current moment.
    /// </para>
    /// </remarks>
    private Queue<Vector<T>> _internalSensorHistory;

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
        int historyWindowSize = 5,
        ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _includeTemporalData = includeTemporalData;
        _historyWindowSize = includeTemporalData ? Math.Max(1, historyWindowSize) : 0;
        _internalSensorHistory = new Queue<Vector<T>>(_historyWindowSize);

        if (includeTemporalData)
        {
            ArchitectureValidator.ValidateInputType(architecture, InputType.ThreeDimensional, nameof(OccupancyNeuralNetwork<T>));
        }
        else
        {
            ArchitectureValidator.ValidateInputType(architecture, InputType.OneDimensional, nameof(OccupancyNeuralNetwork<T>));
        }

        InitializeLayers();
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
    /// Performs a forward pass through the network for temporal data.
    /// </summary>
    /// <param name="input">The input tensor to process with shape [batchSize, timeSteps, features].</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <exception cref="TensorShapeMismatchException">Thrown when the input shape doesn't match the expected shape.</exception>
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
    public Vector<T> UpdatePrediction(Vector<T> currentReading, Queue<Vector<T>>? sensorHistory = null)
    {
        if (_includeTemporalData && sensorHistory == null)
        {
            // Use internal history if no external history is provided
            sensorHistory = _internalSensorHistory;
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
            return Predict(Tensor<T>.FromVector(currentReading)).ToVector();
        }
    }

    /// <summary>
    /// Updates the parameters of the neural network based on computed gradients.
    /// </summary>
    /// <param name="gradients">The gradients to apply to the network parameters.</param>
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
    /// Makes a prediction using the occupancy neural network.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// This method processes input data through the network to predict occupancy. If temporal
    /// data is enabled, it will ensure the input has the appropriate format before processing.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes your sensor data and produces an occupancy prediction.
    /// Depending on how your network is configured, it will either:
    /// 
    /// 1. Process a single moment's sensor readings (like current temperature, CO2, etc.)
    /// 2. Process a sequence of readings over time to detect temporal patterns
    /// 
    /// The output typically indicates how many people are in the space or the probability
    /// of the space being occupied. This method handles all the complex mathematical
    /// transformations needed to convert raw sensor data into meaningful occupancy information.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Set network to inference mode
        bool originalTrainingMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            // Handle different input formats based on temporal configuration
            if (_includeTemporalData)
            {
                // Check if input is properly shaped for temporal processing
                if (input.Rank == 3 && input.Shape[1] == _historyWindowSize)
                {
                    // Input is already in correct temporal format
                    return ForwardTemporal(input);
                }
                else if (input.Rank == 2 && input.Shape[0] == _historyWindowSize)
                {
                    // Input is a sequence of feature vectors, reshape to [1, timeSteps, features]
                    var reshapedInput = new Tensor<T>([1, _historyWindowSize, input.Shape[1]]);
                    for (int t = 0; t < _historyWindowSize; t++)
                    {
                        for (int f = 0; f < input.Shape[1]; f++)
                        {
                            reshapedInput[0, t, f] = input[t, f];
                        }
                    }
                    return ForwardTemporal(reshapedInput);
                }
                else
                {
                    throw new TensorShapeMismatchException(
                        new[] { -1, _historyWindowSize, Architecture.InputSize },
                        input.Shape,
                        nameof(OccupancyNeuralNetwork<T>),
                        nameof(Predict)
                    );
                }
            }
            else
            {
                // Handle non-temporal input
                if (input.Rank > 1)
                {
                    // If input is multi-dimensional but network is not temporal,
                    // assume batch processing and process each vector separately
                    int batchSize = input.Shape[0];
                    int features = input.Rank > 1 ? input.Shape[1] : input.Shape[0];

                    // Create output tensor for batch results
                    var output = new Tensor<T>([batchSize, Architecture.OutputSize]);

                    // Process each input in the batch
                    for (int b = 0; b < batchSize; b++)
                    {
                        // Extract single vector from batch
                        var singleInput = new Vector<T>(features);
                        for (int f = 0; f < features; f++)
                        {
                            singleInput[f] = input[b, f];
                        }

                        // Process through layers
                        var result = ProcessSingleInput(singleInput);

                        // Store in output tensor
                        for (int o = 0; o < result.Length; o++)
                        {
                            output[b, o] = result[o];
                        }
                    }

                    return output;
                }
                else
                {
                    // Single vector input
                    var result = ProcessSingleInput(input.ToVector());
                    return Tensor<T>.FromVector(result);
                }
            }
        }
        finally
        {
            // Restore original training mode
            SetTrainingMode(originalTrainingMode);
        }
    }

    /// <summary>
    /// Processes a single input vector through the network.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// This helper method processes a single input vector through all layers of the network.
    /// It converts the vector to a tensor, passes it through each layer, and returns the result.
    /// </para>
    /// </remarks>
    private Vector<T> ProcessSingleInput(Vector<T> input)
    {
        // Validate input size
        if (input.Length != Architecture.InputSize)
        {
            throw new ArgumentException($"Input size mismatch. Expected {Architecture.InputSize}, got {input.Length}.");
        }

        // Forward pass through all layers
        var current = Tensor<T>.FromVector(input);

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current.ToVector();
    }

    /// <summary>
    /// Trains the neural network on sensor data and occupancy labels.
    /// </summary>
    /// <param name="input">The input tensor containing sensor data.</param>
    /// <param name="expectedOutput">The expected output tensor containing occupancy information.</param>
    /// <remarks>
    /// <para>
    /// This method trains the occupancy neural network by comparing its predictions with expected
    /// occupancy values. It handles both temporal and non-temporal training data formats.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the network to accurately detect occupancy 
    /// by showing it examples of sensor readings and the correct occupancy levels.
    /// 
    /// The training process follows these steps:
    /// 1. The network processes the sensor readings
    /// 2. Its prediction is compared to the actual occupancy value
    /// 3. The difference (error) is used to adjust the network's internal settings
    /// 4. This process repeats with many examples until the network becomes accurate
    /// 
    /// The network automatically handles different data formats depending on whether it's
    /// configured to analyze patterns over time (temporal) or just current readings.
    /// Over time, the network becomes better at recognizing the subtle patterns in
    /// sensor data that indicate human presence.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure we're in training mode
        SetTrainingMode(true);

        // Process based on temporal configuration
        if (_includeTemporalData)
        {
            TrainTemporal(input, expectedOutput);
        }
        else
        {
            TrainNonTemporal(input, expectedOutput);
        }
    }

    /// <summary>
    /// Trains the network on temporal (sequence) data.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, timeSteps, features].</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    private void TrainTemporal(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Validate input shape
        var expectedShape = new[] { input.Shape[0], _historyWindowSize, Architecture.InputSize };
        if (input.Shape.Length != 3 || !input.Shape.SequenceEqual(expectedShape))
        {
            throw new TensorShapeMismatchException(
                expectedShape,
                input.Shape,
                nameof(OccupancyNeuralNetwork<T>),
                nameof(TrainTemporal)
            );
        }

        // Forward pass
        var output = ForwardTemporal(input);

        // Calculate loss using the loss function
        Vector<T> predictedVector = output.ToVector();
        Vector<T> expectedVector = expectedOutput.ToVector();
        T loss = LossFunction.CalculateLoss(predictedVector, expectedVector);

        // Set the LastLoss property
        LastLoss = loss;

        // Calculate error gradients using the loss function's derivative
        Vector<T> gradients = LossFunction.CalculateDerivative(predictedVector, expectedVector);

        // Backpropagation
        Backpropagate(Tensor<T>.FromVector(gradients));

        // Update parameters with optimizer
        T learningRate = NumOps.FromDouble(0.01);
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Trains the network on non-temporal (single time step) data.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, features].</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    private void TrainNonTemporal(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var output = Predict(input);

        // Calculate loss using the loss function
        Vector<T> predictedVector = output.ToVector();
        Vector<T> expectedVector = expectedOutput.ToVector();
        T loss = LossFunction.CalculateLoss(predictedVector, expectedVector);

        // Set the LastLoss property
        LastLoss = loss;

        // Calculate error gradients using the loss function's derivative
        Vector<T> gradients = LossFunction.CalculateDerivative(predictedVector, expectedVector);

        // Backpropagation
        Backpropagate(Tensor<T>.FromVector(gradients));

        // Update parameters with optimizer
        T learningRate = NumOps.FromDouble(0.01);
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Calculates the error between predicted and expected outputs.
    /// </summary>
    /// <param name="predicted">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>A tensor containing the errors.</returns>
    private Tensor<T> CalculateError(Tensor<T> predicted, Tensor<T> expected)
    {
        // Ensure tensors have the same shape
        if (!predicted.Shape.SequenceEqual(expected.Shape))
        {
            throw new TensorShapeMismatchException(
                expected.Shape,
                predicted.Shape,
                nameof(OccupancyNeuralNetwork<T>),
                nameof(CalculateError)
            );
        }

        // Calculate error (expected - predicted)
        var error = new Tensor<T>(predicted.Shape);

        for (int i = 0; i < predicted.Length; i++)
        {
            var predictedValue = predicted.GetFlatIndexValue(i);
            var expectedValue = expected.GetFlatIndexValue(i);
            error.SetFlatIndex(i, NumOps.Subtract(expectedValue, predictedValue));
        }

        return error;
    }


    /// <summary>
    /// Gets metadata about the occupancy neural network.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the occupancy neural network, including
    /// its architecture, temporal configuration, and other relevant parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This provides detailed information about your occupancy detection network.
    /// It includes data like:
    /// - The network's structure (layers, neurons, etc.)
    /// - Whether it analyzes patterns over time
    /// - How many past readings it considers for temporal analysis
    /// - The total number of internal parameters
    /// 
    /// This information is useful for documentation, comparing different network configurations,
    /// or debugging issues with the network's performance.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Count layer types
        var layerTypeCount = new Dictionary<string, int>();
        foreach (var layer in Layers)
        {
            string layerType = layer.GetType().Name;
            if (layerTypeCount.ContainsKey(layerType))
            {
                layerTypeCount[layerType]++;
            }
            else
            {
                layerTypeCount[layerType] = 1;
            }
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.OccupancyNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize },
                { "IncludesTemporalData", _includeTemporalData },
                { "HistoryWindowSize", _historyWindowSize },
                { "LayerCount", Layers.Count },
                { "LayerTypes", layerTypeCount },
                { "TotalParameters", ParameterCount },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves occupancy network-specific data to the binary stream,
    /// such as temporal configuration settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method saves special information about your
    /// occupancy detection network to a file. This includes settings like whether
    /// it analyzes patterns over time and how many past readings it considers.
    /// Saving this information allows you to later reload the network exactly
    /// as it was configured.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save temporal configuration
        writer.Write(_includeTemporalData);
        writer.Write(_historyWindowSize);

        // Save any internal sensor history if present
        writer.Write(_internalSensorHistory.Count);

        foreach (var reading in _internalSensorHistory)
        {
            writer.Write(reading.Length);
            for (int i = 0; i < reading.Length; i++)
            {
                writer.Write(Convert.ToDouble(reading[i]));
            }
        }
    }

    /// <summary>
    /// Deserializes network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads occupancy network-specific data from the binary stream,
    /// such as temporal configuration settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method loads the previously saved special information
    /// about your occupancy detection network from a file. It restores settings like
    /// whether the network analyzes patterns over time and how many past readings it 
    /// considers. This allows you to continue using a network exactly where you left off,
    /// with all its settings and internal state intact.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Load temporal configuration
        _includeTemporalData = reader.ReadBoolean();
        _historyWindowSize = reader.ReadInt32();

        // Initialize sensor history queue
        _internalSensorHistory = new Queue<Vector<T>>(_historyWindowSize);

        // Load any saved sensor history
        int historyCount = reader.ReadInt32();

        for (int h = 0; h < historyCount; h++)
        {
            int readingLength = reader.ReadInt32();
            var reading = new Vector<T>(readingLength);

            for (int i = 0; i < readingLength; i++)
            {
                reading[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            _internalSensorHistory.Enqueue(reading);
        }
    }

    /// <summary>
    /// Creates a new instance of the OccupancyNeuralNetwork with the same architecture and temporal configuration.
    /// </summary>
    /// <returns>A new instance of the occupancy neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new occupancy neural network with the same architecture and temporal
    /// data processing configuration as the current instance. The new instance starts with fresh layers
    /// and an empty sensor history buffer, making it useful for creating multiple networks with identical
    /// configurations or for resetting a network while preserving its structure.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a brand new occupancy detection network with the same settings.
    /// 
    /// Think of it like creating a copy of your current network's blueprint:
    /// - It has the same structure (layers and neurons)
    /// - It uses the same approach to time-based analysis (if enabled)
    /// - It looks at the same number of past readings when analyzing patterns
    /// 
    /// However, the new network starts fresh with:
    /// - Newly initialized weights and parameters
    /// - An empty history buffer (no past sensor readings)
    /// 
    /// This is useful when you want to start over with a clean network that has
    /// the same design but hasn't learned anything yet, or when you need multiple
    /// identical networks for different spaces or comparison purposes.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new OccupancyNeuralNetwork<T>(
            Architecture,
            _includeTemporalData,
            _historyWindowSize,
            LossFunction);
    }
}
