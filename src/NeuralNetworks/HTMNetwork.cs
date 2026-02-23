using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Hierarchical Temporal Memory (HTM) network, a biologically-inspired sequence learning algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Hierarchical Temporal Memory (HTM) is a machine learning model that mimics the structural and algorithmic properties 
/// of the neocortex. It is particularly designed for sequence learning, prediction, and anomaly detection in 
/// time-series data. HTM networks consist of two main components: a Spatial Pooler that creates sparse distributed 
/// representations of inputs, and a Temporal Memory that learns sequences of these representations.
/// </para>
/// <para><b>For Beginners:</b> HTM is a special type of neural network inspired by how the human brain works.
/// 
/// Think of HTM like a system that learns patterns over time:
/// - It's similar to how your brain recognizes songs or predicts what comes next in a familiar sequence
/// - It's especially good at learning from time-series data (information that changes over time)
/// - It can recognize patterns even when they contain noise or slight variations
/// 
/// HTM networks have two main parts:
/// - Spatial Pooler: This converts incoming data into a special format that highlights important patterns
///   (like how your brain might focus on the melody of a song rather than background noise)
/// - Temporal Memory: This learns sequences and can predict what might come next
///   (like how you can anticipate the next note in a familiar song)
/// 
/// HTM is particularly useful for:
/// - Anomaly detection (finding unusual patterns)
/// - Sequence prediction (guessing what comes next)
/// - Pattern recognition in noisy data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HTMNetwork<T> : NeuralNetworkBase<T>
{
    private readonly HTMNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the size of the input vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input size defines the dimensionality of the input vectors that the HTM network can process.
    /// This value is determined by the input shape specified in the network architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many data points the network receives at once.
    /// 
    /// For example:
    /// - If you're analyzing a stock price, this might be 1 (just the price)
    /// - If you're analyzing weather, this might be several values (temperature, humidity, pressure, etc.)
    /// - If you're processing an image, this would be the total number of pixels
    /// 
    /// Think of this as the number of "sensors" your HTM network has to detect information from the world.
    /// </para>
    /// </remarks>
    private int _inputSize { get; }

    /// <summary>
    /// Gets the number of columns in the spatial pooler.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The column count determines the number of columns in the spatial pooler component of the HTM network.
    /// Each column represents a group of cells that share the same feedforward receptive field. A higher column count 
    /// increases the representational capacity of the network but also increases computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the number of "pattern detectors" in the network.
    /// 
    /// Columns in HTM are groups of cells that work together to recognize specific patterns in the input.
    /// 
    /// Think of each column as a specialist that learns to recognize particular features:
    /// - Some columns might learn to recognize rising trends
    /// - Others might learn to recognize sudden changes
    /// - Others might learn to recognize specific repeated patterns
    /// 
    /// More columns mean the network can recognize more different patterns, but requires more
    /// computational power. The default value of 2048 works well for many applications.
    /// </para>
    /// </remarks>
    private int _columnCount { get; set; }

    /// <summary>
    /// Gets the number of cells per column in the temporal memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property specifies how many cells are in each column within the temporal memory component.
    /// Each cell can learn different temporal contexts for the same spatial input pattern. More cells per column
    /// allow the network to disambiguate between different sequences that share common elements.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how well the network can understand context in sequences.
    /// 
    /// In HTM, each column contains multiple cells. These cells help the network understand
    /// that the same input can mean different things depending on what came before it.
    /// 
    /// For example, in English the word "bank" means different things in these sequences:
    /// - "I'll deposit money at the bank"
    /// - "I'll sit on the river bank"
    /// 
    /// Having multiple cells per column helps the network learn these different contexts.
    /// More cells (the default is 32) let the network handle more complex sequences
    /// where the same pattern can have different meanings based on context.
    /// </para>
    /// </remarks>
    private int _cellsPerColumn { get; set; }

    /// <summary>
    /// Gets the target sparsity for the spatial pooler output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sparsity threshold defines the target percentage of active columns in the spatial pooler output.
    /// A typical value is around 2% (0.02), which means that roughly 2% of columns will be active at any given time.
    /// Sparse representations are a key feature of HTM networks, as they enable efficient learning and generalization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how "selective" the network is about what it considers important.
    /// 
    /// Sparsity is a fundamental concept in HTM:
    /// - It means that only a small percentage of neurons are active at any time
    /// - In the human brain, typically only about 2% of neurons are active at once
    /// - This default value (0.02 or 2%) mimics this biological principle
    /// 
    /// Low sparsity (like 2%) helps the network:
    /// - Focus on the most important patterns
    /// - Generalize better to new, similar examples
    /// - Use memory and processing power efficiently
    /// - Avoid getting confused by noise
    /// 
    /// Think of it like highlighting only the most important words in a textbook rather than
    /// everything - it helps you focus on what really matters.
    /// </para>
    /// </remarks>
    private double _sparsityThreshold { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="HTMNetwork{T}"/> class with the specified architecture and parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the network.</param>
    /// <param name="columnCount">The number of columns in the spatial pooler. Default is 2048.</param>
    /// <param name="cellsPerColumn">The number of cells per column in the temporal memory. Default is 32.</param>
    /// <param name="sparsityThreshold">The target sparsity for the spatial pooler output. Default is 0.02 (2%).</param>
    /// <exception cref="InvalidOperationException">Thrown when the input shape is not specified in the architecture.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates an HTM network with the specified architecture and parameters. It initializes
    /// the spatial pooler and temporal memory components based on the provided configuration values.
    /// The default parameters are based on common values used in HTM research and applications.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new HTM network with your chosen settings.
    /// 
    /// When creating an HTM network, you can customize several important aspects:
    /// 
    /// 1. Architecture: The basic structure of your network
    /// 
    /// 2. Column Count: How many pattern detectors (default is 2048)
    ///    - More columns = more capacity to learn different patterns
    ///    - Fewer columns = faster but less capable
    /// 
    /// 3. Cells Per Column: How much context the network can understand (default is 32)
    ///    - More cells = better at understanding complex sequences
    ///    - Fewer cells = simpler but may confuse similar sequences
    /// 
    /// 4. Sparsity Threshold: How selective the network is (default is 0.02 or 2%)
    ///    - Lower values = more selective, focuses on fewer elements
    ///    - Higher values = less selective, considers more elements important
    /// 
    /// The default values work well for many applications, but you can adjust them
    /// based on your specific needs.
    /// </para>
    /// </remarks>
    public HTMNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int columnCount = 2048,
        int cellsPerColumn = 32,
        double sparsityThreshold = 0.02,
        ILossFunction<T>? lossFunction = null,
        HTMNetworkOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new HTMNetworkOptions();
        Options = _options;
        var inputShape = Architecture.GetInputShape();
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for HTM network.");
        }

        _inputSize = inputShape[0];
        _columnCount = columnCount;
        _cellsPerColumn = cellsPerColumn;
        _sparsityThreshold = sparsityThreshold;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the HTM network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the HTM network. If the architecture provides specific layers,
    /// those are used directly. Otherwise, default layers appropriate for an HTM network are created,
    /// including a spatial pooler layer and a temporal memory layer configured with the specified parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your HTM network.
    /// 
    /// When initializing the network:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates standard HTM layers automatically
    /// 
    /// The standard HTM layers typically include:
    /// 1. Spatial Pooler: Converts input to sparse distributed representations
    /// 2. Temporal Memory: Learns sequences and makes predictions
    /// 3. Optional additional layers: May include encoder/decoder layers or other processing
    /// 
    /// This process is like assembling all the components before the network starts learning.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultHTMLayers(Architecture, _columnCount, _cellsPerColumn, _sparsityThreshold));
        }
    }

    /// <summary>
    /// Processes an input vector through the network and updates the network's internal state based on the input.
    /// </summary>
    /// <param name="input">The input vector to learn from.</param>
    /// <exception cref="ArgumentException">Thrown when the input size doesn't match the expected size.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't contain the expected layer types.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the learning process in an HTM network. It first propagates the input through
    /// the spatial pooler and temporal memory layers. Then it performs learning operations on both components,
    /// allowing the network to adapt to the input sequence. The temporal memory layer's state is updated to
    /// maintain the sequence context for the next input.
    /// </para>
    /// <para><b>For Beginners:</b> This method both predicts AND learns from new data.
    /// 
    /// When you call this method with new input data:
    /// 
    /// 1. The network first processes the input through the Spatial Pooler
    ///    - Converts your input to a sparse representation
    ///    - Highlights the most important patterns
    /// 
    /// 2. Then it processes the result through the Temporal Memory
    ///    - Makes predictions based on what it has learned so far
    ///    - Creates a representation of the current state
    /// 
    /// 3. The Temporal Memory learns from this new information
    ///    - Updates its understanding of sequences
    ///    - Adjusts its predictions for the future
    /// 
    /// 4. The network stores the current state to use as context for the next input
    ///    - This is how it maintains an understanding of sequences over time
    /// 
    /// 5. The Spatial Pooler also learns from the input
    ///    - Improves its ability to identify important patterns
    ///    - Adapts to changing input statistics
    /// 
    /// This continuous learning process allows the network to improve over time as it sees more data.
    /// </para>
    /// </remarks>
    public void Learn(Vector<T> input)
    {
        if (input.Length != _inputSize)
            throw new ArgumentException($"Input size mismatch. Expected {_inputSize}, got {input.Length}.");

        // Forward pass through Spatial Pooler
        if (!(Layers[0] is SpatialPoolerLayer<T> spatialPoolerLayer))
            throw new InvalidOperationException("The first layer is not a SpatialPoolerLayer.");
        var spatialPoolerOutput = spatialPoolerLayer.Forward(Tensor<T>.FromVector(input)).ToVector();

        // Forward pass through Temporal Memory
        if (!(Layers[1] is TemporalMemoryLayer<T> temporalMemoryLayer))
            throw new InvalidOperationException("The second layer is not a TemporalMemoryLayer.");
        var temporalMemoryOutput = temporalMemoryLayer.Forward(Tensor<T>.FromVector(spatialPoolerOutput)).ToVector();

        // Learning in Temporal Memory
        temporalMemoryLayer.Learn(spatialPoolerOutput, temporalMemoryLayer.PreviousState);

        // Update the previous state for the next iteration
        temporalMemoryLayer.PreviousState = temporalMemoryOutput;

        // Learning in Spatial Pooler
        spatialPoolerLayer.Learn(input);

        // Forward pass through remaining layers
        var current = temporalMemoryOutput;
        for (int i = 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
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
    /// For HTM networks, this is typically used for fine-tuning parameters after initial learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the configuration values in the network.
    /// 
    /// While HTM networks primarily learn through the specific Learning methods in each layer,
    /// this method allows you to directly update the parameters of all layers at once:
    /// 
    /// 1. It takes a long list of parameter values
    /// 2. Determines which values belong to which layers
    /// 3. Updates each layer with its corresponding values
    /// 
    /// This might be used for:
    /// - Fine-tuning a network after initial training
    /// - Applying optimized parameters found through experimentation
    /// - Resetting certain aspects of the network
    /// 
    /// This method is less commonly used in HTM networks compared to other types of neural networks,
    /// as HTMs typically learn through their specialized learning mechanisms.
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
    /// Makes a prediction using the current state of the HTM network.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>A tensor containing the network's prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through the HTM network without updating its internal state.
    /// It returns a prediction tensor that typically includes information about active columns,
    /// predicted columns, and anomaly detection results.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the network's current knowledge to make a prediction.
    /// 
    /// Unlike the Learn method, Predict:
    /// - Does not update the network's memory or knowledge
    /// - Processes the input and returns a result based on what it has learned so far
    /// - Can be used to test the network's predictions on new data
    /// 
    /// The output typically contains information about:
    /// - Which patterns were activated by the input
    /// - Whether the network considers this input unusual (anomalous)
    /// - The network's prediction of what might come next
    /// </para>
    /// </remarks>
    private Vector<T> Predict(Vector<T> input)
    {
        // Validate input
        if (input.Length != _inputSize)
            throw new ArgumentException($"Input size mismatch. Expected {_inputSize}, got {input.Length}.");

        // Flag to indicate we're in prediction mode (not learning)
        bool originalTrainingMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            // Forward pass through the Spatial Pooler
            if (!(Layers[0] is SpatialPoolerLayer<T> spatialPoolerLayer))
                throw new InvalidOperationException("The first layer is not a SpatialPoolerLayer.");

            var spatialPoolerOutput = spatialPoolerLayer.Forward(Tensor<T>.FromVector(input)).ToVector();

            // Forward pass through the Temporal Memory
            if (!(Layers[1] is TemporalMemoryLayer<T> temporalMemoryLayer))
                throw new InvalidOperationException("The second layer is not a TemporalMemoryLayer.");

            // Get predictions before processing input (what the network expected)
            var predictedCells = temporalMemoryLayer.GetPredictions();

            // Process input to get current state
            var temporalMemoryOutput = temporalMemoryLayer.Forward(Tensor<T>.FromVector(spatialPoolerOutput)).ToVector();

            // Calculate anomaly score (if supported)
            T anomalyScore = NumOps.Zero;
            if (Layers.Count > 2 && Layers[2] is AnomalyDetectorLayer<T> anomalyLayer)
            {
                var anomalyOutput = anomalyLayer.Forward(Tensor<T>.FromVector(temporalMemoryOutput)).ToVector();
                if (anomalyOutput.Length > 0)
                {
                    anomalyScore = anomalyOutput[0];
                }
            }

            // Create output vector with prediction results
            // Format depends on the specific implementation, but might include:
            // - Active columns
            // - Predicted columns
            // - Anomaly score

            // Simple example format:
            // [anomaly_score, active_columns..., predicted_columns...]
            int outputSize = 1 + _columnCount * 2;
            var result = new Vector<T>(outputSize);

            // Set anomaly score
            result[0] = anomalyScore;

            // Set active columns (from spatial pooler output)
            for (int i = 0; i < _columnCount && i < spatialPoolerOutput.Length; i++)
            {
                result[1 + i] = spatialPoolerOutput[i];
            }

            // Set predicted columns (if available)
            if (predictedCells != null && predictedCells.Length >= _columnCount)
            {
                for (int i = 0; i < _columnCount; i++)
                {
                    result[1 + _columnCount + i] = predictedCells[i];
                }
            }

            return result;
        }
        finally
        {
            // Restore original training mode
            SetTrainingMode(originalTrainingMode);
        }
    }

    /// <summary>
    /// Makes a prediction using the HTM network.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>A tensor containing the network's prediction.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Ensure the input is a vector or can be converted to one
        Vector<T> inputVector;
        if (input.Rank == 1)
        {
            inputVector = input.ToVector();
        }
        else if (input.Rank == 2 && input.Shape[0] == 1)
        {
            // Handle single-row batch
            inputVector = new Vector<T>(input.Shape[1]);
            for (int i = 0; i < input.Shape[1]; i++)
            {
                inputVector[i] = input[0, i];
            }
        }
        else
        {
            throw new ArgumentException("Input tensor must be a vector or a single-row batch.");
        }

        // Use the vector prediction method
        var predictionVector = Predict(inputVector);

        // Convert back to tensor
        return Tensor<T>.FromVector(predictionVector);
    }

    /// <summary>
    /// Trains the HTM network on a sequence of inputs.
    /// </summary>
    /// <param name="input">The input tensor, which may contain a batch of inputs or a single input.</param>
    /// <param name="expectedOutput">Not used in HTM networks as they are self-supervised.</param>
    /// <remarks>
    /// <para>
    /// HTM networks are self-supervised and learn to predict future inputs based on past patterns.
    /// This method processes inputs sequentially through the network, allowing it to build a model
    /// of the temporal patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the network to recognize patterns in sequential data.
    /// 
    /// Unlike traditional neural networks, HTMs:
    /// - Don't need labels or expected outputs
    /// - Learn by trying to predict what comes next in a sequence
    /// - Build their own understanding of the patterns in your data
    /// 
    /// This method processes each input in sequence, helping the network learn
    /// temporal patterns and relationships in your data over time.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // HTM networks don't use supervised learning with expected outputs
        // They learn by observing sequences of inputs

        // Track total anomaly score for batch processing
        T totalAnomalyScore = NumOps.Zero;
        int processedInputs = 0;

        // Process input based on its shape
        if (input.Rank == 1)
        {
            // Single input vector
            Learn(input.ToVector());

            // Calculate anomaly score
            totalAnomalyScore = CalculateAnomalyScore();
            processedInputs = 1;
        }
        else if (input.Rank == 2)
        {
            // Batch of inputs - process them sequentially
            int batchSize = input.Shape[0];

            for (int i = 0; i < batchSize; i++)
            {
                // Extract the i-th input from the batch
                var singleInput = new Vector<T>(input.Shape[1]);
                for (int j = 0; j < input.Shape[1]; j++)
                {
                    singleInput[j] = input[i, j];
                }

                // Learn from this input
                Learn(singleInput);

                // Add to total anomaly score
                totalAnomalyScore = NumOps.Add(totalAnomalyScore, CalculateAnomalyScore());
                processedInputs++;
            }
        }
        else
        {
            throw new ArgumentException("Input tensor must be either a vector or a batch of vectors.");
        }

        // Calculate average anomaly score for all processed inputs
        if (processedInputs > 0)
        {
            LastLoss = NumOps.Divide(totalAnomalyScore, NumOps.FromDouble(processedInputs));
        }
    }

    /// <summary>
    /// Calculates the current anomaly score based on the network's state.
    /// </summary>
    /// <returns>An anomaly score where higher values indicate greater prediction error.</returns>
    private T CalculateAnomalyScore()
    {
        T anomalyScore = NumOps.Zero;

        // Check if we have an anomaly detector layer
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is AnomalyDetectorLayer<T> anomalyLayer)
            {
                // Get the current output from the previous layer
                Vector<T> currentState;
                if (i > 0 && Layers[i - 1] is TemporalMemoryLayer<T> temporalMemoryLayer &&
                    temporalMemoryLayer.PreviousState != null)
                {
                    currentState = temporalMemoryLayer.PreviousState;
                    var anomalyOutput = anomalyLayer.Forward(Tensor<T>.FromVector(currentState)).ToVector();
                    if (anomalyOutput.Length > 0)
                    {
                        anomalyScore = anomalyOutput[0];
                        return anomalyScore;
                    }
                }
            }
        }

        // If no anomaly detector layer, calculate a simple prediction error
        if (NumOps.Equals(anomalyScore, NumOps.Zero) && Layers.Count >= 2 &&
            Layers[1] is TemporalMemoryLayer<T> temporalMemoryLayer2)
        {
            // Get the predicted cells from before this input was processed
            var predictedCells = temporalMemoryLayer2.GetPredictions();

            // If we have predictions, calculate a simple match score
            if (predictedCells != null && predictedCells.Length > 0 &&
                temporalMemoryLayer2.PreviousState != null)
            {
                // Calculate overlap between prediction and actual
                int matches = 0;
                int total = 0;

                for (int i = 0; i < Math.Min(predictedCells.Length, temporalMemoryLayer2.PreviousState.Length); i++)
                {
                    if (!NumOps.Equals(predictedCells[i], NumOps.Zero) &&
                        !NumOps.Equals(temporalMemoryLayer2.PreviousState[i], NumOps.Zero))
                    {
                        matches++;
                    }
                    if (!NumOps.Equals(predictedCells[i], NumOps.Zero) ||
                        !NumOps.Equals(temporalMemoryLayer2.PreviousState[i], NumOps.Zero))
                    {
                        total++;
                    }
                }

                // Calculate error rate (1.0 - match rate)
                if (total > 0)
                {
                    double matchRate = (double)matches / total;
                    anomalyScore = NumOps.FromDouble(1.0 - matchRate);
                }
            }
        }

        return anomalyScore;
    }

    /// <summary>
    /// Gets metadata about the HTM network.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the HTM network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the HTM network, including its architecture,
    /// layer configuration, and HTM-specific parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your HTM network.
    /// 
    /// The metadata includes:
    /// - What kind of network this is (HTM)
    /// - Details about its structure (columns, cells, etc.)
    /// - Configuration settings
    /// - Basic statistics about the network
    /// 
    /// This is useful for documentation, comparison with other models,
    /// and keeping track of different network configurations.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.HTMNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputSize", _inputSize },
                { "ColumnCount", _columnCount },
                { "CellsPerColumn", _cellsPerColumn },
                { "SparsityThreshold", _sparsityThreshold },
                { "LayerCount", Layers.Count },
                { "TotalParameters", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes HTM-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write HTM-specific parameters
        writer.Write(_columnCount);
        writer.Write(_cellsPerColumn);
        writer.Write(_sparsityThreshold);

        // Serialize any additional HTM state

        // Look for the temporal memory layer
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is TemporalMemoryLayer<T> temporalMemoryLayer)
            {
                // Mark that we found a temporal memory layer
                writer.Write(true);

                // Serialize the temporal memory's state
                if (temporalMemoryLayer.PreviousState != null)
                {
                    writer.Write(true); // Has previous state
                    writer.Write(temporalMemoryLayer.PreviousState.Length);

                    // Write each element of the previous state
                    for (int j = 0; j < temporalMemoryLayer.PreviousState.Length; j++)
                    {
                        writer.Write(Convert.ToDouble(temporalMemoryLayer.PreviousState[j]));
                    }
                }
                else
                {
                    writer.Write(false); // No previous state
                }

                break; // Stop after finding the first temporal memory layer
            }
        }

        // If no temporal memory layer was found
        writer.Write(false);
    }

    /// <summary>
    /// Deserializes HTM-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read HTM-specific parameters
        _columnCount = reader.ReadInt32();
        _cellsPerColumn = reader.ReadInt32();
        _sparsityThreshold = reader.ReadDouble();

        // Deserialize additional HTM state

        // Check if there was a temporal memory layer
        bool hasTemporalMemoryLayer = reader.ReadBoolean();
        if (hasTemporalMemoryLayer)
        {
            // Look for the temporal memory layer in the current network
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i] is TemporalMemoryLayer<T> temporalMemoryLayer)
                {
                    // Check if there was a previous state
                    bool hasPreviousState = reader.ReadBoolean();
                    if (hasPreviousState)
                    {
                        int stateLength = reader.ReadInt32();
                        var previousState = new Vector<T>(stateLength);

                        // Read each element of the previous state
                        for (int j = 0; j < stateLength; j++)
                        {
                            previousState[j] = NumOps.FromDouble(reader.ReadDouble());
                        }

                        // Set the previous state
                        temporalMemoryLayer.PreviousState = previousState;
                    }

                    break; // Stop after restoring the first temporal memory layer
                }
            }
        }
    }

    /// <summary>
    /// Creates a new instance of the HTM Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new HTM Network instance with the same architecture and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the HTM Network with the same architecture and HTM-specific
    /// parameters as the current instance. It's used in scenarios where a fresh copy of the model is needed
    /// while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the HTM network with the same setup.
    /// 
    /// Think of it like creating a clone of the network:
    /// - The new network has the same architecture (structure)
    /// - It has the same number of columns, cells per column, and sparsity threshold
    /// - But it's a completely separate instance with its own state
    /// - It starts with clean internal memory and connections
    /// 
    /// This is useful when you want to:
    /// - Train the same network design on different datasets
    /// - Compare how the same network structure learns from different sequences
    /// - Start with a fresh network that has the same configuration but no learned patterns
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new HTMNetwork<T>(
            this.Architecture,
            _columnCount,
            _cellsPerColumn,
            _sparsityThreshold);
    }
}
