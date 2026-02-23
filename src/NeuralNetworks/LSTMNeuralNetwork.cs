using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Long Short-Term Memory (LSTM) Neural Network, which is specialized for processing
/// sequential data like text, time series, or audio.
/// </summary>
/// <remarks>
/// <para>
/// Long Short-Term Memory networks are a special kind of recurrent neural network designed to overcome
/// the vanishing gradient problem that traditional RNNs face. LSTMs have a complex internal structure 
/// with specialized "gates" that regulate the flow of information, allowing them to remember patterns
/// over long sequences and selectively forget irrelevant information.
/// </para>
/// <para><b>For Beginners:</b> An LSTM Neural Network is a special type of neural network designed for understanding sequences and patterns that unfold over time.
/// 
/// Think of an LSTM like a smart notepad that can:
/// - Remember important information for long periods
/// - Forget irrelevant details
/// - Update its notes with new information
/// - Decide what parts of its memory to use for making predictions
/// 
/// For example, when processing a sentence like "The clouds are in the sky", an LSTM can:
/// - Remember "The clouds" as the subject even after seeing several more words
/// - Understand that "are" should agree with the plural "clouds" 
/// - Predict that "sky" might come after "in the" because clouds are typically in the sky
/// 
/// LSTMs are particularly good at:
/// - Text generation and language modeling
/// - Speech recognition
/// - Time series prediction (like stock prices or weather)
/// - Translation between languages
/// - Any task where the order of data matters and patterns may span across long sequences
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
public class LSTMNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private readonly LSTMOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The activation function to apply to cell state outputs. Default is tanh.
    /// </summary>
    private IActivationFunction<T>? ScalarActivation { get; }

    /// <summary>
    /// The vector activation function to apply to cell state outputs.
    /// </summary>
    private IVectorActivationFunction<T>? VectorActivation { get; }

    /// <summary>
    /// The activation function to apply to the forget gate. Default is sigmoid.
    /// </summary>
    private IActivationFunction<T>? ForgetGateActivation { get; }

    /// <summary>
    /// The activation function to apply to the input gate. Default is sigmoid.
    /// </summary>
    private IActivationFunction<T>? InputGateActivation { get; }

    /// <summary>
    /// The activation function to apply to the cell gate. Default is tanh.
    /// </summary>
    private IActivationFunction<T>? CellGateActivation { get; }

    /// <summary>
    /// The activation function to apply to the output gate. Default is sigmoid.
    /// </summary>
    private IActivationFunction<T>? OutputGateActivation { get; }

    /// <summary>
    /// The activation function to apply to the forget gate. Default is sigmoid.
    /// </summary>
    private IVectorActivationFunction<T>? ForgetGateVectorActivation { get; }

    /// <summary>
    /// The activation function to apply to the input gate. Default is sigmoid.
    /// </summary>
    private IVectorActivationFunction<T>? InputGateVectorActivation { get; }

    /// <summary>
    /// The activation function to apply to the cell gate. Default is tanh.
    /// </summary>
    private IVectorActivationFunction<T>? CellGateVectorActivation { get; }

    /// <summary>
    /// The activation function to apply to the output gate. Default is sigmoid.
    /// </summary>
    private IVectorActivationFunction<T>? OutputGateVectorActivation { get; }

    /// <summary>
    /// Creates a new LSTM Neural Network with customizable loss and activation functions,
    /// using scalar activation functions.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how the network is structured.
    /// </param>
    /// <param name="lossFunction">
    /// The loss function to use for training the network. If null, Mean Squared Error will be used.
    /// </param>
    /// <param name="outputActivation">
    /// The scalar activation function to apply to LSTM cell state outputs. If null, a hyperbolic tangent (tanh)
    /// activation will be used.
    /// </param>
    /// <param name="forgetGateActivation">
    /// The activation function to apply to the forget gate. If null, sigmoid will be used.
    /// </param>
    /// <param name="inputGateActivation">
    /// The activation function to apply to the input gate. If null, sigmoid will be used.
    /// </param>
    /// <param name="cellGateActivation">
    /// The activation function to apply to the cell gate. If null, tanh will be used.
    /// </param>
    /// <param name="outputGateActivation">
    /// The activation function to apply to the output gate. If null, sigmoid will be used.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor allows full customization of the LSTM network's activation functions and loss function.
    /// Each gate in the LSTM cell can have a different activation function, allowing for experimentation
    /// with novel LSTM architectures.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor gives you complete control over your LSTM network.
    /// 
    /// You can customize:
    /// - The loss function (how the network measures errors)
    /// - The output activation (how cell states are transformed to outputs)
    /// - Each gate's activation function:
    ///   * Forget gate: Controls what information to discard from the cell state
    ///   * Input gate: Controls what new information to store in the cell state
    ///   * Cell gate: Creates candidate values that could be added to the state
    ///   * Output gate: Controls what parts of the cell state to output
    /// 
    /// This level of customization is useful for advanced users experimenting with
    /// different LSTM variants or optimizing for specific tasks.
    /// </para>
    /// </remarks>
    public LSTMNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? outputActivation = null,
        IActivationFunction<T>? forgetGateActivation = null,
        IActivationFunction<T>? inputGateActivation = null,
        IActivationFunction<T>? cellGateActivation = null,
        IActivationFunction<T>? outputGateActivation = null,
        LSTMOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new LSTMOptions();
        Options = _options;

        // Set activation functions (or defaults)
        ScalarActivation = outputActivation ?? new TanhActivation<T>();

        // Set gate activation functions (or defaults)
        ForgetGateActivation = forgetGateActivation ?? new SigmoidActivation<T>();
        InputGateActivation = inputGateActivation ?? new SigmoidActivation<T>();
        CellGateActivation = cellGateActivation ?? new TanhActivation<T>();
        OutputGateActivation = outputGateActivation ?? new SigmoidActivation<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new LSTM Neural Network with customizable loss and activation functions,
    /// using vector activation functions.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how the network is structured.
    /// </param>
    /// <param name="lossFunction">
    /// The loss function to use for training the network. If null, Mean Squared Error will be used.
    /// </param>
    /// <param name="outputVectorActivation">
    /// The vector activation function to apply to LSTM cell state outputs. If null, a hyperbolic tangent (tanh)
    /// activation will be used.
    /// </param>
    /// <param name="forgetGateVectorActivation">
    /// The activation function to apply to the forget gate. If null, sigmoid will be used.
    /// </param>
    /// <param name="inputGateVectorActivation">
    /// The activation function to apply to the input gate. If null, sigmoid will be used.
    /// </param>
    /// <param name="cellGateVectorActivation">
    /// The activation function to apply to the cell gate. If null, tanh will be used.
    /// </param>
    /// <param name="outputGateVectorActivation">
    /// The activation function to apply to the output gate. If null, sigmoid will be used.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor allows full customization of the LSTM network's activation functions and loss function.
    /// Each gate in the LSTM cell can have a different activation function, allowing for experimentation
    /// with novel LSTM architectures.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor gives you complete control over your LSTM network.
    /// 
    /// You can customize:
    /// - The loss function (how the network measures errors)
    /// - The output activation (how cell states are transformed to outputs)
    /// - Each gate's activation function:
    ///   * Forget gate: Controls what information to discard from the cell state
    ///   * Input gate: Controls what new information to store in the cell state
    ///   * Cell gate: Creates candidate values that could be added to the state
    ///   * Output gate: Controls what parts of the cell state to output
    /// 
    /// This level of customization is useful for advanced users experimenting with
    /// different LSTM variants or optimizing for specific tasks.
    /// </para>
    /// </remarks>
    public LSTMNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? outputVectorActivation = null,
        IVectorActivationFunction<T>? forgetGateVectorActivation = null,
        IVectorActivationFunction<T>? inputGateVectorActivation = null,
        IVectorActivationFunction<T>? cellGateVectorActivation = null,
        IVectorActivationFunction<T>? outputGateVectorActivation = null,
        LSTMOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new LSTMOptions();
        Options = _options;

        // Set activation functions (or defaults)
        VectorActivation = outputVectorActivation ?? new TanhActivation<T>();

        // Set gate activation functions (or defaults)
        ForgetGateVectorActivation = forgetGateVectorActivation ?? new SigmoidActivation<T>();
        InputGateVectorActivation = inputGateVectorActivation ?? new SigmoidActivation<T>();
        CellGateVectorActivation = cellGateVectorActivation ?? new TanhActivation<T>();
        OutputGateVectorActivation = outputGateVectorActivation ?? new SigmoidActivation<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Sets up the layers of the LSTM network based on the provided architecture.
    /// If no layers are specified in the architecture, default LSTM layers will be created.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the network layers according to the provided architecture. If the architecture
    /// includes a specific set of layers, those are used directly. Otherwise, the method creates a default
    /// LSTM layer configuration, which typically includes embeddings (for text data), one or more LSTM layers,
    /// and appropriate output layers based on the task type.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of your LSTM network.
    /// 
    /// When initializing the network:
    /// - If you provided specific layers in the architecture, those are used
    /// - If not, the network creates standard LSTM layers automatically
    /// 
    /// The standard LSTM setup typically includes:
    /// 1. Input processing layers (like embedding layers for text)
    /// 2. One or more LSTM layers that process the sequence
    /// 3. Output layers that produce the final prediction
    /// 
    /// This is like assembling all the components of your network before training begins.
    /// Each layer has a specific role in processing your sequential data.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultLSTMNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the internal parameters (weights and biases) of the network with new values.
    /// This is typically used after training to apply optimized parameters.
    /// </summary>
    /// <param name="parameters">
    /// A vector containing all parameters for all layers in the network.
    /// The parameters must be in the correct order matching the network's layer structure.
    /// </param>
    /// <remarks>
    /// <para>
    /// This method distributes a vector of parameters to the appropriate layers in the network. It determines
    /// how many parameters each layer needs, extracts the corresponding segment from the input parameter vector,
    /// and updates each layer with its respective parameters. This is commonly used after optimization algorithms
    /// have calculated improved weights for the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learned values in the network.
    /// 
    /// During training, an LSTM network learns many values (called parameters) that determine
    /// how it processes information. These include:
    /// - Weights that control how inputs affect the network
    /// - Gate parameters that control what information to remember or forget
    /// - Output parameters that determine how predictions are made
    /// 
    /// This method:
    /// 1. Takes a long list of all these parameters
    /// 2. Figures out which parameters belong to which layers
    /// 3. Updates each layer with its corresponding parameters
    /// 
    /// Think of it like updating the settings on different parts of a machine
    /// based on what it has learned through experience.
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
    /// Processes input through the LSTM network to generate predictions.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through the LSTM network.</returns>
    /// <remarks>
    /// <para>
    /// This method implements a forward pass through the LSTM network. It handles both single inputs
    /// and batched sequences, processing the data through each layer while managing the LSTM's
    /// internal state. For sequential data, it processes the input step by step while carrying the
    /// hidden state across time steps.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the LSTM network to make predictions.
    /// 
    /// The prediction process works like this:
    /// 1. Data enters the network (like a sequence of words or time series data)
    /// 2. Each LSTM layer processes the sequence step by step, maintaining internal state
    /// 3. The network remembers important information while processing the sequence
    /// 4. Finally, the output layers convert the LSTM's final state into the desired output format
    /// 
    /// Unlike standard neural networks, LSTMs can "remember" information from earlier in the sequence
    /// when making predictions later in the sequence, which is crucial for tasks like text understanding
    /// or time series forecasting.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Set to inference mode
        SetTrainingMode(false);

        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Simple layer-by-layer forward pass
        // Each layer (including LSTM layers) handles its own time-stepping internally
        // This matches the industry-standard approach used by GRU networks
        var current = input;

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Processes a batched sequence through the LSTM network.
    /// </summary>
    /// <param name="input">The input tensor with shape [batch_size, sequence_length, features].</param>
    /// <returns>The output tensor after processing.</returns>
    private Tensor<T> ProcessBatchedSequence(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape[1];

        // Initialize hidden states and cell states for all LSTM layers
        var states = InitializeStates(batchSize);

        // Create storage for outputs at each time step
        var outputs = new List<Tensor<T>>(sequenceLength);

        // Process each time step
        for (int t = 0; t < sequenceLength; t++)
        {
            // Extract the current time step for all batches
            var timeStep = ExtractTimeStep(input, t);

            // Process this time step through all layers
            var result = ProcessTimeStep(timeStep, states);

            // Store output
            outputs.Add(result);
        }

        // Stack outputs along time dimension if they should be returned as a sequence
        // or return just the final output if only the final result is needed
        if (Architecture.ShouldReturnFullSequence)
        {
            return StackAlongTimeDimension(outputs);
        }
        else
        {
            return outputs[outputs.Count - 1];
        }
    }

    /// <summary>
    /// Processes a single sequence through the LSTM network.
    /// </summary>
    /// <param name="input">The input tensor with shape [sequence_length, features].</param>
    /// <returns>The output tensor after processing.</returns>
    private Tensor<T> ProcessSequence(Tensor<T> input)
    {
        // Add batch dimension of size 1
        var batchedInput = AddBatchDimension(input);

        // Process as a batched sequence
        var result = ProcessBatchedSequence(batchedInput);

        // Remove batch dimension if it was added
        if (result.Shape.Length > 2)
        {
            return RemoveBatchDimension(result);
        }
        else
        {
            return result;
        }
    }

    /// <summary>
    /// Processes a batch of single time step inputs.
    /// </summary>
    /// <param name="input">The input tensor with shape [batch_size, features].</param>
    /// <returns>The output tensor after processing.</returns>
    private Tensor<T> ProcessSingleTimeStepBatch(Tensor<T> input)
    {
        // Add time dimension of size 1
        var sequenceInput = AddTimeDimension(input);

        // Process as a batched sequence
        var result = ProcessBatchedSequence(sequenceInput);

        // Remove time dimension if necessary
        if (result.Shape.Length > 2)
        {
            return RemoveTimeDimension(result);
        }
        else
        {
            return result;
        }
    }

    /// <summary>
    /// Processes a single input sample through the LSTM network.
    /// </summary>
    /// <param name="input">The input tensor with shape [features].</param>
    /// <returns>The output tensor after processing.</returns>
    private Tensor<T> ProcessSingleSample(Tensor<T> input)
    {
        // Add batch and time dimensions (both of size 1)
        var batchedSequenceInput = AddBatchAndTimeDimensions(input);

        // Process as a batched sequence
        var result = ProcessBatchedSequence(batchedSequenceInput);

        // Remove batch and time dimensions
        return RemoveBatchAndTimeDimensions(result);
    }

    /// <summary>
    /// Initializes LSTM states (hidden state and cell state) for all layers.
    /// </summary>
    /// <param name="batchSize">The batch size to initialize states for.</param>
    /// <returns>Dictionary mapping layer indices to their states.</returns>
    private Dictionary<int, (Tensor<T> h, Tensor<T> c)> InitializeStates(int batchSize)
    {
        var states = new Dictionary<int, (Tensor<T> h, Tensor<T> c)>();

        // Count LSTM layers
        int lstmLayers = CountLSTMLayers();

        // Get hidden size from architecture
        int hiddenSize = GetLSTMHiddenSize();

        // Initialize states for each LSTM layer
        int lstmIndex = 0;
        for (int i = 0; i < Layers.Count; i++)
        {
            if (IsLSTMLayer(Layers[i]))
            {
                // Create zero tensors for hidden state and cell state
                // Shape: [batch_size, hidden_size]
                var hiddenState = new Tensor<T>(new int[] { batchSize, hiddenSize });
                var cellState = new Tensor<T>(new int[] { batchSize, hiddenSize });

                // Initialize with zeros
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        hiddenState[b, h] = NumOps.Zero;
                        cellState[b, h] = NumOps.Zero;
                    }
                }

                states[i] = (hiddenState, cellState);
                lstmIndex++;
            }
        }

        return states;
    }

    /// <summary>
    /// Counts the number of LSTM layers in the network.
    /// </summary>
    /// <returns>The count of LSTM layers.</returns>
    private int CountLSTMLayers()
    {
        int count = 0;

        foreach (var layer in Layers)
        {
            if (IsLSTMLayer(layer))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Determines if a layer is an LSTM layer or any other type of recurrent layer.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer is a recurrent layer; otherwise, false.</returns>
    private bool IsLSTMLayer(ILayer<T> layer)
    {
        // Check for specific layer type implementations
        if (layer is LSTMLayer<T> || layer is RecurrentLayer<T> || layer is GRULayer<T>)
        {
            return true;
        }

        // Check for custom layer types that might include LSTM in their name
        string layerType = layer.GetType().Name;
        return layerType.Contains("LSTM") ||
               layerType.Contains("Recurrent") ||
               layerType.Contains("GRU");
    }

    /// <summary>
    /// Gets the hidden size for LSTM layers from the architecture or layer configuration.
    /// </summary>
    /// <returns>The hidden size value.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no recurrent layers are found in the network or when hidden size cannot be determined.
    /// </exception>
    private int GetLSTMHiddenSize()
    {
        var metadata = GetModelMetadata();
        // First, try to get the hidden size from the architecture
        if (metadata.AdditionalInfo != null &&
            metadata.AdditionalInfo.TryGetValue("LSTMHiddenSize", out var hiddenSizeObj) &&
            hiddenSizeObj is int hiddenSize)
        {
            return hiddenSize;
        }

        // Next, check if there's an explicit architecture parameter
        var hiddenLayerSizes = Architecture.GetHiddenLayerSizes();
        if (hiddenLayerSizes != null && hiddenLayerSizes.Length > 0)
        {
            // Use the size of the first hidden layer as a default
            return hiddenLayerSizes[0];
        }

        // Next, look for LSTM layer and get its output shape
        foreach (var layer in Layers)
        {
            if (IsLSTMLayer(layer))
            {
                var outputShape = layer.GetOutputShape();

                // For LSTM layers, output shape is typically [batch_size, hidden_size] or
                // [batch_size, sequence_length, hidden_size]
                if (outputShape.Length >= 2)
                {
                    // The hidden size is usually the last dimension
                    return outputShape[outputShape.Length - 1];
                }
            }
        }

        // If no LSTM layers are found or their shape can't be determined, throw an exception
        throw new InvalidOperationException(
            "Could not determine LSTM hidden size. Please specify LSTMHiddenSize in the architecture's AdditionalInfo.");
    }

    /// <summary>
    /// Processes input through an LSTM cell, applying proper gating mechanisms.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="hiddenState">The previous hidden state.</param>
    /// <param name="cellState">The previous cell state.</param>
    /// <param name="layerIndex">The index of the LSTM layer.</param>
    /// <returns>Tuple containing output, new hidden state, and new cell state.</returns>
    private (Tensor<T> output, Tensor<T> h, Tensor<T> c) ProcessLSTMCell(
        Tensor<T> input, Tensor<T> hiddenState, Tensor<T> cellState, int layerIndex)
    {
        // Get layer parameters
        int hiddenSize = hiddenState.Shape[1];
        int batchSize = hiddenState.Shape[0];
        int inputSize = input.Shape[1];

        // Create concatenated input (input + hidden state) for all gates
        var combinedInput = new Tensor<T>([batchSize, inputSize + hiddenSize]);

        // Concatenate input and hidden state along feature dimension
        for (int b = 0; b < batchSize; b++)
        {
            // Copy input values
            for (int i = 0; i < inputSize; i++)
            {
                combinedInput[b, i] = input[b, i];
            }

            // Copy hidden state values
            for (int h = 0; h < hiddenSize; h++)
            {
                combinedInput[b, inputSize + h] = hiddenState[b, h];
            }
        }

        // Forward through the layer to get all gate values
        var gateOutputs = Layers[layerIndex].Forward(combinedInput);

        // Verify gate outputs have expected shape
        if (gateOutputs.Shape[1] < 4 * hiddenSize)
        {
            throw new InvalidOperationException(
                $"Expected gate outputs of size {4 * hiddenSize}, but got {gateOutputs.Shape[1]}");
        }

        // Create tensors for each gate
        var forgetGate = new Tensor<T>([batchSize, hiddenSize]);
        var inputGate = new Tensor<T>([batchSize, hiddenSize]);
        var cellGate = new Tensor<T>([batchSize, hiddenSize]);
        var outputGate = new Tensor<T>([batchSize, hiddenSize]);

        // Extract and split gate values using efficient per-row copying
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                // Extract gates from the combined output
                forgetGate[b, h] = gateOutputs[b, h];
                inputGate[b, h] = gateOutputs[b, hiddenSize + h];
                cellGate[b, h] = gateOutputs[b, 2 * hiddenSize + h];
                outputGate[b, h] = gateOutputs[b, 3 * hiddenSize + h];
            }
        }

        // Apply appropriate activations to each gate
        forgetGate = NeuralNetworkHelper<T>.ApplyActivation(forgetGate, ForgetGateActivation, ForgetGateVectorActivation);
        inputGate = NeuralNetworkHelper<T>.ApplyActivation(inputGate, InputGateActivation, InputGateVectorActivation);
        cellGate = NeuralNetworkHelper<T>.ApplyActivation(cellGate, CellGateActivation, CellGateVectorActivation);
        outputGate = NeuralNetworkHelper<T>.ApplyActivation(outputGate, OutputGateActivation, OutputGateVectorActivation);

        // Create new cell state and hidden state tensors
        var newCellState = new Tensor<T>(cellState.Shape);
        var newHiddenState = new Tensor<T>(hiddenState.Shape);

        // Efficiently handle vector or scalar activation
        if (VectorActivation != null)
        {
            // Process each batch item separately
            for (int b = 0; b < batchSize; b++)
            {
                // Create vectors for the current batch item
                var cellStateVector = new Vector<T>(hiddenSize);
                var newCellStateVector = new Vector<T>(hiddenSize);

                // First compute the new cell state
                for (int h = 0; h < hiddenSize; h++)
                {
                    // c_t = f_t * c_{t-1} + i_t * g_t
                    T forgetComponent = NumOps.Multiply(forgetGate[b, h], cellState[b, h]);
                    T inputComponent = NumOps.Multiply(inputGate[b, h], cellGate[b, h]);
                    newCellState[b, h] = NumOps.Add(forgetComponent, inputComponent);

                    // Prepare vector for vector activation
                    newCellStateVector[h] = newCellState[b, h];
                }

                // Apply vector activation to the entire cell state vector
                var activatedVector = VectorActivation.Activate(newCellStateVector);

                // Calculate hidden state using activated cell state
                for (int h = 0; h < hiddenSize; h++)
                {
                    // h_t = o_t * tanh(c_t)
                    newHiddenState[b, h] = NumOps.Multiply(outputGate[b, h], activatedVector[h]);
                }
            }
        }
        else
        {
            // Use scalar activation for each element individually
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    // c_t = f_t * c_{t-1} + i_t * g_t
                    T forgetComponent = NumOps.Multiply(forgetGate[b, h], cellState[b, h]);
                    T inputComponent = NumOps.Multiply(inputGate[b, h], cellGate[b, h]);
                    newCellState[b, h] = NumOps.Add(forgetComponent, inputComponent);

                    // Apply activation to cell state
                    T activatedCell = ScalarActivation != null
                        ? ScalarActivation.Activate(newCellState[b, h])
                        : (new TanhActivation<T>()).Activate(newCellState[b, h]);

                    // h_t = o_t * tanh(c_t)
                    newHiddenState[b, h] = NumOps.Multiply(outputGate[b, h], activatedCell);
                }
            }
        }

        return (newHiddenState, newHiddenState, newCellState);
    }

    /// <summary>
    /// Extracts a specific time step from a batched sequence.
    /// </summary>
    /// <param name="input">The input tensor with shape [batch_size, sequence_length, features].</param>
    /// <param name="timeStep">The time step to extract.</param>
    /// <returns>A tensor containing the specified time step for all batches.</returns>
    private Tensor<T> ExtractTimeStep(Tensor<T> input, int timeStep)
    {
        int batchSize = input.Shape[0];
        int features = input.Shape[2];

        // Create result tensor with shape [batch_size, features]
        var result = new Tensor<T>(new int[] { batchSize, features });

        // Copy data for this time step
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                result[b, f] = input[b, timeStep, f];
            }
        }

        return result;
    }

    /// <summary>
    /// Processes a single time step through all layers of the network.
    /// </summary>
    /// <param name="input">The input tensor for this time step.</param>
    /// <param name="states">The current LSTM states for all layers.</param>
    /// <returns>The output tensor after processing this time step.</returns>
    private Tensor<T> ProcessTimeStep(Tensor<T> input, Dictionary<int, (Tensor<T> h, Tensor<T> c)> states)
    {
        Tensor<T> current = input;

        // Process through each layer
        for (int i = 0; i < Layers.Count; i++)
        {
            if (IsLSTMLayer(Layers[i]))
            {
                // For LSTM layers, we need to handle the state
                if (states.TryGetValue(i, out var state))
                {
                    // Process through LSTM cell
                    var result = ProcessLSTMCell(current, state.h, state.c, i);

                    // Update state for next time step
                    states[i] = (result.h, result.c);

                    // Pass hidden state to next layer
                    current = result.output;
                }
                else
                {
                    // Fallback if state is not available
                    current = Layers[i].Forward(current);
                }
            }
            else
            {
                // For non-LSTM layers, just forward pass
                current = Layers[i].Forward(current);
            }
        }

        return current;
    }

    /// <summary>
    /// Stacks a list of tensors along the time dimension.
    /// </summary>
    /// <param name="tensors">The list of tensors to stack.</param>
    /// <returns>A stacked tensor with an additional time dimension.</returns>
    private Tensor<T> StackAlongTimeDimension(List<Tensor<T>> tensors)
    {
        if (tensors.Count == 0)
        {
            return new Tensor<T>([0]);
        }

        // Get shape information from the first tensor
        int[] shape = tensors[0].Shape;

        // Create result shape with time dimension inserted at position 1
        int[] resultShape = new int[shape.Length + 1];
        resultShape[0] = shape[0]; // Batch size
        resultShape[1] = tensors.Count; // Sequence length

        for (int i = 1; i < shape.Length; i++)
        {
            resultShape[i + 1] = shape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data from each tensor
        for (int t = 0; t < tensors.Count; t++)
        {
            var tensor = tensors[t];

            // Copy based on tensor rank
            if (shape.Length == 2)
            {
                int batchSize = shape[0];
                int features = shape[1];

                for (int b = 0; b < batchSize; b++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        result[b, t, f] = tensor[b, f];
                    }
                }
            }
            else
            {
                throw new NotSupportedException("Only 2D tensor stacking is currently supported");
            }
        }

        return result;
    }

    /// <summary>
    /// Adds a batch dimension to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with an additional batch dimension.</returns>
    private Tensor<T> AddBatchDimension(Tensor<T> input)
    {
        int[] inputShape = input.Shape;
        int[] resultShape = new int[inputShape.Length + 1];

        // Add batch dimension of size 1
        resultShape[0] = 1;

        // Copy remaining dimensions
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 1] = inputShape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data
        if (inputShape.Length == 2)
        {
            int seqLength = inputShape[0];
            int features = inputShape[1];

            for (int s = 0; s < seqLength; s++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[0, s, f] = input[s, f];
                }
            }
        }
        else if (inputShape.Length == 1)
        {
            int features = inputShape[0];

            for (int f = 0; f < features; f++)
            {
                result[0, f] = input[f];
            }
        }

        return result;
    }

    /// <summary>
    /// Removes a batch dimension from a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with the batch dimension removed.</returns>
    private Tensor<T> RemoveBatchDimension(Tensor<T> input)
    {
        int[] inputShape = input.Shape;

        // Ensure first dimension is batch and has size 1
        if (inputShape[0] != 1)
        {
            throw new ArgumentException("Cannot remove batch dimension with size != 1");
        }

        int[] resultShape = new int[inputShape.Length - 1];

        // Copy dimensions except batch
        for (int i = 1; i < inputShape.Length; i++)
        {
            resultShape[i - 1] = inputShape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data
        if (inputShape.Length == 3)
        {
            int seqLength = inputShape[1];
            int features = inputShape[2];

            for (int s = 0; s < seqLength; s++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[s, f] = input[0, s, f];
                }
            }
        }
        else if (inputShape.Length == 2)
        {
            int features = inputShape[1];

            for (int f = 0; f < features; f++)
            {
                result[f] = input[0, f];
            }
        }

        return result;
    }

    /// <summary>
    /// Adds a time dimension to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with an additional time dimension.</returns>
    private Tensor<T> AddTimeDimension(Tensor<T> input)
    {
        int[] inputShape = input.Shape;
        int[] resultShape = new int[inputShape.Length + 1];

        // First dimension is batch
        resultShape[0] = inputShape[0];

        // Add time dimension of size 1
        resultShape[1] = 1;

        // Copy remaining dimensions
        for (int i = 1; i < inputShape.Length; i++)
        {
            resultShape[i + 1] = inputShape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data
        if (inputShape.Length == 2)
        {
            int batchSize = inputShape[0];
            int features = inputShape[1];

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[b, 0, f] = input[b, f];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Removes a time dimension from a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with the time dimension removed.</returns>
    private Tensor<T> RemoveTimeDimension(Tensor<T> input)
    {
        int[] inputShape = input.Shape;

        // Ensure second dimension is time and has size 1
        if (inputShape[1] != 1)
        {
            throw new ArgumentException("Cannot remove time dimension with size != 1");
        }

        int[] resultShape = new int[inputShape.Length - 1];

        // Copy first dimension (batch)
        resultShape[0] = inputShape[0];

        // Copy dimensions except time
        for (int i = 2; i < inputShape.Length; i++)
        {
            resultShape[i - 1] = inputShape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data
        if (inputShape.Length == 3)
        {
            int batchSize = inputShape[0];
            int features = inputShape[2];

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    result[b, f] = input[b, 0, f];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Adds both batch and time dimensions to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with additional batch and time dimensions.</returns>
    private Tensor<T> AddBatchAndTimeDimensions(Tensor<T> input)
    {
        int[] inputShape = input.Shape;
        int[] resultShape = new int[inputShape.Length + 2];

        // Add batch and time dimensions of size 1
        resultShape[0] = 1;
        resultShape[1] = 1;

        // Copy remaining dimensions
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 2] = inputShape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data
        if (inputShape.Length == 1)
        {
            int features = inputShape[0];

            for (int f = 0; f < features; f++)
            {
                result[0, 0, f] = input[f];
            }
        }

        return result;
    }

    /// <summary>
    /// Removes both batch and time dimensions from a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with batch and time dimensions removed.</returns>
    private Tensor<T> RemoveBatchAndTimeDimensions(Tensor<T> input)
    {
        int[] inputShape = input.Shape;

        // Ensure first and second dimensions are batch and time with size 1
        if (inputShape[0] != 1 || inputShape[1] != 1)
        {
            throw new ArgumentException("Cannot remove dimensions with size != 1");
        }

        int[] resultShape = new int[inputShape.Length - 2];

        // Copy dimensions except batch and time
        for (int i = 2; i < inputShape.Length; i++)
        {
            resultShape[i - 2] = inputShape[i];
        }

        // Create result tensor
        var result = new Tensor<T>(resultShape);

        // Copy data
        if (inputShape.Length == 3)
        {
            int features = inputShape[2];

            for (int f = 0; f < features; f++)
            {
                result[f] = input[0, 0, f];
            }
        }

        return result;
    }

    /// <summary>
    /// Trains the LSTM network on input-output pairs.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method trains the LSTM network using backpropagation through time (BPTT).
    /// It performs a forward pass to get predictions, calculates the error, and backpropagates
    /// the gradients through the network over time to update the weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the LSTM network to make accurate predictions.
    /// 
    /// The training process works like this:
    /// 1. Input data (like sequences of words or time steps) is fed into the network
    /// 2. The network makes predictions at each time step
    /// 3. These predictions are compared with the expected outputs to calculate the error
    /// 4. The error is "backpropagated" through time, adjusting the network's internal values
    /// 5. This process repeats for many examples, gradually improving the network's performance
    /// 
    /// The key difference from training regular neural networks is that LSTM training needs
    /// to account for connections across time steps, as earlier inputs influence later outputs.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Set to training mode
        SetTrainingMode(true);

        // Forward pass to get predictions
        var predictions = Predict(input);

        var flattenedPredictions = predictions.ToVector();
        var flattenedExpected = expectedOutput.ToVector();

        // Calculate loss
        LastLoss = LossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        // Calculate output gradients
        var gradientVector = LossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);
        var outputGradients = new Tensor<T>(predictions.Shape, gradientVector);

        // Backpropagation through time
        BackpropagateOverTime(outputGradients, input);

        // Update parameters
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Performs backpropagation through time for LSTM networks.
    /// </summary>
    /// <param name="outputGradients">The gradients from the output layer.</param>
    /// <param name="input">The original input used in the forward pass.</param>
    private void BackpropagateOverTime(Tensor<T> outputGradients, Tensor<T> input)
    {
        // Determine dimensions
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape.Length > 1 ? input.Shape[1] : 1;
        int inputFeatures = input.Shape.Length > 2 ? input.Shape[2] : (input.Shape.Length > 1 ? input.Shape[1] : input.Length);

        // Handle different scenarios for output gradients based on returnSequences setting
        Tensor<T>[] gradientPerTimeStep;

        if (Architecture.ShouldReturnFullSequence)
        {
            // If returning full sequence, split output gradients by time step
            gradientPerTimeStep = new Tensor<T>[sequenceLength];
            for (int t = 0; t < sequenceLength; t++)
            {
                // Extract gradient for current time step
                var shape = new int[outputGradients.Shape.Length - 1];
                shape[0] = batchSize;
                if (shape.Length > 1)
                {
                    Array.Copy(outputGradients.Shape, 2, shape, 1, shape.Length - 1);
                }

                gradientPerTimeStep[t] = new Tensor<T>(shape);

                // Copy gradient data for this time step
                for (int b = 0; b < batchSize; b++)
                {
                    if (shape.Length == 1)
                    {
                        gradientPerTimeStep[t][b] = outputGradients[b, t];
                    }
                    else
                    {
                        // Handle higher dimensional outputs
                        int outputFeatures = outputGradients.Shape[2];
                        for (int f = 0; f < outputFeatures; f++)
                        {
                            gradientPerTimeStep[t][b, f] = outputGradients[b, t, f];
                        }
                    }
                }
            }
        }
        else
        {
            // If not returning full sequence, use the provided gradient only for the last time step
            gradientPerTimeStep = new Tensor<T>[sequenceLength];
            for (int t = 0; t < sequenceLength - 1; t++)
            {
                // Zero gradients for all steps except the last
                gradientPerTimeStep[t] = new Tensor<T>(outputGradients.Shape);
            }
            gradientPerTimeStep[sequenceLength - 1] = outputGradients;
        }

        // Initialize hidden state and cell state gradients for tracking between time steps
        var hiddenStateGradients = new Dictionary<int, Tensor<T>>();
        var cellStateGradients = new Dictionary<int, Tensor<T>>();

        // Find LSTM layers
        var lstmLayerIndices = new List<int>();
        for (int i = 0; i < Layers.Count; i++)
        {
            if (IsLSTMLayer(Layers[i]))
            {
                int hiddenSize = GetLSTMLayerHiddenSize(i);
                hiddenStateGradients[i] = new Tensor<T>([batchSize, hiddenSize]);
                cellStateGradients[i] = new Tensor<T>([batchSize, hiddenSize]);
                lstmLayerIndices.Add(i);
            }
        }

        // Check if we have stored activations
        if (_storedActivations == null || _storedActivations.Count == 0)
        {
            // If no activations are stored, fall back to a simpler approach
            FallbackBackPropagation(outputGradients, input);
            return;
        }

        // Backpropagate through time (from last time step to first)
        for (int t = sequenceLength - 1; t >= 0; t--)
        {
            // Extract input for this time step
            Tensor<T> timeStepInput;
            if (input.Shape.Length > 2)
            {
                timeStepInput = input.Slice(1, t, t + 1).Reshape([batchSize, inputFeatures]);
            }
            else if (input.Shape.Length == 2)
            {
                timeStepInput = new Tensor<T>([batchSize, 1]);
                for (int b = 0; b < batchSize; b++)
                {
                    timeStepInput[b, 0] = input[b, t];
                }
            }
            else
            {
                timeStepInput = input; // Single input, no time dimension
            }

            // Current gradient for this time step
            Tensor<T> currentGradient = gradientPerTimeStep[t];

            // Add hidden state gradients from future time steps
            foreach (var lstmIndex in lstmLayerIndices)
            {
                if (hiddenStateGradients.ContainsKey(lstmIndex) && !IsZeroTensor(hiddenStateGradients[lstmIndex]))
                {
                    // We need to transform the hidden state gradient to match the output gradient shape
                    // This might require a specific transformation based on your architecture
                    var transformedGradient = TransformHiddenGradientToOutputGradient(
                        hiddenStateGradients[lstmIndex],
                        currentGradient.Shape);

                    currentGradient = currentGradient.Add(transformedGradient);
                }
            }

            // Backpropagate through layers in reverse order for this time step
            Tensor<T> layerGradient = currentGradient;

            for (int l = Layers.Count - 1; l >= 0; l--)
            {
                // The key modification is here - we're using the existing Backward method of each layer
                layerGradient = Layers[l].Backward(layerGradient);

                // For LSTM layers, we need to update our tracking of state gradients
                if (lstmLayerIndices.Contains(l))
                {
                    // Save hidden state gradient for the previous time step
                    // This is a simplification - in a full implementation, we would compute proper
                    // gradients for hidden and cell states based on the LSTM equations
                    hiddenStateGradients[l] = layerGradient.Clone();
                }
            }
        }

        // Update parameters for each layer
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i].SupportsTraining)
            {
                // Apply parameter updates using the existing method
                T learningRate = NumOps.FromDouble(0.01); // Use proper learning rate from optimizer
                Layers[i].UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Fallback implementation of backpropagation when no stored activations are available.
    /// </summary>
    /// <param name="outputGradients">The gradients from the output layer.</param>
    /// <param name="input">The original input used in the forward pass.</param>
    private void FallbackBackPropagation(Tensor<T> outputGradients, Tensor<T> input)
    {
        // Simple backpropagation without considering time steps
        Tensor<T> gradients = outputGradients;

        // Backpropagate through each layer in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }

        // Update parameters for each layer
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i].SupportsTraining)
            {
                T learningRate = NumOps.FromDouble(0.01); // Use proper learning rate from optimizer
                Layers[i].UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Transforms a hidden state gradient to match the shape of an output gradient.
    /// </summary>
    /// <param name="hiddenGradient">The hidden state gradient from an LSTM layer.</param>
    /// <param name="outputShape">The shape of the output gradient that needs to be matched.</param>
    /// <returns>A transformed gradient matching the output shape.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms hidden state gradients from LSTM layers to match the shape expected
    /// by subsequent layers during backpropagation. When LSTM hidden states have a different
    /// dimensionality than the output, this method performs the necessary transformation.
    /// </para>
    /// <para>
    /// The transformation handles three cases:
    /// 1. **Matching shapes**: Direct copy (no transformation needed)
    /// 2. **Hidden larger than output**: Gradient averaging across hidden dimensions
    /// 3. **Output larger than hidden**: Gradient distribution/replication
    /// </para>
    /// <para>
    /// For production LSTMs with explicit output projection layers (W_out * h_t + b_out),
    /// this transformation approximates the backpropagation through that projection.
    /// If your architecture includes dense output layers after LSTM, those layers
    /// handle the transformation automatically via their Backward() methods.
    /// </para>
    /// </remarks>
    private Tensor<T> TransformHiddenGradientToOutputGradient(Tensor<T> hiddenGradient, int[] outputShape)
    {
        // Create tensor with target shape
        var transformedGradient = new Tensor<T>(outputShape);

        // Case 1: Shapes match exactly - direct copy
        if (hiddenGradient.Shape.SequenceEqual(outputShape))
        {
            for (int i = 0; i < hiddenGradient.Length; i++)
            {
                transformedGradient[i] = hiddenGradient[i];
            }
            return transformedGradient;
        }

        // Case 2: Different shapes - need to transform
        // Get the feature dimensions (last dimension)
        int hiddenFeatures = hiddenGradient.Shape[hiddenGradient.Shape.Length - 1];
        int outputFeatures = outputShape[outputShape.Length - 1];

        // Calculate first dimension (could be batch size, sequence length, or combined depending on layout)
        int outputFirstDim = outputShape.Length > 0 ? outputShape[0] : 1;
        int hiddenFirstDim = hiddenGradient.Shape.Length > 0 ? hiddenGradient.Shape[0] : 1;

        // Ensure first dimensions are compatible
        if (hiddenFirstDim != outputFirstDim && hiddenFirstDim != 1 && outputFirstDim != 1)
        {
            // Incompatible batch sizes - use fallback
            int minLength = Math.Min(hiddenGradient.Length, transformedGradient.Length);
            for (int i = 0; i < minLength; i++)
            {
                transformedGradient[i] = hiddenGradient[i];
            }
            return transformedGradient;
        }

        // Transform based on feature dimension relationship
        if (hiddenFeatures == outputFeatures)
        {
            // Same feature dimension, might differ in batch/sequence dims
            int minLength = Math.Min(hiddenGradient.Length, transformedGradient.Length);
            for (int i = 0; i < minLength; i++)
            {
                transformedGradient[i] = hiddenGradient[i];
            }
        }
        else if (hiddenFeatures > outputFeatures)
        {
            // Hidden state has more features - average/pool gradients
            // NOTE: This is an approximation using uniform averaging. For learned projections
            // (e.g., dense layers with weight matrix W), the mathematically correct approach
            // would be gradOutput = W^T @ gradHidden. This approximation is suitable when
            // no explicit projection layers exist, or as a fallback for debugging.
            // If your architecture includes dense output layers after LSTM, those layers
            // handle the gradient transformation via their Backward() methods.
            for (int b = 0; b < Math.Min(outputFirstDim, hiddenFirstDim); b++)
            {
                for (int outF = 0; outF < outputFeatures; outF++)
                {
                    T accumulatedGrad = NumOps.Zero;
                    int startHiddenF = (outF * hiddenFeatures) / outputFeatures;
                    int endHiddenF = ((outF + 1) * hiddenFeatures) / outputFeatures;

                    // Average gradients from corresponding hidden features
                    for (int hidF = startHiddenF; hidF < endHiddenF && hidF < hiddenFeatures; hidF++)
                    {
                        int hiddenIdx = b * hiddenFeatures + hidF;
                        if (hiddenIdx < hiddenGradient.Length)
                        {
                            accumulatedGrad = NumOps.Add(accumulatedGrad, hiddenGradient[hiddenIdx]);
                        }
                    }

                    // Store averaged gradient
                    int outIdx = b * outputFeatures + outF;
                    if (outIdx < transformedGradient.Length)
                    {
                        int count = endHiddenF - startHiddenF;
                        transformedGradient[outIdx] = count > 0
                            ? NumOps.Divide(accumulatedGrad, NumOps.FromDouble(count))
                            : accumulatedGrad;
                    }
                }
            }
        }
        else
        {
            // Output has more features - replicate gradients
            // NOTE: This is an approximation using gradient replication. For learned projections,
            // the correct approach would use the transpose of the projection matrix.
            // This approximation is suitable when no explicit projection layers exist.
            for (int b = 0; b < Math.Min(outputFirstDim, hiddenFirstDim); b++)
            {
                for (int outF = 0; outF < outputFeatures; outF++)
                {
                    // Map output feature to corresponding hidden feature
                    int hidF = (outF * hiddenFeatures) / outputFeatures;
                    int hiddenIdx = b * hiddenFeatures + hidF;

                    if (hiddenIdx < hiddenGradient.Length)
                    {
                        int outIdx = b * outputFeatures + outF;
                        if (outIdx < transformedGradient.Length)
                        {
                            transformedGradient[outIdx] = hiddenGradient[hiddenIdx];
                        }
                    }
                }
            }
        }

        return transformedGradient;
    }

    /// <summary>
    /// Checks if a tensor contains only zero values.
    /// </summary>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if the tensor contains only zeros; otherwise, false.</returns>
    private bool IsZeroTensor(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            if (!MathHelper.AlmostEqual(tensor[i], NumOps.Zero))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Dictionary to store activations and states from the forward pass.
    /// </summary>
    private Dictionary<string, Tensor<T>> _storedActivations = new Dictionary<string, Tensor<T>>();

    /// <summary>
    /// Performs the backward pass through a single LSTM cell.
    /// </summary>
    /// <param name="dhNext">The gradient of the loss with respect to the hidden state.</param>
    /// <param name="dhPrev">The gradient from the next time step's hidden state.</param>
    /// <param name="dcPrev">The gradient from the next time step's cell state.</param>
    /// <param name="x">The input at this time step.</param>
    /// <param name="h">The hidden state at this time step.</param>
    /// <param name="c">The cell state at this time step.</param>
    /// <param name="hPrev">The hidden state from the previous time step.</param>
    /// <param name="cPrev">The cell state from the previous time step.</param>
    /// <param name="i">The input gate activation.</param>
    /// <param name="f">The forget gate activation.</param>
    /// <param name="g">The cell gate activation.</param>
    /// <param name="o">The output gate activation.</param>
    /// <param name="layerIndex">The index of the LSTM layer.</param>
    /// <returns>A tuple containing gradients for the hidden state, cell state, and input.</returns>
    private (Tensor<T> dh, Tensor<T> dc, Tensor<T> dx) BackwardLSTMCell(
        Tensor<T> dhNext,
        Tensor<T>? dhPrev,
        Tensor<T>? dcPrev,
        Tensor<T> x,
        Tensor<T> h,
        Tensor<T> c,
        Tensor<T> hPrev,
        Tensor<T> cPrev,
        Tensor<T>? i,
        Tensor<T>? f,
        Tensor<T>? g,
        Tensor<T>? o,
        int layerIndex)
    {
        int batchSize = h.Shape[0];
        int hiddenSize = h.Shape[1];
        int inputSize = x.Shape[1];

        // Initialize gradients
        Tensor<T> dh = dhPrev != null ? dhNext.Add(dhPrev) : dhNext.Clone();
        Tensor<T> dc = dcPrev != null ? dcPrev.Clone() : new Tensor<T>([batchSize, hiddenSize]);

        // Calculate gradients for the gates
        Tensor<T> do_ = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> di = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> df = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> dg = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> dc_tanh = new Tensor<T>([batchSize, hiddenSize]);

        // Calculate activated cell state
        Tensor<T> c_tanh = new Tensor<T>([batchSize, hiddenSize]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                c_tanh[b, j] = ScalarActivation != null
                    ? ScalarActivation.Activate(c[b, j])
                    : (new TanhActivation<T>()).Activate(c[b, j]);
            }
        }

        // Ensure we have gate activations
        if (i == null || f == null || g == null || o == null)
        {
            throw new InvalidOperationException("Gate activations must be stored during forward pass for backpropagation.");
        }

        // Backward pass through output gate
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                do_[b, j] = NumOps.Multiply(dh[b, j], c_tanh[b, j]);

                // Gradient for tanh(c)
                dc_tanh[b, j] = NumOps.Multiply(dh[b, j], o[b, j]);

                // Gradient for c
                T tanh_derivative;
                if (ScalarActivation != null)
                {
                    tanh_derivative = ScalarActivation.Derivative(c[b, j]);
                }
                else
                {
                    T tanh_value = (new TanhActivation<T>()).Activate(c[b, j]);
                    tanh_derivative = NumOps.Subtract(NumOps.One, NumOps.Multiply(tanh_value, tanh_value));
                }

                // Add gradient from next cell state
                T dc_value = NumOps.Multiply(dc_tanh[b, j], tanh_derivative);
                if (dcPrev != null)
                {
                    dc_value = NumOps.Add(dc_value, dc[b, j]);
                }
                dc[b, j] = dc_value;

                // Gradients for gates
                di[b, j] = NumOps.Multiply(dc[b, j], g[b, j]);
                dg[b, j] = NumOps.Multiply(dc[b, j], i[b, j]);
                df[b, j] = NumOps.Multiply(dc[b, j], cPrev[b, j]);

                // Gradient to previous cell state
                dc[b, j] = NumOps.Multiply(dc[b, j], f[b, j]);
            }
        }

        // Apply gate activation derivatives
        Tensor<T> di_input = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> df_input = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> dg_input = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> do_input = new Tensor<T>([batchSize, hiddenSize]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
                di_input[b, j] = NumOps.Multiply(di[b, j],
                    NumOps.Multiply(i[b, j], NumOps.Subtract(NumOps.One, i[b, j])));

                df_input[b, j] = NumOps.Multiply(df[b, j],
                    NumOps.Multiply(f[b, j], NumOps.Subtract(NumOps.One, f[b, j])));

                do_input[b, j] = NumOps.Multiply(do_[b, j],
                    NumOps.Multiply(o[b, j], NumOps.Subtract(NumOps.One, o[b, j])));

                // Tanh derivative: 1 - tanh(x)^2
                T tanh_derivative;
                if (CellGateActivation != null)
                {
                    tanh_derivative = CellGateActivation.Derivative(g[b, j]);
                }
                else
                {
                    tanh_derivative = NumOps.Subtract(NumOps.One, NumOps.Multiply(g[b, j], g[b, j]));
                }

                dg_input[b, j] = NumOps.Multiply(dg[b, j], tanh_derivative);
            }
        }

        // Get LSTM layer parameters
        var parameters = GetLSTMLayerParameters(layerIndex);

        // Calculate gradient for input x
        Tensor<T> dx = new Tensor<T>([batchSize, inputSize]);

        // Calculate gradients for weights
        Tensor<T> dWi = new Tensor<T>([hiddenSize, inputSize]);
        Tensor<T> dWf = new Tensor<T>([hiddenSize, inputSize]);
        Tensor<T> dWg = new Tensor<T>([hiddenSize, inputSize]);
        Tensor<T> dWo = new Tensor<T>([hiddenSize, inputSize]);

        Tensor<T> dUi = new Tensor<T>([hiddenSize, hiddenSize]);
        Tensor<T> dUf = new Tensor<T>([hiddenSize, hiddenSize]);
        Tensor<T> dUg = new Tensor<T>([hiddenSize, hiddenSize]);
        Tensor<T> dUo = new Tensor<T>([hiddenSize, hiddenSize]);

        Tensor<T> dbi = new Tensor<T>([hiddenSize]);
        Tensor<T> dbf = new Tensor<T>([hiddenSize]);
        Tensor<T> dbg = new Tensor<T>([hiddenSize]);
        Tensor<T> dbo = new Tensor<T>([hiddenSize]);

        // Calculate gradients for input, weights, and biases
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                // Bias gradients
                dbi[j] = NumOps.Add(dbi[j], di_input[b, j]);
                dbf[j] = NumOps.Add(dbf[j], df_input[b, j]);
                dbg[j] = NumOps.Add(dbg[j], dg_input[b, j]);
                dbo[j] = NumOps.Add(dbo[j], do_input[b, j]);

                // Input gradients
                for (int k = 0; k < inputSize; k++)
                {
                    T dxi = NumOps.Multiply(parameters.Wi[j, k], di_input[b, j]);
                    T dxf = NumOps.Multiply(parameters.Wf[j, k], df_input[b, j]);
                    T dxg = NumOps.Multiply(parameters.Wg[j, k], dg_input[b, j]);
                    T dxo = NumOps.Multiply(parameters.Wo[j, k], do_input[b, j]);

                    dx[b, k] = NumOps.Add(dx[b, k], NumOps.Add(NumOps.Add(dxi, dxf), NumOps.Add(dxg, dxo)));

                    // Weight gradients (input)
                    dWi[j, k] = NumOps.Add(dWi[j, k], NumOps.Multiply(di_input[b, j], x[b, k]));
                    dWf[j, k] = NumOps.Add(dWf[j, k], NumOps.Multiply(df_input[b, j], x[b, k]));
                    dWg[j, k] = NumOps.Add(dWg[j, k], NumOps.Multiply(dg_input[b, j], x[b, k]));
                    dWo[j, k] = NumOps.Add(dWo[j, k], NumOps.Multiply(do_input[b, j], x[b, k]));
                }

                // Weight gradients (hidden)
                for (int k = 0; k < hiddenSize; k++)
                {
                    dUi[j, k] = NumOps.Add(dUi[j, k], NumOps.Multiply(di_input[b, j], hPrev[b, k]));
                    dUf[j, k] = NumOps.Add(dUf[j, k], NumOps.Multiply(df_input[b, j], hPrev[b, k]));
                    dUg[j, k] = NumOps.Add(dUg[j, k], NumOps.Multiply(dg_input[b, j], hPrev[b, k]));
                    dUo[j, k] = NumOps.Add(dUo[j, k], NumOps.Multiply(do_input[b, j], hPrev[b, k]));
                }
            }
        }

        // Store weight gradients for parameter updates
        StoreLSTMGradients(layerIndex, dWi, dWf, dWg, dWo, dUi, dUf, dUg, dUo, dbi, dbf, dbg, dbo);

        return (dh, dc, dx);
    }

    /// <summary>
    /// Gets the hidden size for a specific LSTM layer.
    /// </summary>
    /// <param name="layerIndex">The index of the LSTM layer.</param>
    /// <returns>The hidden size of the LSTM layer.</returns>
    private int GetLSTMLayerHiddenSize(int layerIndex)
    {
        // Get from layer's output shape if available
        if (layerIndex >= 0 && layerIndex < Layers.Count)
        {
            var outputShape = Layers[layerIndex].GetOutputShape();
            if (outputShape.Length >= 2)
            {
                return outputShape[1];
            }
        }

        // Fallback to architecture's hidden layer size
        var hiddenLayerSizes = Architecture.GetHiddenLayerSizes();
        if (hiddenLayerSizes != null && hiddenLayerSizes.Length > 0)
        {
            return hiddenLayerSizes[0];
        }

        // Default if no information is available
        return 128;
    }

    /// <summary>
    /// Represents LSTM layer parameters.
    /// </summary>
    private class LSTMParameters
    {
        public Matrix<T> Wi { get; set; } = Matrix<T>.Empty();
        public Matrix<T> Wf { get; set; } = Matrix<T>.Empty();
        public Matrix<T> Wg { get; set; } = Matrix<T>.Empty();
        public Matrix<T> Wo { get; set; } = Matrix<T>.Empty();

        public Matrix<T> Ui { get; set; } = Matrix<T>.Empty();
        public Matrix<T> Uf { get; set; } = Matrix<T>.Empty();
        public Matrix<T> Ug { get; set; } = Matrix<T>.Empty();
        public Matrix<T> Uo { get; set; } = Matrix<T>.Empty();

        public Vector<T> Bi { get; set; } = Vector<T>.Empty();
        public Vector<T> Bf { get; set; } = Vector<T>.Empty();
        public Vector<T> Bg { get; set; } = Vector<T>.Empty();
        public Vector<T> Bo { get; set; } = Vector<T>.Empty();
    }

    /// <summary>
    /// Gets the LSTM layer parameters for a specific layer.
    /// </summary>
    /// <param name="layerIndex">The index of the LSTM layer.</param>
    /// <returns>The parameters of the LSTM layer.</returns>
    private LSTMParameters GetLSTMLayerParameters(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Layers.Count || !IsLSTMLayer(Layers[layerIndex]))
        {
            throw new ArgumentException("Invalid LSTM layer index.");
        }

        // For a production implementation, we would extract these from the actual layer
        // This is a simplified approach that assumes specific layer structure
        var layer = Layers[layerIndex];
        int hiddenSize = GetLSTMLayerHiddenSize(layerIndex);
        int inputSize = layer.GetInputShape()[1];

        // Extract parameters from layer
        var allParams = layer.GetParameters();

        // Assuming parameter order: Wi, Wf, Wg, Wo, Ui, Uf, Ug, Uo, bi, bf, bg, bo
        int matrixSize = hiddenSize * inputSize;
        int recurrentSize = hiddenSize * hiddenSize;

        return new LSTMParameters
        {
            Wi = ExtractMatrix(allParams, 0, hiddenSize, inputSize),
            Wf = ExtractMatrix(allParams, matrixSize, hiddenSize, inputSize),
            Wg = ExtractMatrix(allParams, 2 * matrixSize, hiddenSize, inputSize),
            Wo = ExtractMatrix(allParams, 3 * matrixSize, hiddenSize, inputSize),

            Ui = ExtractMatrix(allParams, 4 * matrixSize, hiddenSize, hiddenSize),
            Uf = ExtractMatrix(allParams, 4 * matrixSize + recurrentSize, hiddenSize, hiddenSize),
            Ug = ExtractMatrix(allParams, 4 * matrixSize + 2 * recurrentSize, hiddenSize, hiddenSize),
            Uo = ExtractMatrix(allParams, 4 * matrixSize + 3 * recurrentSize, hiddenSize, hiddenSize),

            Bi = ExtractVector(allParams, 4 * matrixSize + 4 * recurrentSize, hiddenSize),
            Bf = ExtractVector(allParams, 4 * matrixSize + 4 * recurrentSize + hiddenSize, hiddenSize),
            Bg = ExtractVector(allParams, 4 * matrixSize + 4 * recurrentSize + 2 * hiddenSize, hiddenSize),
            Bo = ExtractVector(allParams, 4 * matrixSize + 4 * recurrentSize + 3 * hiddenSize, hiddenSize)
        };
    }

    /// <summary>
    /// Extracts a matrix from a parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector.</param>
    /// <param name="startIndex">The starting index in the parameter vector.</param>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="cols">The number of columns in the matrix.</param>
    /// <returns>The extracted matrix.</returns>
    private Matrix<T> ExtractMatrix(Vector<T> parameters, int startIndex, int rows, int cols)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = parameters[startIndex + i * cols + j];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Extracts a vector from a parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector.</param>
    /// <param name="startIndex">The starting index in the parameter vector.</param>
    /// <param name="length">The length of the vector.</param>
    /// <returns>The extracted vector.</returns>
    private Vector<T> ExtractVector(Vector<T> parameters, int startIndex, int length)
    {
        var vector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = parameters[startIndex + i];
        }

        return vector;
    }

    /// <summary>
    /// Stores LSTM gradients for parameter updates.
    /// </summary>
    /// <param name="layerIndex">The index of the LSTM layer.</param>
    /// <param name="dWi">Gradient for input gate weights (input).</param>
    /// <param name="dWf">Gradient for forget gate weights (input).</param>
    /// <param name="dWg">Gradient for cell gate weights (input).</param>
    /// <param name="dWo">Gradient for output gate weights (input).</param>
    /// <param name="dUi">Gradient for input gate weights (recurrent).</param>
    /// <param name="dUf">Gradient for forget gate weights (recurrent).</param>
    /// <param name="dUg">Gradient for cell gate weights (recurrent).</param>
    /// <param name="dUo">Gradient for output gate weights (recurrent).</param>
    /// <param name="dbi">Gradient for input gate bias.</param>
    /// <param name="dbf">Gradient for forget gate bias.</param>
    /// <param name="dbg">Gradient for cell gate bias.</param>
    /// <param name="dbo">Gradient for output gate bias.</param>
    private void StoreLSTMGradients(
        int layerIndex,
        Tensor<T> dWi, Tensor<T> dWf, Tensor<T> dWg, Tensor<T> dWo,
        Tensor<T> dUi, Tensor<T> dUf, Tensor<T> dUg, Tensor<T> dUo,
        Tensor<T> dbi, Tensor<T> dbf, Tensor<T> dbg, Tensor<T> dbo)
    {
        if (layerIndex < 0 || layerIndex >= Layers.Count || !IsLSTMLayer(Layers[layerIndex]))
        {
            return;
        }

        // In a production implementation, we would use the layer's specific method to set gradients
        var layer = Layers[layerIndex];

        if (layer.SupportsTraining)
        {
            // Create gradient vector in the expected format
            int hiddenSize = GetLSTMLayerHiddenSize(layerIndex);
            int inputSize = layer.GetInputShape()[1];

            int totalParamCount = 4 * (hiddenSize * inputSize + hiddenSize * hiddenSize + hiddenSize);
            Vector<T> gradients = new Vector<T>(totalParamCount);

            // Get each component as a vector using built-in methods
            var dWiVector = dWi.ToMatrix().ToColumnVector();
            var dWfVector = dWf.ToMatrix().ToColumnVector();
            var dWgVector = dWg.ToMatrix().ToColumnVector();
            var dWoVector = dWo.ToMatrix().ToColumnVector();

            var dUiVector = dUi.ToMatrix().ToColumnVector();
            var dUfVector = dUf.ToMatrix().ToColumnVector();
            var dUgVector = dUg.ToMatrix().ToColumnVector();
            var dUoVector = dUo.ToMatrix().ToColumnVector();

            // Combine all vectors into the final gradient vector
            int index = 0;

            // Copy input weights
            var dWiLength = dWiVector.Length;
            for (int i = 0; i < dWiLength; i++) gradients[index++] = dWiVector[i];

            var dWfLength = dWfVector.Length;
            for (int i = 0; i < dWfLength; i++) gradients[index++] = dWfVector[i];

            var dWgLength = dWgVector.Length;
            for (int i = 0; i < dWgLength; i++) gradients[index++] = dWgVector[i];

            var dWoLength = dWoVector.Length;
            for (int i = 0; i < dWoLength; i++) gradients[index++] = dWoVector[i];

            // Copy recurrent weights
            var dUiLength = dUiVector.Length;
            for (int i = 0; i < dUiLength; i++) gradients[index++] = dUiVector[i];

            var dUfLength = dUfVector.Length;
            for (int i = 0; i < dUfLength; i++) gradients[index++] = dUfVector[i];

            var dUgLength = dUgVector.Length;
            for (int i = 0; i < dUgLength; i++) gradients[index++] = dUgVector[i];

            var dUoLength = dUoVector.Length;
            for (int i = 0; i < dUoLength; i++) gradients[index++] = dUoVector[i];

            // Copy biases
            var dbiLength = dbi.ToVector().Length;
            for (int i = 0; i < dbiLength; i++) gradients[index++] = dbi.ToVector()[i];

            var dbfLength = dbf.ToVector().Length;
            for (int i = 0; i < dbfLength; i++) gradients[index++] = dbf.ToVector()[i];

            var dbgLength = dbg.ToVector().Length;
            for (int i = 0; i < dbgLength; i++) gradients[index++] = dbg.ToVector()[i];

            var dboLength = dbo.ToVector().Length;
            for (int i = 0; i < dboLength; i++) gradients[index++] = dbo.ToVector()[i];

            T learningRate = NumOps.FromDouble(0.01); // Use appropriate learning rate

            // Get current parameters
            Vector<T> currentParams = layer.GetParameters();

            // Update parameters using vectorized operations: params = params - learningRate * gradients
            int updateLength = Math.Min(gradients.Length, currentParams.Length);
            var gradientSlice = gradients.GetSubVector(0, updateLength);
            var paramSlice = currentParams.GetSubVector(0, updateLength);
            var scaledGradients = (Vector<T>)Engine.Multiply(gradientSlice, learningRate);
            var updatedSlice = (Vector<T>)Engine.Subtract(paramSlice, scaledGradients);

            // Copy updated values back
            for (int i = 0; i < updateLength; i++)
            {
                currentParams[i] = updatedSlice[i];
            }

            // Apply updated parameters to the layer
            layer.UpdateParameters(currentParams);
        }
    }

    /// <summary>
    /// Updates the network parameters based on calculated gradients.
    /// </summary>
    private void UpdateNetworkParameters()
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
    /// Gets metadata about the LSTM model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the LSTM model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the LSTM model, including its architecture,
    /// layer configuration, and other relevant parameters. This information is useful for model
    /// management, tracking experiments, and reporting.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your LSTM network.
    /// 
    /// The metadata includes:
    /// - What this model is designed to do
    /// - Details about the network architecture
    /// - Information about the layers and their sizes
    /// - The total number of parameters (learnable values)
    /// 
    /// This information is useful for:
    /// - Documentation
    /// - Comparing different LSTM configurations
    /// - Debugging and analysis
    /// - Sharing your model with others
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Count LSTM layers and get their sizes
        int lstmLayerCount = 0;
        var lstmSizes = new List<int>();

        foreach (var layer in Layers)
        {
            if (IsLSTMLayer(layer))
            {
                lstmLayerCount++;
                var outputShape = layer.GetOutputShape();
                if (outputShape.Length == 0)
                {
                    throw new InvalidOperationException("LSTM layer output shape is empty.");
                }

                // Hidden size is the last dimension of output shape
                int hiddenSize = outputShape[^1];
                lstmSizes.Add(hiddenSize);
            }
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.LSTMNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LSTMLayerCount", lstmLayerCount },
                { "LSTMLayerSizes", lstmSizes },
                { "TotalLayers", Layers.Count },
                { "TotalParameters", ParameterCount },
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes LSTM-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the LSTM model to a binary stream. It serializes network-specific
    /// parameters, including layer configurations and internal state trackers. This allows the complete
    /// state to be restored later for continued training or inference.
    /// </para>
    /// <para><b>For Beginners:</b> This saves your LSTM network to a file.
    /// 
    /// When saving the LSTM model:
    /// - All the network's learned parameters are saved
    /// - Layer structure and configuration are saved
    /// - Any internal state information is saved
    /// 
    /// This allows you to:
    /// - Save your progress after training
    /// - Share trained models with others
    /// - Load the model later for additional training or making predictions
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
    }

    /// <summary>
    /// Deserializes LSTM-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the state of a previously saved LSTM model from a binary stream. It restores
    /// the network-specific parameters, including layer configurations and internal state trackers,
    /// allowing the model to continue from exactly where it left off.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a saved LSTM network from a file.
    /// 
    /// When loading the LSTM model:
    /// - All the network's learned parameters are restored
    /// - Layer structure and configuration are restored
    /// - Any internal state information is restored
    /// 
    /// This lets you:
    /// - Continue working with a model you trained earlier
    /// - Use models that someone else has trained
    /// - Apply a trained model to new data for predictions
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
    }

    /// <summary>
    /// Creates a new instance of the LSTM Neural Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new LSTM Neural Network instance with the same architecture and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the LSTM Neural Network with the same architecture and activation
    /// functions as the current instance. It's used in scenarios where a fresh copy of the model is needed
    /// while maintaining the same configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a brand new copy of the LSTM network with the same setup.
    /// 
    /// Think of it like creating a clone of the network:
    /// - The new network has the same architecture (structure)
    /// - It has the same activation functions for all gates
    /// - It uses the same loss function
    /// - But it's a completely separate instance with its own parameters
    /// 
    /// This is useful when you want to:
    /// - Create multiple networks with identical settings
    /// - Compare how different initializations affect learning
    /// - Set up ensemble learning with multiple similar networks
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Determine which constructor to use based on whether we're using scalar or vector activations
        if (VectorActivation != null || ForgetGateVectorActivation != null ||
            InputGateVectorActivation != null || CellGateVectorActivation != null ||
            OutputGateVectorActivation != null)
        {
            // Use the vector activation constructor
            return new LSTMNeuralNetwork<T>(
                this.Architecture,
                LossFunction,
                VectorActivation,
                ForgetGateVectorActivation,
                InputGateVectorActivation,
                CellGateVectorActivation,
                OutputGateVectorActivation);
        }
        else
        {
            // Use the scalar activation constructor
            return new LSTMNeuralNetwork<T>(
                this.Architecture,
                LossFunction,
                ScalarActivation,
                ForgetGateActivation,
                InputGateActivation,
                CellGateActivation,
                OutputGateActivation);
        }
    }
}
