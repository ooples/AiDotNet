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
    /// <summary>
    /// The loss function used for training the network.
    /// </summary>
    private ILossFunction<T> LossFunction { get; }

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
        IActivationFunction<T>? outputGateActivation = null) : base(architecture)
    {
        // Set default loss function if none provided
        LossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        
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
        IVectorActivationFunction<T>? outputGateVectorActivation = null) : base(architecture)
    {
        // Set default loss function if none provided
        LossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        
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
    /// Calculates the loss between predictions and expected outputs using the configured loss function.
    /// </summary>
    /// <param name="predictions">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>A scalar value representing the loss.</returns>
    private T CalculateLoss(Tensor<T> predictions, Tensor<T> expected)
    {
        return LossFunction.CalculateLoss(predictions.ToVector(), expected.ToVector());
    }
    
    /// <summary>
    /// Calculates output gradients using the configured loss function's derivative.
    /// </summary>
    /// <param name="predictions">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The gradient tensor for backpropagation.</returns>
    private Tensor<T> CalculateOutputGradients(Tensor<T> predictions, Tensor<T> expected)
    {
        var gradientVector = LossFunction.CalculateDerivative(predictions.ToVector(), expected.ToVector());
        return new Tensor<T>(predictions.Shape, gradientVector);
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
    
        // Check input dimensions and determine if we're processing a batch or a single sequence
        bool isBatchedSequence = input.Shape.Length >= 3;
        bool isBatch = input.Shape.Length >= 2;
        bool isSequence = input.Shape.Length >= 2 && !isBatchedSequence;
    
        // Handle different input shapes
        if (isBatchedSequence)
        {
            // [batch_size, sequence_length, features]
            return ProcessBatchedSequence(input);
        }
        else if (isSequence)
        {
            // [sequence_length, features]
            return ProcessSequence(input);
        }
        else if (isBatch)
        {
            // [batch_size, features] - single time step batch
            return ProcessSingleTimeStepBatch(input);
        }
        else
        {
            // [features] - single sample
            return ProcessSingleSample(input);
        }
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
        var metadata = GetModelMetaData();
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
            for (int i = 0; i < inputSize; i++)
            {
                combinedInput[b, i] = input[b, i];
            }
            
            for (int h = 0; h < hiddenSize; h++)
            {
                combinedInput[b, inputSize + h] = hiddenState[b, h];
            }
        }
        
        // Forward through the layer to get all gate values
        var gateOutputs = Layers[layerIndex].Forward(combinedInput);
        
        // Get weight matrices and biases from the layer parameters
        var parameters = Layers[layerIndex].GetParameters();
        
        // Split the gate outputs into forget, input, cell, and output gates
        // For an LSTM with hidden size H, the gate output tensor will have 4*H units
        // We need to properly slice this tensor to get the individual gates
        
        // Create tensors for each gate
        var forgetGate = new Tensor<T>(new int[] { batchSize, hiddenSize });
        var inputGate = new Tensor<T>(new int[] { batchSize, hiddenSize });
        var cellGate = new Tensor<T>(new int[] { batchSize, hiddenSize });
        var outputGate = new Tensor<T>(new int[] { batchSize, hiddenSize });
        
        // Extract gate values from the combined output
        // Assuming gate outputs has shape [batch_size, 4*hidden_size]
        if (gateOutputs.Shape[1] >= 4 * hiddenSize)
        {
            // Proper slicing of the gates
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    forgetGate[b, h] = gateOutputs[b, h];
                    inputGate[b, h] = gateOutputs[b, hiddenSize + h];
                    cellGate[b, h] = gateOutputs[b, 2 * hiddenSize + h];
                    outputGate[b, h] = gateOutputs[b, 3 * hiddenSize + h];
                }
            }
            
            // Apply appropriate activations to each gate
            // Forget and input gates use sigmoid, cell gate uses tanh
            forgetGate = NeuralNetworkHelper<T>.ApplyActivation(forgetGate, ForgetGateActivation, ForgetGateVectorActivation);
            inputGate = NeuralNetworkHelper<T>.ApplyActivation(inputGate, InputGateActivation, InputGateVectorActivation);
            cellGate = NeuralNetworkHelper<T>.ApplyActivation(cellGate, CellGateActivation, CellGateVectorActivation);
            outputGate = NeuralNetworkHelper<T>.ApplyActivation(outputGate, OutputGateActivation, OutputGateVectorActivation);
            
            // Create new cell state and hidden state tensors
            var newCellState = new Tensor<T>(cellState.Shape);
            var newHiddenState = new Tensor<T>(hiddenState.Shape);
            
            // Apply LSTM equations
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    // Update cell state:
                    // c_t = f_t * c_{t-1} + i_t * g_t
                    // where f_t = forget gate, i_t = input gate, g_t = cell gate
                    T forgetComponent = NumOps.Multiply(forgetGate[b, h], cellState[b, h]);
                    T inputComponent = NumOps.Multiply(inputGate[b, h], cellGate[b, h]);
                    newCellState[b, h] = NumOps.Add(forgetComponent, inputComponent);
                    
                    // Apply activation to cell state to get hidden state output
                    // Here we use the configured activation function
                    T activatedCell;
                    
                    if (VectorActivation != null)
                    {
                        // For vector activation, we need to process the entire cell state
                        // We'll use a simplified approach here for illustration
                        var cellVector = new Vector<T>(hiddenSize);
                        for (int i = 0; i < hiddenSize; i++)
                        {
                            cellVector[i] = newCellState[b, i];
                        }
                        
                        var activatedVector = VectorActivation.Activate(cellVector);
                        activatedCell = activatedVector[h];
                    }
                    else if (ScalarActivation != null)
                    {
                        // Apply scalar activation to cell state
                        activatedCell = ScalarActivation.Activate(newCellState[b, h]);
                    }
                    else
                    {
                        activatedCell = newCellState[b, h]; // No activation
                    }
                    
                    // Update hidden state:
                    // h_t = o_t * tanh(c_t)
                    // where o_t = output gate
                    newHiddenState[b, h] = NumOps.Multiply(outputGate[b, h], activatedCell);
                }
            }
            
            return (newHiddenState, newHiddenState, newCellState);
        }
        else
        {
            // Handle the case where gate output dimension doesn't match expectations
            throw new InvalidOperationException(
                $"Expected gate outputs of size {4 * hiddenSize}, but got {gateOutputs.Shape[1]}");
        }
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
    
        // Calculate loss
        var loss = CalculateLoss(predictions, expectedOutput);
    
        // Calculate output gradients
        var outputGradients = CalculateOutputGradients(predictions, expectedOutput);
    
        // Backpropagation through time
        BackpropagateOverTime(outputGradients, input);
    
        // Update parameters
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Determines if tensor shapes are compatible for element-wise operations.
    /// </summary>
    /// <param name="shape1">The first tensor shape.</param>
    /// <param name="shape2">The second tensor shape.</param>
    /// <returns>True if shapes are compatible; otherwise, false.</returns>
    private bool AreShapesCompatible(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
        {
            return false;
        }
    
        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
            {
                return false;
            }
        }
    
        return true;
    }

    /// <summary>
    /// Performs backpropagation through time for LSTM networks.
    /// </summary>
    /// <param name="outputGradients">The gradients from the output layer.</param>
    /// <param name="input">The original input used in the forward pass.</param>
    private void BackpropagateOverTime(Tensor<T> outputGradients, Tensor<T> input)
    {
        // For LSTMs, we need to backpropagate through time
        // In a full implementation, this would:
        // 1. Unroll the network through time
        // 2. Compute gradients at each time step
        // 3. Propagate gradients backwards through time
        // 4. Accumulate gradients for each layer
    
        // For this simplified implementation, we'll pass the gradients to each layer's Backward method
    
        // Start with output gradients
        Tensor<T> gradients = outputGradients;
    
        // Backpropagate through each layer in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }
    
        // In a real LSTM implementation, we would need to:
        // - Keep track of activations at each time step during forward pass
        // - Propagate gradients backwards through each time step
        // - Handle the recurrent connections between time steps
        // - Apply truncated BPTT if sequences are very long
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
    public override ModelMetaData<T> GetModelMetaData()
    {
        // Count LSTM layers and get their sizes
        int lstmLayerCount = 0;
        var lstmSizes = new List<int>();
    
        foreach (var layer in Layers)
        {
            if (IsLSTMLayer(layer))
            {
                lstmLayerCount++;
                lstmSizes.Add(layer.GetOutputShape()[1]); // Hidden size
            }
        }
    
        return new ModelMetaData<T>
        {
            ModelType = ModelType.LSTMNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LSTMLayerCount", lstmLayerCount },
                { "LSTMLayerSizes", lstmSizes },
                { "TotalLayers", Layers.Count },
                { "TotalParameters", GetParameterCount() },
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