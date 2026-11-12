using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 20, 10 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new LSTMNeuralNetwork&lt;float&gt;())
///     .Build(trainX, trainY);
/// var output = result.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Long Short-Term Memory", "https://www.bioinf.jku.at/publications/older/2604.pdf", Year = 1997, Authors = "Sepp Hochreiter, Jurgen Schmidhuber")]
public partial class LSTMNeuralNetwork<T> : SequenceModelLayoutBase<T>
{
    private readonly LSTMOptions _options;

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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public LSTMNeuralNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 1), outputActivation: (IActivationFunction<T>?)null)
    {
    }

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

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Set to inference mode
        SetTrainingMode(false);

        // Simple layer-by-layer forward pass (wrapped in the #1622 verify-then-trust compiled gate;
        // no-op unless acceleration is engaged). Each layer (including LSTM layers) handles its own
        // time-stepping internally — matching the industry-standard approach used by GRU networks.
        return Accelerate(input, () =>
        {
            var current = input;

            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        });
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
                hiddenState.Fill(NumOps.Zero);
                cellState.Fill(NumOps.Zero);

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

        // Rent concatenated input tensor to avoid GC pressure in hot loop
        var combinedInput = TensorAllocator.Rent<T>([batchSize, inputSize + hiddenSize]);

        // Concatenate input and hidden state along feature dimension using Span
        var combinedSpan = combinedInput.Data.Span;
        var inputSpan = input.Data.Span;
        var hiddenSpan = hiddenState.Data.Span;
        for (int b = 0; b < batchSize; b++)
        {
            int combinedOffset = b * (inputSize + hiddenSize);
            inputSpan.Slice(b * inputSize, inputSize).CopyTo(combinedSpan.Slice(combinedOffset, inputSize));
            hiddenSpan.Slice(b * hiddenSize, hiddenSize).CopyTo(combinedSpan.Slice(combinedOffset + inputSize, hiddenSize));
        }

        // Forward through the layer to get all gate values
        var gateOutputs = Layers[layerIndex].Forward(combinedInput);

        // Verify gate outputs have expected shape
        if (gateOutputs.Shape[1] < 4 * hiddenSize)
        {
            throw new InvalidOperationException(
                $"Expected gate outputs of size {4 * hiddenSize}, but got {gateOutputs.Shape[1]}");
        }

        // Rent gate tensors to avoid GC pressure in hot loop
        var forgetGate = TensorAllocator.Rent<T>([batchSize, hiddenSize]);
        var inputGate = TensorAllocator.Rent<T>([batchSize, hiddenSize]);
        var cellGate = TensorAllocator.Rent<T>([batchSize, hiddenSize]);
        var outputGate = TensorAllocator.Rent<T>([batchSize, hiddenSize]);

        // Extract gate values using Span for zero-copy slicing
        var gateSpan = gateOutputs.Data.Span;
        var fgSpan = forgetGate.Data.Span;
        var igSpan = inputGate.Data.Span;
        var cgSpan = cellGate.Data.Span;
        var ogSpan = outputGate.Data.Span;
        int gateStride = 4 * hiddenSize;
        for (int b = 0; b < batchSize; b++)
        {
            int gateOffset = b * gateStride;
            int hOffset = b * hiddenSize;
            gateSpan.Slice(gateOffset, hiddenSize).CopyTo(fgSpan.Slice(hOffset, hiddenSize));
            gateSpan.Slice(gateOffset + hiddenSize, hiddenSize).CopyTo(igSpan.Slice(hOffset, hiddenSize));
            gateSpan.Slice(gateOffset + 2 * hiddenSize, hiddenSize).CopyTo(cgSpan.Slice(hOffset, hiddenSize));
            gateSpan.Slice(gateOffset + 3 * hiddenSize, hiddenSize).CopyTo(ogSpan.Slice(hOffset, hiddenSize));
        }

        // Apply appropriate activations to each gate
        forgetGate = NeuralNetworkHelper<T>.ApplyActivation(forgetGate, ForgetGateActivation, ForgetGateVectorActivation);
        inputGate = NeuralNetworkHelper<T>.ApplyActivation(inputGate, InputGateActivation, InputGateVectorActivation);
        cellGate = NeuralNetworkHelper<T>.ApplyActivation(cellGate, CellGateActivation, CellGateVectorActivation);
        outputGate = NeuralNetworkHelper<T>.ApplyActivation(outputGate, OutputGateActivation, OutputGateVectorActivation);

        // Rent new state tensors
        var newCellState = TensorAllocator.Rent<T>(cellState._shape);
        var newHiddenState = TensorAllocator.Rent<T>(hiddenState._shape);

        // Efficiently handle vector or scalar activation
        if (VectorActivation != null)
        {
            // VECTORIZED: c_t = f_t * c_{t-1} + i_t * g_t using Engine
            var forgetComponent = Engine.TensorMultiply(forgetGate, cellState);
            var inputComponent = Engine.TensorMultiply(inputGate, cellGate);
            newCellState = Engine.TensorAdd(forgetComponent, inputComponent);

            // Apply vector activation to cell state, then h_t = o_t * tanh(c_t)
            var activatedCellState = VectorActivation.Activate(newCellState);
            newHiddenState = Engine.TensorMultiply(outputGate, activatedCellState);
        }
        else
        {
            // VECTORIZED: c_t = f_t * c_{t-1} + i_t * g_t
            var forgetComponent = Engine.TensorMultiply(forgetGate, cellState);
            var inputComponent = Engine.TensorMultiply(inputGate, cellGate);
            newCellState = Engine.TensorAdd(forgetComponent, inputComponent);

            // Apply scalar activation element-wise, then h_t = o_t * activated(c_t)
            IActivationFunction<T> activation = ScalarActivation ?? new TanhActivation<T>();
            var activatedCellState = newCellState.Transform((x, _) => activation.Activate(x));
            newHiddenState = Engine.TensorMultiply(outputGate, activatedCellState);
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

        // Copy data for this time step: [batch_size, features]
        if (batchSize == 0 || features == 0)
            return TensorAllocator.Rent<T>(new int[] { batchSize, features });

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            return Engine.Reshape(
                Engine.TensorSlice(input, [0, timeStep, 0], [batchSize, 1, features]), [batchSize, features]).Clone();
        }
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
        int[] shape = tensors[0]._shape;

        // Create result shape with time dimension inserted at position 1
        int[] resultShape = new int[shape.Length + 1];
        resultShape[0] = shape[0]; // Batch size
        resultShape[1] = tensors.Count; // Sequence length

        for (int i = 1; i < shape.Length; i++)
        {
            resultShape[i + 1] = shape[i];
        }

        if (shape.Length != 2)
        {
            throw new NotSupportedException("Only 2D tensor stacking is currently supported");
        }

        // Stack the [batch, features] steps along a new time axis 1 (a plain copy, off the tape).
        if (resultShape[0] == 0 || resultShape[2] == 0)
            return TensorAllocator.Rent<T>(resultShape);

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            return Tensor<T>.Stack(tensors.ToArray(), 1);
        }
    }

    /// <summary>
    /// Adds a batch dimension to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with an additional batch dimension.</returns>
    private Tensor<T> AddBatchDimension(Tensor<T> input)
    {
        int[] inputShape = input._shape;
        int[] resultShape = new int[inputShape.Length + 1];

        // Add batch dimension of size 1
        resultShape[0] = 1;

        // Copy remaining dimensions
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 1] = inputShape[i];
        }

        // Copy data (ranks 1 and 2; other ranks get an uncopied result, as before)
        return inputShape.Length == 1 || inputShape.Length == 2
            ? DetachedReshapeCopy(input, resultShape)
            : TensorAllocator.Rent<T>(resultShape);
    }

    /// <summary>
    /// Removes a batch dimension from a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with the batch dimension removed.</returns>
    private Tensor<T> RemoveBatchDimension(Tensor<T> input)
    {
        int[] inputShape = input._shape;

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

        // Copy data (ranks 2 and 3; other ranks get an uncopied result, as before)
        return inputShape.Length == 2 || inputShape.Length == 3
            ? DetachedReshapeCopy(input, resultShape)
            : TensorAllocator.Rent<T>(resultShape);
    }

    /// <summary>
    /// Adds a time dimension to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with an additional time dimension.</returns>
    private Tensor<T> AddTimeDimension(Tensor<T> input)
    {
        int[] inputShape = input._shape;
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

        // Copy data (rank 2; other ranks get an uncopied result, as before)
        return inputShape.Length == 2
            ? DetachedReshapeCopy(input, resultShape)
            : TensorAllocator.Rent<T>(resultShape);
    }

    /// <summary>
    /// Removes a time dimension from a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with the time dimension removed.</returns>
    private Tensor<T> RemoveTimeDimension(Tensor<T> input)
    {
        int[] inputShape = input._shape;

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

        // Copy data (rank 3; other ranks get an uncopied result, as before)
        return inputShape.Length == 3
            ? DetachedReshapeCopy(input, resultShape)
            : TensorAllocator.Rent<T>(resultShape);
    }

    /// <summary>
    /// Adds both batch and time dimensions to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with additional batch and time dimensions.</returns>
    private Tensor<T> AddBatchAndTimeDimensions(Tensor<T> input)
    {
        int[] inputShape = input._shape;
        int[] resultShape = new int[inputShape.Length + 2];

        // Add batch and time dimensions of size 1
        resultShape[0] = 1;
        resultShape[1] = 1;

        // Copy remaining dimensions
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 2] = inputShape[i];
        }

        // Copy data (rank 1; other ranks get an uncopied result, as before)
        return inputShape.Length == 1
            ? DetachedReshapeCopy(input, resultShape)
            : TensorAllocator.Rent<T>(resultShape);
    }

    /// <summary>
    /// Removes both batch and time dimensions from a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor with batch and time dimensions removed.</returns>
    private Tensor<T> RemoveBatchAndTimeDimensions(Tensor<T> input)
    {
        int[] inputShape = input._shape;

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

        // Copy data (rank 3; other ranks get an uncopied result, as before)
        return inputShape.Length == 3
            ? DetachedReshapeCopy(input, resultShape)
            : TensorAllocator.Rent<T>(resultShape);
    }

    /// <summary>
    /// The same elements in a new shape of equal size, as a plain copy (off the tape), as the
    /// element-wise copies were.
    /// </summary>
    private Tensor<T> DetachedReshapeCopy(Tensor<T> input, int[] shape)
    {
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            return Engine.Reshape(input, shape).Clone();
        }
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
        if (hiddenGradient._shape.SequenceEqual(outputShape))
        {
            CopyFlatPrefix(hiddenGradient, transformedGradient, hiddenGradient.Length);
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
            CopyFlatPrefix(hiddenGradient, transformedGradient, Math.Min(hiddenGradient.Length, transformedGradient.Length));
            return transformedGradient;
        }

        // Rows b < min(first dims) of the flat [rows, features] layouts that both tensors hold.
        int rows = Math.Min(outputFirstDim, hiddenFirstDim);
        if (hiddenFeatures > 0) rows = Math.Min(rows, hiddenGradient.Length / hiddenFeatures);
        if (outputFeatures > 0) rows = Math.Min(rows, transformedGradient.Length / outputFeatures);

        // Transform based on feature dimension relationship
        if (hiddenFeatures == outputFeatures)
        {
            // Same feature dimension, might differ in batch/sequence dims
            CopyFlatPrefix(hiddenGradient, transformedGradient, Math.Min(hiddenGradient.Length, transformedGradient.Length));
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
            // Output feature outF averages hidden features [outF * H / O, (outF + 1) * H / O): a sum through
            // a bucket indicator, divided by the bucket size.
            if (rows > 0 && outputFeatures > 0)
            {
                var bucketOfHidden = Enumerable.Range(0, hiddenFeatures)
                    .Select(hidF => Enumerable.Range(0, outputFeatures)
                        .First(outF => hidF < ((outF + 1) * hiddenFeatures) / outputFeatures))
                    .ToArray();
                using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                {
                    var indicator = Engine.TensorOneHot<T>(new Tensor<int>(bucketOfHidden, [hiddenFeatures]), outputFeatures);
                    var counts = Engine.ReduceSum(indicator, new[] { 0 }, keepDims: true);   // [1, O], all >= 1
                    var hiddenRows = FlatRows(hiddenGradient, rows, hiddenFeatures);
                    Engine.TensorDivide(Engine.TensorMatMul(hiddenRows, indicator), counts)
                        .CopyTo(transformedGradient.AsWritableSpan().Slice(0, rows * outputFeatures));
                }
            }
        }
        else
        {
            // Output has more features - replicate gradients
            // NOTE: This is an approximation using gradient replication. For learned projections,
            // the correct approach would use the transpose of the projection matrix.
            // This approximation is suitable when no explicit projection layers exist.
            // Output feature outF copies hidden feature outF * H / O.
            if (rows > 0 && outputFeatures > 0 && hiddenFeatures > 0)
            {
                var sourceFeature = Enumerable.Range(0, outputFeatures)
                    .Select(outF => (outF * hiddenFeatures) / outputFeatures).ToArray();
                using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                {
                    Engine.TensorGather(FlatRows(hiddenGradient, rows, hiddenFeatures), new Tensor<int>(sourceFeature, [outputFeatures]), axis: 1)
                        .CopyTo(transformedGradient.AsWritableSpan().Slice(0, rows * outputFeatures));
                }
            }
        }

        return transformedGradient;
    }

    /// <summary>The first rows * features elements of a tensor, row-major, as [rows, features].</summary>
    private Tensor<T> FlatRows(Tensor<T> source, int rows, int features)
    {
        return Engine.Reshape(
            Engine.TensorSlice(Engine.Reshape(source, [source.Length]), [0], [rows * features]), [rows, features]);
    }

    /// <summary>Copies the first count elements of source into destination (row-major order).</summary>
    private void CopyFlatPrefix(Tensor<T> source, Tensor<T> destination, int count)
    {
        if (count <= 0) return;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            Engine.TensorSlice(Engine.Reshape(source, [source.Length]), [0], [count])
                .CopyTo(destination.AsWritableSpan().Slice(0, count));
        }
    }

    /// <summary>
    /// Checks if a tensor contains only zero values.
    /// </summary>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if the tensor contains only zeros; otherwise, false.</returns>
    private bool IsZeroTensor(Tensor<T> tensor)
    {
        // Every |x - 0| below 1e-8 (MathHelper.AlmostEqual's default); NaN is not.
        if (tensor.Length == 0) return true;
        var nearZero = Engine.TensorLessThan(Engine.TensorAbs(tensor), NumOps.FromDouble(1e-8));
        return NumOps.Equals(Engine.TensorSum(nearZero), NumOps.FromDouble(tensor.Length));
    }

    /// <summary>
    /// Dictionary to store activations and states from the forward pass.
    /// </summary>
    [Scratch]
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

        // Calculate activated cell state and its derivative per element (scalar activation dispatch, as before)
        Tensor<T> c_tanh = new Tensor<T>([batchSize, hiddenSize]);
        Tensor<T> cDerivative = new Tensor<T>([batchSize, hiddenSize]);
        var tanh = new TanhActivation<T>();
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                c_tanh[b, j] = ScalarActivation != null
                    ? ScalarActivation.Activate(c[b, j])
                    : tanh.Activate(c[b, j]);
            }
        }

        // Ensure we have gate activations
        if (i == null || f == null || g == null || o == null)
        {
            throw new InvalidOperationException("Gate activations must be stored during forward pass for backpropagation.");
        }

        // Gradient for c through tanh(c): the activation's derivative, or 1 - tanh(c)^2
        if (ScalarActivation != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    cDerivative[b, j] = ScalarActivation.Derivative(c[b, j]);
                }
            }
        }

        // Cell gate derivative: the activation's derivative per element, or 1 - g^2
        Tensor<T>? gDerivative = null;
        if (CellGateActivation != null)
        {
            gDerivative = new Tensor<T>([batchSize, hiddenSize]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    gDerivative[b, j] = CellGateActivation.Derivative(g[b, j]);
                }
            }
        }

        // Get LSTM layer parameters
        var parameters = GetLSTMLayerParameters(layerIndex);

        Tensor<T> dx, dWi, dWf, dWg, dWo, dUi, dUf, dUg, dUo, dbi, dbf, dbg, dbo;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var ones = new Tensor<T>([batchSize, hiddenSize]);
            ones.Fill(NumOps.One);
            if (ScalarActivation == null)
                cDerivative = Engine.TensorSubtract(ones, Engine.TensorMultiply(c_tanh, c_tanh));

            // Backward pass through output gate
            var do_ = Engine.TensorMultiply(dh, c_tanh);

            // Gradient for tanh(c), then for c (plus the gradient from the next cell state)
            var dc_tanh = Engine.TensorMultiply(dh, o);
            var dcCurrent = Engine.TensorMultiply(dc_tanh, cDerivative);
            if (dcPrev != null)
                dcCurrent = Engine.TensorAdd(dcCurrent, dc);

            // Gradients for gates
            var di = Engine.TensorMultiply(dcCurrent, g);
            var dg = Engine.TensorMultiply(dcCurrent, i);
            var df = Engine.TensorMultiply(dcCurrent, cPrev);

            // Gradient to previous cell state
            dc = Engine.TensorMultiply(dcCurrent, f);

            // Apply gate activation derivatives. Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
            var di_input = Engine.TensorMultiply(di, Engine.TensorMultiply(i, Engine.TensorSubtract(ones, i)));
            var df_input = Engine.TensorMultiply(df, Engine.TensorMultiply(f, Engine.TensorSubtract(ones, f)));
            var do_input = Engine.TensorMultiply(do_, Engine.TensorMultiply(o, Engine.TensorSubtract(ones, o)));
            var dg_input = Engine.TensorMultiply(
                dg, gDerivative ?? Engine.TensorSubtract(ones, Engine.TensorMultiply(g, g)));

            // Bias gradients: sums over the batch
            dbi = Engine.ReduceSum(di_input, new[] { 0 }, keepDims: false);
            dbf = Engine.ReduceSum(df_input, new[] { 0 }, keepDims: false);
            dbg = Engine.ReduceSum(dg_input, new[] { 0 }, keepDims: false);
            dbo = Engine.ReduceSum(do_input, new[] { 0 }, keepDims: false);

            // Input gradients: dx = sum over gates of dgate_input @ W_gate
            var wi = Tensor<T>.FromMatrix(parameters.Wi);
            var wf = Tensor<T>.FromMatrix(parameters.Wf);
            var wg = Tensor<T>.FromMatrix(parameters.Wg);
            var wo = Tensor<T>.FromMatrix(parameters.Wo);
            dx = Engine.TensorAdd(
                Engine.TensorAdd(Engine.TensorMatMul(di_input, wi), Engine.TensorMatMul(df_input, wf)),
                Engine.TensorAdd(Engine.TensorMatMul(dg_input, wg), Engine.TensorMatMul(do_input, wo)));

            // Weight gradients: dgate_input^T @ x (input) and dgate_input^T @ hPrev (hidden)
            var xRows = Engine.TensorSlice(x, [0, 0], [batchSize, inputSize]);
            var hRows = Engine.TensorSlice(hPrev, [0, 0], [batchSize, hiddenSize]);
            dWi = Engine.TensorMatMul(Engine.TensorTranspose(di_input), xRows);
            dWf = Engine.TensorMatMul(Engine.TensorTranspose(df_input), xRows);
            dWg = Engine.TensorMatMul(Engine.TensorTranspose(dg_input), xRows);
            dWo = Engine.TensorMatMul(Engine.TensorTranspose(do_input), xRows);
            dUi = Engine.TensorMatMul(Engine.TensorTranspose(di_input), hRows);
            dUf = Engine.TensorMatMul(Engine.TensorTranspose(df_input), hRows);
            dUg = Engine.TensorMatMul(Engine.TensorTranspose(dg_input), hRows);
            dUo = Engine.TensorMatMul(Engine.TensorTranspose(do_input), hRows);
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
        if (rows == 0 || cols == 0) return new Matrix<T>(rows, cols);
        return Tensor<T>.FromVector(parameters.GetSubVector(startIndex, rows * cols), [rows, cols]).ToMatrix();
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
        if (length == 0) return new Vector<T>(0);
        return parameters.GetSubVector(startIndex, length);
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

            // Combine all vectors into the final gradient vector: input weights, recurrent weights, biases
            var parts = new[]
            {
                dWiVector, dWfVector, dWgVector, dWoVector,
                dUiVector, dUfVector, dUgVector, dUoVector,
                dbi.ToVector(), dbf.ToVector(), dbg.ToVector(), dbo.ToVector()
            };
            int index = 0;
            var gradientSpan = gradients.AsWritableSpan();
            foreach (var part in parts)
            {
                if (index + part.Length > gradientSpan.Length)
                    throw new ArgumentOutOfRangeException(nameof(layerIndex),
                        $"LSTM gradients ({index + part.Length}+ values) exceed the layer's {gradientSpan.Length} parameters.");
                part.AsSpan().CopyTo(gradientSpan.Slice(index));
                index += part.Length;
            }

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
            updatedSlice.AsSpan().Slice(0, updateLength).CopyTo(currentParams.AsWritableSpan());

            // Apply updated parameters to the layer
            layer.UpdateParameters(currentParams);
        }
    }

    /// <summary>
    /// Updates the network parameters based on calculated gradients.
    /// </summary>
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LSTMLayerCount", lstmLayerCount },
                { "LSTMLayerSizes", lstmSizes },
                { "TotalLayers", Layers.Count },
                { "TotalParameters", ParameterCount },
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize }
            },
            ModelDataProvider = () => SerializeForMetadata()
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

}
