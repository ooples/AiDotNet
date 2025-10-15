namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Long Short-Term Memory (LSTM) layer for processing sequential data.
/// </summary>
/// <remarks>
/// <para>
/// The LSTM layer is a specialized type of recurrent neural network (RNN) that is designed to capture long-term
/// dependencies in sequential data. It uses a cell state and a series of gates (forget, input, and output) to control
/// the flow of information through the network, allowing it to remember important patterns over long sequences while
/// forgetting irrelevant information.
/// </para>
/// <para><b>For Beginners:</b> An LSTM layer is like a smart memory system for your AI.
/// 
/// Think of it like a notepad with special features:
/// - It can remember important information for a long time (unlike simpler neural networks)
/// - It can forget irrelevant details (using its "forget gate")
/// - It can decide what new information to write down (using its "input gate")
/// - It can decide what information to share (using its "output gate")
/// 
/// LSTMs are great for:
/// - Text generation and language understanding
/// - Time series prediction (like stock prices)
/// - Speech recognition
/// - Any task where the order and context of information matters
/// 
/// For example, when processing the sentence "The clouds are in the ___", an LSTM would remember
/// that "clouds" appeared earlier, helping it predict "sky" as the missing word.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LSTMLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The size of each input vector (number of features).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the size of each input vector, which is the number of features in the input data.
    /// It is used to determine the dimensions of the weight matrices and for various calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This stores how many features each data point has.
    /// 
    /// For example:
    /// - In text processing, this might be the size of word embeddings
    /// - In time series data, this would be the number of measurements at each time step
    /// - In image processing, this could be the number of features after convolutional layers
    /// 
    /// If you're processing words represented as 100-dimensional vectors, this value would be 100.
    /// </para>
    /// </remarks>
    private readonly int _inputSize;

    /// <summary>
    /// The size of the hidden state and cell state (number of LSTM units).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the size of the hidden state and cell state, which is also the number of LSTM units
    /// in the layer. It determines the capacity of the LSTM to learn and remember patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This stores how much "memory" the LSTM layer has.
    /// 
    /// The hidden size:
    /// - Determines how much information the LSTM can remember
    /// - Affects the capacity to learn complex patterns
    /// - Larger values can capture more complexity but require more computation
    /// 
    /// Think of it as the "brain size" of your LSTM - bigger means more capacity,
    /// but also more computation and training time.
    /// </para>
    /// </remarks>
    private readonly int _hiddenSize;

    /// <summary>
    /// Weights for the forget gate input connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the input vector to the forget gate. These weights determine
    /// how much of the previous cell state is forgotten based on the current input.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help the LSTM decide what to forget based on new input.
    /// 
    /// The forget gate:
    /// - Decides what information to throw away from the cell state
    /// - Uses these weights to process the current input
    /// - Values close to 0 mean "forget", values close to 1 mean "keep"
    /// 
    /// For example, when processing text, these weights might help forget information
    /// about previous subjects when a new subject is introduced.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsFi = default!;

    /// <summary>
    /// Weights for the input gate input connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the input vector to the input gate. These weights determine
    /// how much of the new candidate values are added to the cell state based on the current input.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help the LSTM decide what new information to store based on input.
    /// 
    /// The input gate:
    /// - Decides what new information to add to the cell state
    /// - Uses these weights to process the current input
    /// - Controls how much of the new candidate values should be added
    /// 
    /// For example, when processing text, these weights might help store important
    /// information about a new subject being discussed.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsIi = default!;

    /// <summary>
    /// Weights for the cell state input connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the input vector to the cell state candidate. These weights
    /// determine what new candidate values could be added to the cell state based on the current input.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help create potential new information to store.
    /// 
    /// The cell state candidate:
    /// - Creates new values that might be added to the cell state
    /// - Uses these weights to process the current input
    /// - Generates potential new information to remember
    /// 
    /// These weights help transform the input into a format that can be stored
    /// in the cell's memory.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsCi = default!;

    /// <summary>
    /// Weights for the output gate input connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the input vector to the output gate. These weights determine
    /// how much of the cell state is exposed as output based on the current input.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help the LSTM decide what to output based on input.
    /// 
    /// The output gate:
    /// - Decides what parts of the cell state to output
    /// - Uses these weights to process the current input
    /// - Controls how much information is shared with the next layer
    /// 
    /// For example, when processing text, these weights might help decide which
    /// parts of the accumulated context are relevant for predicting the next word.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsOi = default!;

    /// <summary>
    /// Weights for the forget gate hidden state connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the previous hidden state to the forget gate. These weights
    /// determine how much of the previous cell state is forgotten based on the previous hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help the LSTM decide what to forget based on previous context.
    /// 
    /// The forget gate also considers:
    /// - The previous hidden state (context from earlier time steps)
    /// - Uses these weights to process that context
    /// - Helps make forgetting decisions based on what the network already knows
    /// 
    /// These weights complement the input weights, allowing decisions to be based
    /// on both current input and previous context.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsFh = default!;

    /// <summary>
    /// Weights for the input gate hidden state connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the previous hidden state to the input gate. These weights
    /// determine how much of the new candidate values are added to the cell state based on the previous hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help the LSTM decide what to store based on previous context.
    /// 
    /// The input gate also considers:
    /// - The previous hidden state (context from earlier time steps)
    /// - Uses these weights to process that context
    /// - Helps make storing decisions based on what the network already knows
    /// 
    /// For example, these weights might help decide if new information conflicts with
    /// or complements existing knowledge.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsIh = default!;

    /// <summary>
    /// Weights for the cell state hidden state connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the previous hidden state to the cell state candidate. These
    /// weights determine what new candidate values could be added to the cell state based on the previous hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help create potential new information based on previous context.
    /// 
    /// The cell state candidate also considers:
    /// - The previous hidden state (context from earlier time steps)
    /// - Uses these weights to process that context
    /// - Helps generate new information based on existing knowledge
    /// 
    /// These weights work with the input weights to create candidate values
    /// that consider both current input and previous context.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsCh = default!;

    /// <summary>
    /// Weights for the output gate hidden state connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weights that connect the previous hidden state to the output gate. These weights
    /// determine how much of the cell state is exposed as output based on the previous hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These weights help the LSTM decide what to output based on previous context.
    /// 
    /// The output gate also considers:
    /// - The previous hidden state (context from earlier time steps)
    /// - Uses these weights to process that context
    /// - Helps decide what information to share based on existing knowledge
    /// 
    /// These weights complement the input weights for the output gate, allowing
    /// output decisions based on both current input and previous context.
    /// </para>
    /// </remarks>
    private Tensor<T> _weightsOh = default!;

    /// <summary>
    /// Bias for the forget gate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the bias values for the forget gate. These biases provide a baseline activation level
    /// for the forget gate, independent of the input or hidden state. In LSTMs, the forget gate bias is often
    /// initialized to a positive value to encourage remembering by default.
    /// </para>
    /// <para><b>For Beginners:</b> These values provide a starting point for the forget gate.
    /// 
    /// The bias:
    /// - Adds a base level to the gate's activation
    /// - Helps control the default behavior (remember or forget)
    /// - Is often set to positive values initially to help remember information
    /// 
    /// Think of it as setting the default position of the gate before any input
    /// or context is considered.
    /// </para>
    /// </remarks>
    private Tensor<T> _biasF = default!;

    /// <summary>
    /// Bias for the input gate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the bias values for the input gate. These biases provide a baseline activation level
    /// for the input gate, independent of the input or hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These values provide a starting point for the input gate.
    /// 
    /// The bias:
    /// - Adds a base level to the gate's activation
    /// - Helps control the default behavior (store or ignore new information)
    /// - Can be adjusted during training to optimize performance
    /// 
    /// This sets the default tendency to add new information before considering
    /// the specific input or context.
    /// </para>
    /// </remarks>
    private Tensor<T> _biasI = default!;

    /// <summary>
    /// Bias for the cell state candidate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the bias values for the cell state candidate calculation. These biases provide a baseline
    /// for the candidate values, independent of the input or hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These values provide a starting point for the cell candidate.
    /// 
    /// The bias:
    /// - Adds a base level to the candidate values
    /// - Helps set default patterns for new information
    /// - Adjusts during training to capture common patterns
    /// 
    /// These biases influence the default content of new information before
    /// considering specific inputs or context.
    /// </para>
    /// </remarks>
    private Tensor<T> _biasC = default!;

    /// <summary>
    /// Bias for the output gate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the bias values for the output gate. These biases provide a baseline activation level
    /// for the output gate, independent of the input or hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> These values provide a starting point for the output gate.
    /// 
    /// The bias:
    /// - Adds a base level to the gate's activation
    /// - Helps control the default behavior (output or hide information)
    /// - Can be adjusted during training to optimize performance
    /// 
    /// This sets the default tendency to share information before considering
    /// the specific input or context.
    /// </para>
    /// </remarks>
    private Tensor<T> _biasO = default!;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass. It is needed during the backward pass
    /// to compute gradients correctly. This field is reset when ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the most recent data that was fed into the layer.
    /// 
    /// The layer needs to remember the input:
    /// - To calculate how each input value affected the output
    /// - To determine how to propagate gradients during training
    /// - To ensure the backward pass works correctly
    /// 
    /// This is like keeping track of what you were given so you can explain
    /// how you processed it if asked.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The hidden state from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the hidden state from the most recent forward pass. It contains the output at each time
    /// step and is used during both the forward and backward passes. This field is reset when ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the output state at each time step.
    /// 
    /// The hidden state:
    /// - Contains the output of the LSTM at each time step
    /// - Carries information from earlier steps to later ones
    /// - Is used to make predictions or feed into the next layer
    /// 
    /// Think of it as the LSTM's short-term memory that's updated at each step
    /// and passed to the next step.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastHiddenState;

    /// <summary>
    /// The cell state from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the cell state from the most recent forward pass. The cell state is the main memory
    /// component of the LSTM, capable of retaining information over long sequences. This field is reset when
    /// ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the long-term memory of the LSTM at each time step.
    /// 
    /// The cell state:
    /// - Is the LSTM's main memory component
    /// - Can hold information for long sequences
    /// - Is carefully controlled by the gates
    /// 
    /// Think of it as the LSTM's long-term memory that can preserve important
    /// information even as new inputs are processed.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastCellState;

    /// <summary>
    /// The sigmoid activation function for element-wise operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the sigmoid activation function used for the gates in the LSTM. The sigmoid function
    /// outputs values between 0 and 1, which are used as "gates" to control the flow of information.
    /// </para>
    /// <para><b>For Beginners:</b> This function squeezes values between 0 and 1 for the gates.
    /// 
    /// The sigmoid function:
    /// - Converts any input to a value between 0 and 1
    /// - Is used for all gates (forget, input, output)
    /// - Allows the gates to be partially open or closed
    /// 
    /// For example, a sigmoid output of 0.8 means a gate is 80% open, allowing
    /// most but not all information to flow through.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _sigmoidActivation;

    /// <summary>
    /// The tanh activation function for element-wise operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the tanh activation function used for the cell state candidate in the LSTM. The tanh
    /// function outputs values between -1 and 1, which helps regulate the flow of information.
    /// </para>
    /// <para><b>For Beginners:</b> This function squeezes values between -1 and 1 for cell state calculations.
    /// 
    /// The tanh function:
    /// - Converts any input to a value between -1 and 1
    /// - Is used for cell state candidates
    /// - Allows for both positive and negative information
    /// 
    /// This helps the network learn more complex patterns by allowing values
    /// to either increase or decrease the cell state.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _tanhActivation;

    /// <summary>
    /// The sigmoid activation function for vector operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the vector-based sigmoid activation function, which operates on entire tensors at once
    /// rather than element by element. This can be more efficient for certain operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the regular sigmoid function but faster for large datasets.
    /// 
    /// The vector sigmoid function:
    /// - Does the same job as the regular sigmoid function
    /// - Processes entire groups of values at once
    /// - Can be more efficient on modern hardware
    /// 
    /// The mathematical result is the same, but the computation can be faster.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _sigmoidVectorActivation;

    /// <summary>
    /// The tanh activation function for vector operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the vector-based tanh activation function, which operates on entire tensors at once
    /// rather than element by element. This can be more efficient for certain operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the regular tanh function but faster for large datasets.
    /// 
    /// The vector tanh function:
    /// - Does the same job as the regular tanh function
    /// - Processes entire groups of values at once
    /// - Can be more efficient on modern hardware
    /// 
    /// The mathematical result is the same, but the computation can be faster.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _tanhVectorActivation;

    /// <summary>
    /// Flag indicating whether to use vector or scalar activation functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a flag indicating whether to use vector-based activation functions or scalar (element-wise)
    /// activation functions. Vector<double> functions operate on entire tensors at once, which can be more efficient for
    /// certain operations.
    /// </para>
    /// <para><b>For Beginners:</b> This flag chooses between two ways of applying activation functions.
    /// 
    /// When set to true:
    /// - Vector<double> activation functions are used
    /// - Processing is done on entire tensors at once
    /// - Can be faster for large models
    /// 
    /// When set to false:
    /// - Scalar activation functions are used
    /// - Processing is done one value at a time
    /// - More straightforward but potentially slower
    /// 
    /// The end result is mathematically the same, but the computation method differs.
    /// </para>
    /// </remarks>
    private readonly bool _useVectorActivation;

    /// <summary>
    /// Gets a dictionary containing the gradients for all trainable parameters after a backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the gradients computed during the backward pass, which indicate how each parameter
    /// should be updated to minimize the loss function. The dictionary keys correspond to parameter names, and
    /// the values are tensors containing the gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a learning notebook for the layer.
    /// 
    /// During training:
    /// - The layer calculates how it needs to change its internal values
    /// - These changes (gradients) are stored in this dictionary
    /// - Later, these values are used to update the weights and make the layer smarter
    /// 
    /// Each key in the dictionary refers to a specific part of the LSTM that needs updating,
    /// and the corresponding value shows how much and in what direction to change it.
    /// </para>
    /// </remarks>
    public Dictionary<string, Tensor<T>> Gradients { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property returns true because the LSTM layer has trainable parameters (weights and biases) that can be
    /// updated during training through backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the layer can learn from training data.
    /// 
    /// A value of true means:
    /// - This layer has internal values (weights and biases) that get updated during training
    /// - It can improve its performance as it sees more data
    /// - It actively participates in the learning process
    /// 
    /// Unlike some layers that just do fixed calculations, LSTM layers can adapt and learn
    /// from patterns in your data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="LSTMLayer{T}"/> class with scalar activation functions.
    /// </summary>
    /// <param name="inputSize">The size of each input vector (number of features).</param>
    /// <param name="hiddenSize">The size of the hidden state (number of LSTM units).</param>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="activation">The activation function to use for the cell state, defaults to tanh if not specified.</param>
    /// <param name="recurrentActivation">The activation function to use for the gates, defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an LSTM layer with the specified dimensions and activation functions. It initializes
    /// all the weights and biases needed for the LSTM gates (forget, input, cell state, and output). The weights
    /// are initialized using the Xavier/Glorot initialization technique, which helps with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new LSTM layer with your desired settings using standard activation functions.
    /// 
    /// When setting up this layer:
    /// - inputSize is how many features each data point has
    /// - hiddenSize is how much "memory" each LSTM unit will have
    /// - inputShape defines the expected dimensions of your data
    /// - activation controls how the cell state is processed (usually tanh)
    /// - recurrentActivation controls how the gates operate (usually sigmoid)
    /// 
    /// For example, if you're processing words represented as 100-dimensional vectors,
    /// inputSize would be 100. If you want 200 LSTM units, hiddenSize would be 200.
    /// </para>
    /// </remarks>
    public LSTMLayer(int inputSize, int hiddenSize, int[] inputShape,
        IActivationFunction<T>? activation = null,
        IActivationFunction<T>? recurrentActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, hiddenSize), activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _useVectorActivation = false;

        _sigmoidActivation = recurrentActivation ?? new SigmoidActivation<T>();
        _tanhActivation = activation ?? new TanhActivation<T>();

        Gradients = [];

        // Initialize weights
        _weightsFi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsIi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsCi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsOi = new Tensor<T>([_hiddenSize, _inputSize]);

        _weightsFh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsIh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsCh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsOh = new Tensor<T>([_hiddenSize, _hiddenSize]);

        // Initialize biases
        _biasF = new Tensor<T>([_hiddenSize]);
        _biasI = new Tensor<T>([_hiddenSize]);
        _biasC = new Tensor<T>([_hiddenSize]);
        _biasO = new Tensor<T>([_hiddenSize]);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LSTMLayer{T}"/> class with vector activation functions.
    /// </summary>
    /// <param name="inputSize">The size of each input vector (number of features).</param>
    /// <param name="hiddenSize">The size of the hidden state (number of LSTM units).</param>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="activation">The vector activation function to use for the cell state, defaults to tanh if not specified.</param>
    /// <param name="recurrentActivation">The vector activation function to use for the gates, defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an LSTM layer that uses vector activation functions, which operate on entire tensors
    /// at once rather than element by element. This can be more efficient for certain operations and allows for more
    /// complex activation patterns that consider relationships between different elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new LSTM layer using advanced vector-based activation functions.
    /// 
    /// Vector<double> activation functions:
    /// - Process entire groups of numbers at once, rather than one at a time
    /// - Can be more efficient on certain hardware
    /// - May capture more complex relationships between different values
    /// 
    /// When you might use this constructor instead of the standard one:
    /// - When working with very large models
    /// - When you need maximum performance
    /// - When using specialized activation functions that work on vectors
    /// 
    /// The basic functionality is the same as the standard constructor, but with
    /// potentially better performance for large-scale applications.
    /// </para>
    /// </remarks>
    public LSTMLayer(int inputSize, int hiddenSize, int[] inputShape,
        IVectorActivationFunction<T>? activation = null,
        IVectorActivationFunction<T>? recurrentActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, hiddenSize), activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _useVectorActivation = true;

        _sigmoidVectorActivation = recurrentActivation ?? new SigmoidActivation<T>();
        _tanhVectorActivation = activation ?? new TanhActivation<T>();

        Gradients = [];

        // Initialize weights
        _weightsFi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsIi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsCi = new Tensor<T>([_hiddenSize, _inputSize]);
        _weightsOi = new Tensor<T>([_hiddenSize, _inputSize]);

        _weightsFh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsIh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsCh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _weightsOh = new Tensor<T>([_hiddenSize, _hiddenSize]);

        // Initialize biases
        _biasF = new Tensor<T>([_hiddenSize]);
        _biasI = new Tensor<T>([_hiddenSize]);
        _biasC = new Tensor<T>([_hiddenSize]);
        _biasO = new Tensor<T>([_hiddenSize]);

        InitializeWeights();
    }

    /// <summary>
    /// Calculates the output shape of the LSTM layer based on the input shape and hidden size.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="hiddenSize">The size of the hidden state (number of LSTM units).</param>
    /// <returns>The calculated output shape for the LSTM layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape by preserving the batch size and time steps dimensions of the input,
    /// but changes the feature dimension to match the hidden size. This reflects that the LSTM processes each time step
    /// and outputs a vector of size hiddenSize for each one.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data that will come out of this layer.
    /// 
    /// When an LSTM processes data:
    /// - It keeps the same number of samples (batch size)
    /// - It keeps the same number of time steps
    /// - It changes the number of features to match the hidden size
    /// 
    /// For example, if your input has shape [32, 10, 100] (32 samples, 10 time steps, 100 features),
    /// and your hiddenSize is 200, the output shape would be [32, 10, 200].
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int hiddenSize)
    {
        // Preserve batch size and time steps, change last dimension to hiddenSize
        var outputShape = new int[inputShape.Length];
        Array.Copy(inputShape, outputShape, inputShape.Length);
        outputShape[outputShape.Length - 1] = hiddenSize;

        return outputShape;
    }

    /// <summary>
    /// Initializes the weights of the LSTM layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes all the weights and biases of the LSTM layer using Xavier/Glorot initialization.
    /// This initialization technique helps prevent vanishing or exploding gradients by setting the initial values
    /// to random numbers scaled by a factor based on the input and hidden sizes.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's weights.
    /// 
    /// When creating a neural network:
    /// - Weights need good starting values for effective training
    /// - Xavier/Glorot initialization is a smart way to choose these values
    /// - It scales the random values based on the size of the network
    /// - This helps prevent training problems like vanishing or exploding gradients
    /// 
    /// Think of it like setting up a balanced starting point that will help the
    /// network learn more effectively.
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        // Xavier/Glorot initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputSize + _hiddenSize)));

        InitializeWeight(_weightsFi, scale);
        InitializeWeight(_weightsIi, scale);
        InitializeWeight(_weightsCi, scale);
        InitializeWeight(_weightsOi, scale);
        InitializeWeight(_weightsFh, scale);
        InitializeWeight(_weightsIh, scale);
        InitializeWeight(_weightsCh, scale);
        InitializeWeight(_weightsOh, scale);

        InitializeBias(_biasF);
        InitializeBias(_biasI);
        InitializeBias(_biasC);
        InitializeBias(_biasO);
    }

    /// <summary>
    /// Initializes a weight tensor with scaled random values.
    /// </summary>
    /// <param name="weight">The weight tensor to initialize.</param>
    /// <param name="scale">The scale factor for initialization.</param>
    /// <remarks>
    /// <para>
    /// This method initializes a weight tensor with random values between -0.5 and 0.5, scaled by the provided
    /// scale factor. This is part of the Xavier/Glorot initialization technique, which helps improve training
    /// convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a weight tensor with smart random values.
    /// 
    /// When initializing weights:
    /// - Each value is set to a small random number
    /// - The numbers are centered around zero (between -0.5 and 0.5) then scaled
    /// - The scale factor adjusts the range based on the network size
    /// 
    /// This approach helps the network start learning effectively from the beginning.
    /// </para>
    /// </remarks>
    private void InitializeWeight(Tensor<T> weight, T scale)
    {
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Initializes a bias tensor with zeros.
    /// </summary>
    /// <param name="bias">The bias tensor to initialize.</param>
    /// <remarks>
    /// <para>
    /// This method initializes a bias tensor with zeros. While weights typically need random initialization,
    /// biases are often initialized to zero or small constant values, as they don't suffer from the symmetry
    /// issues that would cause problems with all-zero weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets all values in a bias tensor to zero.
    /// 
    /// Unlike weights, which need random values:
    /// - Biases can start at zero
    /// - During training, they'll adjust as needed
    /// - Starting at zero is a neutral position
    /// 
    /// Think of biases like the "default settings" that get adjusted during training.
    /// </para>
    /// </remarks>
    private void InitializeBias(Tensor<T> bias)
    {
        for (int i = 0; i < bias.Length; i++)
        {
            bias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Performs the forward pass of the LSTM layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after LSTM processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the LSTM layer. It processes the input sequence one time step at a time,
    /// updating the hidden state and cell state for each step. The hidden state at each time step is collected to form
    /// the output tensor. The input, hidden state, and cell state are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the LSTM layer.
    /// 
    /// During the forward pass:
    /// - The layer processes the input sequence step by step
    /// - For each step, it updates its internal memory (hidden state and cell state)
    /// - It produces an output for each step in the sequence
    /// - It remembers the inputs and states for later use during training
    /// 
    /// For example, if processing a sentence, the LSTM would process one word at a time,
    /// updating its understanding of the context with each word, and producing an output
    /// that reflects that understanding.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int timeSteps = input.Shape[1];

        var output = new Tensor<T>([batchSize, timeSteps, _hiddenSize]);
        _lastHiddenState = new Tensor<T>([batchSize, _hiddenSize]);
        _lastCellState = new Tensor<T>([batchSize, _hiddenSize]);

        for (int t = 0; t < timeSteps; t++)
        {
            var xt = input.GetSlice(t);
            (_lastHiddenState, _lastCellState) = LSTMCell(xt, _lastHiddenState, _lastCellState);
            output.SetSlice(t, _lastHiddenState);
        }

        return output;
    }

    /// <summary>
    /// Implements a single LSTM cell computation for one time step.
    /// </summary>
    /// <param name="xt">The input at the current time step.</param>
    /// <param name="prevH">The hidden state from the previous time step.</param>
    /// <param name="prevC">The cell state from the previous time step.</param>
    /// <returns>A tuple containing the new hidden state and cell state.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core LSTM cell computation for a single time step. It calculates the
    /// forget gate, input gate, cell state candidate, and output gate, then uses these to update the cell state
    /// and hidden state. The computations can use either element-wise activation functions or vector activation
    /// functions, depending on how the layer was configured.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes a single time step through the LSTM cell.
    /// 
    /// An LSTM cell works through these steps:
    /// 1. Forget gate (f): Decides what information to throw away from the cell state
    /// 2. Input gate (i): Decides what new information to store in the cell state
    /// 3. Cell candidate (c): Creates new values that could be added to the state
    /// 4. Cell state update: Updates the old cell state into the new cell state
    /// 5. Output gate (o): Decides what parts of the cell state to output
    /// 6. Hidden state update: Creates the new hidden state based on the cell state
    /// 
    /// For example, when processing a word in a sentence, the LSTM might:
    /// - Forget information that's no longer relevant
    /// - Add important details about the new word
    /// - Update its internal understanding
    /// - Output information needed for the next steps
    /// </para>
    /// </remarks>
    private (Tensor<T> hiddenState, Tensor<T> cellState) LSTMCell(Tensor<T> xt, Tensor<T> prevH, Tensor<T> prevC)
    {
        var f = xt.MatrixMultiply(_weightsFi).Add(prevH.MatrixMultiply(_weightsFh)).Add(_biasF);
        var i = xt.MatrixMultiply(_weightsIi).Add(prevH.MatrixMultiply(_weightsIh)).Add(_biasI);
        var c = xt.MatrixMultiply(_weightsCi).Add(prevH.MatrixMultiply(_weightsCh)).Add(_biasC);
        var o = xt.MatrixMultiply(_weightsOi).Add(prevH.MatrixMultiply(_weightsOh)).Add(_biasO);

        if (_useVectorActivation)
        {
            f = _sigmoidVectorActivation!.Activate(f);
            i = _sigmoidVectorActivation!.Activate(i);
            c = _tanhVectorActivation!.Activate(c);
            o = _sigmoidVectorActivation!.Activate(o);
        }
        else
        {
            f = f.Transform((x, _) => _sigmoidActivation!.Activate(x));
            i = i.Transform((x, _) => _sigmoidActivation!.Activate(x));
            c = c.Transform((x, _) => _tanhActivation!.Activate(x));
            o = o.Transform((x, _) => _sigmoidActivation!.Activate(x));
        }

        var newC = f.ElementwiseMultiply(prevC).Add(i.ElementwiseMultiply(c));
        var newH = o.ElementwiseMultiply(_useVectorActivation 
            ? _tanhVectorActivation!.Activate(newC) 
            : newC.Transform((x, _) => _tanhActivation!.Activate(x)));

        return (newH, newC);
    }

   /// <summary>
    /// Performs the backward pass of the LSTM layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the LSTM layer, which is used during training to propagate
    /// error gradients back through the network. It processes the sequence in reverse order (from the last time step
    /// to the first), calculating gradients for all parameters and the input. The gradients are stored for use
    /// in the UpdateParameters method.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's inputs
    /// should change to reduce errors.
    /// 
    /// During the backward pass:
    /// - The layer processes the sequence in reverse order (last step to first)
    /// - At each step, it calculates how each part contributed to the error
    /// - It computes gradients for all weights, biases, and inputs
    /// - These gradients show how to adjust the parameters to improve performance
    /// 
    /// This process is part of the "backpropagation through time" algorithm that helps
    /// recurrent neural networks learn from their mistakes.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastCellState == null)
        {
            throw new InvalidOperationException("Backward pass called before forward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int timeSteps = _lastInput.Shape[1];
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        var dWeightsFi = new Tensor<T>(_weightsFi.Shape);
        var dWeightsIi = new Tensor<T>(_weightsIi.Shape);
        var dWeightsCi = new Tensor<T>(_weightsCi.Shape);
        var dWeightsOi = new Tensor<T>(_weightsOi.Shape);
        var dWeightsFh = new Tensor<T>(_weightsFh.Shape);
        var dWeightsIh = new Tensor<T>(_weightsIh.Shape);
        var dWeightsCh = new Tensor<T>(_weightsCh.Shape);
        var dWeightsOh = new Tensor<T>(_weightsOh.Shape);
        var dBiasF = new Tensor<T>(_biasF.Shape);
        var dBiasI = new Tensor<T>(_biasI.Shape);
        var dBiasC = new Tensor<T>(_biasC.Shape);
        var dBiasO = new Tensor<T>(_biasO.Shape);

        var dNextH = new Tensor<T>([batchSize, _hiddenSize]);
        var dNextC = new Tensor<T>([batchSize, _hiddenSize]);

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            var dh = outputGradient.GetSlice(t).Add(dNextH);
            var xt = _lastInput.GetSlice(t);
            var prevH = t > 0 ? _lastHiddenState.GetSlice(t - 1) : new Tensor<T>([batchSize, _hiddenSize]);
            var prevC = t > 0 ? _lastCellState.GetSlice(t - 1) : new Tensor<T>([batchSize, _hiddenSize]);

            var (dxt, dprevH, dprevC, dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo) = 
                BackwardStep(dh, dNextC, xt, prevH, prevC);

            inputGradient.SetSlice(t, dxt);
            dNextH = dprevH;
            dNextC = dprevC;

            dWeightsFi.Add(dWfi);
            dWeightsIi.Add(dWii);
            dWeightsCi.Add(dWci);
            dWeightsOi.Add(dWoi);
            dWeightsFh.Add(dWfh);
            dWeightsIh.Add(dWih);
            dWeightsCh.Add(dWch);
            dWeightsOh.Add(dWoh);
            dBiasF.Add(dbf);
            dBiasI.Add(dbi);
            dBiasC.Add(dbc);
            dBiasO.Add(dbo);
        }

        // Store gradients for use in UpdateParameters
        Gradients = new Dictionary<string, Tensor<T>>
        {
            {"weightsFi", dWeightsFi}, {"weightsIi", dWeightsIi}, {"weightsCi", dWeightsCi}, {"weightsOi", dWeightsOi},
            {"weightsFh", dWeightsFh}, {"weightsIh", dWeightsIh}, {"weightsCh", dWeightsCh}, {"weightsOh", dWeightsOh},
            {"biasF", dBiasF}, {"biasI", dBiasI}, {"biasC", dBiasC}, {"biasO", dBiasO}
        };

        return inputGradient;
    }

    /// <summary>
    /// Implements a single backward step for the LSTM cell.
    /// </summary>
    /// <param name="dh">The gradient of the loss with respect to the hidden state.</param>
    /// <param name="dc_next">The gradient of the loss with respect to the next cell state.</param>
    /// <param name="x">The input at the current time step.</param>
    /// <param name="prev_h">The hidden state from the previous time step.</param>
    /// <param name="prev_c">The cell state from the previous time step.</param>
    /// <returns>A tuple containing gradients for various components.</returns>
    /// <remarks>
    /// <para>
    /// This method implements a single backward step for the LSTM cell, computing gradients for all parameters
    /// and states. It first recomputes the forward pass values (needed for calculating derivatives), then
    /// computes the gradients by applying the chain rule of differentiation.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how each part of the LSTM cell contributed to errors.
    /// 
    /// During the backward step:
    /// 1. It recomputes the forward calculations to get the gate values
    /// 2. It calculates how much each gate value affected the error
    /// 3. It computes gradients for all weights and biases
    /// 4. It prepares gradients to pass back to earlier time steps
    /// 
    /// This complex process is based on calculus (chain rule) and is the heart of
    /// how LSTMs learn from their mistakes. The gradients show how to adjust each
    /// weight and bias to improve performance.
    /// </para>
    /// </remarks>
    private (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) 
        BackwardStep(Tensor<T> dh, Tensor<T> dc_next, Tensor<T> x, Tensor<T> prev_h, Tensor<T> prev_c)
    {
        // Forward pass calculations (needed for backward pass)
        var concat = Tensor<T>.Concatenate([x, prev_h], 1);
        var f = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation, 
            concat.Multiply(Tensor<T>.Concatenate([_weightsFi, _weightsFh], 0)).Add(_biasF));
        var i = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            concat.Multiply(Tensor<T>.Concatenate([_weightsIi, _weightsIh], 0)).Add(_biasI));
        var c_bar = ActivateTensorConditional(_tanhVectorActivation, _tanhActivation,
            concat.Multiply(Tensor<T>.Concatenate([_weightsCi, _weightsCh], 0)).Add(_biasC));
        var o = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            concat.Multiply(Tensor<T>.Concatenate([_weightsOi, _weightsOh], 0)).Add(_biasO));
        var c = f.PointwiseMultiply(prev_c).Add(i.PointwiseMultiply(c_bar));
        var h = o.PointwiseMultiply(ActivateTensor(_tanhActivation, c));

        // Backward pass
        var do_ = dh.PointwiseMultiply(ActivateTensor(_tanhActivation, c));
        var dc = dh.PointwiseMultiply(o).PointwiseMultiply(DerivativeTensor(_tanhActivation, c)).Add(dc_next);
        var dc_bar = dc.PointwiseMultiply(i);
        var di = dc.PointwiseMultiply(c_bar);
        var df = dc.PointwiseMultiply(prev_c);
        var dprev_c = dc.PointwiseMultiply(f);

        // Gate derivatives
        var di_input = DerivativeTensor(_sigmoidActivation, i).PointwiseMultiply(di);
        var df_input = DerivativeTensor(_sigmoidActivation, f).PointwiseMultiply(df);
        var do_input = DerivativeTensor(_sigmoidActivation, o).PointwiseMultiply(do_);
        var dc_bar_input = DerivativeTensor(_tanhActivation, c_bar).PointwiseMultiply(dc_bar);

        // Compute gradients for weights and biases
        var dWeights = concat.Transpose(new[] { 1, 0 }).Multiply(Tensor<T>.Concatenate(new[] { di_input, df_input, dc_bar_input, do_input }, 1));
        var dWfi = dWeights.Slice(0, 0, _inputSize).Slice(1, 0, _hiddenSize);
        var dWii = dWeights.Slice(0, 0, _inputSize).Slice(1, _hiddenSize, _hiddenSize * 2);
        var dWci = dWeights.Slice(0, 0, _inputSize).Slice(1, _hiddenSize * 2, _hiddenSize * 3);
        var dWoi = dWeights.Slice(0, 0, _inputSize).Slice(1, _hiddenSize * 3, _hiddenSize * 4);
        var dWfh = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, 0, _hiddenSize);
        var dWih = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, _hiddenSize, _hiddenSize * 2);
        var dWch = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, _hiddenSize * 2, _hiddenSize * 3);
        var dWoh = dWeights.Slice(0, _inputSize, _inputSize + _hiddenSize).Slice(1, _hiddenSize * 3, _hiddenSize * 4);

        var dBiases = Tensor<T>.Concatenate(new[] { di_input.Sum(new[] { 0 }), df_input.Sum(new[] { 0 }), dc_bar_input.Sum(new[] { 0 }), do_input.Sum(new[] { 0 }) }, 1);
        var dbf = dBiases.Slice(1, 0, _hiddenSize);
        var dbi = dBiases.Slice(1, _hiddenSize, _hiddenSize * 2);
        var dbc = dBiases.Slice(1, _hiddenSize * 2, _hiddenSize * 3);
        var dbo = dBiases.Slice(1, _hiddenSize * 3, _hiddenSize * 4);

        // Compute gradient for input
        var dInputs = di_input.Multiply(_weightsIi.Transpose(new[] { 1, 0 }))
            .Add(df_input.Multiply(_weightsFi.Transpose(new[] { 1, 0 })))
            .Add(dc_bar_input.Multiply(_weightsCi.Transpose(new[] { 1, 0 })))
            .Add(do_input.Multiply(_weightsOi.Transpose(new[] { 1, 0 })));

        var dx = dInputs.Slice(1, 0, _inputSize);
        var dprev_h = dInputs.Slice(1, _inputSize, _inputSize + _hiddenSize);

        return (dx, dprev_h, dprev_c, dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo);
    }

    /// <summary>
    /// Applies activation to a tensor using either vector or scalar activation functions based on configuration.
    /// </summary>
    /// <param name="vectorActivation">The vector activation function to use if vector activation is enabled.</param>
    /// <param name="scalarActivation">The scalar activation function to use if vector activation is disabled.</param>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>The activated tensor.</returns>
    /// <remarks>
    /// <para>
    /// This helper method applies either a vector activation function or a scalar activation function to a tensor,
    /// depending on the layer's configuration. Vector<double> activation functions operate on entire tensors at once, while
    /// scalar activation functions operate element by element.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the right type of activation function to a tensor.
    /// 
    /// The LSTM can use two types of activation functions:
    /// - Scalar functions that process one number at a time
    /// - Vector<double> functions that process entire groups at once
    /// 
    /// This method checks which type you're using and applies the appropriate function.
    /// Vector<double> functions can be faster for large datasets but work the same way in principle.
    /// </para>
    /// </remarks>
    private Tensor<T> ActivateTensorConditional(IVectorActivationFunction<T>? vectorActivation, IActivationFunction<T>? scalarActivation, Tensor<T> input)
    {
        if (_useVectorActivation)
        {
            return ActivateTensor(vectorActivation, input);
        }
        else
        {
            return ActivateTensor(scalarActivation, input);
        }
    }

    /// <summary>
    /// Updates the parameters of the LSTM layer based on the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates all the weights and biases of the LSTM layer based on the gradients computed during
    /// the backward pass. The learning rate controls the size of the parameter updates. Each parameter is updated
    /// by subtracting the product of its gradient and the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - Each weight and bias is adjusted based on its gradient
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually adjusting its
    /// internal values to better process the input sequences.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var kvp in Gradients)
        {
            var paramName = kvp.Key;
            var gradient = kvp.Value;

            switch (paramName)
            {
                case "weightsFi":
                    _weightsFi = _weightsFi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsIi":
                    _weightsIi = _weightsIi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsCi":
                    _weightsCi = _weightsCi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsOi":
                    _weightsOi = _weightsOi.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsFh":
                    _weightsFh = _weightsFh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsIh":
                    _weightsIh = _weightsIh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsCh":
                    _weightsCh = _weightsCh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "weightsOh":
                    _weightsOh = _weightsOh.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasF":
                    _biasF = _biasF.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasI":
                    _biasI = _biasI.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasC":
                    _biasC = _biasC.Subtract(gradient.Multiply(learningRate));
                    break;
                case "biasO":
                    _biasO = _biasO.Subtract(gradient.Multiply(learningRate));
                    break;
            }
        }
    }

    /// <summary>
    /// Serializes the LSTM layer's parameters to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves all weights and biases of the LSTM layer to a binary stream. This allows the layer's
    /// state to be saved to a file and loaded later, which is useful for saving trained models or for
    /// transferring parameters between different instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the layer's learned values to a file.
    /// 
    /// Serialization is like taking a snapshot of the layer's current state:
    /// - All weights and biases are written to a file
    /// - The exact format ensures they can be loaded back correctly
    /// - This lets you save a trained model for later use
    /// 
    /// For example, after training your model for hours or days, you can save it
    /// and then load it later without having to retrain.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        SerializationHelper<T>.SerializeTensor(writer, _weightsFi);
        SerializationHelper<T>.SerializeTensor(writer, _weightsIi);
        SerializationHelper<T>.SerializeTensor(writer, _weightsCi);
        SerializationHelper<T>.SerializeTensor(writer, _weightsOi);
        SerializationHelper<T>.SerializeTensor(writer, _weightsFh);
        SerializationHelper<T>.SerializeTensor(writer, _weightsIh);
        SerializationHelper<T>.SerializeTensor(writer, _weightsCh);
        SerializationHelper<T>.SerializeTensor(writer, _weightsOh);
        SerializationHelper<T>.SerializeTensor(writer, _biasF);
        SerializationHelper<T>.SerializeTensor(writer, _biasI);
        SerializationHelper<T>.SerializeTensor(writer, _biasC);
        SerializationHelper<T>.SerializeTensor(writer, _biasO);
    }

    /// <summary>
    /// Deserializes the LSTM layer's parameters from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads all weights and biases of the LSTM layer from a binary stream. This allows the layer
    /// to restore its state from a previously saved file, which is useful for loading trained models or for
    /// transferring parameters between different instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads previously saved values into the layer.
    /// 
    /// Deserialization is like restoring a saved snapshot:
    /// - All weights and biases are read from a file
    /// - The layer's internal state is set to match what was saved
    /// - This lets you use a previously trained model without retraining
    /// 
    /// For example, you could train a model on a powerful computer, save it,
    /// and then load it on a less powerful device for actual use.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        _weightsFi = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsIi = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsCi = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsOi = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsFh = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsIh = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsCh = SerializationHelper<T>.DeserializeTensor(reader);
        _weightsOh = SerializationHelper<T>.DeserializeTensor(reader);
        _biasF = SerializationHelper<T>.DeserializeTensor(reader);
        _biasI = SerializationHelper<T>.DeserializeTensor(reader);
        _biasC = SerializationHelper<T>.DeserializeTensor(reader);
        _biasO = SerializationHelper<T>.DeserializeTensor(reader);
    }

    /// <summary>
    /// Gets all trainable parameters of the LSTM layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) from the LSTM layer and combines them
    /// into a single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights in a uniform format.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learned values into a single list.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network has learned during training
    /// - Include all weights and biases for each gate in the LSTM
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk in a simple format
    /// - Advanced optimization techniques that need access to all parameters
    /// - Sharing parameters between different models
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weightsFi.Length + _weightsIi.Length + _weightsCi.Length + _weightsOi.Length +
                          _weightsFh.Length + _weightsIh.Length + _weightsCh.Length + _weightsOh.Length +
                          _biasF.Length + _biasI.Length + _biasC.Length + _biasO.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy weights parameters
        CopyTensorToVector(_weightsFi, parameters, ref index);
        CopyTensorToVector(_weightsIi, parameters, ref index);
        CopyTensorToVector(_weightsCi, parameters, ref index);
        CopyTensorToVector(_weightsOi, parameters, ref index);
        CopyTensorToVector(_weightsFh, parameters, ref index);
        CopyTensorToVector(_weightsIh, parameters, ref index);
        CopyTensorToVector(_weightsCh, parameters, ref index);
        CopyTensorToVector(_weightsOh, parameters, ref index);
    
        // Copy bias parameters
        CopyTensorToVector(_biasF, parameters, ref index);
        CopyTensorToVector(_biasI, parameters, ref index);
        CopyTensorToVector(_biasC, parameters, ref index);
        CopyTensorToVector(_biasO, parameters, ref index);
    
        return parameters;
    }

    /// <summary>
    /// Copies values from a tensor to a vector.
    /// </summary>
    /// <param name="tensor">The source tensor.</param>
    /// <param name="vector">The destination vector.</param>
    /// <param name="startIndex">The starting index in the vector.</param>
    /// <remarks>
    /// <para>
    /// This helper method copies all values from a tensor to a section of a vector, starting at the specified index.
    /// The index is updated to point to the position after the last copied value, allowing multiple tensors to be
    /// copied sequentially into the same vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method copies values from a multi-dimensional array to a simple list.
    /// 
    /// When transferring values:
    /// - Each value from the tensor is placed in the vector
    /// - The values are copied in a specific order
    /// - The index keeps track of the current position in the vector
    /// 
    /// This is used when collecting all parameters into a single list for saving or optimization.
    /// </para>
    /// </remarks>
    private void CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, ref int startIndex)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            vector[startIndex++] = tensor[i];
        }
    }

    /// <summary>
    /// Sets the trainable parameters of the LSTM layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (weights and biases) of the LSTM layer from a single vector.
    /// It extracts the appropriate portions of the input vector for each parameter. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learned values from a single list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The method distributes values to the appropriate weights and biases
    /// - This allows you to restore a previously saved model
    /// 
    /// For example, after loading a parameter vector from a file, this method
    /// would update all the internal weights and biases of the LSTM to match
    /// what was saved.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _weightsFi.Length + _weightsIi.Length + _weightsCi.Length + _weightsOi.Length +
                          _weightsFh.Length + _weightsIh.Length + _weightsCh.Length + _weightsOh.Length +
                          _biasF.Length + _biasI.Length + _biasC.Length + _biasO.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weights parameters
        CopyVectorToTensor(parameters, _weightsFi, ref index);
        CopyVectorToTensor(parameters, _weightsIi, ref index);
        CopyVectorToTensor(parameters, _weightsCi, ref index);
        CopyVectorToTensor(parameters, _weightsOi, ref index);
        CopyVectorToTensor(parameters, _weightsFh, ref index);
        CopyVectorToTensor(parameters, _weightsIh, ref index);
        CopyVectorToTensor(parameters, _weightsCh, ref index);
        CopyVectorToTensor(parameters, _weightsOh, ref index);
    
        // Set bias parameters
        CopyVectorToTensor(parameters, _biasF, ref index);
        CopyVectorToTensor(parameters, _biasI, ref index);
        CopyVectorToTensor(parameters, _biasC, ref index);
        CopyVectorToTensor(parameters, _biasO, ref index);
    }

    /// <summary>
    /// Copies values from a vector to a tensor.
    /// </summary>
    /// <param name="vector">The source vector.</param>
    /// <param name="tensor">The destination tensor.</param>
    /// <param name="startIndex">The starting index in the vector.</param>
    /// <remarks>
    /// <para>
    /// This helper method copies values from a section of a vector to a tensor, starting at the specified index
    /// in the vector. The index is updated to point to the position after the last copied value, allowing multiple
    /// tensors to be populated sequentially from the same vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method copies values from a simple list to a multi-dimensional array.
    /// 
    /// When transferring values:
    /// - Each value from the vector is placed in the tensor
    /// - The values are copied in a specific order
    /// - The index keeps track of the current position in the vector
    /// 
    /// This is used when loading parameters from a saved model or after optimization.
    /// </para>
    /// </remarks>
    private void CopyVectorToTensor(Vector<T> vector, Tensor<T> tensor, ref int startIndex)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = vector[startIndex++];
        }
    }

    /// <summary>
    /// Resets the internal state of the LSTM layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears any cached data from previous forward passes, essentially resetting the layer
    /// to its initial state. This is useful when starting to process a new sequence or when implementing
    /// stateful recurrent networks where you want to explicitly control when states are reset.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and hidden states are cleared
    /// - Gradients from previous training steps are cleared
    /// - The layer forgets any information from previous sequences
    /// 
    /// This is important when:
    /// - Processing a new, unrelated sequence
    /// - Starting a new training episode
    /// - You want the network to forget its previous context
    /// 
    /// For example, if you've processed one paragraph and want to start with a completely
    /// new paragraph, you should reset the state to prevent the new paragraph from being
    /// influenced by the previous one.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastHiddenState = null;
        _lastCellState = null;
        Gradients.Clear();
    }
}