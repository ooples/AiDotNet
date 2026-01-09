using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
    private Tensor<T> _weightsFi;

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
    private Tensor<T> _weightsIi;

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
    private Tensor<T> _weightsCi;

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
    private Tensor<T> _weightsOi;

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
    private Tensor<T> _weightsFh;

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
    private Tensor<T> _weightsIh;

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
    private Tensor<T> _weightsCh;

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
    private Tensor<T> _weightsOh;

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
    private Tensor<T> _biasF;

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
    private Tensor<T> _biasI;

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
    private Tensor<T> _biasC;

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
    private Tensor<T> _biasO;

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
    private Tensor<T>? _lastCellState;

    /// <summary>
    /// Cached hidden states for all time steps (Batch, Time, Hidden).
    /// </summary>
    private Tensor<T>? _cachedHiddenStates;

    /// <summary>
    /// Cached cell states for all time steps (Batch, Time, Hidden).
    /// </summary>
    private Tensor<T>? _cachedCellStates;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

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
    /// activation functions. Vector functions operate on entire tensors at once, which can be more efficient for
    /// certain operations.
    /// </para>
    /// <para><b>For Beginners:</b> This flag chooses between two ways of applying activation functions.
    /// 
    /// When set to true:
    /// - Vector activation functions are used
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

    #region GPU Training Fields

    // GPU-resident weight tensors
    private GpuTensor<T>? _gpuWeightsFi;
    private GpuTensor<T>? _gpuWeightsIi;
    private GpuTensor<T>? _gpuWeightsCi;
    private GpuTensor<T>? _gpuWeightsOi;
    private GpuTensor<T>? _gpuWeightsFh;
    private GpuTensor<T>? _gpuWeightsIh;
    private GpuTensor<T>? _gpuWeightsCh;
    private GpuTensor<T>? _gpuWeightsOh;
    private GpuTensor<T>? _gpuBiasF;
    private GpuTensor<T>? _gpuBiasI;
    private GpuTensor<T>? _gpuBiasC;
    private GpuTensor<T>? _gpuBiasO;

    // GPU-resident gradient tensors
    private GpuTensor<T>? _gpuWeightsFiGradient;
    private GpuTensor<T>? _gpuWeightsIiGradient;
    private GpuTensor<T>? _gpuWeightsCiGradient;
    private GpuTensor<T>? _gpuWeightsOiGradient;
    private GpuTensor<T>? _gpuWeightsFhGradient;
    private GpuTensor<T>? _gpuWeightsIhGradient;
    private GpuTensor<T>? _gpuWeightsChGradient;
    private GpuTensor<T>? _gpuWeightsOhGradient;
    private GpuTensor<T>? _gpuBiasFGradient;
    private GpuTensor<T>? _gpuBiasIGradient;
    private GpuTensor<T>? _gpuBiasCGradient;
    private GpuTensor<T>? _gpuBiasOGradient;

    // GPU-resident optimizer state tensors (velocity for SGD momentum, M/V for Adam)
    private GpuTensor<T>? _gpuWeightsFiVelocity;
    private GpuTensor<T>? _gpuWeightsIiVelocity;
    private GpuTensor<T>? _gpuWeightsCiVelocity;
    private GpuTensor<T>? _gpuWeightsOiVelocity;
    private GpuTensor<T>? _gpuWeightsFhVelocity;
    private GpuTensor<T>? _gpuWeightsIhVelocity;
    private GpuTensor<T>? _gpuWeightsChVelocity;
    private GpuTensor<T>? _gpuWeightsOhVelocity;
    private GpuTensor<T>? _gpuBiasFVelocity;
    private GpuTensor<T>? _gpuBiasIVelocity;
    private GpuTensor<T>? _gpuBiasCVelocity;
    private GpuTensor<T>? _gpuBiasOVelocity;

    // Adam M/V buffers
    private GpuTensor<T>? _gpuWeightsFiM;
    private GpuTensor<T>? _gpuWeightsFiV;
    private GpuTensor<T>? _gpuWeightsIiM;
    private GpuTensor<T>? _gpuWeightsIiV;
    private GpuTensor<T>? _gpuWeightsCiM;
    private GpuTensor<T>? _gpuWeightsCiV;
    private GpuTensor<T>? _gpuWeightsOiM;
    private GpuTensor<T>? _gpuWeightsOiV;
    private GpuTensor<T>? _gpuWeightsFhM;
    private GpuTensor<T>? _gpuWeightsFhV;
    private GpuTensor<T>? _gpuWeightsIhM;
    private GpuTensor<T>? _gpuWeightsIhV;
    private GpuTensor<T>? _gpuWeightsChM;
    private GpuTensor<T>? _gpuWeightsChV;
    private GpuTensor<T>? _gpuWeightsOhM;
    private GpuTensor<T>? _gpuWeightsOhV;
    private GpuTensor<T>? _gpuBiasFM;
    private GpuTensor<T>? _gpuBiasFV;
    private GpuTensor<T>? _gpuBiasIM;
    private GpuTensor<T>? _gpuBiasIV;
    private GpuTensor<T>? _gpuBiasCM;
    private GpuTensor<T>? _gpuBiasCV;
    private GpuTensor<T>? _gpuBiasOM;
    private GpuTensor<T>? _gpuBiasOV;

    // Cached forward pass state for backpropagation (per timestep arrays)
    private IGpuTensor<T>? _gpuLastInput;
    private IGpuTensor<T>[]? _gpuCachedForgetGates;
    private IGpuTensor<T>[]? _gpuCachedInputGates;
    private IGpuTensor<T>[]? _gpuCachedCellCandidates;
    private IGpuTensor<T>[]? _gpuCachedOutputGates;
    private IGpuTensor<T>[]? _gpuCachedCellStates;
    private IGpuTensor<T>[]? _gpuCachedHiddenStates;
    private IGpuTensor<T>? _gpuInitialHiddenState;
    private IGpuTensor<T>? _gpuInitialCellState;

    #endregion

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
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <value>
    /// The total number of parameters across all weight matrices and bias vectors.
    /// For an LSTM with input size I and hidden size H, this is:
    /// 4 * (H * I) + 4 * (H * H) + 4 * H = 4 * H * (I + H + 1)
    /// </value>
    /// <remarks>
    /// <para>
    /// The LSTM has 4 gates (forget, input, cell, output), each with:
    /// - Input-to-hidden weights: [hiddenSize × inputSize]
    /// - Hidden-to-hidden weights: [hiddenSize × hiddenSize]
    /// - Biases: [hiddenSize]
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        4 * (_hiddenSize * _inputSize) +  // 4 input weight matrices
        4 * (_hiddenSize * _hiddenSize) + // 4 hidden weight matrices
        4 * _hiddenSize;                  // 4 bias vectors

    /// <summary>
    /// Gets the forget gate input weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsFi => _weightsFi;

    /// <summary>
    /// Gets the input gate input weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsIi => _weightsIi;

    /// <summary>
    /// Gets the cell gate input weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsCi => _weightsCi;

    /// <summary>
    /// Gets the output gate input weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsOi => _weightsOi;

    /// <summary>
    /// Gets the forget gate hidden weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsFh => _weightsFh;

    /// <summary>
    /// Gets the input gate hidden weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsIh => _weightsIh;

    /// <summary>
    /// Gets the cell gate hidden weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsCh => _weightsCh;

    /// <summary>
    /// Gets the output gate hidden weights for weight loading.
    /// </summary>
    public Tensor<T> WeightsOh => _weightsOh;

    /// <summary>
    /// Gets the forget gate bias for weight loading.
    /// </summary>
    public Tensor<T> BiasF => _biasF;

    /// <summary>
    /// Gets the input gate bias for weight loading.
    /// </summary>
    public Tensor<T> BiasI => _biasI;

    /// <summary>
    /// Gets the cell gate bias for weight loading.
    /// </summary>
    public Tensor<T> BiasC => _biasC;

    /// <summary>
    /// Gets the output gate bias for weight loading.
    /// </summary>
    public Tensor<T> BiasO => _biasO;

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
        IActivationFunction<T>? recurrentActivation = null,
        IEngine? engine = null)
        : base(inputShape, CalculateOutputShape(inputShape, hiddenSize), activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _useVectorActivation = false;

        _sigmoidActivation = recurrentActivation ?? new SigmoidActivation<T>();
        _tanhActivation = activation ?? new TanhActivation<T>();

        Gradients = new Dictionary<string, Tensor<T>>();

        // Initialize weights
        _weightsFi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });
        _weightsIi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });
        _weightsCi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });
        _weightsOi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });

        _weightsFh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });
        _weightsIh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });
        _weightsCh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });
        _weightsOh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });

        // Initialize biases
        _biasF = new Tensor<T>(new int[] { _hiddenSize });
        _biasI = new Tensor<T>(new int[] { _hiddenSize });
        _biasC = new Tensor<T>(new int[] { _hiddenSize });
        _biasO = new Tensor<T>(new int[] { _hiddenSize });

        InitializeWeights();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_weightsFi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsFh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biasF, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasI, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasC, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasO, PersistentTensorRole.Biases);
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
    /// Vector activation functions:
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
        IVectorActivationFunction<T>? recurrentActivation = null,
        IEngine? engine = null)
        : base(inputShape, CalculateOutputShape(inputShape, hiddenSize), activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _useVectorActivation = true;

        _sigmoidVectorActivation = recurrentActivation ?? new SigmoidActivation<T>();
        _tanhVectorActivation = activation ?? new TanhActivation<T>();

        Gradients = new Dictionary<string, Tensor<T>>();

        // Initialize weights
        _weightsFi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });
        _weightsIi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });
        _weightsCi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });
        _weightsOi = new Tensor<T>(new int[] { _hiddenSize, _inputSize });

        _weightsFh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });
        _weightsIh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });
        _weightsCh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });
        _weightsOh = new Tensor<T>(new int[] { _hiddenSize, _hiddenSize });

        // Initialize biases
        _biasF = new Tensor<T>(new int[] { _hiddenSize });
        _biasI = new Tensor<T>(new int[] { _hiddenSize });
        _biasC = new Tensor<T>(new int[] { _hiddenSize });
        _biasO = new Tensor<T>(new int[] { _hiddenSize });

        InitializeWeights();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_weightsFi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsFh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsIh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsCh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_weightsOh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biasF, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasI, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasC, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_biasO, PersistentTensorRole.Biases);
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
        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(2.0, (_inputSize + _hiddenSize))));

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
        // Create random tensor using Tensor<T>.CreateRandom [0, 1]
        var randomTensor = Tensor<T>.CreateRandom(weight.Shape[0], weight.Shape[1]);

        // Shift to [-0.5, 0.5] range: random - 0.5
        var halfTensor = new Tensor<T>(weight.Shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor and copy to weight tensor using flat indexer
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = scaled[i];
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
        // Use tensor Fill method to initialize bias with zeros
        bias.Fill(NumOps.Zero);
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
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims into batch for rank > 3
        Tensor<T> input3D;
        int batchSize;
        int timeSteps;

        if (rank == 2)
        {
            // 2D input for LSTM is always interpreted as [timeSteps, features] with batchSize=1
            // This is the standard sequence format for LSTM processing.
            // If users want batch processing, they should provide 3D input [batchSize, timeSteps, features]
            batchSize = 1;
            timeSteps = input.Shape[0];
            int featureSize = input.Shape[1];
            input3D = input.Reshape([1, timeSteps, featureSize]);
        }
        else if (rank == 3)
        {
            // Standard 3D input [batchSize, timeSteps, inputSize]
            batchSize = input.Shape[0];
            timeSteps = input.Shape[1];
            input3D = input;
        }
        else
        {
            // Higher-rank tensor: collapse leading dims into batch
            timeSteps = input.Shape[rank - 2];
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            input3D = input.Reshape([flatBatch, timeSteps, _inputSize]);
        }

        _lastInput = input3D;

        var output = new Tensor<T>(new int[] { batchSize, timeSteps, _hiddenSize });

        _cachedHiddenStates = new Tensor<T>(new int[] { batchSize, timeSteps, _hiddenSize });
        _cachedCellStates = new Tensor<T>(new int[] { batchSize, timeSteps, _hiddenSize });

        var currentH = new Tensor<T>(new int[] { batchSize, _hiddenSize });
        var currentC = new Tensor<T>(new int[] { batchSize, _hiddenSize });

        // Pre-transpose weights for efficiency
        var WfiT = Engine.TensorTranspose(_weightsFi);
        var WiiT = Engine.TensorTranspose(_weightsIi);
        var WciT = Engine.TensorTranspose(_weightsCi);
        var WoiT = Engine.TensorTranspose(_weightsOi);
        var WfhT = Engine.TensorTranspose(_weightsFh);
        var WihT = Engine.TensorTranspose(_weightsIh);
        var WchT = Engine.TensorTranspose(_weightsCh);
        var WohT = Engine.TensorTranspose(_weightsOh);
        var biasF2D = _biasF.Reshape([1, _hiddenSize]);
        var biasI2D = _biasI.Reshape([1, _hiddenSize]);
        var biasC2D = _biasC.Reshape([1, _hiddenSize]);
        var biasO2D = _biasO.Reshape([1, _hiddenSize]);

        for (int t = 0; t < timeSteps; t++)
        {
            // Slice along the time dimension (dim 1), keeping batch dimension
            // For input [batchSize, timeSteps, inputSize], this returns [batchSize, inputSize]
            var xt = input3D.GetSliceAlongDimension(t, 1);

            // Forget Gate - using TensorBroadcastAdd for bias (supports [batch, hidden] + [hidden] broadcasting)
            var f = Engine.TensorMatMul(xt, WfiT);
            f = Engine.TensorAdd(f, Engine.TensorMatMul(currentH, WfhT));
            f = Engine.TensorBroadcastAdd(f, biasF2D);
            f = Engine.Sigmoid(f);

            // Input Gate
            var i = Engine.TensorMatMul(xt, WiiT);
            i = Engine.TensorAdd(i, Engine.TensorMatMul(currentH, WihT));
            i = Engine.TensorBroadcastAdd(i, biasI2D);
            i = Engine.Sigmoid(i);

            // Cell Candidate
            var c_tilde = Engine.TensorMatMul(xt, WciT);
            c_tilde = Engine.TensorAdd(c_tilde, Engine.TensorMatMul(currentH, WchT));
            c_tilde = Engine.TensorBroadcastAdd(c_tilde, biasC2D);
            c_tilde = Engine.Tanh(c_tilde);

            // Output Gate
            var o = Engine.TensorMatMul(xt, WoiT);
            o = Engine.TensorAdd(o, Engine.TensorMatMul(currentH, WohT));
            o = Engine.TensorBroadcastAdd(o, biasO2D);
            o = Engine.Sigmoid(o);

            // Update Cell State
            var f_prevC = Engine.TensorMultiply(f, currentC);
            var i_cTilde = Engine.TensorMultiply(i, c_tilde);
            currentC = Engine.TensorAdd(f_prevC, i_cTilde);

            // Update Hidden State
            var tanhC = Engine.Tanh(currentC);
            currentH = Engine.TensorMultiply(o, tanhC);

            // Store results along the time dimension (dimension 1)
            output.SetSlice(1, t, currentH);
            _cachedHiddenStates.SetSlice(1, t, currentH);
            _cachedCellStates.SetSlice(1, t, currentC);
        }

        _lastHiddenState = currentH;
        _lastCellState = currentC;

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Output shape: [...leadingDims, timeSteps, hiddenSize]
            int[] newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 2] = timeSteps;
            newShape[_originalInputShape.Length - 1] = _hiddenSize;
            output = output.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // 2D input -> 2D output (remove added batch dim)
            output = output.Reshape([timeSteps, _hiddenSize]);
        }

        return output;
    }

    /// <summary>
    /// Performs a GPU-resident forward pass using GPU-accelerated LSTM operations.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor.</param>
    /// <returns>GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// All data stays on the GPU throughout the computation, avoiding expensive CPU-GPU transfers.
    /// The LSTM gates (forget, input, cell, output) are computed using GPU matrix operations.</para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];

        // Determine batch size and time steps from input shape
        int batchSize, timeSteps;
        int rank = input.Shape.Length;
        bool reshaped2D = false;
        bool reshapedHigherRank = false;
        int[]? originalShape = null;

        if (rank == 2)
        {
            // 2D input [timeSteps, inputSize] -> single batch
            batchSize = 1;
            timeSteps = input.Shape[0];
            reshaped2D = true;
            originalShape = input.Shape;
        }
        else if (rank == 3)
        {
            // Standard 3D input [batchSize, timeSteps, inputSize]
            batchSize = input.Shape[0];
            timeSteps = input.Shape[1];
        }
        else
        {
            // Higher-rank tensor: collapse leading dims into batch
            originalShape = input.Shape;
            reshapedHigherRank = true;
            timeSteps = input.Shape[rank - 2];
            batchSize = 1;
            for (int d = 0; d < rank - 2; d++)
                batchSize *= input.Shape[d];
        }

        // Validate input size
        int expectedInputSize = reshaped2D ? input.Shape[1] : (reshapedHigherRank ? input.Shape[rank - 1] : input.Shape[2]);
        if (expectedInputSize != _inputSize)
        {
            throw new ArgumentException(
                $"Expected input size {_inputSize}, but got {expectedInputSize}.");
        }

        // Upload weights to GPU (transposed for efficient matmul)
        using var WfiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsFi).ToArray()));
        using var WiiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsIi).ToArray()));
        using var WciBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsCi).ToArray()));
        using var WoiBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsOi).ToArray()));
        using var WfhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsFh).ToArray()));
        using var WihBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsIh).ToArray()));
        using var WchBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsCh).ToArray()));
        using var WohBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_weightsOh).ToArray()));
        using var biasFBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasF.ToArray()));
        using var biasIBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasI.ToArray()));
        using var biasCBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasC.ToArray()));
        using var biasOBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biasO.ToArray()));

        // Allocate hidden and cell state buffers
        int hiddenBufferSize = batchSize * _hiddenSize;
        var currentHBuffer = backend.AllocateBuffer(hiddenBufferSize);
        var currentCBuffer = backend.AllocateBuffer(hiddenBufferSize);
        backend.Fill(currentHBuffer, 0.0f, hiddenBufferSize); // Initialize to zero
        backend.Fill(currentCBuffer, 0.0f, hiddenBufferSize);

        // Allocate output buffer
        int outputSize = batchSize * timeSteps * _hiddenSize;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Temporary buffers for gate computations
        var tempBuffer1 = backend.AllocateBuffer(hiddenBufferSize);
        var tempBuffer2 = backend.AllocateBuffer(hiddenBufferSize);

        // For training mode, allocate arrays to cache per-timestep states
        bool cacheForTraining = IsTrainingMode;
        if (cacheForTraining)
        {
            ClearGpuTrainingCache();
            _gpuLastInput = input;
            _gpuCachedForgetGates = new IGpuTensor<T>[timeSteps];
            _gpuCachedInputGates = new IGpuTensor<T>[timeSteps];
            _gpuCachedCellCandidates = new IGpuTensor<T>[timeSteps];
            _gpuCachedOutputGates = new IGpuTensor<T>[timeSteps];
            _gpuCachedCellStates = new IGpuTensor<T>[timeSteps];
            _gpuCachedHiddenStates = new IGpuTensor<T>[timeSteps];

            // Cache initial states (zeros)
            var initHBuffer = backend.AllocateBuffer(hiddenBufferSize);
            var initCBuffer = backend.AllocateBuffer(hiddenBufferSize);
            backend.Fill(initHBuffer, 0.0f, hiddenBufferSize);
            backend.Fill(initCBuffer, 0.0f, hiddenBufferSize);
            _gpuInitialHiddenState = new GpuTensor<T>(backend, initHBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
            _gpuInitialCellState = new GpuTensor<T>(backend, initCBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
        }

        try
        {
            // Process each time step
            for (int t = 0; t < timeSteps; t++)
            {
                // Get input slice for time step t
                // Input shape is [batch, timeSteps, inputSize], so slice at offset t * batchSize * inputSize
                int inputSliceOffset = t * batchSize * _inputSize;

                // Create view into input for this timestep
                var inputSlice = input.CreateView(inputSliceOffset, [batchSize, _inputSize]);

                // Allocate gate buffers (persistent if training, temporary otherwise)
                var fGateBuffer = backend.AllocateBuffer(hiddenBufferSize);
                var iGateBuffer = backend.AllocateBuffer(hiddenBufferSize);
                var cTildeBuffer = backend.AllocateBuffer(hiddenBufferSize);
                var oGateBuffer = backend.AllocateBuffer(hiddenBufferSize);
                var newCBuffer = backend.AllocateBuffer(hiddenBufferSize);
                var newHBuffer = backend.AllocateBuffer(hiddenBufferSize);

                // Forget Gate: f = sigmoid(x*Wfi^T + h*Wfh^T + biasF)
                backend.Gemm(inputSlice.Buffer, WfiBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(currentHBuffer, WfhBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, fGateBuffer, hiddenBufferSize);
                backend.BiasAdd(fGateBuffer, biasFBuffer, fGateBuffer, batchSize, _hiddenSize);
                backend.Sigmoid(fGateBuffer, fGateBuffer, hiddenBufferSize);

                // Input Gate: i = sigmoid(x*Wii^T + h*Wih^T + biasI)
                backend.Gemm(inputSlice.Buffer, WiiBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(currentHBuffer, WihBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, iGateBuffer, hiddenBufferSize);
                backend.BiasAdd(iGateBuffer, biasIBuffer, iGateBuffer, batchSize, _hiddenSize);
                backend.Sigmoid(iGateBuffer, iGateBuffer, hiddenBufferSize);

                // Cell Candidate: c_tilde = tanh(x*Wci^T + h*Wch^T + biasC)
                backend.Gemm(inputSlice.Buffer, WciBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(currentHBuffer, WchBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, cTildeBuffer, hiddenBufferSize);
                backend.BiasAdd(cTildeBuffer, biasCBuffer, cTildeBuffer, batchSize, _hiddenSize);
                backend.Tanh(cTildeBuffer, cTildeBuffer, hiddenBufferSize);

                // Output Gate: o = sigmoid(x*Woi^T + h*Woh^T + biasO)
                backend.Gemm(inputSlice.Buffer, WoiBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(currentHBuffer, WohBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, oGateBuffer, hiddenBufferSize);
                backend.BiasAdd(oGateBuffer, biasOBuffer, oGateBuffer, batchSize, _hiddenSize);
                backend.Sigmoid(oGateBuffer, oGateBuffer, hiddenBufferSize);

                // Update Cell State: C = f * C_prev + i * c_tilde
                backend.Multiply(fGateBuffer, currentCBuffer, tempBuffer1, hiddenBufferSize);
                backend.Multiply(iGateBuffer, cTildeBuffer, tempBuffer2, hiddenBufferSize);
                backend.Add(tempBuffer1, tempBuffer2, newCBuffer, hiddenBufferSize);

                // Update Hidden State: H = o * tanh(C)
                backend.Tanh(newCBuffer, tempBuffer1, hiddenBufferSize);
                backend.Multiply(oGateBuffer, tempBuffer1, newHBuffer, hiddenBufferSize);

                // Cache states for training
                if (cacheForTraining && _gpuCachedForgetGates is not null && _gpuCachedInputGates is not null &&
                    _gpuCachedCellCandidates is not null && _gpuCachedOutputGates is not null &&
                    _gpuCachedCellStates is not null && _gpuCachedHiddenStates is not null)
                {
                    _gpuCachedForgetGates[t] = new GpuTensor<T>(backend, fGateBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                    _gpuCachedInputGates[t] = new GpuTensor<T>(backend, iGateBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                    _gpuCachedCellCandidates[t] = new GpuTensor<T>(backend, cTildeBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                    _gpuCachedOutputGates[t] = new GpuTensor<T>(backend, oGateBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                    _gpuCachedCellStates[t] = new GpuTensor<T>(backend, newCBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                    _gpuCachedHiddenStates[t] = new GpuTensor<T>(backend, newHBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                }
                else
                {
                    // Dispose gate buffers if not caching
                    fGateBuffer.Dispose();
                    iGateBuffer.Dispose();
                    cTildeBuffer.Dispose();
                    oGateBuffer.Dispose();
                }

                // Copy new states to current state buffers (use the cached buffer if training)
                if (cacheForTraining && _gpuCachedCellStates is not null && _gpuCachedHiddenStates is not null &&
                    _gpuCachedCellStates[t] is not null && _gpuCachedHiddenStates[t] is not null)
                {
                    backend.Copy(_gpuCachedCellStates[t].Buffer, currentCBuffer, hiddenBufferSize);
                    backend.Copy(_gpuCachedHiddenStates[t].Buffer, currentHBuffer, hiddenBufferSize);
                }
                else
                {
                    backend.Copy(newCBuffer, currentCBuffer, hiddenBufferSize);
                    backend.Copy(newHBuffer, currentHBuffer, hiddenBufferSize);
                    newCBuffer.Dispose();
                    newHBuffer.Dispose();
                }

                // Store hidden state in output at position t
                int outputOffset = t * hiddenBufferSize;
                backend.Copy2DStrided(currentHBuffer, outputBuffer, 1, hiddenBufferSize, outputSize, outputOffset);
            }

            // Determine output shape
            int[] outputShape;
            if (reshaped2D)
            {
                outputShape = [timeSteps, _hiddenSize];
            }
            else if (reshapedHigherRank && originalShape != null)
            {
                outputShape = new int[originalShape.Length];
                for (int d = 0; d < originalShape.Length - 2; d++)
                    outputShape[d] = originalShape[d];
                outputShape[originalShape.Length - 2] = timeSteps;
                outputShape[originalShape.Length - 1] = _hiddenSize;
            }
            else
            {
                outputShape = [batchSize, timeSteps, _hiddenSize];
            }

            // Cleanup temporary buffers
            currentHBuffer.Dispose();
            currentCBuffer.Dispose();
            tempBuffer1.Dispose();
            tempBuffer2.Dispose();

            return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            // Cleanup buffers on error
            currentHBuffer.Dispose();
            currentCBuffer.Dispose();
            tempBuffer1.Dispose();
            tempBuffer2.Dispose();
            outputBuffer.Dispose();
            ClearGpuTrainingCache();
            throw;
        }
    }

    /// <summary>
    /// Clears the cached GPU tensors used for training.
    /// </summary>
    private void ClearGpuTrainingCache()
    {
        _gpuLastInput = null;

        if (_gpuCachedForgetGates != null)
        {
            foreach (var t in _gpuCachedForgetGates)
                (t as IDisposable)?.Dispose();
            _gpuCachedForgetGates = null;
        }

        if (_gpuCachedInputGates != null)
        {
            foreach (var t in _gpuCachedInputGates)
                (t as IDisposable)?.Dispose();
            _gpuCachedInputGates = null;
        }

        if (_gpuCachedCellCandidates != null)
        {
            foreach (var t in _gpuCachedCellCandidates)
                (t as IDisposable)?.Dispose();
            _gpuCachedCellCandidates = null;
        }

        if (_gpuCachedOutputGates != null)
        {
            foreach (var t in _gpuCachedOutputGates)
                (t as IDisposable)?.Dispose();
            _gpuCachedOutputGates = null;
        }

        if (_gpuCachedCellStates != null)
        {
            foreach (var t in _gpuCachedCellStates)
                (t as IDisposable)?.Dispose();
            _gpuCachedCellStates = null;
        }

        if (_gpuCachedHiddenStates != null)
        {
            foreach (var t in _gpuCachedHiddenStates)
                (t as IDisposable)?.Dispose();
            _gpuCachedHiddenStates = null;
        }

        (_gpuInitialHiddenState as IDisposable)?.Dispose();
        _gpuInitialHiddenState = null;

        (_gpuInitialCellState as IDisposable)?.Dispose();
        _gpuInitialCellState = null;
    }

    /// <summary>
    /// GPU-resident backward pass for LSTM using Backpropagation Through Time (BPTT).
    /// Computes gradients for all weights while keeping tensors on GPU.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient from upstream layer.</param>
    /// <returns>GPU-resident gradient to pass to previous layer.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the GPU-accelerated backward pass of the LSTM layer.
    /// It processes the sequence in reverse order (from the last time step to the first),
    /// computing gradients for all gate parameters (forget, input, cell, output),
    /// hidden states, and cell states.
    /// </para>
    /// <para><b>For Beginners:</b> This is the GPU version of backpropagation through time.
    /// All computations stay on the GPU for maximum performance.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.Backend ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuLastInput == null || _gpuCachedForgetGates == null || _gpuCachedInputGates == null ||
            _gpuCachedCellCandidates == null || _gpuCachedOutputGates == null ||
            _gpuCachedCellStates == null || _gpuCachedHiddenStates == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        int batchSize = _gpuLastInput.Shape[0];
        int timeSteps = _gpuCachedForgetGates.Length;
        int hiddenBufferSize = batchSize * _hiddenSize;
        int inputBufferSize = batchSize * _inputSize;

        // Initialize gradient accumulators for weights (zeros)
        var dWfiBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dWiiBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dWciBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dWoiBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dWfhBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dWihBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dWchBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dWohBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dBfBuffer = backend.AllocateBuffer(_hiddenSize);
        var dBiBuffer = backend.AllocateBuffer(_hiddenSize);
        var dBcBuffer = backend.AllocateBuffer(_hiddenSize);
        var dBoBuffer = backend.AllocateBuffer(_hiddenSize);

        backend.Fill(dWfiBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dWiiBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dWciBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dWoiBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dWfhBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dWihBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dWchBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dWohBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dBfBuffer, 0.0f, _hiddenSize);
        backend.Fill(dBiBuffer, 0.0f, _hiddenSize);
        backend.Fill(dBcBuffer, 0.0f, _hiddenSize);
        backend.Fill(dBoBuffer, 0.0f, _hiddenSize);

        // Allocate input gradient buffer
        int inputGradientSize = batchSize * timeSteps * _inputSize;
        var inputGradientBuffer = backend.AllocateBuffer(inputGradientSize);
        backend.Fill(inputGradientBuffer, 0.0f, inputGradientSize);

        // Upload transposed weights to GPU for backward matmul
        using var WfiT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsFi.ToArray()));
        using var WiiT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsIi.ToArray()));
        using var WciT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsCi.ToArray()));
        using var WoiT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsOi.ToArray()));
        using var WfhT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsFh.ToArray()));
        using var WihT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsIh.ToArray()));
        using var WchT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsCh.ToArray()));
        using var WohT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weightsOh.ToArray()));

        // Temporary buffers
        var dHNext = backend.AllocateBuffer(hiddenBufferSize);
        var dCNext = backend.AllocateBuffer(hiddenBufferSize);
        var tempBuffer1 = backend.AllocateBuffer(hiddenBufferSize);
        var tempBuffer2 = backend.AllocateBuffer(hiddenBufferSize);
        // Weight gradient buffer needs to hold max of input and hidden weight sizes
        int maxWeightSize = Math.Max(_inputSize, _hiddenSize) * _hiddenSize;
        var weightGradBuffer = backend.AllocateBuffer(maxWeightSize);
        // Temp buffer for input gradient computation (batchSize x inputSize)
        var inputGradTemp = backend.AllocateBuffer(inputBufferSize);
        var dO = backend.AllocateBuffer(hiddenBufferSize);
        var dC = backend.AllocateBuffer(hiddenBufferSize);
        var dF = backend.AllocateBuffer(hiddenBufferSize);
        var dI = backend.AllocateBuffer(hiddenBufferSize);
        var dCTilde = backend.AllocateBuffer(hiddenBufferSize);
        var tanhC = backend.AllocateBuffer(hiddenBufferSize);

        // Additional buffers for transpose operations (GemmTN replacement)
        var inputTranspose = backend.AllocateBuffer(inputBufferSize);
        var hiddenTranspose = backend.AllocateBuffer(hiddenBufferSize);
        var gateGradTranspose = backend.AllocateBuffer(hiddenBufferSize);
        // Temp buffer for hidden gradient computation (batchSize x hiddenSize)
        var hiddenGradTemp = backend.AllocateBuffer(hiddenBufferSize);

        backend.Fill(dHNext, 0.0f, hiddenBufferSize);
        backend.Fill(dCNext, 0.0f, hiddenBufferSize);

        try
        {
            // Process each time step in reverse order (BPTT)
            for (int t = timeSteps - 1; t >= 0; t--)
            {
                // Get output gradient for this timestep
                int gradOffset = t * hiddenBufferSize;
                var gradSlice = outputGradient.CreateView(gradOffset, [batchSize, _hiddenSize]);

                // Add gradient from next timestep to current output gradient
                backend.Add(gradSlice.Buffer, dHNext, dHNext, hiddenBufferSize);

                // Get cached states
                var f_t = _gpuCachedForgetGates[t];
                var i_t = _gpuCachedInputGates[t];
                var cTilde_t = _gpuCachedCellCandidates[t];
                var o_t = _gpuCachedOutputGates[t];
                var c_t = _gpuCachedCellStates[t];
                var h_t = _gpuCachedHiddenStates[t];

                // Get previous states
                IGpuTensor<T> c_prev = t > 0 ? _gpuCachedCellStates[t - 1] : _gpuInitialCellState!;
                IGpuTensor<T> h_prev = t > 0 ? _gpuCachedHiddenStates[t - 1] : _gpuInitialHiddenState!;

                // dO = dH * tanh(C) * sigmoid'(o) where sigmoid'(o) = o * (1 - o)
                backend.Tanh(c_t.Buffer, tanhC, hiddenBufferSize);
                backend.Multiply(dHNext, tanhC, tempBuffer1, hiddenBufferSize);
                // SigmoidBackward computes: dO = tempBuffer1 * o * (1 - o)
                backend.SigmoidBackward(tempBuffer1, o_t.Buffer, dO, hiddenBufferSize);

                // dC = dH * o * tanh'(C) + dC_next where tanh'(C) = 1 - tanh(C)^2
                backend.Multiply(dHNext, o_t.Buffer, tempBuffer1, hiddenBufferSize);
                // TanhBackward computes: dC = tempBuffer1 * (1 - tanh(c_t)^2)
                // Note: tanhC already contains tanh(c_t) from above
                backend.TanhBackward(tempBuffer1, tanhC, dC, hiddenBufferSize);
                backend.Add(dC, dCNext, dC, hiddenBufferSize);

                // dF = dC * C_prev * sigmoid'(f) where sigmoid'(f) = f * (1 - f)
                backend.Multiply(dC, c_prev.Buffer, tempBuffer1, hiddenBufferSize);
                backend.SigmoidBackward(tempBuffer1, f_t.Buffer, dF, hiddenBufferSize);

                // dI = dC * cTilde * sigmoid'(i) where sigmoid'(i) = i * (1 - i)
                backend.Multiply(dC, cTilde_t.Buffer, tempBuffer1, hiddenBufferSize);
                backend.SigmoidBackward(tempBuffer1, i_t.Buffer, dI, hiddenBufferSize);

                // dCTilde = dC * i * tanh'(cTilde) where tanh'(cTilde) = 1 - cTilde^2
                backend.Multiply(dC, i_t.Buffer, tempBuffer1, hiddenBufferSize);
                backend.TanhBackward(tempBuffer1, cTilde_t.Buffer, dCTilde, hiddenBufferSize);

                // dC_next = dC * f (for next iteration, which is previous timestep)
                backend.Multiply(dC, f_t.Buffer, dCNext, hiddenBufferSize);

                // Get input for this timestep
                int inputOffset = t * inputBufferSize;
                var inputSlice = _gpuLastInput.CreateView(inputOffset, [batchSize, _inputSize]);

                // Accumulate weight gradients using outer products
                // dW_i += x^T @ dGate (for input weights)
                // dW_h += h_prev^T @ dGate (for hidden weights)
                // dB += sum(dGate, axis=0) (for biases)

                // Forget gate gradients
                // dWfi += input^T @ dF: Transpose input then Gemm
                backend.Transpose(inputSlice.Buffer, inputTranspose, batchSize, _inputSize);
                backend.Gemm(inputTranspose, dF, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWfiBuffer, weightGradBuffer, dWfiBuffer, _hiddenSize * _inputSize);
                // dWfh += h_prev^T @ dF: Transpose h_prev then Gemm
                backend.Transpose(h_prev.Buffer, hiddenTranspose, batchSize, _hiddenSize);
                backend.Gemm(hiddenTranspose, dF, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dWfhBuffer, weightGradBuffer, dWfhBuffer, _hiddenSize * _hiddenSize);
                // dBf += sum(dF, axis=0): Transpose then SumAxis
                backend.Transpose(dF, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBfBuffer, tempBuffer1, dBfBuffer, _hiddenSize);

                // Input gate gradients
                // dWii += input^T @ dI: input already transposed above
                backend.Gemm(inputTranspose, dI, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWiiBuffer, weightGradBuffer, dWiiBuffer, _hiddenSize * _inputSize);
                // dWih += h_prev^T @ dI: h_prev already transposed above
                backend.Gemm(hiddenTranspose, dI, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dWihBuffer, weightGradBuffer, dWihBuffer, _hiddenSize * _hiddenSize);
                // dBi += sum(dI, axis=0): Transpose then SumAxis
                backend.Transpose(dI, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBiBuffer, tempBuffer1, dBiBuffer, _hiddenSize);

                // Cell candidate gradients
                // dWci += input^T @ dCTilde: input already transposed above
                backend.Gemm(inputTranspose, dCTilde, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWciBuffer, weightGradBuffer, dWciBuffer, _hiddenSize * _inputSize);
                // dWch += h_prev^T @ dCTilde: h_prev already transposed above
                backend.Gemm(hiddenTranspose, dCTilde, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dWchBuffer, weightGradBuffer, dWchBuffer, _hiddenSize * _hiddenSize);
                // dBc += sum(dCTilde, axis=0): Transpose then SumAxis
                backend.Transpose(dCTilde, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBcBuffer, tempBuffer1, dBcBuffer, _hiddenSize);

                // Output gate gradients
                // dWoi += input^T @ dO: input already transposed above
                backend.Gemm(inputTranspose, dO, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWoiBuffer, weightGradBuffer, dWoiBuffer, _hiddenSize * _inputSize);
                // dWoh += h_prev^T @ dO: h_prev already transposed above
                backend.Gemm(hiddenTranspose, dO, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dWohBuffer, weightGradBuffer, dWohBuffer, _hiddenSize * _hiddenSize);
                // dBo += sum(dO, axis=0): Transpose then SumAxis
                backend.Transpose(dO, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBoBuffer, tempBuffer1, dBoBuffer, _hiddenSize);

                // Compute input gradient for this timestep
                // dX = dF @ Wfi + dI @ Wii + dCTilde @ Wci + dO @ Woi
                var inputGradSlice = backend.AllocateBuffer(inputBufferSize);
                backend.Gemm(dF, WfiT, inputGradSlice, batchSize, _inputSize, _hiddenSize);
                backend.Gemm(dI, WiiT, inputGradTemp, batchSize, _inputSize, _hiddenSize);
                backend.Add(inputGradSlice, inputGradTemp, inputGradSlice, inputBufferSize);
                backend.Gemm(dCTilde, WciT, inputGradTemp, batchSize, _inputSize, _hiddenSize);
                backend.Add(inputGradSlice, inputGradTemp, inputGradSlice, inputBufferSize);
                backend.Gemm(dO, WoiT, inputGradTemp, batchSize, _inputSize, _hiddenSize);
                backend.Add(inputGradSlice, inputGradTemp, inputGradSlice, inputBufferSize);

                // Store input gradient
                backend.Copy2DStrided(inputGradSlice, inputGradientBuffer, 1, inputBufferSize, inputGradientSize, inputOffset);
                inputGradSlice.Dispose();

                // Compute dH_next for previous timestep
                // dH_prev = dF @ Wfh + dI @ Wih + dCTilde @ Wch + dO @ Woh
                backend.Gemm(dF, WfhT, dHNext, batchSize, _hiddenSize, _hiddenSize);
                backend.Gemm(dI, WihT, hiddenGradTemp, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(dHNext, hiddenGradTemp, dHNext, hiddenBufferSize);
                backend.Gemm(dCTilde, WchT, hiddenGradTemp, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(dHNext, hiddenGradTemp, dHNext, hiddenBufferSize);
                backend.Gemm(dO, WohT, hiddenGradTemp, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(dHNext, hiddenGradTemp, dHNext, hiddenBufferSize);
            }

            // Store gradient tensors for UpdateParametersGpu
            _gpuWeightsFiGradient = new GpuTensor<T>(backend, dWfiBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsIiGradient = new GpuTensor<T>(backend, dWiiBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsCiGradient = new GpuTensor<T>(backend, dWciBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsOiGradient = new GpuTensor<T>(backend, dWoiBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsFhGradient = new GpuTensor<T>(backend, dWfhBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsIhGradient = new GpuTensor<T>(backend, dWihBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsChGradient = new GpuTensor<T>(backend, dWchBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWeightsOhGradient = new GpuTensor<T>(backend, dWohBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasFGradient = new GpuTensor<T>(backend, dBfBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasIGradient = new GpuTensor<T>(backend, dBiBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasCGradient = new GpuTensor<T>(backend, dBcBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBiasOGradient = new GpuTensor<T>(backend, dBoBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);

            // Cleanup temporary buffers
            dHNext.Dispose();
            dCNext.Dispose();
            tempBuffer1.Dispose();
            tempBuffer2.Dispose();
            weightGradBuffer.Dispose();
            inputGradTemp.Dispose();
            inputTranspose.Dispose();
            hiddenTranspose.Dispose();
            gateGradTranspose.Dispose();
            hiddenGradTemp.Dispose();
            dO.Dispose();
            dC.Dispose();
            dF.Dispose();
            dI.Dispose();
            dCTilde.Dispose();
            tanhC.Dispose();

            return new GpuTensor<T>(backend, inputGradientBuffer, [batchSize, timeSteps, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
        }
        catch
        {
            dWfiBuffer.Dispose();
            dWiiBuffer.Dispose();
            dWciBuffer.Dispose();
            dWoiBuffer.Dispose();
            dWfhBuffer.Dispose();
            dWihBuffer.Dispose();
            dWchBuffer.Dispose();
            dWohBuffer.Dispose();
            dBfBuffer.Dispose();
            dBiBuffer.Dispose();
            dBcBuffer.Dispose();
            dBoBuffer.Dispose();
            inputGradientBuffer.Dispose();
            dHNext.Dispose();
            dCNext.Dispose();
            tempBuffer1.Dispose();
            tempBuffer2.Dispose();
            weightGradBuffer.Dispose();
            inputGradTemp.Dispose();
            inputTranspose.Dispose();
            hiddenTranspose.Dispose();
            gateGradTranspose.Dispose();
            hiddenGradTemp.Dispose();
            dO.Dispose();
            dC.Dispose();
            dF.Dispose();
            dI.Dispose();
            dCTilde.Dispose();
            tanhC.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-resident parameter update with polymorphic optimizer support.
    /// Updates all weight tensors directly on GPU using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.Backend ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuWeightsFiGradient == null || _gpuWeightsIiGradient == null ||
            _gpuWeightsCiGradient == null || _gpuWeightsOiGradient == null ||
            _gpuWeightsFhGradient == null || _gpuWeightsIhGradient == null ||
            _gpuWeightsChGradient == null || _gpuWeightsOhGradient == null ||
            _gpuBiasFGradient == null || _gpuBiasIGradient == null ||
            _gpuBiasCGradient == null || _gpuBiasOGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuWeightsFi ??= new GpuTensor<T>(backend, _weightsFi, GpuTensorRole.Weight);
        _gpuWeightsIi ??= new GpuTensor<T>(backend, _weightsIi, GpuTensorRole.Weight);
        _gpuWeightsCi ??= new GpuTensor<T>(backend, _weightsCi, GpuTensorRole.Weight);
        _gpuWeightsOi ??= new GpuTensor<T>(backend, _weightsOi, GpuTensorRole.Weight);
        _gpuWeightsFh ??= new GpuTensor<T>(backend, _weightsFh, GpuTensorRole.Weight);
        _gpuWeightsIh ??= new GpuTensor<T>(backend, _weightsIh, GpuTensorRole.Weight);
        _gpuWeightsCh ??= new GpuTensor<T>(backend, _weightsCh, GpuTensorRole.Weight);
        _gpuWeightsOh ??= new GpuTensor<T>(backend, _weightsOh, GpuTensorRole.Weight);
        _gpuBiasF ??= new GpuTensor<T>(backend, _biasF, GpuTensorRole.Bias);
        _gpuBiasI ??= new GpuTensor<T>(backend, _biasI, GpuTensorRole.Bias);
        _gpuBiasC ??= new GpuTensor<T>(backend, _biasC, GpuTensorRole.Bias);
        _gpuBiasO ??= new GpuTensor<T>(backend, _biasO, GpuTensorRole.Bias);

        // Ensure optimizer state buffers exist
        EnsureLstmOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        config.ApplyUpdate(backend, _gpuWeightsFi.Buffer, _gpuWeightsFiGradient.Buffer, BuildLstmOptimizerState("Wfi"), _weightsFi.Length);
        config.ApplyUpdate(backend, _gpuWeightsIi.Buffer, _gpuWeightsIiGradient.Buffer, BuildLstmOptimizerState("Wii"), _weightsIi.Length);
        config.ApplyUpdate(backend, _gpuWeightsCi.Buffer, _gpuWeightsCiGradient.Buffer, BuildLstmOptimizerState("Wci"), _weightsCi.Length);
        config.ApplyUpdate(backend, _gpuWeightsOi.Buffer, _gpuWeightsOiGradient.Buffer, BuildLstmOptimizerState("Woi"), _weightsOi.Length);
        config.ApplyUpdate(backend, _gpuWeightsFh.Buffer, _gpuWeightsFhGradient.Buffer, BuildLstmOptimizerState("Wfh"), _weightsFh.Length);
        config.ApplyUpdate(backend, _gpuWeightsIh.Buffer, _gpuWeightsIhGradient.Buffer, BuildLstmOptimizerState("Wih"), _weightsIh.Length);
        config.ApplyUpdate(backend, _gpuWeightsCh.Buffer, _gpuWeightsChGradient.Buffer, BuildLstmOptimizerState("Wch"), _weightsCh.Length);
        config.ApplyUpdate(backend, _gpuWeightsOh.Buffer, _gpuWeightsOhGradient.Buffer, BuildLstmOptimizerState("Woh"), _weightsOh.Length);
        config.ApplyUpdate(backend, _gpuBiasF.Buffer, _gpuBiasFGradient.Buffer, BuildLstmOptimizerState("Bf"), _biasF.Length);
        config.ApplyUpdate(backend, _gpuBiasI.Buffer, _gpuBiasIGradient.Buffer, BuildLstmOptimizerState("Bi"), _biasI.Length);
        config.ApplyUpdate(backend, _gpuBiasC.Buffer, _gpuBiasCGradient.Buffer, BuildLstmOptimizerState("Bc"), _biasC.Length);
        config.ApplyUpdate(backend, _gpuBiasO.Buffer, _gpuBiasOGradient.Buffer, BuildLstmOptimizerState("Bo"), _biasO.Length);

        // Sync back to CPU tensors for compatibility
        _weightsFi = _gpuWeightsFi.ToTensor();
        _weightsIi = _gpuWeightsIi.ToTensor();
        _weightsCi = _gpuWeightsCi.ToTensor();
        _weightsOi = _gpuWeightsOi.ToTensor();
        _weightsFh = _gpuWeightsFh.ToTensor();
        _weightsIh = _gpuWeightsIh.ToTensor();
        _weightsCh = _gpuWeightsCh.ToTensor();
        _weightsOh = _gpuWeightsOh.ToTensor();
        _biasF = _gpuBiasF.ToTensor();
        _biasI = _gpuBiasI.ToTensor();
        _biasC = _gpuBiasC.ToTensor();
        _biasO = _gpuBiasO.ToTensor();
    }

    /// <summary>
    /// Ensures GPU optimizer state buffers exist for all LSTM parameters.
    /// </summary>
    private void EnsureLstmOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int inputWeightSize = _hiddenSize * _inputSize;
        int hiddenWeightSize = _hiddenSize * _hiddenSize;
        int biasSize = _hiddenSize;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                // Velocity buffers
                _gpuWeightsFiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightsFiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
                // SquaredAvg buffers for RMSprop
                _gpuWeightsFiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adagrad:
                // AccumulatedGrad buffers for Adagrad
                _gpuWeightsFiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;
        }
    }

    /// <summary>
    /// Builds the optimizer state for a specific LSTM parameter.
    /// </summary>
    private GpuOptimizerState BuildLstmOptimizerState(string paramName)
    {
        return paramName switch
        {
            "Wfi" => new GpuOptimizerState { Velocity = _gpuWeightsFiVelocity?.Buffer, M = _gpuWeightsFiM?.Buffer, V = _gpuWeightsFiV?.Buffer, SquaredAvg = _gpuWeightsFiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsFiVelocity?.Buffer },
            "Wii" => new GpuOptimizerState { Velocity = _gpuWeightsIiVelocity?.Buffer, M = _gpuWeightsIiM?.Buffer, V = _gpuWeightsIiV?.Buffer, SquaredAvg = _gpuWeightsIiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsIiVelocity?.Buffer },
            "Wci" => new GpuOptimizerState { Velocity = _gpuWeightsCiVelocity?.Buffer, M = _gpuWeightsCiM?.Buffer, V = _gpuWeightsCiV?.Buffer, SquaredAvg = _gpuWeightsCiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsCiVelocity?.Buffer },
            "Woi" => new GpuOptimizerState { Velocity = _gpuWeightsOiVelocity?.Buffer, M = _gpuWeightsOiM?.Buffer, V = _gpuWeightsOiV?.Buffer, SquaredAvg = _gpuWeightsOiVelocity?.Buffer, AccumulatedGrad = _gpuWeightsOiVelocity?.Buffer },
            "Wfh" => new GpuOptimizerState { Velocity = _gpuWeightsFhVelocity?.Buffer, M = _gpuWeightsFhM?.Buffer, V = _gpuWeightsFhV?.Buffer, SquaredAvg = _gpuWeightsFhVelocity?.Buffer, AccumulatedGrad = _gpuWeightsFhVelocity?.Buffer },
            "Wih" => new GpuOptimizerState { Velocity = _gpuWeightsIhVelocity?.Buffer, M = _gpuWeightsIhM?.Buffer, V = _gpuWeightsIhV?.Buffer, SquaredAvg = _gpuWeightsIhVelocity?.Buffer, AccumulatedGrad = _gpuWeightsIhVelocity?.Buffer },
            "Wch" => new GpuOptimizerState { Velocity = _gpuWeightsChVelocity?.Buffer, M = _gpuWeightsChM?.Buffer, V = _gpuWeightsChV?.Buffer, SquaredAvg = _gpuWeightsChVelocity?.Buffer, AccumulatedGrad = _gpuWeightsChVelocity?.Buffer },
            "Woh" => new GpuOptimizerState { Velocity = _gpuWeightsOhVelocity?.Buffer, M = _gpuWeightsOhM?.Buffer, V = _gpuWeightsOhV?.Buffer, SquaredAvg = _gpuWeightsOhVelocity?.Buffer, AccumulatedGrad = _gpuWeightsOhVelocity?.Buffer },
            "Bf" => new GpuOptimizerState { Velocity = _gpuBiasFVelocity?.Buffer, M = _gpuBiasFM?.Buffer, V = _gpuBiasFV?.Buffer, SquaredAvg = _gpuBiasFVelocity?.Buffer, AccumulatedGrad = _gpuBiasFVelocity?.Buffer },
            "Bi" => new GpuOptimizerState { Velocity = _gpuBiasIVelocity?.Buffer, M = _gpuBiasIM?.Buffer, V = _gpuBiasIV?.Buffer, SquaredAvg = _gpuBiasIVelocity?.Buffer, AccumulatedGrad = _gpuBiasIVelocity?.Buffer },
            "Bc" => new GpuOptimizerState { Velocity = _gpuBiasCVelocity?.Buffer, M = _gpuBiasCM?.Buffer, V = _gpuBiasCV?.Buffer, SquaredAvg = _gpuBiasCVelocity?.Buffer, AccumulatedGrad = _gpuBiasCVelocity?.Buffer },
            "Bo" => new GpuOptimizerState { Velocity = _gpuBiasOVelocity?.Buffer, M = _gpuBiasOM?.Buffer, V = _gpuBiasOV?.Buffer, SquaredAvg = _gpuBiasOVelocity?.Buffer, AccumulatedGrad = _gpuBiasOVelocity?.Buffer },
            _ => new GpuOptimizerState()
        };
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
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using Backpropagation Through Time (BPTT) for LSTM.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using manual gradient calculations optimized for
    /// LSTM networks. It performs backpropagation through time (BPTT), processing the
    /// sequence in reverse order and computing gradients for all gate parameters (forget, input,
    /// cell, output), hidden states, and cell states.
    /// </para>
    /// <para>
    /// Autodiff Note: LSTM backward pass involves complex gate interactions and cell state dynamics.
    /// Implementing this with automatic differentiation would require handling temporal dependencies,
    /// gate-specific gradient flows, and the memory cell update mechanism. The manual implementation
    /// provides efficient and correct gradient calculations for all LSTM components.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _cachedHiddenStates == null || _cachedCellStates == null)
        {
            throw new InvalidOperationException("Backward pass called before forward pass.");
        }

        // Normalize outputGradient to 3D to match canonical _lastInput shape
        var outGrad3D = outputGradient;
        int origRank = _originalInputShape?.Length ?? 3;
        if (_originalInputShape != null && origRank == 2)
        {
            // 2D output gradient -> 3D (add batch dim)
            outGrad3D = outputGradient.Reshape([1, outputGradient.Shape[0], outputGradient.Shape[1]]);
        }
        else if (_originalInputShape != null && origRank > 3)
        {
            // Higher-rank output gradient -> 3D (flatten leading dims)
            int flatBatch = 1;
            for (int d = 0; d < origRank - 2; d++)
                flatBatch *= _originalInputShape[d];
            outGrad3D = outputGradient.Reshape([flatBatch, outputGradient.Shape[origRank - 2], outputGradient.Shape[origRank - 1]]);
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

        var dNextH = new Tensor<T>(new int[] { batchSize, _hiddenSize });
        var dNextC = new Tensor<T>(new int[] { batchSize, _hiddenSize });

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            // Slice along time dimension (dim 1) for all 3D tensors
            var dh = outGrad3D.GetSliceAlongDimension(t, 1).Add(dNextH);
            var xt = _lastInput.GetSliceAlongDimension(t, 1);
            // Use cached states - slice along time dimension
            var prevH = t > 0 ? _cachedHiddenStates.GetSliceAlongDimension(t - 1, 1) : new Tensor<T>(new int[] { batchSize, _hiddenSize });
            var prevC = t > 0 ? _cachedCellStates.GetSliceAlongDimension(t - 1, 1) : new Tensor<T>(new int[] { batchSize, _hiddenSize });

            var (dxt, dprevH, dprevC, dWfi, dWii, dWci, dWoi, dWfh, dWih, dWch, dWoh, dbf, dbi, dbc, dbo) =
                BackwardStep(dh, dNextC, xt, prevH, prevC);

            inputGradient.SetSlice(1, t, dxt);
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

        // Restore higher-rank gradients to their original shape
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation for LSTM.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients for the LSTM layer.
    /// It processes the sequence through time, creating a computation graph for each time step
    /// that includes all gate computations (forget, input, cell, output) and state updates.
    /// The autodiff system then handles gradient propagation through these operations.
    /// </para>
    /// <para>
    /// Note: This implementation is slower than BackwardManual but provides:
    /// - Gradient verification capability
    /// - Easier experimentation with gate modifications
    /// - Educational value for understanding LSTM gradient flow
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastCellState == null ||
            _cachedHiddenStates == null || _cachedCellStates == null)
        {
            throw new InvalidOperationException("Backward pass called before forward pass.");
        }

        // Normalize outputGradient to 3D to match canonical _lastInput shape
        var outGrad3D = outputGradient;
        int origRank = _originalInputShape?.Length ?? 3;
        if (_originalInputShape != null && origRank == 2)
        {
            // 2D output gradient -> 3D (add batch dim)
            outGrad3D = outputGradient.Reshape([1, outputGradient.Shape[0], outputGradient.Shape[1]]);
        }
        else if (_originalInputShape != null && origRank > 3)
        {
            // Higher-rank output gradient -> 3D (flatten leading dims)
            int flatBatch = 1;
            for (int d = 0; d < origRank - 2; d++)
                flatBatch *= _originalInputShape[d];
            outGrad3D = outputGradient.Reshape([flatBatch, outputGradient.Shape[origRank - 2], outputGradient.Shape[origRank - 1]]);
        }

        int batchSize = _lastInput.Shape[0];
        int timeSteps = _lastInput.Shape[1];

        // Initialize gradient accumulators for all parameters
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

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Process each time step using autodiff
        for (int t = timeSteps - 1; t >= 0; t--)
        {
            // Get input and states for this time step - slice along time dimension (dim 1)
            var xt = _lastInput.GetSliceAlongDimension(t, 1);
            // Use cached states for previous time step (they are 3D: [batch, time, hidden])
            var prevH = t > 0 ? _cachedHiddenStates.GetSliceAlongDimension(t - 1, 1) : new Tensor<T>(new int[] { batchSize, _hiddenSize });
            var prevC = t > 0 ? _cachedCellStates.GetSliceAlongDimension(t - 1, 1) : new Tensor<T>(new int[] { batchSize, _hiddenSize });
            var gradSlice = outGrad3D.GetSliceAlongDimension(t, 1);

            // Convert parameters to computation nodes with gradient tracking
            var inputNode = Autodiff.TensorOperations<T>.Variable(xt, "input", requiresGradient: true);
            var prevHNode = Autodiff.TensorOperations<T>.Variable(prevH, "prevH", requiresGradient: true);
            var prevCNode = Autodiff.TensorOperations<T>.Variable(prevC, "prevC", requiresGradient: true);

            var weightsFiNode = Autodiff.TensorOperations<T>.Variable(_weightsFi, "weightsFi", requiresGradient: true);
            var weightsIiNode = Autodiff.TensorOperations<T>.Variable(_weightsIi, "weightsIi", requiresGradient: true);
            var weightsCiNode = Autodiff.TensorOperations<T>.Variable(_weightsCi, "weightsCi", requiresGradient: true);
            var weightsOiNode = Autodiff.TensorOperations<T>.Variable(_weightsOi, "weightsOi", requiresGradient: true);

            var weightsFhNode = Autodiff.TensorOperations<T>.Variable(_weightsFh, "weightsFh", requiresGradient: true);
            var weightsIhNode = Autodiff.TensorOperations<T>.Variable(_weightsIh, "weightsIh", requiresGradient: true);
            var weightsChNode = Autodiff.TensorOperations<T>.Variable(_weightsCh, "weightsCh", requiresGradient: true);
            var weightsOhNode = Autodiff.TensorOperations<T>.Variable(_weightsOh, "weightsOh", requiresGradient: true);

            var biasFNode = Autodiff.TensorOperations<T>.Variable(_biasF, "biasF", requiresGradient: true);
            var biasINode = Autodiff.TensorOperations<T>.Variable(_biasI, "biasI", requiresGradient: true);
            var biasCNode = Autodiff.TensorOperations<T>.Variable(_biasC, "biasC", requiresGradient: true);
            var biasONode = Autodiff.TensorOperations<T>.Variable(_biasO, "biasO", requiresGradient: true);

            // Compute LSTM gates using autodiff operations
            // Forget gate: f = sigmoid(xt @ Wfi + prevH @ Wfh + bf)
            var fInput = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, Autodiff.TensorOperations<T>.Transpose(weightsFiNode));
            var fHidden = Autodiff.TensorOperations<T>.MatrixMultiply(prevHNode, Autodiff.TensorOperations<T>.Transpose(weightsFhNode));
            var fPreActivation = Autodiff.TensorOperations<T>.Add(Autodiff.TensorOperations<T>.Add(fInput, fHidden), biasFNode);
            var f = Autodiff.TensorOperations<T>.Sigmoid(fPreActivation);

            // Input gate: i = sigmoid(xt @ Wii + prevH @ Wih + bi)
            var iInput = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, Autodiff.TensorOperations<T>.Transpose(weightsIiNode));
            var iHidden = Autodiff.TensorOperations<T>.MatrixMultiply(prevHNode, Autodiff.TensorOperations<T>.Transpose(weightsIhNode));
            var iPreActivation = Autodiff.TensorOperations<T>.Add(Autodiff.TensorOperations<T>.Add(iInput, iHidden), biasINode);
            var i = Autodiff.TensorOperations<T>.Sigmoid(iPreActivation);

            // Cell candidate: c_tilde = tanh(xt @ Wci + prevH @ Wch + bc)
            var cInput = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, Autodiff.TensorOperations<T>.Transpose(weightsCiNode));
            var cHidden = Autodiff.TensorOperations<T>.MatrixMultiply(prevHNode, Autodiff.TensorOperations<T>.Transpose(weightsChNode));
            var cPreActivation = Autodiff.TensorOperations<T>.Add(Autodiff.TensorOperations<T>.Add(cInput, cHidden), biasCNode);
            var cTilde = Autodiff.TensorOperations<T>.Tanh(cPreActivation);

            // Output gate: o = sigmoid(xt @ Woi + prevH @ Woh + bo)
            var oInput = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, Autodiff.TensorOperations<T>.Transpose(weightsOiNode));
            var oHidden = Autodiff.TensorOperations<T>.MatrixMultiply(prevHNode, Autodiff.TensorOperations<T>.Transpose(weightsOhNode));
            var oPreActivation = Autodiff.TensorOperations<T>.Add(Autodiff.TensorOperations<T>.Add(oInput, oHidden), biasONode);
            var o = Autodiff.TensorOperations<T>.Sigmoid(oPreActivation);

            // Cell state update: newC = f * prevC + i * c_tilde
            var forgetGated = Autodiff.TensorOperations<T>.ElementwiseMultiply(f, prevCNode);
            var inputGated = Autodiff.TensorOperations<T>.ElementwiseMultiply(i, cTilde);
            var newC = Autodiff.TensorOperations<T>.Add(forgetGated, inputGated);

            // Hidden state update: newH = o * tanh(newC)
            var newCActivated = Autodiff.TensorOperations<T>.Tanh(newC);
            var newH = Autodiff.TensorOperations<T>.ElementwiseMultiply(o, newCActivated);

            // Set gradient at output and propagate backward
            newH.Gradient = gradSlice;

            // Perform topological sort and backward pass (inlined)
            var visited = new HashSet<Autodiff.ComputationNode<T>>();
            var topoOrder = new List<Autodiff.ComputationNode<T>>();
            var topoStack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
            topoStack.Push((newH, false));

            while (topoStack.Count > 0)
            {
                var (currentNode, processed) = topoStack.Pop();
                if (visited.Contains(currentNode)) continue;

                if (processed)
                {
                    visited.Add(currentNode);
                    topoOrder.Add(currentNode);
                }
                else
                {
                    topoStack.Push((currentNode, true));
                    foreach (var parent in currentNode.Parents)
                    {
                        if (!visited.Contains(parent))
                        {
                            topoStack.Push((parent, false));
                        }
                    }
                }
            }

            for (int idx = topoOrder.Count - 1; idx >= 0; idx--)
            {
                var node = topoOrder[idx];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            // Accumulate gradients - set along time dimension (dim 1)
            if (inputNode.Gradient != null)
            {
                inputGradient.SetSlice(1, t, inputNode.Gradient);
            }

            if (weightsFiNode.Gradient != null) dWeightsFi = dWeightsFi.Add(weightsFiNode.Gradient);
            if (weightsIiNode.Gradient != null) dWeightsIi = dWeightsIi.Add(weightsIiNode.Gradient);
            if (weightsCiNode.Gradient != null) dWeightsCi = dWeightsCi.Add(weightsCiNode.Gradient);
            if (weightsOiNode.Gradient != null) dWeightsOi = dWeightsOi.Add(weightsOiNode.Gradient);
            if (weightsFhNode.Gradient != null) dWeightsFh = dWeightsFh.Add(weightsFhNode.Gradient);
            if (weightsIhNode.Gradient != null) dWeightsIh = dWeightsIh.Add(weightsIhNode.Gradient);
            if (weightsChNode.Gradient != null) dWeightsCh = dWeightsCh.Add(weightsChNode.Gradient);
            if (weightsOhNode.Gradient != null) dWeightsOh = dWeightsOh.Add(weightsOhNode.Gradient);
            if (biasFNode.Gradient != null) dBiasF = dBiasF.Add(biasFNode.Gradient);
            if (biasINode.Gradient != null) dBiasI = dBiasI.Add(biasINode.Gradient);
            if (biasCNode.Gradient != null) dBiasC = dBiasC.Add(biasCNode.Gradient);
            if (biasONode.Gradient != null) dBiasO = dBiasO.Add(biasONode.Gradient);
        }

        // Store gradients for UpdateParameters
        Gradients = new Dictionary<string, Tensor<T>>
        {
            {"weightsFi", dWeightsFi}, {"weightsIi", dWeightsIi}, {"weightsCi", dWeightsCi}, {"weightsOi", dWeightsOi},
            {"weightsFh", dWeightsFh}, {"weightsIh", dWeightsIh}, {"weightsCh", dWeightsCh}, {"weightsOh", dWeightsOh},
            {"biasF", dBiasF}, {"biasI", dBiasI}, {"biasC", dBiasC}, {"biasO", dBiasO}
        };

        // Restore higher-rank gradients to their original shape
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

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
        // concat has shape [batchSize, inputSize + hiddenSize]
        var concat = Tensor<T>.Concatenate(new[] { x, prev_h }, 1);
        var biasF2D = _biasF.Reshape([1, _hiddenSize]);
        var biasI2D = _biasI.Reshape([1, _hiddenSize]);
        var biasC2D = _biasC.Reshape([1, _hiddenSize]);
        var biasO2D = _biasO.Reshape([1, _hiddenSize]);

        // For LSTM gate computation: f = σ(concat @ W^T + b)
        // W_fi has shape [hiddenSize, inputSize], W_fh has shape [hiddenSize, hiddenSize]
        // After transpose: W_fi^T = [inputSize, hiddenSize], W_fh^T = [hiddenSize, hiddenSize]
        // Concatenate along axis 0: [inputSize + hiddenSize, hiddenSize]
        // concat @ W_combined = [batchSize, inputSize + hiddenSize] @ [inputSize + hiddenSize, hiddenSize] = [batchSize, hiddenSize]
        // Gate computations with proper bias broadcasting
        // concat @ W_combined = [batchSize, inputSize + hiddenSize] @ [inputSize + hiddenSize, hiddenSize] = [batchSize, hiddenSize]
        // _biasF is [hiddenSize], need to broadcast across batch dimension
        var f = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            Engine.TensorBroadcastAdd(concat.Multiply(Tensor<T>.Concatenate(new[] { _weightsFi.Transpose(new[] { 1, 0 }), _weightsFh.Transpose(new[] { 1, 0 }) }, 0)), biasF2D));
        var i = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            Engine.TensorBroadcastAdd(concat.Multiply(Tensor<T>.Concatenate(new[] { _weightsIi.Transpose(new[] { 1, 0 }), _weightsIh.Transpose(new[] { 1, 0 }) }, 0)), biasI2D));
        var c_bar = ActivateTensorConditional(_tanhVectorActivation, _tanhActivation,
            Engine.TensorBroadcastAdd(concat.Multiply(Tensor<T>.Concatenate(new[] { _weightsCi.Transpose(new[] { 1, 0 }), _weightsCh.Transpose(new[] { 1, 0 }) }, 0)), biasC2D));
        var o = ActivateTensorConditional(_sigmoidVectorActivation, _sigmoidActivation,
            Engine.TensorBroadcastAdd(concat.Multiply(Tensor<T>.Concatenate(new[] { _weightsOi.Transpose(new[] { 1, 0 }), _weightsOh.Transpose(new[] { 1, 0 }) }, 0)), biasO2D));
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

        // Sum over batch dimension (0), results are 1D tensors [hiddenSize]
        // Concatenate along axis 0 since these are 1D tensors
        var dBiases = Tensor<T>.Concatenate(new[] { di_input.Sum(new[] { 0 }), df_input.Sum(new[] { 0 }), dc_bar_input.Sum(new[] { 0 }), do_input.Sum(new[] { 0 }) }, 0);
        var dbf = dBiases.Slice(0, 0, _hiddenSize);
        var dbi = dBiases.Slice(0, _hiddenSize, _hiddenSize * 2);
        var dbc = dBiases.Slice(0, _hiddenSize * 2, _hiddenSize * 3);
        var dbo = dBiases.Slice(0, _hiddenSize * 3, _hiddenSize * 4);

        // Compute gradient for input (using input-to-hidden weights)
        // Forward: gate = concat @ W^T + b, so backward: dx = d_gate @ W (no transpose needed)
        // _weightsIi has shape [hiddenSize, inputSize], di_input has shape [batch, hiddenSize]
        // dx = [batch, hiddenSize] @ [hiddenSize, inputSize] = [batch, inputSize]
        var dx = di_input.Multiply(_weightsIi)
            .Add(df_input.Multiply(_weightsFi))
            .Add(dc_bar_input.Multiply(_weightsCi))
            .Add(do_input.Multiply(_weightsOi));

        // Compute gradient for previous hidden state (using hidden-to-hidden weights)
        // _weightsIh has shape [hiddenSize, hiddenSize], di_input has shape [batch, hiddenSize]
        // dprev_h = [batch, hiddenSize] @ [hiddenSize, hiddenSize] = [batch, hiddenSize]
        var dprev_h = di_input.Multiply(_weightsIh)
            .Add(df_input.Multiply(_weightsFh))
            .Add(dc_bar_input.Multiply(_weightsCh))
            .Add(do_input.Multiply(_weightsOh));

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
    /// depending on the layer's configuration. Vector activation functions operate on entire tensors at once, while
    /// scalar activation functions operate element by element.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the right type of activation function to a tensor.
    /// 
    /// The LSTM can use two types of activation functions:
    /// - Scalar functions that process one number at a time
    /// - Vector functions that process entire groups at once
    /// 
    /// This method checks which type you're using and applies the appropriate function.
    /// Vector functions can be faster for large datasets but work the same way in principle.
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

    private Dictionary<string, Tensor<T>> _velocities = new Dictionary<string, Tensor<T>>();

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
        bool useGpu = Engine is DirectGpuTensorEngine;
        DirectGpuTensorEngine? gpuEngine = useGpu ? (DirectGpuTensorEngine)Engine : null;
        float lr = useGpu ? (float)NumOps.ToDouble(learningRate) : 0.0f;

        foreach (var kvp in Gradients)
        {
            var paramName = kvp.Key;
            var gradient = kvp.Value;

            if (useGpu)
            {
                // GPU Update
                Tensor<T> param = paramName switch
                {
                    "weightsFi" => _weightsFi,
                    "weightsIi" => _weightsIi,
                    "weightsCi" => _weightsCi,
                    "weightsOi" => _weightsOi,
                    "weightsFh" => _weightsFh,
                    "weightsIh" => _weightsIh,
                    "weightsCh" => _weightsCh,
                    "weightsOh" => _weightsOh,
                    "biasF" => _biasF,
                    "biasI" => _biasI,
                    "biasC" => _biasC,
                    "biasO" => _biasO,
                    _ => throw new InvalidOperationException($"Unknown parameter: {paramName}")
                };

                if (!_velocities.TryGetValue(paramName, out var velocity))
                {
                    velocity = new Tensor<T>(param.Shape);
                    velocity.Fill(NumOps.Zero);
                    gpuEngine!.RegisterPersistentTensor(velocity, PersistentTensorRole.OptimizerState);
                    _velocities[paramName] = velocity;
                }

                gpuEngine!.SgdMomentumUpdateGpu(param, gradient, velocity, lr, 0.0f, 0.0f);
            }
            else
            {
                // CPU Update
                var scaledGradient = Engine.TensorMultiplyScalar(gradient, learningRate);

                switch (paramName)
                {
                    case "weightsFi":
                        _weightsFi = Engine.TensorSubtract(_weightsFi, scaledGradient);
                        break;
                    case "weightsIi":
                        _weightsIi = Engine.TensorSubtract(_weightsIi, scaledGradient);
                        break;
                    case "weightsCi":
                        _weightsCi = Engine.TensorSubtract(_weightsCi, scaledGradient);
                        break;
                    case "weightsOi":
                        _weightsOi = Engine.TensorSubtract(_weightsOi, scaledGradient);
                        break;
                    case "weightsFh":
                        _weightsFh = Engine.TensorSubtract(_weightsFh, scaledGradient);
                        break;
                    case "weightsIh":
                        _weightsIh = Engine.TensorSubtract(_weightsIh, scaledGradient);
                        break;
                    case "weightsCh":
                        _weightsCh = Engine.TensorSubtract(_weightsCh, scaledGradient);
                        break;
                    case "weightsOh":
                        _weightsOh = Engine.TensorSubtract(_weightsOh, scaledGradient);
                        break;
                    case "biasF":
                        _biasF = Engine.TensorSubtract(_biasF, scaledGradient);
                        break;
                    case "biasI":
                        _biasI = Engine.TensorSubtract(_biasI, scaledGradient);
                        break;
                    case "biasC":
                        _biasC = Engine.TensorSubtract(_biasC, scaledGradient);
                        break;
                    case "biasO":
                        _biasO = Engine.TensorSubtract(_biasO, scaledGradient);
                        break;
                }
            }
        }

        if (!useGpu)
        {
            // Notify GPU that tensor data has changed
            Engine.InvalidatePersistentTensor(_weightsFi);
            Engine.InvalidatePersistentTensor(_weightsIi);
            Engine.InvalidatePersistentTensor(_weightsCi);
            Engine.InvalidatePersistentTensor(_weightsOi);
            Engine.InvalidatePersistentTensor(_weightsFh);
            Engine.InvalidatePersistentTensor(_weightsIh);
            Engine.InvalidatePersistentTensor(_weightsCh);
            Engine.InvalidatePersistentTensor(_weightsOh);
            Engine.InvalidatePersistentTensor(_biasF);
            Engine.InvalidatePersistentTensor(_biasI);
            Engine.InvalidatePersistentTensor(_biasC);
            Engine.InvalidatePersistentTensor(_biasO);
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
        // Use Vector.Concatenate for production-grade parameter extraction
        return Vector<T>.Concatenate(
            new Vector<T>(_weightsFi.ToArray()),
            new Vector<T>(_weightsIi.ToArray()),
            new Vector<T>(_weightsCi.ToArray()),
            new Vector<T>(_weightsOi.ToArray()),
            new Vector<T>(_weightsFh.ToArray()),
            new Vector<T>(_weightsIh.ToArray()),
            new Vector<T>(_weightsCh.ToArray()),
            new Vector<T>(_weightsOh.ToArray()),
            new Vector<T>(_biasF.ToArray()),
            new Vector<T>(_biasI.ToArray()),
            new Vector<T>(_biasC.ToArray()),
            new Vector<T>(_biasO.ToArray())
        );
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
        int inputWeightSize = _hiddenSize * _inputSize;
        int hiddenWeightSize = _hiddenSize * _hiddenSize;
        int biasSize = _hiddenSize;

        int totalParams = inputWeightSize * 4 + hiddenWeightSize * 4 + biasSize * 4;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int idx = 0;

        // Use Tensor.FromVector for production-grade parameter setting
        _weightsFi = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), new int[] { _hiddenSize, _inputSize });
        idx += inputWeightSize;

        _weightsIi = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), new int[] { _hiddenSize, _inputSize });
        idx += inputWeightSize;

        _weightsCi = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), new int[] { _hiddenSize, _inputSize });
        idx += inputWeightSize;

        _weightsOi = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), new int[] { _hiddenSize, _inputSize });
        idx += inputWeightSize;

        _weightsFh = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), new int[] { _hiddenSize, _hiddenSize });
        idx += hiddenWeightSize;

        _weightsIh = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), new int[] { _hiddenSize, _hiddenSize });
        idx += hiddenWeightSize;

        _weightsCh = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), new int[] { _hiddenSize, _hiddenSize });
        idx += hiddenWeightSize;

        _weightsOh = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), new int[] { _hiddenSize, _hiddenSize });
        idx += hiddenWeightSize;

        _biasF = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), new int[] { _hiddenSize });
        idx += biasSize;

        _biasI = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), new int[] { _hiddenSize });
        idx += biasSize;

        _biasC = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), new int[] { _hiddenSize });
        idx += biasSize;

        _biasO = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), new int[] { _hiddenSize });

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_weightsFi);
        Engine.InvalidatePersistentTensor(_weightsIi);
        Engine.InvalidatePersistentTensor(_weightsCi);
        Engine.InvalidatePersistentTensor(_weightsOi);
        Engine.InvalidatePersistentTensor(_weightsFh);
        Engine.InvalidatePersistentTensor(_weightsIh);
        Engine.InvalidatePersistentTensor(_weightsCh);
        Engine.InvalidatePersistentTensor(_weightsOh);
        Engine.InvalidatePersistentTensor(_biasF);
        Engine.InvalidatePersistentTensor(_biasI);
        Engine.InvalidatePersistentTensor(_biasC);
        Engine.InvalidatePersistentTensor(_biasO);
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

        // Clear per-time-step cached states to prevent stale state leakage between sequences
        _cachedHiddenStates = null;
        _cachedCellStates = null;

        Gradients.Clear();
    }

    /// <summary>
    /// Exports the LSTM layer's single time-step computation as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the hidden state at one time step.</returns>
    /// <remarks>
    /// <para>
    /// This method exports a single LSTM cell computation for JIT compilation.
    /// The graph computes: h_t, c_t = LSTMCell(x_t, h_{t-1}, c_{t-1})
    /// using the standard LSTM equations with forget, input, output gates and cell candidate.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_weightsFi == null || _weightsIi == null || _weightsCi == null || _weightsOi == null)
            throw new InvalidOperationException("LSTM weights not initialized. Call Initialize() first.");

        if (_weightsFh == null || _weightsIh == null || _weightsCh == null || _weightsOh == null)
            throw new InvalidOperationException("LSTM recurrent weights not initialized. Call Initialize() first.");

        if (_biasF == null || _biasI == null || _biasC == null || _biasO == null)
            throw new InvalidOperationException("LSTM biases not initialized. Call Initialize() first.");

        // Create placeholders for single time-step inputs
        // x_t shape: [batchSize, inputSize]
        var inputPlaceholder = new Tensor<T>(new int[] { 1, _inputSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "x_t");

        // h_{t-1} shape: [batchSize, hiddenSize]
        var prevHiddenPlaceholder = new Tensor<T>(new int[] { 1, _hiddenSize });
        var prevHiddenNode = TensorOperations<T>.Variable(prevHiddenPlaceholder, "h_prev");

        // c_{t-1} shape: [batchSize, hiddenSize]
        var prevCellPlaceholder = new Tensor<T>(new int[] { 1, _hiddenSize });
        var prevCellNode = TensorOperations<T>.Variable(prevCellPlaceholder, "c_prev");

        // Create weight and bias nodes
        var weightsFiNode = TensorOperations<T>.Variable(_weightsFi, "W_fi");
        var weightsIiNode = TensorOperations<T>.Variable(_weightsIi, "W_ii");
        var weightsCiNode = TensorOperations<T>.Variable(_weightsCi, "W_ci");
        var weightsOiNode = TensorOperations<T>.Variable(_weightsOi, "W_oi");

        var weightsFhNode = TensorOperations<T>.Variable(_weightsFh, "W_fh");
        var weightsIhNode = TensorOperations<T>.Variable(_weightsIh, "W_ih");
        var weightsChNode = TensorOperations<T>.Variable(_weightsCh, "W_ch");
        var weightsOhNode = TensorOperations<T>.Variable(_weightsOh, "W_oh");

        var biasFNode = TensorOperations<T>.Variable(_biasF, "b_f");
        var biasINode = TensorOperations<T>.Variable(_biasI, "b_i");
        var biasCNode = TensorOperations<T>.Variable(_biasC, "b_c");
        var biasONode = TensorOperations<T>.Variable(_biasO, "b_o");

        // Add inputs to the list
        inputNodes.Add(inputNode);
        inputNodes.Add(prevHiddenNode);
        inputNodes.Add(prevCellNode);
        inputNodes.Add(weightsFiNode);
        inputNodes.Add(weightsIiNode);
        inputNodes.Add(weightsCiNode);
        inputNodes.Add(weightsOiNode);
        inputNodes.Add(weightsFhNode);
        inputNodes.Add(weightsIhNode);
        inputNodes.Add(weightsChNode);
        inputNodes.Add(weightsOhNode);
        inputNodes.Add(biasFNode);
        inputNodes.Add(biasINode);
        inputNodes.Add(biasCNode);
        inputNodes.Add(biasONode);

        // Build LSTM computation graph (single time step)
        // Forget gate: f_t = sigmoid(W_fi @ x_t + W_fh @ h_{t-1} + b_f)
        var weightsFiT = TensorOperations<T>.Transpose(weightsFiNode);
        var weightsFhT = TensorOperations<T>.Transpose(weightsFhNode);
        var f_input = TensorOperations<T>.MatrixMultiply(inputNode, weightsFiT);
        var f_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, weightsFhT);
        var f_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(f_input, f_hidden), biasFNode);
        var f_t = TensorOperations<T>.Sigmoid(f_preact);

        // Input gate: i_t = sigmoid(W_ii @ x_t + W_ih @ h_{t-1} + b_i)
        var weightsIiT = TensorOperations<T>.Transpose(weightsIiNode);
        var weightsIhT = TensorOperations<T>.Transpose(weightsIhNode);
        var i_input = TensorOperations<T>.MatrixMultiply(inputNode, weightsIiT);
        var i_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, weightsIhT);
        var i_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(i_input, i_hidden), biasINode);
        var i_t = TensorOperations<T>.Sigmoid(i_preact);

        // Cell candidate: c_tilde = tanh(W_ci @ x_t + W_ch @ h_{t-1} + b_c)
        var weightsCiT = TensorOperations<T>.Transpose(weightsCiNode);
        var weightsChT = TensorOperations<T>.Transpose(weightsChNode);
        var c_input = TensorOperations<T>.MatrixMultiply(inputNode, weightsCiT);
        var c_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, weightsChT);
        var c_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(c_input, c_hidden), biasCNode);
        var c_tilde = TensorOperations<T>.Tanh(c_preact);

        // Output gate: o_t = sigmoid(W_oi @ x_t + W_oh @ h_{t-1} + b_o)
        var weightsOiT = TensorOperations<T>.Transpose(weightsOiNode);
        var weightsOhT = TensorOperations<T>.Transpose(weightsOhNode);
        var o_input = TensorOperations<T>.MatrixMultiply(inputNode, weightsOiT);
        var o_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, weightsOhT);
        var o_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(o_input, o_hidden), biasONode);
        var o_t = TensorOperations<T>.Sigmoid(o_preact);

        // Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c_tilde
        var forget_gated = TensorOperations<T>.ElementwiseMultiply(f_t, prevCellNode);
        var input_gated = TensorOperations<T>.ElementwiseMultiply(i_t, c_tilde);
        var c_t = TensorOperations<T>.Add(forget_gated, input_gated);

        // Hidden state: h_t = o_t ⊙ tanh(c_t)
        var c_t_tanh = TensorOperations<T>.Tanh(c_t);
        var h_t = TensorOperations<T>.ElementwiseMultiply(o_t, c_t_tanh);

        return h_t;
    }

    /// <summary>
    /// Gets whether this layer currently supports JIT compilation.
    /// </summary>
    /// <value>
    /// True for LSTM layers, as single time-step JIT compilation is supported.
    /// </value>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Converts a Matrix to a 2D Tensor for use in computation graphs.
    /// </summary>
    private static Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        // Use Matrix.ToArray() and Tensor.FromVector with reshape for production-grade conversion
        var data = matrix.ToArray();
        return Tensor<T>.FromVector(new Vector<T>(data), [matrix.Rows, matrix.Columns]);
    }

    /// <summary>
    /// Converts a Vector to a 1D Tensor for use in computation graphs.
    /// </summary>
    private static Tensor<T> VectorToTensor(Vector<T> vector)
    {
        return Tensor<T>.FromVector(vector);
    }
}
