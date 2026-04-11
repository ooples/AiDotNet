#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Memory;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, HasTrainingMode = true, ChangesShape = true, Cost = ComputeCost.High, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, new[] { 1, 4 }, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class LSTMLayer<T> : LayerBase<T>
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

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
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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
    private Tensor<T>? _gpuWeightsFi;
    private Tensor<T>? _gpuWeightsIi;
    private Tensor<T>? _gpuWeightsCi;
    private Tensor<T>? _gpuWeightsOi;
    private Tensor<T>? _gpuWeightsFh;
    private Tensor<T>? _gpuWeightsIh;
    private Tensor<T>? _gpuWeightsCh;
    private Tensor<T>? _gpuWeightsOh;
    private Tensor<T>? _gpuBiasF;
    private Tensor<T>? _gpuBiasI;
    private Tensor<T>? _gpuBiasC;
    private Tensor<T>? _gpuBiasO;

    // GPU-resident gradient tensors
    private Tensor<T>? _gpuWeightsFiGradient;
    private Tensor<T>? _gpuWeightsIiGradient;
    private Tensor<T>? _gpuWeightsCiGradient;
    private Tensor<T>? _gpuWeightsOiGradient;
    private Tensor<T>? _gpuWeightsFhGradient;
    private Tensor<T>? _gpuWeightsIhGradient;
    private Tensor<T>? _gpuWeightsChGradient;
    private Tensor<T>? _gpuWeightsOhGradient;
    private Tensor<T>? _gpuBiasFGradient;
    private Tensor<T>? _gpuBiasIGradient;
    private Tensor<T>? _gpuBiasCGradient;
    private Tensor<T>? _gpuBiasOGradient;

    // GPU-resident optimizer state tensors (velocity for SGD momentum, M/V for Adam)
    private Tensor<T>? _gpuWeightsFiVelocity;
    private Tensor<T>? _gpuWeightsIiVelocity;
    private Tensor<T>? _gpuWeightsCiVelocity;
    private Tensor<T>? _gpuWeightsOiVelocity;
    private Tensor<T>? _gpuWeightsFhVelocity;
    private Tensor<T>? _gpuWeightsIhVelocity;
    private Tensor<T>? _gpuWeightsChVelocity;
    private Tensor<T>? _gpuWeightsOhVelocity;
    private Tensor<T>? _gpuBiasFVelocity;
    private Tensor<T>? _gpuBiasIVelocity;
    private Tensor<T>? _gpuBiasCVelocity;
    private Tensor<T>? _gpuBiasOVelocity;

    // Adam M/V buffers
    private Tensor<T>? _gpuWeightsFiM;
    private Tensor<T>? _gpuWeightsFiV;
    private Tensor<T>? _gpuWeightsIiM;
    private Tensor<T>? _gpuWeightsIiV;
    private Tensor<T>? _gpuWeightsCiM;
    private Tensor<T>? _gpuWeightsCiV;
    private Tensor<T>? _gpuWeightsOiM;
    private Tensor<T>? _gpuWeightsOiV;
    private Tensor<T>? _gpuWeightsFhM;
    private Tensor<T>? _gpuWeightsFhV;
    private Tensor<T>? _gpuWeightsIhM;
    private Tensor<T>? _gpuWeightsIhV;
    private Tensor<T>? _gpuWeightsChM;
    private Tensor<T>? _gpuWeightsChV;
    private Tensor<T>? _gpuWeightsOhM;
    private Tensor<T>? _gpuWeightsOhV;
    private Tensor<T>? _gpuBiasFM;
    private Tensor<T>? _gpuBiasFV;
    private Tensor<T>? _gpuBiasIM;
    private Tensor<T>? _gpuBiasIV;
    private Tensor<T>? _gpuBiasCM;
    private Tensor<T>? _gpuBiasCV;
    private Tensor<T>? _gpuBiasOM;
    private Tensor<T>? _gpuBiasOV;

    // Cached forward pass state for backpropagation (per timestep arrays)
    private Tensor<T>? _gpuLastInput;
    private Tensor<T>[]? _gpuCachedForgetGates;
    private Tensor<T>[]? _gpuCachedInputGates;
    private Tensor<T>[]? _gpuCachedCellCandidates;
    private Tensor<T>[]? _gpuCachedOutputGates;
    private Tensor<T>[]? _gpuCachedCellStates;
    private Tensor<T>[]? _gpuCachedHiddenStates;
    private Tensor<T>? _gpuInitialHiddenState;
    private Tensor<T>? _gpuInitialCellState;

    // Cached stacked weights for fused kernel (PyTorch format: i, f, g, o)
    private IGpuBuffer? _gpuStackedWeightsIh;
    private IGpuBuffer? _gpuStackedWeightsHh;
    private IGpuBuffer? _gpuStackedBiasIh;
    private IGpuBuffer? _gpuStackedBiasHh;
    private bool _gpuStackedWeightsValid;

    // Fused kernel output cache buffers
    private IGpuBuffer? _gpuFusedAllH;
    private IGpuBuffer? _gpuFusedAllC;
    private IGpuBuffer? _gpuFusedCacheGates;

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
        var halfTensor = new Tensor<T>(weight._shape);
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
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims into batch for rank > 3
        Tensor<T> input3D;
        int batchSize;
        int timeSteps;

        if (rank == 1)
        {
            // 1D input [features]: treat as single timestep, single batch → [1, 1, features]
            batchSize = 1;
            timeSteps = 1;
            input3D = Engine.Reshape(input, [1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // 2D input [timeSteps, features]: single batch
            batchSize = 1;
            timeSteps = input.Shape[0];
            int featureSize = input.Shape[1];
            input3D = Engine.Reshape(input, [1, timeSteps, featureSize]);
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
            input3D = Engine.Reshape(input, [flatBatch, timeSteps, _inputSize]);
        }

        _lastInput = input3D;

        // Use TensorAllocator.Rent for forward pass tensors to reduce GC pressure
        var output = TensorAllocator.Rent<T>(new int[] { batchSize, timeSteps, _hiddenSize });

        _cachedHiddenStates = TensorAllocator.Rent<T>(new int[] { batchSize, timeSteps, _hiddenSize });
        _cachedCellStates = TensorAllocator.Rent<T>(new int[] { batchSize, timeSteps, _hiddenSize });

        var currentH = TensorAllocator.Rent<T>(new int[] { batchSize, _hiddenSize });
        var currentC = TensorAllocator.Rent<T>(new int[] { batchSize, _hiddenSize });

        // Pre-transpose weights for efficiency
        var WfiT = Engine.TensorTranspose(_weightsFi);
        var WiiT = Engine.TensorTranspose(_weightsIi);
        var WciT = Engine.TensorTranspose(_weightsCi);
        var WoiT = Engine.TensorTranspose(_weightsOi);
        var WfhT = Engine.TensorTranspose(_weightsFh);
        var WihT = Engine.TensorTranspose(_weightsIh);
        var WchT = Engine.TensorTranspose(_weightsCh);
        var WohT = Engine.TensorTranspose(_weightsOh);
        var biasF2D = Engine.Reshape(_biasF, [1, _hiddenSize]);
        var biasI2D = Engine.Reshape(_biasI, [1, _hiddenSize]);
        var biasC2D = Engine.Reshape(_biasC, [1, _hiddenSize]);
        var biasO2D = Engine.Reshape(_biasO, [1, _hiddenSize]);

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
            output = Engine.Reshape(output, newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // 2D input -> 2D output (remove added batch dim)
            output = Engine.Reshape(output, [timeSteps, _hiddenSize]);
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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
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
            originalShape = input._shape;
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
            originalShape = input._shape;
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

        // Prepare stacked weights in PyTorch format (i, f, g, o order)
        PrepareStackedWeightsForGpu(backend);

        // Buffer size calculations
        int hiddenBufferSize = batchSize * _hiddenSize;
        int outputSize = batchSize * timeSteps * _hiddenSize;
        int allHCSize = (timeSteps + 1) * batchSize * _hiddenSize;
        int cacheGatesSize = timeSteps * batchSize * _hiddenSize * 4;

        // Allocate initial hidden and cell state buffers (zeros)
        var hInitBuffer = backend.AllocateBuffer(hiddenBufferSize);
        var cInitBuffer = backend.AllocateBuffer(hiddenBufferSize);
        backend.Fill(hInitBuffer, 0.0f, hiddenBufferSize);
        backend.Fill(cInitBuffer, 0.0f, hiddenBufferSize);

        // Allocate output buffers
        var outputBuffer = backend.AllocateBuffer(outputSize);
        var hFinalBuffer = backend.AllocateBuffer(hiddenBufferSize);
        var cFinalBuffer = backend.AllocateBuffer(hiddenBufferSize);

        // Allocate cache buffers for backward pass (always needed for training)
        bool cacheForTraining = IsTrainingMode;
        IGpuBuffer? allHBuffer = null;
        IGpuBuffer? allCBuffer = null;
        IGpuBuffer? cacheGatesBuffer = null;

        if (cacheForTraining)
        {
            ClearGpuTrainingCache();
            _gpuLastInput = input;

            allHBuffer = backend.AllocateBuffer(allHCSize);
            allCBuffer = backend.AllocateBuffer(allHCSize);
            cacheGatesBuffer = backend.AllocateBuffer(cacheGatesSize);

            // Store fused cache buffers for backward pass
            _gpuFusedAllH = allHBuffer;
            _gpuFusedAllC = allCBuffer;
            _gpuFusedCacheGates = cacheGatesBuffer;

            // Cache initial states
            var initHBufferCopy = backend.AllocateBuffer(hiddenBufferSize);
            var initCBufferCopy = backend.AllocateBuffer(hiddenBufferSize);
            backend.Copy(hInitBuffer, initHBufferCopy, hiddenBufferSize);
            backend.Copy(cInitBuffer, initCBufferCopy, hiddenBufferSize);
            _gpuInitialHiddenState = GpuTensorHelper.UploadToGpu<T>(backend, initHBufferCopy, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
            _gpuInitialCellState = GpuTensorHelper.UploadToGpu<T>(backend, initCBufferCopy, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
        }
        else
        {
            // Allocate temporary cache buffers (will be disposed after forward)
            allHBuffer = backend.AllocateBuffer(allHCSize);
            allCBuffer = backend.AllocateBuffer(allHCSize);
            cacheGatesBuffer = backend.AllocateBuffer(cacheGatesSize);
        }

        try
        {
            // Call fused LSTM sequence kernel (processes all timesteps in one kernel launch)
            backend.LstmForwardSequence(
                input.Buffer, hInitBuffer, cInitBuffer,
                _gpuStackedWeightsIh!, _gpuStackedWeightsHh!, _gpuStackedBiasIh!, _gpuStackedBiasHh!,
                outputBuffer, hFinalBuffer, cFinalBuffer,
                allHBuffer, allCBuffer, cacheGatesBuffer,
                timeSteps, batchSize, _inputSize, _hiddenSize);

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
            hInitBuffer.Dispose();
            cInitBuffer.Dispose();
            hFinalBuffer.Dispose();
            cFinalBuffer.Dispose();

            // If not training, dispose cache buffers
            if (!cacheForTraining)
            {
                allHBuffer.Dispose();
                allCBuffer.Dispose();
                cacheGatesBuffer.Dispose();
            }

            return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            // Cleanup buffers on error
            hInitBuffer.Dispose();
            cInitBuffer.Dispose();
            hFinalBuffer.Dispose();
            cFinalBuffer.Dispose();
            outputBuffer.Dispose();

            if (!cacheForTraining)
            {
                allHBuffer?.Dispose();
                allCBuffer?.Dispose();
                cacheGatesBuffer?.Dispose();
            }
            else
            {
                ClearGpuTrainingCache();
            }

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

        // Note: Do NOT dispose stacked weights here - they're reusable across forward passes
        // and are only invalidated when actual weights are updated via InvalidateGpuStackedWeights()

        _gpuFusedAllH?.Dispose();
        _gpuFusedAllH = null;
        _gpuFusedAllC?.Dispose();
        _gpuFusedAllC = null;
        _gpuFusedCacheGates?.Dispose();
        _gpuFusedCacheGates = null;
    }

    /// <summary>
    /// Prepares stacked weights in PyTorch format (i, f, g, o order) for the fused LSTM kernel.
    /// </summary>
    private void PrepareStackedWeightsForGpu(IDirectGpuBackend backend)
    {
        if (_gpuStackedWeightsValid && _gpuStackedWeightsIh != null)
            return;

        // Stack input-to-hidden weights: [4*hiddenSize, inputSize] in order i, f, g, o
        // Layer order: I (input), F (forget), C (cell/gate), O (output)
        // Kernel order: i, f, g, o -> I, F, C, O
        int weightsIhSize = 4 * _hiddenSize * _inputSize;
        int weightsHhSize = 4 * _hiddenSize * _hiddenSize;
        int biasSize = 4 * _hiddenSize;

        var stackedIh = new float[weightsIhSize];
        var stackedHh = new float[weightsHhSize];
        var stackedBiasIh = new float[biasSize];
        var stackedBiasHh = new float[biasSize]; // PyTorch uses separate ih/hh biases

        // Convert weights to float arrays
        var wIi = DirectGpuEngine.ToFloatArray<T>(_weightsIi.ToArray());
        var wFi = DirectGpuEngine.ToFloatArray<T>(_weightsFi.ToArray());
        var wCi = DirectGpuEngine.ToFloatArray<T>(_weightsCi.ToArray());
        var wOi = DirectGpuEngine.ToFloatArray<T>(_weightsOi.ToArray());

        var wIh = DirectGpuEngine.ToFloatArray<T>(_weightsIh.ToArray());
        var wFh = DirectGpuEngine.ToFloatArray<T>(_weightsFh.ToArray());
        var wCh = DirectGpuEngine.ToFloatArray<T>(_weightsCh.ToArray());
        var wOh = DirectGpuEngine.ToFloatArray<T>(_weightsOh.ToArray());

        var bI = DirectGpuEngine.ToFloatArray<T>(_biasI.ToArray());
        var bF = DirectGpuEngine.ToFloatArray<T>(_biasF.ToArray());
        var bC = DirectGpuEngine.ToFloatArray<T>(_biasC.ToArray());
        var bO = DirectGpuEngine.ToFloatArray<T>(_biasO.ToArray());

        // Stack weights in kernel order (i, f, g, o)
        int inputGateOffset = 0;
        int forgetGateOffset = _hiddenSize * _inputSize;
        int cellGateOffset = 2 * _hiddenSize * _inputSize;
        int outputGateOffset = 3 * _hiddenSize * _inputSize;

        Array.Copy(wIi, 0, stackedIh, inputGateOffset, _hiddenSize * _inputSize);
        Array.Copy(wFi, 0, stackedIh, forgetGateOffset, _hiddenSize * _inputSize);
        Array.Copy(wCi, 0, stackedIh, cellGateOffset, _hiddenSize * _inputSize);
        Array.Copy(wOi, 0, stackedIh, outputGateOffset, _hiddenSize * _inputSize);

        int hhInputGateOffset = 0;
        int hhForgetGateOffset = _hiddenSize * _hiddenSize;
        int hhCellGateOffset = 2 * _hiddenSize * _hiddenSize;
        int hhOutputGateOffset = 3 * _hiddenSize * _hiddenSize;

        Array.Copy(wIh, 0, stackedHh, hhInputGateOffset, _hiddenSize * _hiddenSize);
        Array.Copy(wFh, 0, stackedHh, hhForgetGateOffset, _hiddenSize * _hiddenSize);
        Array.Copy(wCh, 0, stackedHh, hhCellGateOffset, _hiddenSize * _hiddenSize);
        Array.Copy(wOh, 0, stackedHh, hhOutputGateOffset, _hiddenSize * _hiddenSize);

        // Stack biases (put all in biasIh, biasHh will be zeros - matches PyTorch default)
        int biasInputGateOffset = 0;
        int biasForgetGateOffset = _hiddenSize;
        int biasCellGateOffset = 2 * _hiddenSize;
        int biasOutputGateOffset = 3 * _hiddenSize;

        Array.Copy(bI, 0, stackedBiasIh, biasInputGateOffset, _hiddenSize);
        Array.Copy(bF, 0, stackedBiasIh, biasForgetGateOffset, _hiddenSize);
        Array.Copy(bC, 0, stackedBiasIh, biasCellGateOffset, _hiddenSize);
        Array.Copy(bO, 0, stackedBiasIh, biasOutputGateOffset, _hiddenSize);
        // stackedBiasHh stays zeros

        // Allocate GPU buffers
        _gpuStackedWeightsIh?.Dispose();
        _gpuStackedWeightsHh?.Dispose();
        _gpuStackedBiasIh?.Dispose();
        _gpuStackedBiasHh?.Dispose();

        _gpuStackedWeightsIh = backend.AllocateBuffer(stackedIh);
        _gpuStackedWeightsHh = backend.AllocateBuffer(stackedHh);
        _gpuStackedBiasIh = backend.AllocateBuffer(stackedBiasIh);
        _gpuStackedBiasHh = backend.AllocateBuffer(stackedBiasHh);
        _gpuStackedWeightsValid = true;
    }

    /// <summary>
    /// Invalidates and disposes the stacked GPU weight buffers.
    /// Call this after any weight modification (CPU or GPU side) to ensure
    /// PrepareStackedWeightsForGpu will rebuild the stacked buffers.
    /// </summary>
    private void InvalidateGpuStackedWeights()
    {
        _gpuStackedWeightsIh?.Dispose();
        _gpuStackedWeightsIh = null;
        _gpuStackedWeightsHh?.Dispose();
        _gpuStackedWeightsHh = null;
        _gpuStackedBiasIh?.Dispose();
        _gpuStackedBiasIh = null;
        _gpuStackedBiasHh?.Dispose();
        _gpuStackedBiasHh = null;
        _gpuStackedWeightsValid = false;
    }

    /// <summary>
    /// Extracts per-gate gradients from stacked gradient buffers after fused backward kernel.
    /// </summary>
    private void UnstackGradients(
        IDirectGpuBackend backend,
        IGpuBuffer stackedGradIh, IGpuBuffer stackedGradHh, IGpuBuffer stackedGradBias,
        out float[] dWIi, out float[] dWFi, out float[] dWCi, out float[] dWOi,
        out float[] dWIh, out float[] dWFh, out float[] dWCh, out float[] dWOh,
        out float[] dBI, out float[] dBF, out float[] dBC, out float[] dBO)
    {
        int weightsIhSize = 4 * _hiddenSize * _inputSize;
        int weightsHhSize = 4 * _hiddenSize * _hiddenSize;
        int biasSize = 4 * _hiddenSize;

        var stackedIh = new float[weightsIhSize];
        var stackedHh = new float[weightsHhSize];
        var stackedBias = new float[biasSize];

        backend.DownloadBuffer(stackedGradIh, stackedIh);
        backend.DownloadBuffer(stackedGradHh, stackedHh);
        backend.DownloadBuffer(stackedGradBias, stackedBias);

        // Extract per-gate gradients from stacked format
        dWIi = new float[_hiddenSize * _inputSize];
        dWFi = new float[_hiddenSize * _inputSize];
        dWCi = new float[_hiddenSize * _inputSize];
        dWOi = new float[_hiddenSize * _inputSize];

        Array.Copy(stackedIh, 0, dWIi, 0, _hiddenSize * _inputSize);
        Array.Copy(stackedIh, _hiddenSize * _inputSize, dWFi, 0, _hiddenSize * _inputSize);
        Array.Copy(stackedIh, 2 * _hiddenSize * _inputSize, dWCi, 0, _hiddenSize * _inputSize);
        Array.Copy(stackedIh, 3 * _hiddenSize * _inputSize, dWOi, 0, _hiddenSize * _inputSize);

        dWIh = new float[_hiddenSize * _hiddenSize];
        dWFh = new float[_hiddenSize * _hiddenSize];
        dWCh = new float[_hiddenSize * _hiddenSize];
        dWOh = new float[_hiddenSize * _hiddenSize];

        Array.Copy(stackedHh, 0, dWIh, 0, _hiddenSize * _hiddenSize);
        Array.Copy(stackedHh, _hiddenSize * _hiddenSize, dWFh, 0, _hiddenSize * _hiddenSize);
        Array.Copy(stackedHh, 2 * _hiddenSize * _hiddenSize, dWCh, 0, _hiddenSize * _hiddenSize);
        Array.Copy(stackedHh, 3 * _hiddenSize * _hiddenSize, dWOh, 0, _hiddenSize * _hiddenSize);

        dBI = new float[_hiddenSize];
        dBF = new float[_hiddenSize];
        dBC = new float[_hiddenSize];
        dBO = new float[_hiddenSize];

        Array.Copy(stackedBias, 0, dBI, 0, _hiddenSize);
        Array.Copy(stackedBias, _hiddenSize, dBF, 0, _hiddenSize);
        Array.Copy(stackedBias, 2 * _hiddenSize, dBC, 0, _hiddenSize);
        Array.Copy(stackedBias, 3 * _hiddenSize, dBO, 0, _hiddenSize);
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

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuWeightsFiGradient == null || _gpuWeightsIiGradient == null ||
            _gpuWeightsCiGradient == null || _gpuWeightsOiGradient == null ||
            _gpuWeightsFhGradient == null || _gpuWeightsIhGradient == null ||
            _gpuWeightsChGradient == null || _gpuWeightsOhGradient == null ||
            _gpuBiasFGradient == null || _gpuBiasIGradient == null ||
            _gpuBiasCGradient == null || _gpuBiasOGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuWeightsFi ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsFi, GpuTensorRole.Weight);
        _gpuWeightsIi ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsIi, GpuTensorRole.Weight);
        _gpuWeightsCi ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsCi, GpuTensorRole.Weight);
        _gpuWeightsOi ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsOi, GpuTensorRole.Weight);
        _gpuWeightsFh ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsFh, GpuTensorRole.Weight);
        _gpuWeightsIh ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsIh, GpuTensorRole.Weight);
        _gpuWeightsCh ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsCh, GpuTensorRole.Weight);
        _gpuWeightsOh ??= GpuTensorHelper.UploadToGpu<T>(backend, _weightsOh, GpuTensorRole.Weight);
        _gpuBiasF ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasF, GpuTensorRole.Bias);
        _gpuBiasI ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasI, GpuTensorRole.Bias);
        _gpuBiasC ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasC, GpuTensorRole.Bias);
        _gpuBiasO ??= GpuTensorHelper.UploadToGpu<T>(backend, _biasO, GpuTensorRole.Bias);

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
        _weightsFi = _gpuWeightsFi;
        _weightsIi = _gpuWeightsIi;
        _weightsCi = _gpuWeightsCi;
        _weightsOi = _gpuWeightsOi;
        _weightsFh = _gpuWeightsFh;
        _weightsIh = _gpuWeightsIh;
        _weightsCh = _gpuWeightsCh;
        _weightsOh = _gpuWeightsOh;
        _biasF = _gpuBiasF;
        _biasI = _gpuBiasI;
        _biasC = _gpuBiasC;
        _biasO = _gpuBiasO;

        // Invalidate stacked weight buffers since individual weights have been modified
        InvalidateGpuStackedWeights();
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
                _gpuWeightsFiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightsFiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
                // SquaredAvg buffers for RMSprop
                _gpuWeightsFiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adagrad:
                // AccumulatedGrad buffers for Adagrad
                _gpuWeightsFiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsCiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOiVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsFhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsIhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsChVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsOhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasFVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasIVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasCVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasOVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
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
                    velocity = new Tensor<T>(param._shape);
                    velocity.Fill(NumOps.Zero);
                    var gpu1 = gpuEngine ?? throw new InvalidOperationException("GPU engine is not available.");
                    gpu1.RegisterPersistentTensor(velocity, PersistentTensorRole.OptimizerState);
                    _velocities[paramName] = velocity;
                }

                var gpu2 = gpuEngine ?? throw new InvalidOperationException("GPU engine is not available.");
                gpu2.SgdMomentumUpdateGpu(param, gradient, velocity, lr, 0.0f, 0.0f);
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

        // Invalidate stacked weight buffers since weights have been modified
        InvalidateGpuStackedWeights();
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

        // Invalidate stacked weight buffers since weights have been replaced from deserialization
        InvalidateGpuStackedWeights();
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
        // Bulk copy from contiguous tensor storage — avoids ToArray() double-copy
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_weightsFi.Data),
            Vector<T>.FromMemory(_weightsIi.Data),
            Vector<T>.FromMemory(_weightsCi.Data),
            Vector<T>.FromMemory(_weightsOi.Data),
            Vector<T>.FromMemory(_weightsFh.Data),
            Vector<T>.FromMemory(_weightsIh.Data),
            Vector<T>.FromMemory(_weightsCh.Data),
            Vector<T>.FromMemory(_weightsOh.Data),
            Vector<T>.FromMemory(_biasF.Data),
            Vector<T>.FromMemory(_biasI.Data),
            Vector<T>.FromMemory(_biasC.Data),
            Vector<T>.FromMemory(_biasO.Data)
        );
    }

    public override Vector<T> GetParameterGradients()
    {
        if (Gradients == null || Gradients.Count == 0)
            return new Vector<T>(ParameterCount);

        Tensor<T> Get(string key) => Gradients.TryGetValue(key, out var t) ? t : new Tensor<T>([0]);

        // Bulk copy from contiguous tensor storage — avoids ToArray() double-copy
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(Get("weightsFi").Data),
            Vector<T>.FromMemory(Get("weightsIi").Data),
            Vector<T>.FromMemory(Get("weightsCi").Data),
            Vector<T>.FromMemory(Get("weightsOi").Data),
            Vector<T>.FromMemory(Get("weightsFh").Data),
            Vector<T>.FromMemory(Get("weightsIh").Data),
            Vector<T>.FromMemory(Get("weightsCh").Data),
            Vector<T>.FromMemory(Get("weightsOh").Data),
            Vector<T>.FromMemory(Get("biasF").Data),
            Vector<T>.FromMemory(Get("biasI").Data),
            Vector<T>.FromMemory(Get("biasC").Data),
            Vector<T>.FromMemory(Get("biasO").Data)
        );
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        Gradients?.Clear();
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
