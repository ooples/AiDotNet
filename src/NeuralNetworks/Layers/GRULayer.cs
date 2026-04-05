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
/// Represents a Gated Recurrent Unit (GRU) layer for processing sequential data.
/// </summary>
/// <remarks>
/// <para>
/// The GRU (Gated Recurrent Unit) layer is a type of recurrent neural network layer that is designed to
/// capture dependencies over time in sequential data. It addresses the vanishing gradient problem that
/// standard recurrent neural networks face when dealing with long sequences. The GRU uses update and reset
/// gates to control the flow of information, allowing the network to retain relevant information over
/// many time steps while forgetting irrelevant details.
/// </para>
/// <para><b>For Beginners:</b> This layer helps neural networks understand sequences of data, like sentences or time series.
/// 
/// Think of the GRU as having a "memory" that helps it understand context:
/// - When reading a sentence, it remembers important words from earlier
/// - When analyzing stock prices, it remembers relevant trends from previous days
/// - It uses special "gates" to decide what information to keep or forget
/// 
/// For example, in the sentence "The clouds were dark and it started to ___", 
/// the GRU would recognize the context and predict "rain" because it remembers
/// the earlier words about dark clouds.
/// 
/// GRUs are simpler versions of LSTMs (Long Short-Term Memory) but often perform similarly well
/// while being more efficient to train.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, HasTrainingMode = true, ChangesShape = true, Cost = ComputeCost.High, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, false, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class GRULayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensors for the update gate (z), reset gate (r), and candidate hidden state (h).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These weight tensors transform the input at each time step:
    /// - _Wz: Weights for the update gate that determines how much of the previous hidden state to keep
    /// - _Wr: Weights for the reset gate that determines how much of the previous hidden state to reset
    /// - _Wh: Weights for the candidate hidden state that contains new information
    /// </para>
    /// <para><b>For Beginners:</b> These weights transform your input data into useful information.
    ///
    /// Think of these weight tensors as "filters" that extract different types of information:
    /// - _Wz helps decide how much old information to keep
    /// - _Wr helps decide how much old information to reset or forget
    /// - _Wh helps create new information based on the current input
    ///
    /// During training, these weights are adjusted to better recognize important patterns in your data.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wz, _Wr, _Wh;

    /// <summary>
    /// The weight tensors that transform the previous hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These weight tensors process the previous hidden state:
    /// - _Uz: Weights that transform the previous hidden state for the update gate
    /// - _Ur: Weights that transform the previous hidden state for the reset gate
    /// - _Uh: Weights that transform the previous hidden state for the candidate hidden state
    /// </para>
    /// <para><b>For Beginners:</b> These weights process the "memory" from previous time steps.
    ///
    /// These tensors help the GRU work with information from earlier in the sequence:
    /// - _Uz helps decide which parts of the memory to keep
    /// - _Ur helps decide which parts of the memory to reset
    /// - _Uh helps combine memory with new information
    ///
    /// For example, when reading text, these weights help the network remember important context
    /// from words that appeared earlier in the sentence.
    /// </para>
    /// </remarks>
    private Tensor<T> _Uz, _Ur, _Uh;

    /// <summary>
    /// The bias tensors for the update gate (z), reset gate (r), and candidate hidden state (h).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These bias tensors provide an offset to the transformations:
    /// - _bz: Bias for the update gate
    /// - _br: Bias for the reset gate
    /// - _bh: Bias for the candidate hidden state
    /// </para>
    /// <para><b>For Beginners:</b> These biases are like "default settings" for each gate.
    ///
    /// Biases help the network by:
    /// - Providing a starting point for each gate's operation
    /// - Allowing outputs to be non-zero even when inputs are zero
    /// - Giving the model flexibility to fit data better
    ///
    /// They're like the "baseline" settings that get adjusted during training.
    /// </para>
    /// </remarks>
    private Tensor<T> _bz, _br, _bh;

    /// <summary>
    /// Gradients for the weight matrices during backpropagation.
    /// </summary>
    private Tensor<T>? _dWz, _dWr, _dWh, _dUz, _dUr, _dUh;

    /// <summary>
    /// Gradients for the bias vectors during backpropagation.
    /// </summary>
    private Tensor<T>? _dbz, _dbr, _dbh;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The final hidden state from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastHiddenState;

    /// <summary>
    /// The activation values for the update gate (z), reset gate (r), and candidate hidden state (h)
    /// from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastZ, _lastR, _lastH;

    /// <summary>
    /// All hidden states from the last forward pass, used when returning sequences.
    /// </summary>
    private List<Tensor<T>>? _allHiddenStates;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// The size of the input feature vector at each time step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the dimensionality of the input feature vector at each time step in the sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of features in each element of your sequence.
    /// 
    /// For example:
    /// - In text processing, this might be the size of word embeddings (e.g., 300 dimensions)
    /// - In time series, this would be the number of measurements at each time point
    /// - In audio processing, this could be the number of frequency bands
    /// </para>
    /// </remarks>
    private readonly int _inputSize;

    /// <summary>
    /// The size of the hidden state vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines the dimensionality of the GRU's hidden state and, consequently, the capacity
    /// of the layer to capture and represent patterns in the sequence data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the GRU's "memory".
    /// 
    /// The hidden size determines:
    /// - How much information the GRU can remember about the sequence
    /// - The complexity of patterns it can recognize
    /// - The capacity of the model (larger = more capacity but more parameters to train)
    /// 
    /// A larger hidden size allows the layer to capture more complex patterns but requires
    /// more data and time to train effectively.
    /// </para>
    /// </remarks>
    private readonly int _hiddenSize;

    // Cached ones tensor for (1-z) computation — avoids per-timestep allocation
    private Tensor<T>? _cachedOnesForGate;

    /// <summary>
    /// Determines whether the layer returns the full sequence of hidden states or just the final state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If true, the forward pass returns all hidden states for all time steps. If false, only the final
    /// hidden state is returned.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls what information the layer provides as output.
    /// 
    /// Two options:
    /// - true: Returns information about every element in the sequence (useful for sequence-to-sequence tasks)
    /// - false: Returns only the final "summary" after processing the entire sequence
    /// 
    /// For example, in sentiment analysis, you might only need the final state (false)
    /// to classify the sentiment of a whole text, but for translation, you'd want
    /// information about each position (true).
    /// </para>
    /// </remarks>
    private readonly bool _returnSequences;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// The activation function applied to the candidate hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the candidate hidden state. Typically, a tanh activation function
    /// is used to maintain values between -1 and 1, which helps with the stability of the recurrent network.
    /// </para>
    /// <para><b>For Beginners:</b> This function shapes the new information the GRU creates.
    /// 
    /// The activation function:
    /// - Adds non-linearity, helping the network learn complex patterns
    /// - Usually tanh (hyperbolic tangent), which outputs values between -1 and 1
    /// - Helps keep the values in a manageable range to prevent explosion
    /// 
    /// Think of it as a way to normalize and shape the information before it's combined with memory.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _activation;

    /// <summary>
    /// The activation function applied to the update and reset gates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the update and reset gates. Typically, a sigmoid activation function
    /// is used, which outputs values between 0 and 1, allowing the gates to control the flow of information.
    /// </para>
    /// <para><b>For Beginners:</b> This function controls how the gates operate.
    /// 
    /// The recurrent activation:
    /// - Usually sigmoid, which outputs values between 0 and 1
    /// - Creates "gate" values that act like percentages or dials
    /// - Value of 0 means "block completely" while 1 means "let everything through"
    /// 
    /// This helps the GRU decide how much information to remember versus forget at each step.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _recurrentActivation;

    /// <summary>
    /// The vector activation function applied to the candidate hidden state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector activation function is applied to the entire candidate hidden state vector at once rather
    /// than element-wise. This can capture dependencies between different elements of the hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> A more advanced way to shape new information.
    /// 
    /// A vector activation:
    /// - Works on the entire vector at once, not just individual values
    /// - Can capture relationships between different features
    /// - Potentially more powerful than element-wise activations
    /// 
    /// This is useful for capturing complex patterns that depend on multiple features interacting together.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _vectorActivation;

    /// <summary>
    /// The vector activation function applied to the update and reset gates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector activation function is applied to the entire update and reset gate vectors at once rather
    /// than element-wise. This can capture dependencies between different elements of the gates.
    /// </para>
    /// <para><b>For Beginners:</b> A more advanced way to control the gates.
    /// 
    /// A vector recurrent activation:
    /// - Works on entire gate vectors at once
    /// - Can make decisions based on all features together
    /// - Potentially more powerful than individual feature activations
    /// 
    /// This allows for more sophisticated memory management, where decisions about
    /// what to remember depend on the relationships between multiple features.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _vectorRecurrentActivation;

    #region GPU Training Fields

    // GPU-resident weight tensors
    private Tensor<T>? _gpuWz;
    private Tensor<T>? _gpuWr;
    private Tensor<T>? _gpuWh;
    private Tensor<T>? _gpuUz;
    private Tensor<T>? _gpuUr;
    private Tensor<T>? _gpuUh;
    private Tensor<T>? _gpuBz;
    private Tensor<T>? _gpuBr;
    private Tensor<T>? _gpuBh;

    // GPU-resident gradient tensors
    private Tensor<T>? _gpuWzGradient;
    private Tensor<T>? _gpuWrGradient;
    private Tensor<T>? _gpuWhGradient;
    private Tensor<T>? _gpuUzGradient;
    private Tensor<T>? _gpuUrGradient;
    private Tensor<T>? _gpuUhGradient;
    private Tensor<T>? _gpuBzGradient;
    private Tensor<T>? _gpuBrGradient;
    private Tensor<T>? _gpuBhGradient;

    // GPU-resident optimizer state tensors (SGD/NAG/LARS velocity)
    private Tensor<T>? _gpuWzVelocity;
    private Tensor<T>? _gpuWrVelocity;
    private Tensor<T>? _gpuWhVelocity;
    private Tensor<T>? _gpuUzVelocity;
    private Tensor<T>? _gpuUrVelocity;
    private Tensor<T>? _gpuUhVelocity;
    private Tensor<T>? _gpuBzVelocity;
    private Tensor<T>? _gpuBrVelocity;
    private Tensor<T>? _gpuBhVelocity;

    // Adam/AdamW M (first moment) tensors
    private Tensor<T>? _gpuWzM;
    private Tensor<T>? _gpuWrM;
    private Tensor<T>? _gpuWhM;
    private Tensor<T>? _gpuUzM;
    private Tensor<T>? _gpuUrM;
    private Tensor<T>? _gpuUhM;
    private Tensor<T>? _gpuBzM;
    private Tensor<T>? _gpuBrM;
    private Tensor<T>? _gpuBhM;

    // Adam/AdamW V (second moment) tensors
    private Tensor<T>? _gpuWzV;
    private Tensor<T>? _gpuWrV;
    private Tensor<T>? _gpuWhV;
    private Tensor<T>? _gpuUzV;
    private Tensor<T>? _gpuUrV;
    private Tensor<T>? _gpuUhV;
    private Tensor<T>? _gpuBzV;
    private Tensor<T>? _gpuBrV;
    private Tensor<T>? _gpuBhV;

    // Cached forward pass state for backpropagation (BPTT)
    private Tensor<T>? _gpuLastInput;
    private Tensor<T>[]? _gpuCachedZGates;
    private Tensor<T>[]? _gpuCachedRGates;
    private Tensor<T>[]? _gpuCachedHCandidates;
    private Tensor<T>[]? _gpuCachedHiddenStates;
    private Tensor<T>? _gpuInitialHiddenState;

    // Cached stacked weights for fused kernel (PyTorch format: r, z, n)
    private IGpuBuffer? _gpuStackedWeightsIh;
    private IGpuBuffer? _gpuStackedWeightsHh;
    private IGpuBuffer? _gpuStackedBiasIh;
    private IGpuBuffer? _gpuStackedBiasHh;
    private bool _gpuStackedWeightsValid;

    // Fused kernel output cache buffers for backward pass
    private IGpuBuffer? _gpuFusedAllH;
    private IGpuBuffer? _gpuFusedCacheGates;

    #endregion

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>
    /// The total number of weight and bias parameters in the GRU layer.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property calculates the total number of trainable parameters in the GRU layer, which includes
    /// all the weights and biases for the gates and candidate hidden state.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers the layer needs to learn.
    /// 
    /// The formula counts:
    /// - Weights connecting inputs to the GRU (Wz, Wr, Wh)
    /// - Weights connecting the previous hidden state (Uz, Ur, Uh)
    /// - Bias values for each gate and candidate state (bz, br, bh)
    /// 
    /// A higher parameter count means the model can capture more complex patterns
    /// but requires more data and time to train effectively.
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        _hiddenSize * _inputSize * 3 +  // Wz, Wr, Wh
        _hiddenSize * _hiddenSize * 3 + // Uz, Ur, Uh
        _hiddenSize * 3;                // bz, br, bh

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The GRULayer always returns true because it contains trainable weights and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The GRU layer always supports training because it has weights and biases that can be updated.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="GRULayer{T}"/> class with the specified dimensions, return behavior, and element-wise activation functions.
    /// </summary>
    /// <param name="inputSize">The size of the input feature vector at each time step.</param>
    /// <param name="hiddenSize">The size of the hidden state vector.</param>
    /// <param name="returnSequences">If <c>true</c>, returns all hidden states; if <c>false</c>, returns only the final hidden state.</param>
    /// <param name="activation">The activation function for the candidate hidden state. Defaults to tanh if not specified.</param>
    /// <param name="recurrentActivation">The activation function for the gates. Defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GRU layer with the specified dimensions and element-wise activation functions.
    /// The weights are initialized randomly with a scale factor based on the hidden size, and the biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU layer with standard activation functions.
    /// 
    /// When creating a GRU layer, you specify:
    /// - inputSize: How many features each element in your sequence has
    /// - hiddenSize: How large the GRU's "memory" should be
    /// - returnSequences: Whether you want information about every element or just a final summary
    /// - activation: How to shape new information (default is tanh, outputting values between -1 and 1)
    /// - recurrentActivation: How the gates should work (default is sigmoid, outputting values between 0 and 1)
    /// 
    /// For example, if processing sentences where each word is represented by a 100-dimensional vector,
    /// and you want a 200-dimensional memory, you would use inputSize=100 and hiddenSize=200.
    /// </para>
    /// </remarks>
    public GRULayer(int inputSize, int hiddenSize,
                    bool returnSequences = false,
                    IActivationFunction<T>? activation = null,
                    IActivationFunction<T>? recurrentActivation = null)
        : base([inputSize], [hiddenSize], activation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _returnSequences = returnSequences;
        _activation = activation ?? new TanhActivation<T>();
        _recurrentActivation = recurrentActivation ?? new SigmoidActivation<T>();

        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(1.0, _hiddenSize)));

        _Wz = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wr = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wh = InitializeTensor(_hiddenSize, _inputSize, scale);

        _Uz = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Ur = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Uh = InitializeTensor(_hiddenSize, _hiddenSize, scale);

        _bz = new Tensor<T>([_hiddenSize]);
        _br = new Tensor<T>([_hiddenSize]);
        _bh = new Tensor<T>([_hiddenSize]);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_Wz, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wr, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Uz, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Ur, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Uh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bz, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_br, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_bh, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GRULayer{T}"/> class with the specified dimensions, return behavior, and vector activation functions.
    /// </summary>
    /// <param name="inputSize">The size of the input feature vector at each time step.</param>
    /// <param name="hiddenSize">The size of the hidden state vector.</param>
    /// <param name="returnSequences">If <c>true</c>, returns all hidden states; if <c>false</c>, returns only the final hidden state.</param>
    /// <param name="vectorActivation">The vector activation function for the candidate hidden state. Defaults to tanh if not specified.</param>
    /// <param name="vectorRecurrentActivation">The vector activation function for the gates. Defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GRU layer with the specified dimensions and vector activation functions.
    /// Vector activation functions operate on entire vectors rather than individual elements, which can capture
    /// dependencies between different elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GRU layer with more advanced vector-based activation functions.
    /// 
    /// Vector activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - May be more powerful for complex patterns
    /// 
    /// This constructor is useful when you need the layer to understand how different
    /// features interact with each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public GRULayer(int inputSize, int hiddenSize,
                    bool returnSequences = false,
                    IVectorActivationFunction<T>? vectorActivation = null,
                    IVectorActivationFunction<T>? vectorRecurrentActivation = null)
        : base([inputSize], [hiddenSize], vectorActivation ?? new TanhActivation<T>())
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _returnSequences = returnSequences;
        _vectorActivation = vectorActivation ?? new TanhActivation<T>();
        _vectorRecurrentActivation = vectorRecurrentActivation ?? new SigmoidActivation<T>();

        T scale = NumOps.Sqrt(NumOps.FromDouble(NumericalStabilityHelper.SafeDiv(1.0, _hiddenSize)));

        _Wz = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wr = InitializeTensor(_hiddenSize, _inputSize, scale);
        _Wh = InitializeTensor(_hiddenSize, _inputSize, scale);

        _Uz = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Ur = InitializeTensor(_hiddenSize, _hiddenSize, scale);
        _Uh = InitializeTensor(_hiddenSize, _hiddenSize, scale);

        _bz = new Tensor<T>([_hiddenSize]);
        _br = new Tensor<T>([_hiddenSize]);
        _bh = new Tensor<T>([_hiddenSize]);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_Wz, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wr, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Wh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Uz, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Ur, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_Uh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bz, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_br, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_bh, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="rows">The number of rows in the tensor.</param>
    /// <param name="cols">The number of columns in the tensor.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <returns>A new tensor with scaled random values.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor with the specified dimensions and fills it with random values
    /// between -0.5 and 0.5, scaled by the provided scale factor. This type of initialization helps
    /// with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a tensor of random starting values for weights.
    ///
    /// The method:
    /// - Creates a new tensor with the specified size
    /// - Fills it with random numbers between -0.5 and 0.5
    /// - Multiplies these numbers by a scale factor to control their size
    ///
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// The scale factor helps prevent values from being too large or too small at the start of training.
    /// </para>
    /// </remarks>
    private Tensor<T> InitializeTensor(int rows, int cols, T scale)
    {
        // Create random tensor using Tensor<T>.CreateRandom [0, 1]
        var randomTensor = Tensor<T>.CreateRandom(rows, cols);

        // Shift to [-0.5, 0.5] range: random - 0.5
        var halfTensor = new Tensor<T>([rows, cols]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor
        return Engine.TensorMultiplyScalar(shifted, scale);
    }

    /// <summary>
    /// Performs the forward pass of the GRU layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape should be [batchSize, sequenceLength, inputSize].</param>
    /// <returns>The output tensor. If returnSequences is true, shape will be [batchSize, sequenceLength, hiddenSize]; otherwise, [batchSize, hiddenSize].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the GRU layer. It processes the input sequence step by step,
    /// updating the hidden state at each time step according to the GRU equations. The update gate (z) controls
    /// how much of the previous hidden state to keep, the reset gate (r) controls how much of the previous hidden
    /// state to reset, and the candidate hidden state (h_candidate) contains new information from the current input.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your sequence data through the GRU.
    /// 
    /// For each element in your sequence (like each word in a sentence):
    /// 1. The update gate (z) decides how much of the old memory to keep
    /// 2. The reset gate (r) decides how much of the old memory to forget
    /// 3. The layer creates new information based on the current input and relevant memory
    /// 4. It combines the kept memory and new information to update its understanding
    /// 
    /// This process repeats for each element in the sequence, with the memory
    /// evolving to capture the relevant context from the entire sequence.
    /// 
    /// The final output depends on the returnSequences setting:
    /// - If true: Returns information about every element in the sequence
    /// - If false: Returns only the final memory state
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape.ToArray();
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims into batch for rank > 3
        Tensor<T> input3D;
        int batchSize;
        int sequenceLength;

        if (rank == 1)
        {
            // 1D input [total] -> treat as single batch, single sequence step
            batchSize = 1;
            sequenceLength = 1;
            int totalSize = input.Shape[0];
            int inferredInputSize = Math.Min(totalSize, _inputSize);

            // Reshape to [1, 1, inputSize], padding if needed
            var data = new T[_inputSize];
            for (int i = 0; i < inferredInputSize && i < input.Length; i++)
            {
                data[i] = input.Data.Span[i];
            }
            input3D = new Tensor<T>(new[] { 1, 1, _inputSize }, new Vector<T>(data));
        }
        else if (rank == 2)
        {
            // 2D input [sequenceLength, inputSize] -> add batch dim
            batchSize = 1;
            sequenceLength = input.Shape[0];
            int actualInputSize = input.Shape[1];

            if (actualInputSize != _inputSize)
            {
                // Adapt the input by truncating or padding to match expected input size
                int targetElements = sequenceLength * _inputSize;

                if (actualInputSize >= _inputSize)
                {
                    // Truncate: take only the first _inputSize features from each timestep
                    var data = new T[targetElements];
                    for (int s = 0; s < sequenceLength; s++)
                    {
                        for (int f = 0; f < _inputSize; f++)
                        {
                            data[s * _inputSize + f] = input.Data.Span[s * actualInputSize + f];
                        }
                    }
                    input3D = new Tensor<T>(new[] { 1, sequenceLength, _inputSize }, new Vector<T>(data));
                }
                else
                {
                    // Pad: add zeros for missing features
                    var data = new T[targetElements];
                    for (int s = 0; s < sequenceLength; s++)
                    {
                        for (int f = 0; f < actualInputSize; f++)
                        {
                            data[s * _inputSize + f] = input.Data.Span[s * actualInputSize + f];
                        }
                        // Remaining features are initialized to default (zero)
                    }
                    input3D = new Tensor<T>(new[] { 1, sequenceLength, _inputSize }, new Vector<T>(data));
                }
            }
            else
            {
                input3D = input.Reshape([1, sequenceLength, _inputSize]);
            }
        }
        else if (rank == 3)
        {
            // Standard 3D input [batchSize, sequenceLength, inputSize]
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            int actualInputSize = input.Shape[2];

            // Handle input size mismatch: if the actual input size doesn't match _inputSize,
            // we need to adapt the input. This can happen when the GRU is used in a pipeline
            // where upstream layers output different dimensions than expected.
            if (actualInputSize != _inputSize)
            {
                // Reshape input to match expected input size by distributing elements
                int totalElements = batchSize * sequenceLength * actualInputSize;
                int targetElements = batchSize * sequenceLength * _inputSize;

                if (totalElements >= targetElements)
                {
                    // Truncate: take only the first _inputSize features from each timestep
                    var data = new T[targetElements];
                    int srcIdx = 0;
                    int dstIdx = 0;
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int s = 0; s < sequenceLength; s++)
                        {
                            for (int f = 0; f < _inputSize; f++)
                            {
                                data[dstIdx++] = input.Data.Span[srcIdx + f];
                            }
                            srcIdx += actualInputSize;
                        }
                    }
                    input3D = new Tensor<T>(new[] { batchSize, sequenceLength, _inputSize }, new Vector<T>(data));
                }
                else
                {
                    // Pad: add zeros for missing features
                    var data = new T[targetElements];
                    int srcIdx = 0;
                    int dstIdx = 0;
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int s = 0; s < sequenceLength; s++)
                        {
                            for (int f = 0; f < actualInputSize; f++)
                            {
                                data[dstIdx + f] = input.Data.Span[srcIdx + f];
                            }
                            srcIdx += actualInputSize;
                            dstIdx += _inputSize;
                        }
                    }
                    input3D = new Tensor<T>(new[] { batchSize, sequenceLength, _inputSize }, new Vector<T>(data));
                }
            }
            else
            {
                input3D = input;
            }
        }
        else
        {
            // Higher-rank tensor: collapse leading dims into batch
            sequenceLength = input.Shape[rank - 2];
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;

            int actualInputSize = input.Shape[rank - 1];
            if (actualInputSize != _inputSize)
            {
                // Adapt the input by truncating or padding
                int totalElements = flatBatch * sequenceLength * actualInputSize;
                int targetElements = flatBatch * sequenceLength * _inputSize;

                if (totalElements >= targetElements)
                {
                    // Truncate
                    var data = new T[targetElements];
                    int srcIdx = 0;
                    int dstIdx = 0;
                    for (int b = 0; b < flatBatch; b++)
                    {
                        for (int s = 0; s < sequenceLength; s++)
                        {
                            for (int f = 0; f < _inputSize; f++)
                            {
                                data[dstIdx++] = input.Data.Span[srcIdx + f];
                            }
                            srcIdx += actualInputSize;
                        }
                    }
                    input3D = new Tensor<T>(new[] { flatBatch, sequenceLength, _inputSize }, new Vector<T>(data));
                }
                else
                {
                    // Pad
                    var data = new T[targetElements];
                    int srcIdx = 0;
                    int dstIdx = 0;
                    for (int b = 0; b < flatBatch; b++)
                    {
                        for (int s = 0; s < sequenceLength; s++)
                        {
                            for (int f = 0; f < actualInputSize; f++)
                            {
                                data[dstIdx + f] = input.Data.Span[srcIdx + f];
                            }
                            srcIdx += actualInputSize;
                            dstIdx += _inputSize;
                        }
                    }
                    input3D = new Tensor<T>(new[] { flatBatch, sequenceLength, _inputSize }, new Vector<T>(data));
                }
            }
            else
            {
                input3D = input.Reshape([flatBatch, sequenceLength, _inputSize]);
            }
        }

        // Cache input only if training
        if (IsTrainingMode)
        {
            _lastInput = input3D;
        }

        // Reset hidden state if needed
        if (_lastHiddenState == null)
        {
            _lastHiddenState = TensorAllocator.Rent<T>([batchSize, _hiddenSize]);
        }

        // Initialize list to store all hidden states if returning sequences
        if (_returnSequences)
        {
            _allHiddenStates = new List<Tensor<T>>(sequenceLength);
        }

        // Process each time step
        Tensor<T> currentHiddenState = _lastHiddenState;

        // Store the last activation values for backpropagation
        Tensor<T>? lastZ = null;
        Tensor<T>? lastR = null;
        Tensor<T>? lastH_candidate = null;

        // Pre-transpose weights for efficiency (weights are [hiddenSize, inputSize], need [inputSize, hiddenSize] for matmul)
        var WzT = Engine.TensorTranspose(_Wz);
        var WrT = Engine.TensorTranspose(_Wr);
        var WhT = Engine.TensorTranspose(_Wh);
        var UzT = Engine.TensorTranspose(_Uz);
        var UrT = Engine.TensorTranspose(_Ur);
        var UhT = Engine.TensorTranspose(_Uh);

        for (int t = 0; t < sequenceLength; t++)
        {
            // Extract current time step input: slice axis 1 (sequence) at timestep t
            var xt = input3D.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);

            var z = ApplyActivation(Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(xt, WzT), Engine.TensorMatMul(currentHiddenState, UzT)), _bz), true);
            var r = ApplyActivation(Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(xt, WrT), Engine.TensorMatMul(currentHiddenState, UrT)), _br), true);
            var h_candidate = ApplyActivation(Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(xt, WhT), Engine.TensorMatMul(r.ElementwiseMultiply(currentHiddenState), UhT)), _bh), false);
            // Compute (1 - z) using cached ones tensor — avoids per-timestep allocation
            if (_cachedOnesForGate == null || !_cachedOnesForGate.Shape.ToArray().SequenceEqual(z.Shape.ToArray()))
            {
                _cachedOnesForGate = Tensor<T>.CreateDefault(z.Shape.ToArray(), NumOps.One);
            }
            var oneMinusZ = Engine.TensorSubtract(_cachedOnesForGate, z);


            var h = z.ElementwiseMultiply(currentHiddenState).Add(

                oneMinusZ.ElementwiseMultiply(h_candidate)

            );

            currentHiddenState = h;

            // Save the last timestep's activations
            if (t == sequenceLength - 1)
            {
                lastZ = z;
                lastR = r;
                lastH_candidate = h_candidate;
            }

            if (_returnSequences && _allHiddenStates != null)
            {
                _allHiddenStates.Add(h.Clone());
            }
        }

        if (IsTrainingMode)
        {
            _lastZ = lastZ;
            _lastR = lastR;
            _lastH = lastH_candidate;
            _lastHiddenState = currentHiddenState;
        }

        // Return either the sequence of hidden states or just the final state
        Tensor<T> output;
        if (_returnSequences && _allHiddenStates != null)
        {
            // Each hidden state is [batchSize, hiddenSize]
            // Concatenate along axis 1, then reshape to [batchSize, sequenceLength, hiddenSize]
            output = Tensor<T>.Concatenate([.. _allHiddenStates], 1).Reshape([batchSize, sequenceLength, _hiddenSize]);
        }
        else
        {
            output = currentHiddenState;
        }

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            int[] newShape;
            if (_returnSequences)
            {
                // Output shape: [...leadingDims, sequenceLength, hiddenSize]
                newShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 2] = sequenceLength;
                newShape[_originalInputShape.Length - 1] = _hiddenSize;
            }
            else
            {
                // Output shape: [...leadingDims, hiddenSize]
                newShape = new int[_originalInputShape.Length - 1];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 2] = _hiddenSize;
            }
            output = output.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2 && !_returnSequences)
        {
            // 2D input -> 1D output (remove batch dim)
            output = output.Reshape([_hiddenSize]);
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors using fused sequence kernel.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after GRU processing.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable.</exception>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input.Shape.ToArray();
        int rank = shape.Length;

        // Determine sequence length, batch size from shape
        int sequenceLength;
        int batchSize;
        if (rank == 1)
        {
            sequenceLength = 1;
            batchSize = 1;
        }
        else if (rank == 2)
        {
            sequenceLength = shape[0];
            batchSize = 1;
        }
        else if (rank == 3)
        {
            sequenceLength = shape[1];
            batchSize = shape[0];
        }
        else
        {
            // Higher rank: collapse leading dims into batch
            sequenceLength = shape[rank - 2];
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= shape[d];
            batchSize = flatBatch;
        }

        int hiddenBufferSize = batchSize * _hiddenSize;
        int fullOutputSize = sequenceLength * batchSize * _hiddenSize;
        int allHSize = (sequenceLength + 1) * batchSize * _hiddenSize;
        int cacheGatesSize = sequenceLength * batchSize * _hiddenSize * 3;

        int[] outputShape;
        int outputSize;
        if (_returnSequences)
        {
            outputShape = [batchSize, sequenceLength, _hiddenSize];
            outputSize = fullOutputSize;
        }
        else
        {
            outputShape = [batchSize, _hiddenSize];
            outputSize = hiddenBufferSize;
        }

        // Cache input for backward pass if training
        if (IsTrainingMode)
        {
            _lastInput = input.Reshape([batchSize, sequenceLength, _inputSize]);
            _originalInputShape = shape;
        }

        // Prepare stacked weights in PyTorch format (r, z, n order)
        PrepareStackedWeightsForGpu(backend);

        // Allocate initial hidden state buffer (zeros)
        var hInitBuffer = backend.AllocateBuffer(hiddenBufferSize);
        backend.Fill(hInitBuffer, 0.0f, hiddenBufferSize);

        // Allocate output buffers
        var fullOutputBuffer = backend.AllocateBuffer(fullOutputSize);
        var hFinalBuffer = backend.AllocateBuffer(hiddenBufferSize);

        // Allocate cache buffers for backward pass
        bool cacheForTraining = IsTrainingMode;
        IGpuBuffer? allHBuffer = null;
        IGpuBuffer? cacheGatesBuffer = null;

        if (cacheForTraining)
        {
            ClearGpuTrainingCache();
            _gpuLastInput = input;

            allHBuffer = backend.AllocateBuffer(allHSize);
            cacheGatesBuffer = backend.AllocateBuffer(cacheGatesSize);

            // Store fused cache buffers for backward pass
            _gpuFusedAllH = allHBuffer;
            _gpuFusedCacheGates = cacheGatesBuffer;

            // Cache initial hidden state
            var initHBufferCopy = backend.AllocateBuffer(hiddenBufferSize);
            backend.Copy(hInitBuffer, initHBufferCopy, hiddenBufferSize);
            _gpuInitialHiddenState = GpuTensorHelper.UploadToGpu<T>(backend, initHBufferCopy, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
        }
        else
        {
            // Allocate temporary cache buffers (will be disposed after forward)
            allHBuffer = backend.AllocateBuffer(allHSize);
            cacheGatesBuffer = backend.AllocateBuffer(cacheGatesSize);
        }

        try
        {
            // Call fused GRU sequence kernel (processes all timesteps in one kernel launch)
            backend.GruForwardSequence(
                input.Buffer, hInitBuffer,
                _gpuStackedWeightsIh!, _gpuStackedWeightsHh!, _gpuStackedBiasIh!, _gpuStackedBiasHh!,
                fullOutputBuffer, hFinalBuffer, allHBuffer, cacheGatesBuffer,
                sequenceLength, batchSize, _inputSize, _hiddenSize);

            // Create output buffer - either full sequence or final hidden state
            IGpuBuffer outputBuffer;
            if (_returnSequences)
            {
                outputBuffer = fullOutputBuffer;
            }
            else
            {
                // Copy final hidden state to output
                outputBuffer = backend.AllocateBuffer(outputSize);
                backend.Copy(hFinalBuffer, outputBuffer, hiddenBufferSize);
                fullOutputBuffer.Dispose();
            }

            // Cleanup temporary buffers
            hInitBuffer.Dispose();
            hFinalBuffer.Dispose();

            // If not training, dispose cache buffers
            if (!cacheForTraining)
            {
                allHBuffer.Dispose();
                cacheGatesBuffer.Dispose();
            }

            return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            // Cleanup buffers on error
            hInitBuffer.Dispose();
            hFinalBuffer.Dispose();
            fullOutputBuffer.Dispose();

            if (!cacheForTraining)
            {
                allHBuffer?.Dispose();
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
    /// Applies the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <param name="isRecurrent">If <c>true</c>, applies the recurrent activation function; otherwise, applies the regular activation function.</param>
    /// <returns>The activated tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method applies either the recurrent activation function (for gates) or the regular activation function
    /// (for the candidate hidden state) to the input tensor. It supports both element-wise and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the appropriate "shaping function" to the values.
    /// 
    /// Depending on what's being processed:
    /// - For gates (update and reset), it uses the recurrent activation (usually sigmoid)
    /// - For new information (candidate hidden state), it uses the regular activation (usually tanh)
    /// 
    /// This helps shape the values into the right ranges:
    /// - Gates need to be between 0 and 1 (like percentages)
    /// - New information is often kept between -1 and 1 for stability
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivation(Tensor<T> input, bool isRecurrent)
    {
        if (isRecurrent)
        {
            if (_vectorRecurrentActivation != null)
            {
                // Use centralized ActivationHelper for optimized activation dispatch
                return ActivationHelper.ApplyActivation(_vectorRecurrentActivation, input, Engine);
            }
            else if (_recurrentActivation != null)
            {
                return input.Transform((x, _) => _recurrentActivation.Activate(x));
            }
        }
        else
        {
            if (_vectorActivation != null)
            {
                // Use centralized ActivationHelper for optimized activation dispatch
                return ActivationHelper.ApplyActivation(_vectorActivation, input, Engine);
            }
            else if (_activation != null)
            {
                return input.Transform((x, _) => _activation.Activate(x));
            }
        }

        throw new InvalidOperationException("No activation function specified");
    }

    /// <summary>
    /// Computes a gate activation for the GRU layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="hidden">The hidden state tensor.</param>
    /// <param name="W">The input weight matrix.</param>
    /// <param name="U">The hidden state weight matrix.</param>
    /// <param name="b">The bias vector.</param>
    /// <param name="isRecurrent">If true, applies recurrent activation; otherwise, applies regular activation.</param>
    /// <returns>The computed gate activation.</returns>
    private Tensor<T> ComputeGate(Tensor<T> input, Tensor<T> hidden, Tensor<T> W, Tensor<T> U, Tensor<T> b, bool isRecurrent)
    {
        // W is [hiddenSize, inputSize], U is [hiddenSize, hiddenSize]
        // input is [batch, inputSize], hidden is [batch, hiddenSize]
        // Need to transpose to get [inputSize, hiddenSize] and [hiddenSize, hiddenSize]
        var WT = Engine.TensorTranspose(W);
        var UT = Engine.TensorTranspose(U);
        var gate = Engine.TensorBroadcastAdd(Engine.TensorAdd(Engine.TensorMatMul(input, WT), Engine.TensorMatMul(hidden, UT)), b);
        return ApplyActivation(gate, isRecurrent);
    }

    private Tensor<T>? _WzVelocity, _WrVelocity, _WhVelocity, _UzVelocity, _UrVelocity, _UhVelocity;
    private Tensor<T>? _bzVelocity, _brVelocity, _bhVelocity;

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all the weight matrices and bias vectors of the GRU layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// This is typically called after the backward pass during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - All weights and biases are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to understand and predict sequences.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_dWz == null || _dWr == null || _dWh == null ||
            _dUz == null || _dUr == null || _dUh == null ||
            _dbz == null || _dbr == null || _dbh == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_WzVelocity == null)
            {
                _WzVelocity = new Tensor<T>(_Wz.Shape.ToArray()); _WzVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_WzVelocity, PersistentTensorRole.OptimizerState);
                _WrVelocity = new Tensor<T>(_Wr.Shape.ToArray()); _WrVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_WrVelocity, PersistentTensorRole.OptimizerState);
                _WhVelocity = new Tensor<T>(_Wh.Shape.ToArray()); _WhVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_WhVelocity, PersistentTensorRole.OptimizerState);
                _UzVelocity = new Tensor<T>(_Uz.Shape.ToArray()); _UzVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_UzVelocity, PersistentTensorRole.OptimizerState);
                _UrVelocity = new Tensor<T>(_Ur.Shape.ToArray()); _UrVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_UrVelocity, PersistentTensorRole.OptimizerState);
                _UhVelocity = new Tensor<T>(_Uh.Shape.ToArray()); _UhVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_UhVelocity, PersistentTensorRole.OptimizerState);
                _bzVelocity = new Tensor<T>(_bz.Shape.ToArray()); _bzVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_bzVelocity, PersistentTensorRole.OptimizerState);
                _brVelocity = new Tensor<T>(_br.Shape.ToArray()); _brVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_brVelocity, PersistentTensorRole.OptimizerState);
                _bhVelocity = new Tensor<T>(_bh.Shape.ToArray()); _bhVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_bhVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_Wz, _dWz, _WzVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_Wr, _dWr, _WrVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_Wh, _dWh, _WhVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_Uz, _dUz, _UzVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_Ur, _dUr, _UrVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_Uh, _dUh, _UhVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_bz, _dbz, _bzVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_br, _dbr, _brVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_bh, _dbh, _bhVelocity!, lr, 0.0f, 0.0f);
        }
        else
        {
            // Use Engine operations for parameter updates
            var scaledWz = Engine.TensorMultiplyScalar(_dWz, learningRate);
            var scaledWr = Engine.TensorMultiplyScalar(_dWr, learningRate);
            var scaledWh = Engine.TensorMultiplyScalar(_dWh, learningRate);
            var scaledUz = Engine.TensorMultiplyScalar(_dUz, learningRate);
            var scaledUr = Engine.TensorMultiplyScalar(_dUr, learningRate);
            var scaledUh = Engine.TensorMultiplyScalar(_dUh, learningRate);
            var scaledBz = Engine.TensorMultiplyScalar(_dbz, learningRate);
            var scaledBr = Engine.TensorMultiplyScalar(_dbr, learningRate);
            var scaledBh = Engine.TensorMultiplyScalar(_dbh, learningRate);

            _Wz = Engine.TensorSubtract(_Wz, scaledWz);
            _Wr = Engine.TensorSubtract(_Wr, scaledWr);
            _Wh = Engine.TensorSubtract(_Wh, scaledWh);
            _Uz = Engine.TensorSubtract(_Uz, scaledUz);
            _Ur = Engine.TensorSubtract(_Ur, scaledUr);
            _Uh = Engine.TensorSubtract(_Uh, scaledUh);
            _bz = Engine.TensorSubtract(_bz, scaledBz);
            _br = Engine.TensorSubtract(_br, scaledBr);
            _bh = Engine.TensorSubtract(_bh, scaledBh);

            Engine.InvalidatePersistentTensor(_Wz);
            Engine.InvalidatePersistentTensor(_Wr);
            Engine.InvalidatePersistentTensor(_Wh);
            Engine.InvalidatePersistentTensor(_Uz);
            Engine.InvalidatePersistentTensor(_Ur);
            Engine.InvalidatePersistentTensor(_Uh);
            Engine.InvalidatePersistentTensor(_bz);
            Engine.InvalidatePersistentTensor(_br);
            Engine.InvalidatePersistentTensor(_bh);
        }

        // Invalidate stacked weight buffers since weights have been modified
        InvalidateGpuStackedWeights();
    }

    /// <summary>
    /// Updates the parameters of the layer with the given vector of parameter values.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets all the weight matrices and bias vectors of the GRU layer from a single vector of parameters.
    /// The parameters are arranged in the following order: Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you directly set all the learnable values in the layer.
    /// 
    /// The parameters vector contains all weights and biases in a specific order:
    /// 1. Weights for input to update gate (Wz)
    /// 2. Weights for input to reset gate (Wr)
    /// 3. Weights for input to candidate hidden state (Wh)
    /// 4. Weights for hidden state to update gate (Uz)
    /// 5. Weights for hidden state to reset gate (Ur)
    /// 6. Weights for hidden state to candidate hidden state (Uh)
    /// 7. Biases for update gate (bz)
    /// 8. Biases for reset gate (br)
    /// 9. Biases for candidate hidden state (bh)
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Setting specific parameter values for testing
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int inputWeightSize = _hiddenSize * _inputSize;
        int hiddenWeightSize = _hiddenSize * _hiddenSize;
        int biasSize = _hiddenSize;
        int idx = 0;

        // Extract and reshape weight tensors using Tensor.FromVector
        _Wz = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), [_hiddenSize, _inputSize]);
        idx += inputWeightSize;

        _Wr = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), [_hiddenSize, _inputSize]);
        idx += inputWeightSize;

        _Wh = Tensor<T>.FromVector(parameters.Slice(idx, inputWeightSize), [_hiddenSize, _inputSize]);
        idx += inputWeightSize;

        _Uz = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), [_hiddenSize, _hiddenSize]);
        idx += hiddenWeightSize;

        _Ur = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), [_hiddenSize, _hiddenSize]);
        idx += hiddenWeightSize;

        _Uh = Tensor<T>.FromVector(parameters.Slice(idx, hiddenWeightSize), [_hiddenSize, _hiddenSize]);
        idx += hiddenWeightSize;

        _bz = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), [_hiddenSize]);
        idx += biasSize;

        _br = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), [_hiddenSize]);
        idx += biasSize;

        _bh = Tensor<T>.FromVector(parameters.Slice(idx, biasSize), [_hiddenSize]);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_Wz);
        Engine.InvalidatePersistentTensor(_Wr);
        Engine.InvalidatePersistentTensor(_Wh);
        Engine.InvalidatePersistentTensor(_Uz);
        Engine.InvalidatePersistentTensor(_Ur);
        Engine.InvalidatePersistentTensor(_Uh);
        Engine.InvalidatePersistentTensor(_bz);
        Engine.InvalidatePersistentTensor(_br);
        Engine.InvalidatePersistentTensor(_bh);

        // Invalidate stacked weight buffers since weights have been replaced
        InvalidateGpuStackedWeights();
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) and combines them into a single vector.
    /// The parameters are arranged in the following order: Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for
    /// saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// It gathers all parameters in this specific order:
    /// 1. Weights for input to update gate (Wz)
    /// 2. Weights for input to reset gate (Wr)
    /// 3. Weights for input to candidate hidden state (Wh)
    /// 4. Weights for hidden state to update gate (Uz)
    /// 5. Weights for hidden state to reset gate (Ur)
    /// 6. Weights for hidden state to candidate hidden state (Uh)
    /// 7. Biases for update gate (bz)
    /// 8. Biases for reset gate (br)
    /// 9. Biases for candidate hidden state (bh)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    /// <summary>
    /// Returns metadata for serialization including ReturnSequences flag.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ReturnSequences"] = _returnSequences.ToString();
        return metadata;
    }

    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_Wz.ToArray()),
            new Vector<T>(_Wr.ToArray()),
            new Vector<T>(_Wh.ToArray()),
            new Vector<T>(_Uz.ToArray()),
            new Vector<T>(_Ur.ToArray()),
            new Vector<T>(_Uh.ToArray()),
            new Vector<T>(_bz.ToArray()),
            new Vector<T>(_br.ToArray()),
            new Vector<T>(_bh.ToArray())
        );
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        void CopyToTensor(Tensor<T> tensor)
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor.SetFlat(i, parameters[idx++]);
        }

        CopyToTensor(_Wz);
        CopyToTensor(_Wr);
        CopyToTensor(_Wh);
        CopyToTensor(_Uz);
        CopyToTensor(_Ur);
        CopyToTensor(_Uh);
        CopyToTensor(_bz);
        CopyToTensor(_br);
        CopyToTensor(_bh);
    }

    public override Vector<T> GetParameterGradients()
    {
        if (_dWz == null || _dWr == null || _dWh == null ||
            _dUz == null || _dUr == null || _dUh == null ||
            _dbz == null || _dbr == null || _dbh == null)
        {
            return new Vector<T>(ParameterCount);
        }

        return Vector<T>.Concatenate(
            new Vector<T>(_dWz.ToArray()),
            new Vector<T>(_dWr.ToArray()),
            new Vector<T>(_dWh.ToArray()),
            new Vector<T>(_dUz.ToArray()),
            new Vector<T>(_dUr.ToArray()),
            new Vector<T>(_dUh.ToArray()),
            new Vector<T>(_dbz.ToArray()),
            new Vector<T>(_dbr.ToArray()),
            new Vector<T>(_dbh.ToArray())
        );
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _dWz = null; _dWr = null; _dWh = null;
        _dUz = null; _dUr = null; _dUh = null;
        _dbz = null; _dbr = null; _dbh = null;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This includes the last input, hidden state, activation values, and all hidden states if returning sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The hidden state (memory) is cleared
    /// - All stored information about previous inputs is removed
    /// - All gate activations are reset
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// 
    /// For example, if you've processed one sentence and want to start with a new sentence,
    /// you should reset the state to prevent the new sentence from being influenced by the previous one.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastHiddenState = null;
        _lastZ = null;
        _lastR = null;
        _lastH = null;
        _allHiddenStates = null;
    }

    /// <summary>
    /// Applies the derivative of the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to apply the derivative to.</param>
    /// <param name="isRecurrent">If <c>true</c>, applies the derivative of the recurrent activation function; otherwise, applies the derivative of the regular activation function.</param>
    /// <returns>The tensor with the activation derivative applied.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of either the recurrent activation function (for gates) or the regular
    /// activation function (for the candidate hidden state) to the input tensor. This is used during the backward
    /// pass to calculate gradients. It supports both element-wise and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the activation function is to changes.
    ///
    /// During backpropagation (learning):
    /// - We need to know how much a small change in input affects the output
    /// - This method calculates that sensitivity for either the gate functions or the main activation
    /// - It's a crucial part of the math that allows neural networks to learn
    ///
    /// Think of it as measuring the "slope" of the activation function at each point,
    /// which tells us how to adjust the parameters to improve the network's performance.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivationDerivative(Tensor<T> input, bool isRecurrent)
    {
        if (isRecurrent)
        {
            if (_vectorRecurrentActivation != null)
                return _vectorRecurrentActivation.Derivative(input);
            else if (_recurrentActivation != null)
                return input.Transform((x, _) => _recurrentActivation.Derivative(x));
        }
        else
        {
            if (_vectorActivation != null)
                return _vectorActivation.Derivative(input);
            else if (_activation != null)
                return input.Transform((x, _) => _activation.Derivative(x));
        }

        throw new InvalidOperationException("No activation function specified.");
    }

    /// <summary>
    /// Broadcasts a 1D tensor across the batch dimension.
    /// </summary>
    /// <param name="vector">The 1D tensor to broadcast.</param>
    /// <param name="batchSize">The batch size to broadcast to.</param>
    /// <returns>A 2D tensor with the vector broadcast across rows.</returns>
    private Tensor<T> BroadcastVector(Tensor<T> vector, int batchSize)
    {
        // Use Engine.TensorTile to broadcast the vector across the batch dimension
        // First reshape vector from [length] to [1, length], then tile along first axis
        var reshapedVector = vector.Reshape([1, vector.Length]);
        return Engine.TensorTile(reshapedVector, [batchSize, 1]);
    }

    /// <summary>
    /// Creates a tensor of ones with the same shape as the input tensor.
    /// </summary>
    private Tensor<T> CreateOnesLike(Tensor<T> tensor)
    {
        var ones = new Tensor<T>(tensor.Shape.ToArray());
        ones.Fill(NumOps.One);
        return ones;
    }

    /// <summary>
    /// Creates a deep copy of this GRU layer with independent weights and reset state.
    /// </summary>
    /// <returns>A new GRULayer with the same weights but independent of the original.</returns>
    public override LayerBase<T> Clone()
    {
        var clone = (GRULayer<T>)base.Clone();

        // Deep copy all weight tensors
        clone._Wz = _Wz.Clone();
        clone._Wr = _Wr.Clone();
        clone._Wh = _Wh.Clone();
        clone._Uz = _Uz.Clone();
        clone._Ur = _Ur.Clone();
        clone._Uh = _Uh.Clone();
        clone._bz = _bz.Clone();
        clone._br = _br.Clone();
        clone._bh = _bh.Clone();

        // Reset internal state (don't share state between original and clone)
        clone._lastInput = null;
        clone._lastHiddenState = null;
        clone._lastZ = null;
        clone._lastR = null;
        clone._lastH = null;
        clone._allHiddenStates = null;
        clone._originalInputShape = null;

        // Reset gradients
        clone._dWz = null;
        clone._dWr = null;
        clone._dWh = null;
        clone._dUz = null;
        clone._dUr = null;
        clone._dUh = null;
        clone._dbz = null;
        clone._dbr = null;
        clone._dbh = null;

        return clone;
    }

    /// <summary>
    /// Clears the GPU training cache to release GPU memory.
    /// </summary>
    private void ClearGpuTrainingCache()
    {
        if (_gpuCachedZGates != null)
        {
            foreach (var tensor in _gpuCachedZGates)
                tensor?.Dispose();
            _gpuCachedZGates = null;
        }
        if (_gpuCachedRGates != null)
        {
            foreach (var tensor in _gpuCachedRGates)
                tensor?.Dispose();
            _gpuCachedRGates = null;
        }
        if (_gpuCachedHCandidates != null)
        {
            foreach (var tensor in _gpuCachedHCandidates)
                tensor?.Dispose();
            _gpuCachedHCandidates = null;
        }
        if (_gpuCachedHiddenStates != null)
        {
            foreach (var tensor in _gpuCachedHiddenStates)
                tensor?.Dispose();
            _gpuCachedHiddenStates = null;
        }
        _gpuInitialHiddenState?.Dispose();
        _gpuInitialHiddenState = null;
        _gpuLastInput = null; // Don't dispose - owned by caller

        // Note: Do NOT dispose stacked weights here - they're reusable across forward passes
        // and are only invalidated when actual weights are updated via InvalidateGpuStackedWeights()

        _gpuFusedAllH?.Dispose();
        _gpuFusedAllH = null;
        _gpuFusedCacheGates?.Dispose();
        _gpuFusedCacheGates = null;
    }

    /// <summary>
    /// Prepares stacked weights in PyTorch format (r, z, n order) for the fused GRU kernel.
    /// </summary>
    private void PrepareStackedWeightsForGpu(IDirectGpuBackend backend)
    {
        if (_gpuStackedWeightsValid && _gpuStackedWeightsIh != null)
            return;

        // Stack input-to-hidden weights: [3*hiddenSize, inputSize] in order r, z, n
        // Layer has: _Wz (update/z), _Wr (reset/r), _Wh (candidate/n)
        // Kernel order: r, z, n -> Wr, Wz, Wh
        int weightsIhSize = 3 * _hiddenSize * _inputSize;
        int weightsHhSize = 3 * _hiddenSize * _hiddenSize;
        int biasSize = 3 * _hiddenSize;

        var stackedIh = new float[weightsIhSize];
        var stackedHh = new float[weightsHhSize];
        var stackedBiasIh = new float[biasSize];
        var stackedBiasHh = new float[biasSize]; // PyTorch uses separate ih/hh biases

        // Convert weights to float arrays
        var wR = DirectGpuEngine.ToFloatArray<T>(_Wr.ToArray());
        var wZ = DirectGpuEngine.ToFloatArray<T>(_Wz.ToArray());
        var wH = DirectGpuEngine.ToFloatArray<T>(_Wh.ToArray());

        var uR = DirectGpuEngine.ToFloatArray<T>(_Ur.ToArray());
        var uZ = DirectGpuEngine.ToFloatArray<T>(_Uz.ToArray());
        var uH = DirectGpuEngine.ToFloatArray<T>(_Uh.ToArray());

        var bR = DirectGpuEngine.ToFloatArray<T>(_br.ToArray());
        var bZ = DirectGpuEngine.ToFloatArray<T>(_bz.ToArray());
        var bH = DirectGpuEngine.ToFloatArray<T>(_bh.ToArray());

        // Stack weights in kernel order (r, z, n)
        int resetGateOffset = 0;
        int updateGateOffset = _hiddenSize * _inputSize;
        int newGateOffset = 2 * _hiddenSize * _inputSize;

        Array.Copy(wR, 0, stackedIh, resetGateOffset, _hiddenSize * _inputSize);
        Array.Copy(wZ, 0, stackedIh, updateGateOffset, _hiddenSize * _inputSize);
        Array.Copy(wH, 0, stackedIh, newGateOffset, _hiddenSize * _inputSize);

        int hhResetGateOffset = 0;
        int hhUpdateGateOffset = _hiddenSize * _hiddenSize;
        int hhNewGateOffset = 2 * _hiddenSize * _hiddenSize;

        Array.Copy(uR, 0, stackedHh, hhResetGateOffset, _hiddenSize * _hiddenSize);
        Array.Copy(uZ, 0, stackedHh, hhUpdateGateOffset, _hiddenSize * _hiddenSize);
        Array.Copy(uH, 0, stackedHh, hhNewGateOffset, _hiddenSize * _hiddenSize);

        // Stack biases (put all in biasIh, biasHh will be zeros - matches PyTorch default)
        int biasResetGateOffset = 0;
        int biasUpdateGateOffset = _hiddenSize;
        int biasNewGateOffset = 2 * _hiddenSize;

        Array.Copy(bR, 0, stackedBiasIh, biasResetGateOffset, _hiddenSize);
        Array.Copy(bZ, 0, stackedBiasIh, biasUpdateGateOffset, _hiddenSize);
        Array.Copy(bH, 0, stackedBiasIh, biasNewGateOffset, _hiddenSize);
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
        out float[] dWr, out float[] dWz, out float[] dWh,
        out float[] dUr, out float[] dUz, out float[] dUh,
        out float[] dBr, out float[] dBz, out float[] dBh)
    {
        int weightsIhSize = 3 * _hiddenSize * _inputSize;
        int weightsHhSize = 3 * _hiddenSize * _hiddenSize;
        int biasSize = 3 * _hiddenSize;

        var stackedIh = new float[weightsIhSize];
        var stackedHh = new float[weightsHhSize];
        var stackedBias = new float[biasSize];

        backend.DownloadBuffer(stackedGradIh, stackedIh);
        backend.DownloadBuffer(stackedGradHh, stackedHh);
        backend.DownloadBuffer(stackedGradBias, stackedBias);

        // Extract per-gate gradients from stacked format (r, z, n order)
        dWr = new float[_hiddenSize * _inputSize];
        dWz = new float[_hiddenSize * _inputSize];
        dWh = new float[_hiddenSize * _inputSize];

        Array.Copy(stackedIh, 0, dWr, 0, _hiddenSize * _inputSize);
        Array.Copy(stackedIh, _hiddenSize * _inputSize, dWz, 0, _hiddenSize * _inputSize);
        Array.Copy(stackedIh, 2 * _hiddenSize * _inputSize, dWh, 0, _hiddenSize * _inputSize);

        dUr = new float[_hiddenSize * _hiddenSize];
        dUz = new float[_hiddenSize * _hiddenSize];
        dUh = new float[_hiddenSize * _hiddenSize];

        Array.Copy(stackedHh, 0, dUr, 0, _hiddenSize * _hiddenSize);
        Array.Copy(stackedHh, _hiddenSize * _hiddenSize, dUz, 0, _hiddenSize * _hiddenSize);
        Array.Copy(stackedHh, 2 * _hiddenSize * _hiddenSize, dUh, 0, _hiddenSize * _hiddenSize);

        dBr = new float[_hiddenSize];
        dBz = new float[_hiddenSize];
        dBh = new float[_hiddenSize];

        Array.Copy(stackedBias, 0, dBr, 0, _hiddenSize);
        Array.Copy(stackedBias, _hiddenSize, dBz, 0, _hiddenSize);
        Array.Copy(stackedBias, 2 * _hiddenSize, dBh, 0, _hiddenSize);
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

        if (_gpuWzGradient == null || _gpuWrGradient == null || _gpuWhGradient == null ||
            _gpuUzGradient == null || _gpuUrGradient == null || _gpuUhGradient == null ||
            _gpuBzGradient == null || _gpuBrGradient == null || _gpuBhGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuWz ??= GpuTensorHelper.UploadToGpu<T>(backend, _Wz, GpuTensorRole.Weight);
        _gpuWr ??= GpuTensorHelper.UploadToGpu<T>(backend, _Wr, GpuTensorRole.Weight);
        _gpuWh ??= GpuTensorHelper.UploadToGpu<T>(backend, _Wh, GpuTensorRole.Weight);
        _gpuUz ??= GpuTensorHelper.UploadToGpu<T>(backend, _Uz, GpuTensorRole.Weight);
        _gpuUr ??= GpuTensorHelper.UploadToGpu<T>(backend, _Ur, GpuTensorRole.Weight);
        _gpuUh ??= GpuTensorHelper.UploadToGpu<T>(backend, _Uh, GpuTensorRole.Weight);
        _gpuBz ??= GpuTensorHelper.UploadToGpu<T>(backend, _bz, GpuTensorRole.Bias);
        _gpuBr ??= GpuTensorHelper.UploadToGpu<T>(backend, _br, GpuTensorRole.Bias);
        _gpuBh ??= GpuTensorHelper.UploadToGpu<T>(backend, _bh, GpuTensorRole.Bias);

        // Ensure optimizer state exists
        EnsureGruOptimizerState(config, backend);

        // Apply optimizer updates using polymorphic config
        var wzState = BuildGruOptimizerState("Wz");
        var wrState = BuildGruOptimizerState("Wr");
        var whState = BuildGruOptimizerState("Wh");
        var uzState = BuildGruOptimizerState("Uz");
        var urState = BuildGruOptimizerState("Ur");
        var uhState = BuildGruOptimizerState("Uh");
        var bzState = BuildGruOptimizerState("bz");
        var brState = BuildGruOptimizerState("br");
        var bhState = BuildGruOptimizerState("bh");

        config.ApplyUpdate(backend, _gpuWz.Buffer, _gpuWzGradient.Buffer, wzState, _Wz.Length);
        config.ApplyUpdate(backend, _gpuWr.Buffer, _gpuWrGradient.Buffer, wrState, _Wr.Length);
        config.ApplyUpdate(backend, _gpuWh.Buffer, _gpuWhGradient.Buffer, whState, _Wh.Length);
        config.ApplyUpdate(backend, _gpuUz.Buffer, _gpuUzGradient.Buffer, uzState, _Uz.Length);
        config.ApplyUpdate(backend, _gpuUr.Buffer, _gpuUrGradient.Buffer, urState, _Ur.Length);
        config.ApplyUpdate(backend, _gpuUh.Buffer, _gpuUhGradient.Buffer, uhState, _Uh.Length);
        config.ApplyUpdate(backend, _gpuBz.Buffer, _gpuBzGradient.Buffer, bzState, _bz.Length);
        config.ApplyUpdate(backend, _gpuBr.Buffer, _gpuBrGradient.Buffer, brState, _br.Length);
        config.ApplyUpdate(backend, _gpuBh.Buffer, _gpuBhGradient.Buffer, bhState, _bh.Length);

        // Invalidate stacked weight buffers since individual weights have been modified
        InvalidateGpuStackedWeights();
    }

    /// <summary>
    /// Ensures optimizer state tensors are allocated for the given optimizer type.
    /// </summary>
    private void EnsureGruOptimizerState(IGpuOptimizerConfig config, IDirectGpuBackend backend)
    {
        var optimizerType = config.OptimizerType;

        if (optimizerType == GpuOptimizerType.Sgd || optimizerType == GpuOptimizerType.Nag || optimizerType == GpuOptimizerType.Lars)
        {
            // Velocity tensors for momentum-based optimizers
            var inputWeightSize = _hiddenSize * _inputSize;
            var hiddenWeightSize = _hiddenSize * _hiddenSize;
            var biasSize = _hiddenSize;
            _gpuWzVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWrVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUzVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUrVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBzVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBrVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBhVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            var inputWeightSize = _hiddenSize * _inputSize;
            var hiddenWeightSize = _hiddenSize * _hiddenSize;
            var biasSize = _hiddenSize;

            // First moment (M) tensors
            _gpuWzM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWrM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUzM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUrM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBzM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBrM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBhM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);

            // Second moment (V) tensors
            _gpuWzV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWrV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUzV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUrV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBzV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBrV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBhV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
    }

    /// <summary>
    /// Builds optimizer state for a specific parameter tensor.
    /// </summary>
    private GpuOptimizerState BuildGruOptimizerState(string paramName)
    {
        return paramName switch
        {
            "Wz" => new GpuOptimizerState { Velocity = _gpuWzVelocity?.Buffer, M = _gpuWzM?.Buffer, V = _gpuWzV?.Buffer },
            "Wr" => new GpuOptimizerState { Velocity = _gpuWrVelocity?.Buffer, M = _gpuWrM?.Buffer, V = _gpuWrV?.Buffer },
            "Wh" => new GpuOptimizerState { Velocity = _gpuWhVelocity?.Buffer, M = _gpuWhM?.Buffer, V = _gpuWhV?.Buffer },
            "Uz" => new GpuOptimizerState { Velocity = _gpuUzVelocity?.Buffer, M = _gpuUzM?.Buffer, V = _gpuUzV?.Buffer },
            "Ur" => new GpuOptimizerState { Velocity = _gpuUrVelocity?.Buffer, M = _gpuUrM?.Buffer, V = _gpuUrV?.Buffer },
            "Uh" => new GpuOptimizerState { Velocity = _gpuUhVelocity?.Buffer, M = _gpuUhM?.Buffer, V = _gpuUhV?.Buffer },
            "bz" => new GpuOptimizerState { Velocity = _gpuBzVelocity?.Buffer, M = _gpuBzM?.Buffer, V = _gpuBzV?.Buffer },
            "br" => new GpuOptimizerState { Velocity = _gpuBrVelocity?.Buffer, M = _gpuBrM?.Buffer, V = _gpuBrV?.Buffer },
            "bh" => new GpuOptimizerState { Velocity = _gpuBhVelocity?.Buffer, M = _gpuBhM?.Buffer, V = _gpuBhV?.Buffer },
            _ => new GpuOptimizerState()
        };
    }
}
