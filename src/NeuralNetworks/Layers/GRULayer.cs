using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
public class GRULayer<T> : LayerBase<T>
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
    private GpuTensor<T>? _gpuWz;
    private GpuTensor<T>? _gpuWr;
    private GpuTensor<T>? _gpuWh;
    private GpuTensor<T>? _gpuUz;
    private GpuTensor<T>? _gpuUr;
    private GpuTensor<T>? _gpuUh;
    private GpuTensor<T>? _gpuBz;
    private GpuTensor<T>? _gpuBr;
    private GpuTensor<T>? _gpuBh;

    // GPU-resident gradient tensors
    private GpuTensor<T>? _gpuWzGradient;
    private GpuTensor<T>? _gpuWrGradient;
    private GpuTensor<T>? _gpuWhGradient;
    private GpuTensor<T>? _gpuUzGradient;
    private GpuTensor<T>? _gpuUrGradient;
    private GpuTensor<T>? _gpuUhGradient;
    private GpuTensor<T>? _gpuBzGradient;
    private GpuTensor<T>? _gpuBrGradient;
    private GpuTensor<T>? _gpuBhGradient;

    // GPU-resident optimizer state tensors (SGD/NAG/LARS velocity)
    private GpuTensor<T>? _gpuWzVelocity;
    private GpuTensor<T>? _gpuWrVelocity;
    private GpuTensor<T>? _gpuWhVelocity;
    private GpuTensor<T>? _gpuUzVelocity;
    private GpuTensor<T>? _gpuUrVelocity;
    private GpuTensor<T>? _gpuUhVelocity;
    private GpuTensor<T>? _gpuBzVelocity;
    private GpuTensor<T>? _gpuBrVelocity;
    private GpuTensor<T>? _gpuBhVelocity;

    // Adam/AdamW M (first moment) tensors
    private GpuTensor<T>? _gpuWzM;
    private GpuTensor<T>? _gpuWrM;
    private GpuTensor<T>? _gpuWhM;
    private GpuTensor<T>? _gpuUzM;
    private GpuTensor<T>? _gpuUrM;
    private GpuTensor<T>? _gpuUhM;
    private GpuTensor<T>? _gpuBzM;
    private GpuTensor<T>? _gpuBrM;
    private GpuTensor<T>? _gpuBhM;

    // Adam/AdamW V (second moment) tensors
    private GpuTensor<T>? _gpuWzV;
    private GpuTensor<T>? _gpuWrV;
    private GpuTensor<T>? _gpuWhV;
    private GpuTensor<T>? _gpuUzV;
    private GpuTensor<T>? _gpuUrV;
    private GpuTensor<T>? _gpuUhV;
    private GpuTensor<T>? _gpuBzV;
    private GpuTensor<T>? _gpuBrV;
    private GpuTensor<T>? _gpuBhV;

    // Cached forward pass state for backpropagation (BPTT)
    private IGpuTensor<T>? _gpuLastInput;
    private IGpuTensor<T>[]? _gpuCachedZGates;
    private IGpuTensor<T>[]? _gpuCachedRGates;
    private IGpuTensor<T>[]? _gpuCachedHCandidates;
    private IGpuTensor<T>[]? _gpuCachedHiddenStates;
    private IGpuTensor<T>? _gpuInitialHiddenState;

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
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims into batch for rank > 3
        Tensor<T> input3D;
        int batchSize;
        int sequenceLength;

        if (rank == 2)
        {
            // 2D input [sequenceLength, inputSize] -> add batch dim
            batchSize = 1;
            sequenceLength = input.Shape[0];
            input3D = input.Reshape([1, sequenceLength, _inputSize]);
        }
        else if (rank == 3)
        {
            // Standard 3D input [batchSize, sequenceLength, inputSize]
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            input3D = input;
        }
        else
        {
            // Higher-rank tensor: collapse leading dims into batch
            sequenceLength = input.Shape[rank - 2];
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            input3D = input.Reshape([flatBatch, sequenceLength, _inputSize]);
        }

        // Cache input only if training
        if (IsTrainingMode)
        {
            _lastInput = input3D;
        }

        // Reset hidden state if needed
        if (_lastHiddenState == null)
        {
            _lastHiddenState = new Tensor<T>([batchSize, _hiddenSize]);
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
            // Vectorized: compute (1 - z) using Tensor operations

            var ones = new Tensor<T>(z.Shape);

            ones.Fill(NumOps.One);

            var oneMinusZ = ones.Subtract(z);


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
    /// Performs the forward pass on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after GRU processing.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable.</exception>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input.Shape;
        int rank = shape.Length;
        int hiddenSize = _hiddenSize;
        int inputSize = _inputSize;

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
            sequenceLength = shape[1]; // [batch, seq, input] -> standard 3D is [batch, seq, input] in this codebase? 
            // Checking Forward: 
            // if (rank == 3) { batchSize = input.Shape[0]; sequenceLength = input.Shape[1]; }
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
        int inputSliceSize = batchSize * _inputSize;
        int outputSize = sequenceLength * batchSize * _hiddenSize;
        int[] outputShape = [batchSize, sequenceLength, _hiddenSize]; // Default for rank 3

        // Fix output shape logic to match existing ForwardGpu
        if (_returnSequences)
        {
            // Already set above
        }
        else
        {
            outputShape = [batchSize, _hiddenSize];
            outputSize = hiddenBufferSize;
        }

        // Cache input for backward pass if training
        if (IsTrainingMode)
        {
            _lastInput = input.ToTensor().Reshape([batchSize, sequenceLength, _inputSize]);
            _originalInputShape = shape;
        }

        // Allocate output buffer
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Upload transposed weights to GPU
        using var WzBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_Wz).ToArray()));
        using var WrBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_Wr).ToArray()));
        using var WhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_Wh).ToArray()));
        using var UzBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_Uz).ToArray()));
        using var UrBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_Ur).ToArray()));
        using var UhBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(Engine.TensorTranspose(_Uh).ToArray()));

        // Upload biases to GPU
        using var biasZBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_bz.ToArray()));
        using var biasRBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_br.ToArray()));
        using var biasHBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_bh.ToArray()));

        // Allocate hidden state buffers
        var currentHBuffer = backend.AllocateBuffer(hiddenBufferSize);
        backend.Fill(currentHBuffer, 0.0f, hiddenBufferSize); // Initialize to zeros

        // Allocate temporary buffers
        using var tempBuffer1 = backend.AllocateBuffer(hiddenBufferSize);
        using var tempBuffer2 = backend.AllocateBuffer(hiddenBufferSize);
        using var zGateBuffer = backend.AllocateBuffer(hiddenBufferSize);
        using var rGateBuffer = backend.AllocateBuffer(hiddenBufferSize);
        using var hCandidateBuffer = backend.AllocateBuffer(hiddenBufferSize);
        using var rGatedHBuffer = backend.AllocateBuffer(hiddenBufferSize);
        using var onesBuffer = backend.AllocateBuffer(hiddenBufferSize);
        using var oneMinusZBuffer = backend.AllocateBuffer(hiddenBufferSize);
        var newHBuffer = backend.AllocateBuffer(hiddenBufferSize);

        // Fill ones buffer
        backend.Fill(onesBuffer, 1.0f, hiddenBufferSize);

        // Lists to store intermediate states for backward pass
        List<float[]>? cachedZ = null;
        List<float[]>? cachedR = null;
        List<float[]>? cachedHCan = null;
        List<float[]>? cachedH = null;

        // Store buffer snapshots for deferred download (keep data on GPU during loop)
        List<IGpuBuffer>? zBufferSnapshots = null;
        List<IGpuBuffer>? rBufferSnapshots = null;
        List<IGpuBuffer>? hCanBufferSnapshots = null;
        List<IGpuBuffer>? hBufferSnapshots = null;

        // Check if we should cache states on GPU for GPU-resident training
        bool cacheForGpuTraining = IsTrainingMode && Engine is DirectGpuTensorEngine;

        if (IsTrainingMode)
        {
            cachedZ = new List<float[]>(sequenceLength);
            cachedR = new List<float[]>(sequenceLength);
            cachedHCan = new List<float[]>(sequenceLength);
            cachedH = new List<float[]>(sequenceLength);

            // Allocate snapshot buffers to store intermediate states on GPU
            zBufferSnapshots = new List<IGpuBuffer>(sequenceLength);
            rBufferSnapshots = new List<IGpuBuffer>(sequenceLength);
            hCanBufferSnapshots = new List<IGpuBuffer>(sequenceLength);
            hBufferSnapshots = new List<IGpuBuffer>(sequenceLength);

            // Clear and prepare GPU training cache for GPU-resident backprop
            if (cacheForGpuTraining)
            {
                ClearGpuTrainingCache();
                _gpuCachedZGates = new IGpuTensor<T>[sequenceLength];
                _gpuCachedRGates = new IGpuTensor<T>[sequenceLength];
                _gpuCachedHCandidates = new IGpuTensor<T>[sequenceLength];
                _gpuCachedHiddenStates = new IGpuTensor<T>[sequenceLength];
                _gpuLastInput = input;

                // Cache initial hidden state (zeros)
                var initHBuffer = backend.AllocateBuffer(hiddenBufferSize);
                backend.Fill(initHBuffer, 0.0f, hiddenBufferSize);
                _gpuInitialHiddenState = new GpuTensor<T>(backend, initHBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
            }
        }

        try
        {
            // Process each time step
            for (int t = 0; t < sequenceLength; t++)
            {
                // Get input slice for this timestep
                int inputSliceOffset = t * inputSliceSize;
                var inputSlice = input.CreateView(inputSliceOffset, [batchSize, _inputSize]);

                // Update Gate: z = sigmoid(x @ Wz^T + h @ Uz^T + bz)
                backend.Gemm(inputSlice.Buffer, WzBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(currentHBuffer, UzBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, zGateBuffer, hiddenBufferSize);
                backend.BiasAdd(zGateBuffer, biasZBuffer, zGateBuffer, batchSize, _hiddenSize);
                backend.Sigmoid(zGateBuffer, zGateBuffer, hiddenBufferSize);

                // Reset Gate: r = sigmoid(x @ Wr^T + h @ Ur^T + br)
                backend.Gemm(inputSlice.Buffer, WrBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(currentHBuffer, UrBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, rGateBuffer, hiddenBufferSize);
                backend.BiasAdd(rGateBuffer, biasRBuffer, rGateBuffer, batchSize, _hiddenSize);
                backend.Sigmoid(rGateBuffer, rGateBuffer, hiddenBufferSize);

                // Candidate Hidden State: h_candidate = tanh(x @ Wh^T + (r * h) @ Uh^T + bh)
                backend.Multiply(rGateBuffer, currentHBuffer, rGatedHBuffer, hiddenBufferSize);
                backend.Gemm(inputSlice.Buffer, WhBuffer, tempBuffer1, batchSize, _hiddenSize, _inputSize);
                backend.Gemm(rGatedHBuffer, UhBuffer, tempBuffer2, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(tempBuffer1, tempBuffer2, hCandidateBuffer, hiddenBufferSize);
                backend.BiasAdd(hCandidateBuffer, biasHBuffer, hCandidateBuffer, batchSize, _hiddenSize);
                backend.Tanh(hCandidateBuffer, hCandidateBuffer, hiddenBufferSize);

                // Final Hidden State: h = z * h_prev + (1 - z) * h_candidate
                // Compute (1 - z)
                backend.Subtract(onesBuffer, zGateBuffer, oneMinusZBuffer, hiddenBufferSize);
                // z * h_prev
                backend.Multiply(zGateBuffer, currentHBuffer, tempBuffer1, hiddenBufferSize);
                // (1 - z) * h_candidate
                backend.Multiply(oneMinusZBuffer, hCandidateBuffer, tempBuffer2, hiddenBufferSize);
                // h = z * h_prev + (1 - z) * h_candidate
                backend.Add(tempBuffer1, tempBuffer2, newHBuffer, hiddenBufferSize);

                // Store hidden state in output if returning sequences
                if (_returnSequences)
                {
                    int outputOffset = t * hiddenBufferSize;
                    backend.Copy2DStrided(newHBuffer, outputBuffer, 1, hiddenBufferSize, outputSize, outputOffset);
                }

                // Cache states if training - copy to GPU snapshot buffers (defer download until after loop)
                if (IsTrainingMode && zBufferSnapshots != null)
                {
                    // Allocate snapshot buffers and copy current state
                    var zSnapshot = backend.AllocateBuffer(hiddenBufferSize);
                    var rSnapshot = backend.AllocateBuffer(hiddenBufferSize);
                    var hCanSnapshot = backend.AllocateBuffer(hiddenBufferSize);
                    var hSnapshot = backend.AllocateBuffer(hiddenBufferSize);

                    backend.Copy(zGateBuffer, zSnapshot, hiddenBufferSize);
                    backend.Copy(rGateBuffer, rSnapshot, hiddenBufferSize);
                    backend.Copy(hCandidateBuffer, hCanSnapshot, hiddenBufferSize);
                    backend.Copy(newHBuffer, hSnapshot, hiddenBufferSize);

                    zBufferSnapshots.Add(zSnapshot);
                    rBufferSnapshots!.Add(rSnapshot);
                    hCanBufferSnapshots!.Add(hCanSnapshot);
                    hBufferSnapshots!.Add(hSnapshot);

                    // Also cache for GPU-resident training (keep separate GpuTensor references)
                    if (cacheForGpuTraining && _gpuCachedZGates is not null && _gpuCachedRGates is not null &&
                        _gpuCachedHCandidates is not null && _gpuCachedHiddenStates is not null)
                    {
                        var zTensorBuffer = backend.AllocateBuffer(hiddenBufferSize);
                        var rTensorBuffer = backend.AllocateBuffer(hiddenBufferSize);
                        var hCanTensorBuffer = backend.AllocateBuffer(hiddenBufferSize);
                        var hTensorBuffer = backend.AllocateBuffer(hiddenBufferSize);

                        backend.Copy(zGateBuffer, zTensorBuffer, hiddenBufferSize);
                        backend.Copy(rGateBuffer, rTensorBuffer, hiddenBufferSize);
                        backend.Copy(hCandidateBuffer, hCanTensorBuffer, hiddenBufferSize);
                        backend.Copy(newHBuffer, hTensorBuffer, hiddenBufferSize);

                        _gpuCachedZGates[t] = new GpuTensor<T>(backend, zTensorBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                        _gpuCachedRGates[t] = new GpuTensor<T>(backend, rTensorBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                        _gpuCachedHCandidates[t] = new GpuTensor<T>(backend, hCanTensorBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                        _gpuCachedHiddenStates[t] = new GpuTensor<T>(backend, hTensorBuffer, [batchSize, _hiddenSize], GpuTensorRole.Activation, ownsBuffer: true);
                    }
                }

                // Swap hidden state buffers
                var tempH = currentHBuffer;
                currentHBuffer = newHBuffer;
                newHBuffer = tempH;
            }

            // If not returning sequences, copy final hidden state to output
            if (!_returnSequences)
            {
                backend.Copy(currentHBuffer, outputBuffer, hiddenBufferSize);
            }

            // Batch download all cached states AFTER the compute loop completes
            // This keeps all data on GPU during the loop for better performance
            if (IsTrainingMode && zBufferSnapshots != null && sequenceLength > 0)
            {
                // Download all snapshots in sequence (GPU work is done, now we transfer)
                for (int t = 0; t < sequenceLength; t++)
                {
                    cachedZ!.Add(backend.DownloadBuffer(zBufferSnapshots[t]));
                    cachedR!.Add(backend.DownloadBuffer(rBufferSnapshots![t]));
                    cachedHCan!.Add(backend.DownloadBuffer(hCanBufferSnapshots![t]));
                    cachedH!.Add(backend.DownloadBuffer(hBufferSnapshots![t]));

                    // Dispose snapshot buffers after download
                    zBufferSnapshots[t].Dispose();
                    rBufferSnapshots[t].Dispose();
                    hCanBufferSnapshots[t].Dispose();
                    hBufferSnapshots[t].Dispose();
                }
            }

            // Reconstruct CPU tensors for backward pass if training
            if (IsTrainingMode && cachedZ != null && cachedZ.Count > 0)
            {
                // Store last timestep activations for single-step backward compatibility
                _lastZ = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(cachedZ[sequenceLength - 1]), [batchSize, _hiddenSize]);
                _lastR = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(cachedR![sequenceLength - 1]), [batchSize, _hiddenSize]);
                _lastH = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(cachedHCan![sequenceLength - 1]), [batchSize, _hiddenSize]);
                _lastHiddenState = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(cachedH![sequenceLength - 1]), [batchSize, _hiddenSize]);

                // Store full sequence of hidden states if needed
                if (_returnSequences || _allHiddenStates != null)
                {
                    _allHiddenStates = new List<Tensor<T>>(sequenceLength);
                    foreach (var hData in cachedH)
                    {
                        _allHiddenStates.Add(new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(hData), [batchSize, _hiddenSize]));
                    }
                }
            }

            // Dispose the buffer we're not returning
            newHBuffer.Dispose();
            // currentHBuffer is the last hidden state, keep it if needed for internal state tracking
            currentHBuffer.Dispose();

            return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            // Clean up on error
            outputBuffer.Dispose();
            currentHBuffer.Dispose();
            newHBuffer.Dispose();

            // Clean up snapshot buffers on error
            if (zBufferSnapshots != null)
            {
                foreach (var buf in zBufferSnapshots) buf?.Dispose();
                foreach (var buf in rBufferSnapshots!) buf?.Dispose();
                foreach (var buf in hCanBufferSnapshots!) buf?.Dispose();
                foreach (var buf in hBufferSnapshots!) buf?.Dispose();
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
    /// Performs the backward pass of the GRU layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the GRU layer, which is used during training to propagate
    /// error gradients back through the network. It calculates the gradients for all the weights and biases,
    /// and returns the gradient with respect to the layer's input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    ///
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change to reduce the overall error
    /// 2. It calculates how each of its weights and biases should change to produce better output
    /// 3. It calculates how its input should change, which will be used by earlier layers
    ///
    /// This complex calculation essentially runs the GRU's logic in reverse, tracking how
    /// changes to the output would affect each internal part of the layer.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using Backpropagation Through Time (BPTT) for GRU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using manual gradient calculations optimized for
    /// GRU networks. It performs backpropagation through time (BPTT), processing the
    /// sequence in reverse order and computing gradients for all gate parameters (reset, update)
    /// and hidden states.
    /// </para>
    /// <para>
    /// Autodiff Note: GRU backward pass involves complex interactions between reset and update gates.
    /// Implementing this with automatic differentiation would require handling temporal dependencies
    /// and gate-specific gradient flows. The manual implementation provides efficient and correct
    /// gradient calculations for all GRU components.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastZ == null || _lastR == null || _lastH == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];

        // Initialize gradients
        var dWz = new Tensor<T>([_hiddenSize, _inputSize]);
        var dWr = new Tensor<T>([_hiddenSize, _inputSize]);
        var dWh = new Tensor<T>([_hiddenSize, _inputSize]);
        var dUz = new Tensor<T>([_hiddenSize, _hiddenSize]);
        var dUr = new Tensor<T>([_hiddenSize, _hiddenSize]);
        var dUh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        var dbz = new Tensor<T>([_hiddenSize]);
        var dbr = new Tensor<T>([_hiddenSize]);
        var dbh = new Tensor<T>([_hiddenSize]);

        // Initialize input gradients
        var dInputs = new Tensor<T>([batchSize, sequenceLength, _inputSize]);

        // If we don't have sequence data, just process the final time step
        if (!_returnSequences || _allHiddenStates == null || _allHiddenStates.Count == 0)
        {
            // Handle single timestep case (simple backward pass)
            var dh = outputGradient;
            // Vectorized: compute (1 - _lastZ) using Tensor operations
            var ones1 = new Tensor<T>(_lastZ.Shape);
            ones1.Fill(NumOps.One);
            var oneMinusLastZ = ones1.Subtract(_lastZ);

            var dh_candidate = dh.ElementwiseMultiply(oneMinusLastZ);
            var dz = dh.ElementwiseMultiply(_lastHiddenState.Subtract(_lastH));

            var dr = ApplyActivationDerivative(_lastH, isRecurrent: false)
                .ElementwiseMultiply(dh_candidate)
                .ElementwiseMultiply(_lastHiddenState.Multiply(_Uh));

            // dx = dz @ Wz + dr @ Wr + dh_candidate @ Wh
            // Wz is [hiddenSize, inputSize], so dz @ Wz = [batch, hidden] @ [hidden, input] = [batch, input]
            var dx = dz.Multiply(_Wz)
                    .Add(dr.Multiply(_Wr))
                    .Add(dh_candidate.Multiply(_Wh));

            // Reshape dx to match input format for the last timestep using Engine.TensorSetSlice
            var dxReshaped = dx.Reshape([batchSize, 1, _inputSize]);
            dInputs = Engine.TensorSetSlice(dInputs, dxReshaped, [0, sequenceLength - 1, 0]);

            // Calculate gradients for weights and biases
            dWz = _lastInput.Slice(1, sequenceLength - 1, sequenceLength)
                            .Reshape([batchSize, _inputSize])
                            .Transpose([1, 0])
                            .Multiply(dz);
            dWr = _lastInput.Slice(1, sequenceLength - 1, sequenceLength)
                            .Reshape([batchSize, _inputSize])
                            .Transpose([1, 0])
                            .Multiply(dr);
            dWh = _lastInput.Slice(1, sequenceLength - 1, sequenceLength)
                            .Reshape([batchSize, _inputSize])
                            .Transpose([1, 0])
                            .Multiply(dh_candidate);

            // For U gradients, we need the previous hidden state
            // If not available, use zeros
            Tensor<T> prevHidden;
            if (sequenceLength > 1 && _allHiddenStates != null && _allHiddenStates.Count >= sequenceLength - 1)
            {
                prevHidden = _allHiddenStates[sequenceLength - 2];
            }
            else
            {
                prevHidden = new Tensor<T>([batchSize, _hiddenSize]);
            }

            dUz = prevHidden.Transpose([1, 0]).Multiply(dz);
            dUr = prevHidden.Transpose([1, 0]).Multiply(dr);
            dUh = prevHidden.Transpose([1, 0]).Multiply(dh_candidate.ElementwiseMultiply(_lastR));

            dbz = dz.Sum([0]);
            dbr = dr.Sum([0]);
            dbh = dh_candidate.Sum([0]);
        }
        else
        {
            // For returning sequences, we need to backpropagate through all time steps

            // Split the output gradient into time steps if returningSequences is true
            Tensor<T>[] dhOutSequence = new Tensor<T>[sequenceLength];
            if (_returnSequences)
            {
                // Split the gradient for each time step
                for (int t = 0; t < sequenceLength; t++)
                {
                    dhOutSequence[t] = outputGradient.Slice(1, t, t + 1).Reshape([batchSize, _hiddenSize]);
                }
            }
            else
            {
                // If not returning sequences, only the last time step has gradient from output
                for (int t = 0; t < sequenceLength - 1; t++)
                {
                    dhOutSequence[t] = new Tensor<T>([batchSize, _hiddenSize]);
                }
                dhOutSequence[sequenceLength - 1] = outputGradient;
            }

            // Initialize hidden state gradient for the next timestep
            var dhNext = new Tensor<T>([batchSize, _hiddenSize]);

            // Recompute hidden states and activations for each time step (needed for backward pass)
            List<Tensor<T>> timeStepHidden = new List<Tensor<T>>(sequenceLength);
            List<Tensor<T>> timeStepZ = new List<Tensor<T>>(sequenceLength);
            List<Tensor<T>> timeStepR = new List<Tensor<T>>(sequenceLength);
            List<Tensor<T>> timeStepHCandidate = new List<Tensor<T>>(sequenceLength);
            var currentH = new Tensor<T>([batchSize, _hiddenSize]);

            for (int t = 0; t < sequenceLength; t++)
            {
                var xt = _lastInput.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);

                var z = ComputeGate(xt, currentH, _Wz, _Uz, _bz, true);
                var r = ComputeGate(xt, currentH, _Wr, _Ur, _br, true);
                var h_candidate = ComputeGate(xt, r.ElementwiseMultiply(currentH), _Wh, _Uh, _bh, false);

                // Vectorized: compute (1 - z) using Tensor operations
                var ones2 = new Tensor<T>(z.Shape);
                ones2.Fill(NumOps.One);
                var oneMinusZ2 = ones2.Subtract(z);

                var newH = z.ElementwiseMultiply(currentH).Add(
                    oneMinusZ2.ElementwiseMultiply(h_candidate)
                );

                currentH = newH;

                // Add to lists (not assign by index)
                timeStepHidden.Add(currentH.Clone());
                timeStepZ.Add(z.Clone());
                timeStepR.Add(r.Clone());
                timeStepHCandidate.Add(h_candidate.Clone());
            }

            // Backpropagate through time
            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                // Combine gradient from output and from next timestep
                var dh = dhNext.Add(dhOutSequence[t]);

                // Get activations for this timestep
                var xt = _lastInput.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);
                var h = timeStepHidden[t];
                var z = timeStepZ[t];
                var r = timeStepR[t];
                var h_candidate = timeStepHCandidate[t];

                // Get previous hidden state (or zeros for t=0)
                var h_prev = t > 0 ? timeStepHidden[t - 1] : new Tensor<T>([batchSize, _hiddenSize]);

                // Calculate gradients for this timestep
                // Vectorized: compute (1 - z) using Tensor operations
                var ones3 = new Tensor<T>(z.Shape);
                ones3.Fill(NumOps.One);
                var oneMinusZ3 = ones3.Subtract(z);

                var dh_candidate = dh.ElementwiseMultiply(oneMinusZ3);
                var dz = dh.ElementwiseMultiply(h_prev.Subtract(h_candidate));

                var dr = ApplyActivationDerivative(h_candidate, isRecurrent: false)
                    .ElementwiseMultiply(dh_candidate)
                    .ElementwiseMultiply(h_prev.Multiply(_Uh));

                // Input gradient for this timestep: dxt = dz @ Wz + dr @ Wr + dh_candidate @ Wh
                // Wz is [hiddenSize, inputSize], so dz @ Wz = [batch, hidden] @ [hidden, input] = [batch, input]
                var dxt = dz.Multiply(_Wz)
                        .Add(dr.Multiply(_Wr))
                        .Add(dh_candidate.Multiply(_Wh));

                // Store input gradients using Engine.TensorSetSlice
                var dxtReshaped = dxt.Reshape([batchSize, 1, _inputSize]);
                dInputs = Engine.TensorSetSlice(dInputs, dxtReshaped, [0, t, 0]);

                // Gradient for next timestep's hidden state
                dhNext = dz.Multiply(_Uz.Transpose())
                        .Add(dr.Multiply(_Ur.Transpose()))
                        .Add(dh_candidate.ElementwiseMultiply(r).Multiply(_Uh.Transpose()));

                // Accumulate weight gradients
                dWz = dWz.Add(xt.Transpose([1, 0]).Multiply(dz));
                dWr = dWr.Add(xt.Transpose([1, 0]).Multiply(dr));
                dWh = dWh.Add(xt.Transpose([1, 0]).Multiply(dh_candidate));

                dUz = dUz.Add(h_prev.Transpose([1, 0]).Multiply(dz));
                dUr = dUr.Add(h_prev.Transpose([1, 0]).Multiply(dr));
                dUh = dUh.Add(h_prev.Transpose([1, 0]).Multiply(dh_candidate.ElementwiseMultiply(r)));

                dbz = dbz.Add(dz.Sum([0]));
                dbr = dbr.Add(dr.Sum([0]));
                dbh = dbh.Add(dh_candidate.Sum([0]));
            }
        }

        // Store gradients for use in UpdateParameters
        _dWz = dWz; _dWr = dWr; _dWh = dWh;
        _dUz = dUz; _dUr = dUr; _dUh = dUh;
        _dbz = dbz; _dbr = dbr; _dbh = dbh;

        return dInputs;
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
                _WzVelocity = new Tensor<T>(_Wz.Shape); _WzVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_WzVelocity, PersistentTensorRole.OptimizerState);
                _WrVelocity = new Tensor<T>(_Wr.Shape); _WrVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_WrVelocity, PersistentTensorRole.OptimizerState);
                _WhVelocity = new Tensor<T>(_Wh.Shape); _WhVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_WhVelocity, PersistentTensorRole.OptimizerState);
                _UzVelocity = new Tensor<T>(_Uz.Shape); _UzVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_UzVelocity, PersistentTensorRole.OptimizerState);
                _UrVelocity = new Tensor<T>(_Ur.Shape); _UrVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_UrVelocity, PersistentTensorRole.OptimizerState);
                _UhVelocity = new Tensor<T>(_Uh.Shape); _UhVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_UhVelocity, PersistentTensorRole.OptimizerState);
                _bzVelocity = new Tensor<T>(_bz.Shape); _bzVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_bzVelocity, PersistentTensorRole.OptimizerState);
                _brVelocity = new Tensor<T>(_br.Shape); _brVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_brVelocity, PersistentTensorRole.OptimizerState);
                _bhVelocity = new Tensor<T>(_bh.Shape); _bhVelocity.Fill(NumOps.Zero); gpuEngine.RegisterPersistentTensor(_bhVelocity, PersistentTensorRole.OptimizerState);
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
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate for production-grade parameter extraction
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
    /// Exports the GRU layer's single time-step computation as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the hidden state at one time step.</returns>
    /// <remarks>
    /// <para>
    /// This method exports a single GRU cell computation for JIT compilation.
    /// The graph computes: h_t = GRUCell(x_t, h_{t-1})
    /// using the standard GRU equations with update gate, reset gate, and candidate hidden state.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create placeholders for single time-step inputs
        // x_t shape: [batchSize, inputSize]
        var inputPlaceholder = new Tensor<T>(new int[] { 1, _inputSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "x_t");

        // h_{t-1} shape: [batchSize, hiddenSize]
        var prevHiddenPlaceholder = new Tensor<T>(new int[] { 1, _hiddenSize });
        var prevHiddenNode = TensorOperations<T>.Variable(prevHiddenPlaceholder, "h_prev");

        // Create weight and bias nodes (storage is already Tensor<T>)
        var WzNode = TensorOperations<T>.Variable(_Wz, "W_z");
        var WrNode = TensorOperations<T>.Variable(_Wr, "W_r");
        var WhNode = TensorOperations<T>.Variable(_Wh, "W_h");
        var UzNode = TensorOperations<T>.Variable(_Uz, "U_z");
        var UrNode = TensorOperations<T>.Variable(_Ur, "U_r");
        var UhNode = TensorOperations<T>.Variable(_Uh, "U_h");
        var bzNode = TensorOperations<T>.Variable(_bz, "b_z");
        var brNode = TensorOperations<T>.Variable(_br, "b_r");
        var bhNode = TensorOperations<T>.Variable(_bh, "b_h");

        // Add inputs to the list
        inputNodes.Add(inputNode);
        inputNodes.Add(prevHiddenNode);
        inputNodes.Add(WzNode);
        inputNodes.Add(WrNode);
        inputNodes.Add(WhNode);
        inputNodes.Add(UzNode);
        inputNodes.Add(UrNode);
        inputNodes.Add(UhNode);
        inputNodes.Add(bzNode);
        inputNodes.Add(brNode);
        inputNodes.Add(bhNode);

        // Build GRU computation graph (single time step)
        // Update gate: z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)
        var WzT = TensorOperations<T>.Transpose(WzNode);
        var UzT = TensorOperations<T>.Transpose(UzNode);
        var z_input = TensorOperations<T>.MatrixMultiply(inputNode, WzT);
        var z_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, UzT);
        var z_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(z_input, z_hidden), bzNode);
        var z_t = TensorOperations<T>.Sigmoid(z_preact);

        // Reset gate: r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)
        var WrT = TensorOperations<T>.Transpose(WrNode);
        var UrT = TensorOperations<T>.Transpose(UrNode);
        var r_input = TensorOperations<T>.MatrixMultiply(inputNode, WrT);
        var r_hidden = TensorOperations<T>.MatrixMultiply(prevHiddenNode, UrT);
        var r_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(r_input, r_hidden), brNode);
        var r_t = TensorOperations<T>.Sigmoid(r_preact);

        // Candidate hidden state: h_candidate = tanh(W_h @ x_t + U_h @ (r_t ⊙ h_{t-1}) + b_h)
        var WhT = TensorOperations<T>.Transpose(WhNode);
        var UhT = TensorOperations<T>.Transpose(UhNode);
        var h_input = TensorOperations<T>.MatrixMultiply(inputNode, WhT);
        var r_gated = TensorOperations<T>.ElementwiseMultiply(r_t, prevHiddenNode);
        var h_hidden = TensorOperations<T>.MatrixMultiply(r_gated, UhT);
        var h_preact = TensorOperations<T>.Add(TensorOperations<T>.Add(h_input, h_hidden), bhNode);
        var h_candidate = TensorOperations<T>.Tanh(h_preact);

        // Final hidden state: h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h_candidate
        var z_gated = TensorOperations<T>.ElementwiseMultiply(z_t, prevHiddenNode);

        // Compute (1 - z_t)
        var onesTensor = new Tensor<T>(new int[] { 1, _hiddenSize });
        onesTensor.Fill(NumOps.One);
        var onesNode = TensorOperations<T>.Constant(onesTensor);
        var one_minus_z = TensorOperations<T>.Subtract(onesNode, z_t);

        var candidate_gated = TensorOperations<T>.ElementwiseMultiply(one_minus_z, h_candidate);
        var h_t = TensorOperations<T>.Add(z_gated, candidate_gated);

        return h_t;
    }

    /// <summary>
    /// Gets whether this layer currently supports JIT compilation.
    /// </summary>
    /// <value>
    /// True for GRU layers, as single time-step JIT compilation is supported.
    /// </value>
    public override bool SupportsJitCompilation => true;

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
    /// Backward pass implementation using automatic differentiation with full BPTT.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements full Backpropagation Through Time (BPTT) using automatic differentiation.
    /// It builds a complete computation graph across all timesteps and backpropagates through the
    /// entire sequence, correctly computing gradients for all parameters and inputs.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];

        // Initialize gradient accumulators
        _dWz = new Tensor<T>([_hiddenSize, _inputSize]);
        _dWr = new Tensor<T>([_hiddenSize, _inputSize]);
        _dWh = new Tensor<T>([_hiddenSize, _inputSize]);
        _dUz = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _dUr = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _dUh = new Tensor<T>([_hiddenSize, _hiddenSize]);
        _dbz = new Tensor<T>([_hiddenSize]);
        _dbr = new Tensor<T>([_hiddenSize]);
        _dbh = new Tensor<T>([_hiddenSize]);

        // Initialize input gradients tensor
        var dInputs = new Tensor<T>([batchSize, sequenceLength, _inputSize]);

        // Create input nodes for each timestep
        var inputNodes = new List<ComputationNode<T>>(sequenceLength);
        for (int t = 0; t < sequenceLength; t++)
        {
            var xt = _lastInput.Slice(1, t, t + 1).Reshape([batchSize, _inputSize]);
            inputNodes.Add(Autodiff.TensorOperations<T>.Variable(xt, $"input_t{t}", requiresGradient: true));
        }

        // Create parameter nodes (shared across all timesteps)
        var WzNode = Autodiff.TensorOperations<T>.Variable(_Wz, "Wz", requiresGradient: true);
        var WrNode = Autodiff.TensorOperations<T>.Variable(_Wr, "Wr", requiresGradient: true);
        var WhNode = Autodiff.TensorOperations<T>.Variable(_Wh, "Wh", requiresGradient: true);
        var UzNode = Autodiff.TensorOperations<T>.Variable(_Uz, "Uz", requiresGradient: true);
        var UrNode = Autodiff.TensorOperations<T>.Variable(_Ur, "Ur", requiresGradient: true);
        var UhNode = Autodiff.TensorOperations<T>.Variable(_Uh, "Uh", requiresGradient: true);
        var bzNode = Autodiff.TensorOperations<T>.Variable(_bz, "bz", requiresGradient: true);
        var brNode = Autodiff.TensorOperations<T>.Variable(_br, "br", requiresGradient: true);
        var bhNode = Autodiff.TensorOperations<T>.Variable(_bh, "bh", requiresGradient: true);

        // Transpose weight matrices once (shared across timesteps)
        var WzT = Autodiff.TensorOperations<T>.Transpose(WzNode);
        var WrT = Autodiff.TensorOperations<T>.Transpose(WrNode);
        var WhT = Autodiff.TensorOperations<T>.Transpose(WhNode);
        var UzT = Autodiff.TensorOperations<T>.Transpose(UzNode);
        var UrT = Autodiff.TensorOperations<T>.Transpose(UrNode);
        var UhT = Autodiff.TensorOperations<T>.Transpose(UhNode);

        // Broadcast biases for batch operations
        var bzBroadcast = Autodiff.TensorOperations<T>.Variable(
            BroadcastVector(_bz, batchSize), "bz_broadcast", requiresGradient: false);
        var brBroadcast = Autodiff.TensorOperations<T>.Variable(
            BroadcastVector(_br, batchSize), "br_broadcast", requiresGradient: false);
        var bhBroadcast = Autodiff.TensorOperations<T>.Variable(
            BroadcastVector(_bh, batchSize), "bh_broadcast", requiresGradient: false);

        // Build computation graph through all timesteps (forward unrolling)
        var hiddenStates = new List<ComputationNode<T>>(sequenceLength + 1);

        // Initial hidden state (zeros)
        var h0 = new Tensor<T>([batchSize, _hiddenSize]);
        hiddenStates.Add(Autodiff.TensorOperations<T>.Variable(h0, "h0", requiresGradient: false));

        // Process each timestep, building the full computation graph
        for (int t = 0; t < sequenceLength; t++)
        {
            var xt = inputNodes[t];
            var hPrev = hiddenStates[t];

            // Compute update gate: z = sigmoid(xt @ Wz.T + h_prev @ Uz.T + bz)
            var z_input = Autodiff.TensorOperations<T>.MatrixMultiply(xt, WzT);
            var z_hidden = Autodiff.TensorOperations<T>.MatrixMultiply(hPrev, UzT);
            var z_sum = Autodiff.TensorOperations<T>.Add(z_input, z_hidden);
            var z_preact = Autodiff.TensorOperations<T>.Add(z_sum, bzBroadcast);
            var z = Autodiff.TensorOperations<T>.Sigmoid(z_preact);

            // Compute reset gate: r = sigmoid(xt @ Wr.T + h_prev @ Ur.T + br)
            var r_input = Autodiff.TensorOperations<T>.MatrixMultiply(xt, WrT);
            var r_hidden = Autodiff.TensorOperations<T>.MatrixMultiply(hPrev, UrT);
            var r_sum = Autodiff.TensorOperations<T>.Add(r_input, r_hidden);
            var r_preact = Autodiff.TensorOperations<T>.Add(r_sum, brBroadcast);
            var r = Autodiff.TensorOperations<T>.Sigmoid(r_preact);

            // Compute candidate hidden state: h_candidate = tanh(xt @ Wh.T + (r * h_prev) @ Uh.T + bh)
            var h_input = Autodiff.TensorOperations<T>.MatrixMultiply(xt, WhT);
            var r_gated = Autodiff.TensorOperations<T>.ElementwiseMultiply(r, hPrev);
            var h_hidden = Autodiff.TensorOperations<T>.MatrixMultiply(r_gated, UhT);
            var h_sum = Autodiff.TensorOperations<T>.Add(h_input, h_hidden);
            var h_preact = Autodiff.TensorOperations<T>.Add(h_sum, bhBroadcast);
            var h_candidate = Autodiff.TensorOperations<T>.Tanh(h_preact);

            // Compute final hidden state: h = z * h_prev + (1 - z) * h_candidate
            var z_gated = Autodiff.TensorOperations<T>.ElementwiseMultiply(z, hPrev);
            var onesT = CreateOnesLike(z.Value);
            var onesNode = Autodiff.TensorOperations<T>.Constant(onesT);
            var one_minus_z = Autodiff.TensorOperations<T>.Subtract(onesNode, z);
            var candidate_gated = Autodiff.TensorOperations<T>.ElementwiseMultiply(one_minus_z, h_candidate);
            var h_new = Autodiff.TensorOperations<T>.Add(z_gated, candidate_gated);

            hiddenStates.Add(h_new);
        }

        // Determine output node(s) and set gradients
        if (_returnSequences)
        {
            // For return sequences, we need to backprop from all hidden states
            // Split the output gradient for each timestep
            for (int t = 0; t < sequenceLength; t++)
            {
                var gradT = outputGradient.Slice(1, t, t + 1).Reshape([batchSize, _hiddenSize]);
                var hNode = hiddenStates[t + 1]; // +1 because hiddenStates[0] is h0

                // Accumulate gradients if already set
                if (hNode.Gradient == null)
                {
                    hNode.Gradient = gradT;
                }
                else
                {
                    hNode.Gradient = Engine.TensorAdd(hNode.Gradient, gradT);
                }
            }
        }
        else
        {
            // For non-sequence output, only the final hidden state receives gradient
            var finalHidden = hiddenStates[sequenceLength];
            var outputGradFlat = outputGradient.Shape.Length == 2
                ? outputGradient
                : outputGradient.Reshape([batchSize, _hiddenSize]);
            finalHidden.Gradient = outputGradFlat;
        }

        // Perform topological sort and backward pass (inlined)
        // Start from all nodes that have gradients set
        var visited = new HashSet<ComputationNode<T>>();
        var topoOrder = new List<ComputationNode<T>>();
        var topoStack = new Stack<(ComputationNode<T> node, bool processed)>();

        // Add all output nodes to the stack
        for (int t = sequenceLength; t >= 1; t--)
        {
            if (hiddenStates[t].Gradient != null && !visited.Contains(hiddenStates[t]))
            {
                topoStack.Push((hiddenStates[t], false));
            }
        }

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

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract parameter gradients (accumulated across all timesteps via autodiff)
        if (WzNode.Gradient != null) _dWz = WzNode.Gradient;
        if (WrNode.Gradient != null) _dWr = WrNode.Gradient;
        if (WhNode.Gradient != null) _dWh = WhNode.Gradient;
        if (UzNode.Gradient != null) _dUz = UzNode.Gradient;
        if (UrNode.Gradient != null) _dUr = UrNode.Gradient;
        if (UhNode.Gradient != null) _dUh = UhNode.Gradient;
        if (bzNode.Gradient != null) _dbz = bzNode.Gradient.Sum([0]);
        if (brNode.Gradient != null) _dbr = brNode.Gradient.Sum([0]);
        if (bhNode.Gradient != null) _dbh = bhNode.Gradient.Sum([0]);

        // Extract input gradients for each timestep using Engine.TensorSetSlice
        for (int t = 0; t < sequenceLength; t++)
        {
            var inputGrad = inputNodes[t].Gradient;
            if (inputGrad != null)
            {
                var inputGradReshaped = inputGrad.Reshape([batchSize, 1, _inputSize]);
                dInputs = Engine.TensorSetSlice(dInputs, inputGradReshaped, [0, t, 0]);
            }
        }

        return dInputs;
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
        var ones = new Tensor<T>(tensor.Shape);
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
    }

    /// <summary>
    /// GPU-resident backward pass implementing Backpropagation Through Time (BPTT).
    /// Computes gradients for all weights and biases directly on GPU.
    /// </summary>
    /// <param name="outputGradient">Gradient of the loss with respect to the layer output.</param>
    /// <returns>Gradient of the loss with respect to the layer input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuLastInput == null || _gpuCachedZGates == null || _gpuCachedRGates == null ||
            _gpuCachedHCandidates == null || _gpuCachedHiddenStates == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        int batchSize = _gpuLastInput.Shape[0];
        int timeSteps = _gpuCachedZGates.Length;
        int hiddenBufferSize = batchSize * _hiddenSize;
        int inputBufferSize = batchSize * _inputSize;

        // Initialize gradient accumulators for weights (zeros)
        var dWzBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dWrBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dWhBuffer = backend.AllocateBuffer(_hiddenSize * _inputSize);
        var dUzBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dUrBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dUhBuffer = backend.AllocateBuffer(_hiddenSize * _hiddenSize);
        var dBzBuffer = backend.AllocateBuffer(_hiddenSize);
        var dBrBuffer = backend.AllocateBuffer(_hiddenSize);
        var dBhBuffer = backend.AllocateBuffer(_hiddenSize);

        backend.Fill(dWzBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dWrBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dWhBuffer, 0.0f, _hiddenSize * _inputSize);
        backend.Fill(dUzBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dUrBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dUhBuffer, 0.0f, _hiddenSize * _hiddenSize);
        backend.Fill(dBzBuffer, 0.0f, _hiddenSize);
        backend.Fill(dBrBuffer, 0.0f, _hiddenSize);
        backend.Fill(dBhBuffer, 0.0f, _hiddenSize);

        // Allocate input gradient buffer
        int inputGradientSize = batchSize * timeSteps * _inputSize;
        var inputGradientBuffer = backend.AllocateBuffer(inputGradientSize);
        backend.Fill(inputGradientBuffer, 0.0f, inputGradientSize);

        // Upload transposed weights to GPU for backward matmul
        using var WzT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_Wz.ToArray()));
        using var WrT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_Wr.ToArray()));
        using var WhT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_Wh.ToArray()));
        using var UzT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_Uz.ToArray()));
        using var UrT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_Ur.ToArray()));
        using var UhT = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_Uh.ToArray()));

        // Temporary buffers
        var dHNext = backend.AllocateBuffer(hiddenBufferSize);
        var tempBuffer1 = backend.AllocateBuffer(hiddenBufferSize);
        var tempBuffer2 = backend.AllocateBuffer(hiddenBufferSize);
        var dZ = backend.AllocateBuffer(hiddenBufferSize);
        var dR = backend.AllocateBuffer(hiddenBufferSize);
        var dHCandidate = backend.AllocateBuffer(hiddenBufferSize);
        var dHPrev = backend.AllocateBuffer(hiddenBufferSize);

        // Additional buffers for transpose operations
        var inputTranspose = backend.AllocateBuffer(inputBufferSize);
        var hiddenTranspose = backend.AllocateBuffer(hiddenBufferSize);
        var gateGradTranspose = backend.AllocateBuffer(hiddenBufferSize);
        int maxWeightSize = Math.Max(_inputSize, _hiddenSize) * _hiddenSize;
        var weightGradBuffer = backend.AllocateBuffer(maxWeightSize);
        var inputGradTemp = backend.AllocateBuffer(inputBufferSize);
        var hiddenGradTemp = backend.AllocateBuffer(hiddenBufferSize);

        backend.Fill(dHNext, 0.0f, hiddenBufferSize);

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
                var z_t = _gpuCachedZGates[t];
                var r_t = _gpuCachedRGates[t];
                var hCan_t = _gpuCachedHCandidates[t];
                var h_t = _gpuCachedHiddenStates[t];

                // Get previous hidden state
                IGpuTensor<T> h_prev = t > 0 ? _gpuCachedHiddenStates[t - 1] : _gpuInitialHiddenState!;

                // GRU Backward equations:
                // h = z * h_prev + (1 - z) * h_candidate
                // dh_candidate = dh * (1 - z) * tanh'(h_candidate)
                // dz = dh * (h_prev - h_candidate) * sigmoid'(z)
                // dh_prev += dh * z + dh_candidate_through_r
                // dr = d(r * h_prev) @ Uh^T * tanh_grad @ Wh^T (complex)

                // d(h_candidate) = dh * (1 - z) * tanh'(h_candidate)
                // First compute (1 - z)
                using var onesBuffer = backend.AllocateBuffer(hiddenBufferSize);
                backend.Fill(onesBuffer, 1.0f, hiddenBufferSize);
                backend.Subtract(onesBuffer, z_t.Buffer, tempBuffer1, hiddenBufferSize); // tempBuffer1 = 1 - z
                backend.Multiply(dHNext, tempBuffer1, tempBuffer2, hiddenBufferSize); // tempBuffer2 = dh * (1 - z)
                backend.TanhBackward(tempBuffer2, hCan_t.Buffer, dHCandidate, hiddenBufferSize); // dHCandidate = tempBuffer2 * tanh'(hCan)

                // dz = dh * (h_prev - h_candidate) * sigmoid'(z)
                backend.Subtract(h_prev.Buffer, hCan_t.Buffer, tempBuffer1, hiddenBufferSize); // tempBuffer1 = h_prev - h_candidate
                backend.Multiply(dHNext, tempBuffer1, tempBuffer2, hiddenBufferSize); // tempBuffer2 = dh * (h_prev - h_candidate)
                backend.SigmoidBackward(tempBuffer2, z_t.Buffer, dZ, hiddenBufferSize); // dZ = tempBuffer2 * sigmoid'(z)

                // dr = (dHCandidate @ Uh^T) * h_prev * sigmoid'(r)
                // The gradient flows through: h_candidate = tanh(x @ Wh + (r * h_prev) @ Uh + bh)
                // d(r * h_prev) = dHCandidate from tanh -> matmul with Uh
                backend.Gemm(dHCandidate, UhT, tempBuffer1, batchSize, _hiddenSize, _hiddenSize);
                backend.Multiply(tempBuffer1, h_prev.Buffer, tempBuffer2, hiddenBufferSize); // tempBuffer2 = gradient * h_prev
                backend.SigmoidBackward(tempBuffer2, r_t.Buffer, dR, hiddenBufferSize); // dR = tempBuffer2 * sigmoid'(r)

                // Get input for this timestep
                int inputOffset = t * inputBufferSize;
                var inputSlice = _gpuLastInput.CreateView(inputOffset, [batchSize, _inputSize]);

                // Accumulate weight gradients using outer products
                // dWz += input^T @ dZ
                backend.Transpose(inputSlice.Buffer, inputTranspose, batchSize, _inputSize);
                backend.Gemm(inputTranspose, dZ, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWzBuffer, weightGradBuffer, dWzBuffer, _hiddenSize * _inputSize);

                // dUz += h_prev^T @ dZ
                backend.Transpose(h_prev.Buffer, hiddenTranspose, batchSize, _hiddenSize);
                backend.Gemm(hiddenTranspose, dZ, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dUzBuffer, weightGradBuffer, dUzBuffer, _hiddenSize * _hiddenSize);

                // dBz += sum(dZ, axis=0)
                backend.Transpose(dZ, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBzBuffer, tempBuffer1, dBzBuffer, _hiddenSize);

                // dWr += input^T @ dR
                backend.Gemm(inputTranspose, dR, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWrBuffer, weightGradBuffer, dWrBuffer, _hiddenSize * _inputSize);

                // dUr += h_prev^T @ dR
                backend.Gemm(hiddenTranspose, dR, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dUrBuffer, weightGradBuffer, dUrBuffer, _hiddenSize * _hiddenSize);

                // dBr += sum(dR, axis=0)
                backend.Transpose(dR, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBrBuffer, tempBuffer1, dBrBuffer, _hiddenSize);

                // dWh += input^T @ dHCandidate
                backend.Gemm(inputTranspose, dHCandidate, weightGradBuffer, _inputSize, _hiddenSize, batchSize);
                backend.Add(dWhBuffer, weightGradBuffer, dWhBuffer, _hiddenSize * _inputSize);

                // dUh += (r * h_prev)^T @ dHCandidate
                using var rGatedH = backend.AllocateBuffer(hiddenBufferSize);
                backend.Multiply(r_t.Buffer, h_prev.Buffer, rGatedH, hiddenBufferSize);
                backend.Transpose(rGatedH, hiddenTranspose, batchSize, _hiddenSize);
                backend.Gemm(hiddenTranspose, dHCandidate, weightGradBuffer, _hiddenSize, _hiddenSize, batchSize);
                backend.Add(dUhBuffer, weightGradBuffer, dUhBuffer, _hiddenSize * _hiddenSize);

                // dBh += sum(dHCandidate, axis=0)
                backend.Transpose(dHCandidate, gateGradTranspose, batchSize, _hiddenSize);
                backend.SumAxis(gateGradTranspose, tempBuffer1, _hiddenSize, batchSize);
                backend.Add(dBhBuffer, tempBuffer1, dBhBuffer, _hiddenSize);

                // Compute input gradient for this timestep
                // dX = dZ @ Wz + dR @ Wr + dHCandidate @ Wh
                var inputGradSlice = backend.AllocateBuffer(inputBufferSize);
                backend.Gemm(dZ, WzT, inputGradSlice, batchSize, _inputSize, _hiddenSize);
                backend.Gemm(dR, WrT, inputGradTemp, batchSize, _inputSize, _hiddenSize);
                backend.Add(inputGradSlice, inputGradTemp, inputGradSlice, inputBufferSize);
                backend.Gemm(dHCandidate, WhT, inputGradTemp, batchSize, _inputSize, _hiddenSize);
                backend.Add(inputGradSlice, inputGradTemp, inputGradSlice, inputBufferSize);

                // Store input gradient
                backend.Copy2DStrided(inputGradSlice, inputGradientBuffer, 1, inputBufferSize, inputGradientSize, inputOffset);
                inputGradSlice.Dispose();

                // Compute dH_next for previous timestep
                // dH_prev = dh * z + dZ @ Uz + dR @ Ur + (dHCandidate * r) @ Uh
                backend.Multiply(dHNext, z_t.Buffer, dHPrev, hiddenBufferSize);
                backend.Gemm(dZ, UzT, hiddenGradTemp, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(dHPrev, hiddenGradTemp, dHPrev, hiddenBufferSize);
                backend.Gemm(dR, UrT, hiddenGradTemp, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(dHPrev, hiddenGradTemp, dHPrev, hiddenBufferSize);
                // Add gradient through (r * h_prev) @ Uh path
                backend.Multiply(dHCandidate, r_t.Buffer, tempBuffer1, hiddenBufferSize);
                backend.Gemm(tempBuffer1, UhT, hiddenGradTemp, batchSize, _hiddenSize, _hiddenSize);
                backend.Add(dHPrev, hiddenGradTemp, dHPrev, hiddenBufferSize);

                // Copy dH_prev to dHNext for next iteration
                backend.Copy(dHPrev, dHNext, hiddenBufferSize);
            }

            // Store gradient tensors for UpdateParametersGpu
            _gpuWzGradient = new GpuTensor<T>(backend, dWzBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWrGradient = new GpuTensor<T>(backend, dWrBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuWhGradient = new GpuTensor<T>(backend, dWhBuffer, [_hiddenSize, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuUzGradient = new GpuTensor<T>(backend, dUzBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuUrGradient = new GpuTensor<T>(backend, dUrBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuUhGradient = new GpuTensor<T>(backend, dUhBuffer, [_hiddenSize, _hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBzGradient = new GpuTensor<T>(backend, dBzBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBrGradient = new GpuTensor<T>(backend, dBrBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);
            _gpuBhGradient = new GpuTensor<T>(backend, dBhBuffer, [_hiddenSize], GpuTensorRole.Gradient, ownsBuffer: true);

            // Cleanup temporary buffers
            dHNext.Dispose();
            dHPrev.Dispose();
            tempBuffer1.Dispose();
            tempBuffer2.Dispose();
            dZ.Dispose();
            dR.Dispose();
            dHCandidate.Dispose();
            inputTranspose.Dispose();
            hiddenTranspose.Dispose();
            gateGradTranspose.Dispose();
            weightGradBuffer.Dispose();
            inputGradTemp.Dispose();
            hiddenGradTemp.Dispose();

            return new GpuTensor<T>(backend, inputGradientBuffer, [batchSize, timeSteps, _inputSize], GpuTensorRole.Gradient, ownsBuffer: true);
        }
        catch
        {
            dWzBuffer.Dispose();
            dWrBuffer.Dispose();
            dWhBuffer.Dispose();
            dUzBuffer.Dispose();
            dUrBuffer.Dispose();
            dUhBuffer.Dispose();
            dBzBuffer.Dispose();
            dBrBuffer.Dispose();
            dBhBuffer.Dispose();
            inputGradientBuffer.Dispose();
            dHNext.Dispose();
            dHPrev.Dispose();
            tempBuffer1.Dispose();
            tempBuffer2.Dispose();
            dZ.Dispose();
            dR.Dispose();
            dHCandidate.Dispose();
            inputTranspose.Dispose();
            hiddenTranspose.Dispose();
            gateGradTranspose.Dispose();
            weightGradBuffer.Dispose();
            inputGradTemp.Dispose();
            hiddenGradTemp.Dispose();
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

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuWzGradient == null || _gpuWrGradient == null || _gpuWhGradient == null ||
            _gpuUzGradient == null || _gpuUrGradient == null || _gpuUhGradient == null ||
            _gpuBzGradient == null || _gpuBrGradient == null || _gpuBhGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuWz ??= new GpuTensor<T>(backend, _Wz, GpuTensorRole.Weight);
        _gpuWr ??= new GpuTensor<T>(backend, _Wr, GpuTensorRole.Weight);
        _gpuWh ??= new GpuTensor<T>(backend, _Wh, GpuTensorRole.Weight);
        _gpuUz ??= new GpuTensor<T>(backend, _Uz, GpuTensorRole.Weight);
        _gpuUr ??= new GpuTensor<T>(backend, _Ur, GpuTensorRole.Weight);
        _gpuUh ??= new GpuTensor<T>(backend, _Uh, GpuTensorRole.Weight);
        _gpuBz ??= new GpuTensor<T>(backend, _bz, GpuTensorRole.Bias);
        _gpuBr ??= new GpuTensor<T>(backend, _br, GpuTensorRole.Bias);
        _gpuBh ??= new GpuTensor<T>(backend, _bh, GpuTensorRole.Bias);

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
            _gpuWzVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWrVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUzVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUrVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBzVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBrVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBhVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            var inputWeightSize = _hiddenSize * _inputSize;
            var hiddenWeightSize = _hiddenSize * _hiddenSize;
            var biasSize = _hiddenSize;

            // First moment (M) tensors
            _gpuWzM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWrM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUzM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUrM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBzM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBrM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBhM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);

            // Second moment (V) tensors
            _gpuWzV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWrV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuWhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([inputWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUzV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUrV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuUhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([hiddenWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBzV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBrV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBhV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
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
